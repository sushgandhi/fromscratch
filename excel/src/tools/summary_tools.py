"""
Summary tools for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import logging
from .base_tool import BaseTool, ToolInput, ToolOutput
from src.utils.data_utils import get_workbook_sheets, read_excel_file, save_to_s3, convert_numpy_types
from src.utils.claude_client import get_claude_client
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from langchain.schema import HumanMessage

# Configure logging
logger = logging.getLogger(__name__)

# Constants for LLM processing
DEFAULT_SAMPLE_ROWS = 100
MAX_TEXT_SAMPLE_SIZE = 50


class ColumnSummaryInput(ToolInput):
    """Input model for column summary tool"""
    columns: Optional[List[str]] = Field(None, description="Specific columns to summarize")
    use_llm_for_text: bool = Field(default=True, description="Use LLM to analyze text columns")
    text_sample_size: int = Field(default=MAX_TEXT_SAMPLE_SIZE, description="Number of text samples to send to LLM")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class ColumnSummaryTool(BaseTool):
    """Tool for generating column summaries"""
    
    def __init__(self):
        super().__init__()
        self.description = "Generate statistical summaries for columns"
    
    def execute(self, input_data: ColumnSummaryInput) -> ToolOutput:
        """Generate column summary"""
        try:
            df = self.load_data(input_data)
            
            if input_data.columns:
                missing_columns = [col for col in input_data.columns if col not in df.columns]
                if missing_columns:
                    return ToolOutput(
                        success=False,
                        error_message=f"Columns not found: {missing_columns}"
                    )
                df_to_summarize = df[input_data.columns]
            else:
                df_to_summarize = df
            
            summary = self._generate_column_summary(df_to_summarize, input_data.use_llm_for_text, input_data.text_sample_size)
            
            # Summary tools return analysis results, not tabular data
            # Just return compact summary info to avoid context overflow
            compact_summary = {
                "summary_type": "column_analysis",
                "columns_analyzed": list(df_to_summarize.columns),
                "total_columns": len(df_to_summarize.columns),
                "analysis_complete": True
            }
            
            if input_data.output_path:
                save_to_s3(summary, input_data.output_path, format='json')
                return ToolOutput(
                    success=True,
                    data=compact_summary,  # Compact summary only
                    output_path=input_data.output_path,
                    metadata={
                        "columns_analyzed": len(df_to_summarize.columns),
                        "tool_type": "column_summary",
                        "full_summary_path": input_data.output_path
                    }
                )
            else:
                return ToolOutput(
                    success=True,
                    data=compact_summary,  # Compact summary only
                    metadata={
                        "columns_analyzed": len(df_to_summarize.columns),
                        "tool_type": "column_summary"
                    }
                )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Column summary failed: {str(e)}"
            )
    
    def _generate_column_summary(self, df: pd.DataFrame, use_llm_for_text: bool = True, text_sample_size: int = MAX_TEXT_SAMPLE_SIZE) -> Dict[str, Any]:
        """Generate detailed column summary with LLM analysis for text columns"""
        summary = {}
        text_columns_for_llm = []
        
        for column in df.columns:
            col_data = df[column]
            col_summary = {
                "name": column,
                "data_type": str(col_data.dtype),
                "total_count": len(col_data),
                "non_null_count": col_data.count(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
                "unique_count": col_data.nunique()
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                col_summary.update({
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "q25": col_data.quantile(0.25),
                    "q75": col_data.quantile(0.75)
                })
                
                # Add textual summary for numeric columns too
                col_summary["textual_summary"] = (
                    f"The column '{column}' has {len(col_data)} values with a mean of {col_data.mean():.2f} "
                    f"and a standard deviation of {col_data.std():.2f}. Values range from {col_data.min():.2f} to {col_data.max():.2f}."
                )
                
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                text_data = col_data.dropna().astype(str)
                if len(text_data) > 0:
                    col_summary.update({
                        "avg_length": text_data.str.len().mean(),
                        "min_length": text_data.str.len().min(),
                        "max_length": text_data.str.len().max(),
                        "most_common": text_data.value_counts().head(5).to_dict()
                    })
                    
                    # Prepare for LLM analysis if enabled
                    if use_llm_for_text and len(text_data) > 0:
                        sample_values = text_data.head(text_sample_size).tolist()
                        text_columns_for_llm.append({
                            "column_name": column,
                            "sample_values": sample_values,
                            "total_count": len(text_data),
                            "unique_count": col_data.nunique(),
                            "avg_length": text_data.str.len().mean()
                        })
                        
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                date_data = col_data.dropna()
                if len(date_data) > 0:
                    col_summary.update({
                        "earliest_date": date_data.min(),
                        "latest_date": date_data.max(),
                        "date_range_days": (date_data.max() - date_data.min()).days
                    })
            
            summary[column] = convert_numpy_types(col_summary)
        
        # Generate LLM summaries for text columns
        if use_llm_for_text and text_columns_for_llm:
            llm_summaries = self._generate_llm_column_summaries(text_columns_for_llm)
            
            # Merge LLM summaries back into main summary
            for col_name, llm_summary in llm_summaries.items():
                if col_name in summary:
                    summary[col_name]["llm_summary"] = llm_summary
        
        return summary
    
    def _generate_llm_column_summaries(self, text_columns: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate LLM-based summaries for text columns"""
        try:
            # Get Claude client
            llm = get_claude_client()
            
            # Prepare prompt for LLM
            prompt = f"""Analyze the following text columns and provide concise summaries for each column based on the provided sample values. Each column summary should capture the key information, patterns, and purpose of that column.

Return the summaries in the following JSON format:
{{
    "column_summaries": {{
        "column_name_1": "Summary of column 1...",
        "column_name_2": "Summary of column 2..."
    }}
}}

Important guidelines:
1. The JSON must be valid and properly formatted
2. All keys and string values must use double quotes (")
3. Any text containing special characters should be properly escaped
4. Keep summaries concise but informative
5. Focus on what the column represents and key patterns

Columns to analyze:
{json.dumps(text_columns, indent=2)}

Provide only the JSON response, no other text."""

            # Call LLM
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            content = response.content
            
            # Parse JSON response
            try:
                # Handle various response formats
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                else:
                    json_content = content.strip()
                
                llm_result = json.loads(json_content)
                return llm_result.get("column_summaries", {})
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM JSON response: {e}")
                logger.error(f"Raw response: {content}")
                return {}
                
        except Exception as e:
            logger.error(f"Error generating LLM summaries for columns: {e}")
            return {}


class RowSummaryInput(ToolInput):
    """Input model for row summary tool"""
    sample_size: int = Field(default=10, description="Number of sample rows to include")
    create_summary_column: bool = Field(default=False, description="Create a new column with row-by-row LLM summaries")
    summary_column_name: str = Field(default="row_summary", description="Name for the new summary column")
    max_rows_for_llm: int = Field(default=50, description="Maximum number of rows to process with LLM")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class RowSummaryTool(BaseTool):
    """Tool for generating row summaries"""
    
    def __init__(self):
        super().__init__()
        self.description = "Generate row-level summaries and samples"
    
    def execute(self, input_data: RowSummaryInput) -> ToolOutput:
        """Generate row summary and optionally create summary column"""
        try:
            df = self.load_data(input_data)
            
            # Generate basic row summary
            summary = self._generate_row_summary(df, input_data.sample_size)
            
            # Create summary column if requested
            result_data = df.copy()
            if input_data.create_summary_column:
                summary_column = self._generate_row_summaries_column(
                    df, 
                    input_data.summary_column_name,
                    input_data.max_rows_for_llm
                )
                result_data[input_data.summary_column_name] = summary_column
                summary["summary_column_created"] = True
                summary["summary_column_name"] = input_data.summary_column_name
            
            # Handle results based on whether we created new data or just analysis
            if input_data.create_summary_column:
                # We created new columns - treat like data transformation tools
                if input_data.operation_id:
                    # Use new save_with_summary method for large datasets
                    result = self.save_with_summary(result_data, input_data.operation_id)
                    result.metadata.update({
                        "total_rows": len(df),
                        "summary_column_created": True,
                        "summary_column_name": input_data.summary_column_name,
                        "tool_type": "row_summary"
                    })
                    return result
                else:
                    # Fallback to old method
                    return ToolOutput(
                        success=True,
                        data=result_data.to_dict('records'),
                        output_path=None,
                        metadata={
                            "total_rows": len(df),
                            "summary_column_created": True,
                            "summary_column_name": input_data.summary_column_name,
                            "tool_type": "row_summary"
                        }
                    )
            else:
                # Just analysis - return compact summary
                compact_summary = {
                    "summary_type": "row_analysis",
                    "total_rows": len(df),
                    "analysis_complete": True
                }
                
                if input_data.output_path:
                    save_to_s3(summary, input_data.output_path, format='json')
                    return ToolOutput(
                        success=True,
                        data=compact_summary,
                        output_path=input_data.output_path,
                        metadata={
                            "total_rows": len(df),
                            "summary_column_created": False,
                            "tool_type": "row_summary",
                            "full_summary_path": input_data.output_path
                        }
                    )
                else:
                    return ToolOutput(
                        success=True,
                        data=compact_summary,
                        metadata={
                            "total_rows": len(df),
                            "summary_column_created": False,
                            "tool_type": "row_summary"
                        }
                    )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Row summary failed: {str(e)}"
            )
    
    def _generate_row_summary(self, df: pd.DataFrame, sample_size: int) -> Dict[str, Any]:
        """Generate row summary"""
        summary = {
            "total_rows": len(df),
            "complete_rows": len(df.dropna()),
            "incomplete_rows": len(df) - len(df.dropna()),
            "duplicate_rows": df.duplicated().sum(),
            "sample_rows": convert_numpy_types(df.head(sample_size).to_dict('records')),
            "rows_with_most_nulls": []
        }
        
        # Find rows with most null values
        null_counts = df.isnull().sum(axis=1)
        worst_rows = null_counts.nlargest(5)
        for idx, null_count in worst_rows.items():
            if null_count > 0:
                summary["rows_with_most_nulls"].append({
                    "row_index": int(idx),
                    "null_count": int(null_count),
                    "data": convert_numpy_types(df.iloc[idx].to_dict())
                })
        
        return summary
    
    def _generate_row_summaries_column(self, df: pd.DataFrame, column_name: str, max_rows: int) -> List[str]:
        """Generate LLM-based summaries for each row and return as a list"""
        try:
            # Get Claude client
            llm = get_claude_client()
            
            # Limit rows to prevent excessive API calls
            rows_to_process = min(len(df), max_rows)
            df_subset = df.head(rows_to_process)
            
            # Prepare column information for context
            column_info = []
            for col in df.columns:
                col_info = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "description": f"Column '{col}' of type {df[col].dtype}"
                }
                column_info.append(col_info)
            
            # Process rows in batches to avoid token limits
            batch_size = 10
            summaries = []
            
            for batch_start in range(0, rows_to_process, batch_size):
                batch_end = min(batch_start + batch_size, rows_to_process)
                batch_rows = df_subset.iloc[batch_start:batch_end].to_dict('records')
                
                # Prepare batch prompt
                prompt = f"""You are analyzing a dataset and need to create concise summaries for each row. 

Dataset Context:
- Total columns: {len(df.columns)}
- Column information: {json.dumps(column_info, indent=2)}

Please create a brief, informative summary for each row that captures the key information in that row. Each summary should be 1-2 sentences max.

Return your response in the following JSON format:
{{
    "row_summaries": [
        "Summary for row 1...",
        "Summary for row 2...",
        "Summary for row 3..."
    ]
}}

Rows to summarize (batch {batch_start//batch_size + 1}):
{json.dumps(batch_rows, indent=2, default=str)}

Guidelines:
1. Keep summaries concise but informative
2. Focus on the most important/distinctive information in each row
3. Use natural language that explains what the row represents
4. Return exactly {len(batch_rows)} summaries in the JSON array
5. Return only valid JSON

Provide only the JSON response, no other text."""

                # Call LLM for this batch
                messages = [HumanMessage(content=prompt)]
                response = llm.invoke(messages)
                content = response.content
                
                # Parse batch response
                try:
                    # Handle various response formats
                    if "```json" in content:
                        json_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_content = content.split("```")[1].split("```")[0].strip()
                    else:
                        json_content = content.strip()
                    
                    batch_result = json.loads(json_content)
                    batch_summaries = batch_result.get("row_summaries", [])
                    
                    # Ensure we have the right number of summaries
                    if len(batch_summaries) != len(batch_rows):
                        logger.warning(f"Expected {len(batch_rows)} summaries, got {len(batch_summaries)}")
                        # Pad with generic summaries if needed
                        while len(batch_summaries) < len(batch_rows):
                            batch_summaries.append("Row data summary not available")
                    
                    summaries.extend(batch_summaries[:len(batch_rows)])
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing batch LLM response: {e}")
                    # Add generic summaries for failed batch
                    for _ in range(len(batch_rows)):
                        summaries.append("Summary generation failed")
                
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Add generic summaries for failed batch
                    for _ in range(len(batch_rows)):
                        summaries.append("Summary generation failed")
            
            # Pad with empty summaries for remaining rows if we processed fewer than total
            while len(summaries) < len(df):
                summaries.append("Row not processed (exceeds limit)")
            
            return summaries[:len(df)]  # Ensure exact length match
            
        except Exception as e:
            logger.error(f"Error generating row summaries: {e}")
            # Return generic summaries if LLM fails
            return [f"Row {i+1} data" for i in range(len(df))]


class SheetSummaryInput(ToolInput):
    """Input model for sheet summary tool"""
    include_visualizations: bool = Field(default=True, description="Whether to include basic visualizations")
    use_llm_analysis: bool = Field(default=True, description="Use LLM for intelligent data analysis")
    sample_rows: int = Field(default=DEFAULT_SAMPLE_ROWS, description="Number of sample rows for LLM analysis")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class SheetSummaryTool(BaseTool):
    """Tool for generating sheet summaries with numerical and textual analysis"""
    
    def __init__(self):
        super().__init__()
        self.description = "Generate comprehensive sheet summaries with statistics and visualizations"
    
    def execute(self, input_data: SheetSummaryInput) -> ToolOutput:
        """Generate sheet summary"""
        try:
            df = self.load_data(input_data)
            
            # Generate comprehensive summary
            summary = self._generate_sheet_summary(df, input_data.include_visualizations, input_data.use_llm_analysis, input_data.sample_rows)
            
            # Return results
            if input_data.output_path:
                # Save as HTML for rich formatting
                html_path = input_data.output_path.replace('.csv', '.html')
                html_content = self._create_summary_html(summary)
                save_to_s3(html_content, html_path, format='html')
                
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=html_path,
                    metadata={"summary_type": "sheet", "total_rows": len(df), "total_columns": len(df.columns)}
                )
            else:
                # For local execution, save HTML locally and return basic summary data
                local_html_path = f"outputs/sheet_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
                html_content = self._create_summary_html(summary)
                
                # Create output directory
                import os
                os.makedirs("outputs", exist_ok=True)
                
                # Save HTML locally
                with open(local_html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Return summary data as list for Excel workbook compatibility
                summary_list = [
                    {"metric": "total_rows", "value": summary["basic_info"]["rows"]},
                    {"metric": "total_columns", "value": summary["basic_info"]["columns"]},
                    {"metric": "complete_rows", "value": summary["data_quality"]["complete_rows"]},
                    {"metric": "null_percentage", "value": round(summary["data_quality"]["null_percentage"], 2)},
                    {"metric": "numeric_columns", "value": summary["column_analysis"]["numeric_columns"]},
                    {"metric": "text_columns", "value": summary["column_analysis"]["text_columns"]}
                ]
                
                return ToolOutput(
                    success=True,
                    data=summary_list,
                    output_path=local_html_path,
                    metadata={
                        "summary_type": "sheet", 
                        "total_rows": len(df), 
                        "total_columns": len(df.columns),
                        "html_path": local_html_path,
                        "full_summary": summary
                    }
                )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Sheet summary failed: {str(e)}"
            )
    
    def _generate_sheet_summary(self, df: pd.DataFrame, include_viz: bool, use_llm_analysis: bool = True, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> Dict[str, Any]:
        """Generate comprehensive sheet summary"""
        
        # Basic statistics
        summary = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "total_cells": len(df) * len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "data_types": df.dtypes.value_counts().to_dict()
            },
            "data_quality": {
                "complete_rows": len(df.dropna()),
                "incomplete_rows": len(df) - len(df.dropna()),
                "duplicate_rows": df.duplicated().sum(),
                "total_nulls": df.isnull().sum().sum(),
                "null_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            "column_analysis": {},
            "correlations": {},
            "visualizations": {}
        }
        
        # Column analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        text_columns = df.select_dtypes(include=[object]).columns
        
        summary["column_analysis"] = {
            "numeric_columns": len(numeric_columns),
            "text_columns": len(text_columns),
            "date_columns": len(df.select_dtypes(include=['datetime']).columns),
            "numeric_summary": convert_numpy_types(df[numeric_columns].describe().to_dict()) if len(numeric_columns) > 0 else {},
        }
        
        # Correlations for numeric data
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            summary["correlations"] = {
                "correlation_matrix": convert_numpy_types(corr_matrix.to_dict()),
                "strongest_correlations": self._find_strong_correlations(corr_matrix)
            }
        
        # Generate visualizations if requested
        if include_viz and len(df) > 0:
            summary["visualizations"] = self._generate_basic_visualizations(df)
        
        # Generate LLM analysis if requested
        if use_llm_analysis and len(df) > 0:
            summary["llm_analysis"] = self._generate_llm_sheet_analysis(df, sample_rows)
        
        return convert_numpy_types(summary)
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations between columns"""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corrs.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return strong_corrs
    
    def _generate_basic_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate basic visualizations as HTML strings"""
        visualizations = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Distribution of numeric columns
        if len(numeric_columns) > 0:
            for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                visualizations[f"{col}_distribution"] = fig.to_html(include_plotlyjs=True)
        
        # Correlation heatmap
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap")
            visualizations["correlation_heatmap"] = fig.to_html(include_plotlyjs=True)
        
        return visualizations
    
    def _generate_llm_sheet_analysis(self, df: pd.DataFrame, sample_rows: int) -> Dict[str, str]:
        """Generate LLM-based intelligent analysis of the entire sheet"""
        try:
            # Get Claude client
            llm = get_claude_client()
            
            # Prepare sample data for LLM
            sample_data = df.head(sample_rows).to_dict('records')
            column_info = []
            
            for col in df.columns:
                col_data = df[col]
                col_info = {
                    "name": col,
                    "type": str(col_data.dtype),
                    "sample_values": col_data.dropna().head(10).tolist()
                }
                column_info.append(col_info)
            
            # Prepare comprehensive analysis prompt
            prompt = f"""Analyze this dataset and provide intelligent insights. You are looking at a dataset with {len(df)} rows and {len(df.columns)} columns.

Column Information:
{json.dumps(column_info, indent=2, default=str)}

Sample Data (first {min(sample_rows, len(df))} rows):
{json.dumps(sample_data, indent=2, default=str)}

Please provide a comprehensive analysis in the following JSON format:
{{
    "data_insights": {{
        "overall_purpose": "What this dataset appears to be about and its main purpose",
        "key_patterns": "Main patterns or trends you notice in the data",
        "data_quality_assessment": "Assessment of data quality, completeness, and potential issues",
        "business_insights": "Business or domain-specific insights from the data",
        "recommendations": "Recommendations for further analysis or data improvements"
    }}
}}

Guidelines:
1. Be specific and actionable in your insights
2. Focus on business value and practical implications
3. Identify potential data quality issues or anomalies
4. Suggest meaningful analysis that could be performed
5. Keep responses concise but informative
6. Return only valid JSON

Provide only the JSON response, no other text."""

            # Call LLM
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            content = response.content
            
            # Parse JSON response
            try:
                # Handle various response formats
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                else:
                    json_content = content.strip()
                
                llm_result = json.loads(json_content)
                return llm_result.get("data_insights", {})
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM sheet analysis JSON: {e}")
                logger.error(f"Raw response: {content}")
                return {"error": "Failed to parse LLM analysis"}
                
        except Exception as e:
            logger.error(f"Error generating LLM sheet analysis: {e}")
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _create_summary_html(self, summary: Dict[str, Any]) -> str:
        """Create HTML representation of summary"""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sheet Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Sheet Summary Report</h1>
            
            <div class="section">
                <h2>Basic Information</h2>
                <div class="metric">Rows: {{ summary.basic_info.rows }}</div>
                <div class="metric">Columns: {{ summary.basic_info.columns }}</div>
                <div class="metric">Total Cells: {{ summary.basic_info.total_cells }}</div>
            </div>
            
            <div class="section">
                <h2>Data Quality</h2>
                <div class="metric">Complete Rows: {{ summary.data_quality.complete_rows }}</div>
                <div class="metric">Incomplete Rows: {{ summary.data_quality.incomplete_rows }}</div>
                <div class="metric">Duplicate Rows: {{ summary.data_quality.duplicate_rows }}</div>
                <div class="metric">Null Percentage: {{ "%.2f"|format(summary.data_quality.null_percentage) }}%</div>
            </div>
            
            {% if summary.visualizations %}
            <div class="section">
                <h2>Visualizations</h2>
                {% for viz_name, viz_html in summary.visualizations.items() %}
                    <h3>{{ viz_name.replace('_', ' ').title() }}</h3>
                    {{ viz_html|safe }}
                {% endfor %}
            </div>
            {% endif %}
        </body>
        </html>
        """)
        
        return template.render(summary=summary)


class WorkbookSummaryInput(ToolInput):
    """Input model for workbook summary tool"""
    include_visualizations: bool = Field(default=True, description="Whether to include visualizations")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class WorkbookSummaryTool(BaseTool):
    """Tool for generating complete workbook summaries"""
    
    def __init__(self):
        super().__init__()
        self.description = "Generate comprehensive workbook summaries with all sheets analyzed"
    
    def execute(self, input_data: WorkbookSummaryInput) -> ToolOutput:
        """Generate workbook summary"""
        try:
            # For workbook summary, we need a file path
            if not input_data.data_path:
                return ToolOutput(
                    success=False,
                    error_message="Workbook summary requires data_path (cannot work with data list)"
                )
            
            # Get all sheets
            sheet_names = get_workbook_sheets(input_data.data_path)
            
            # Generate summary for each sheet
            workbook_summary = {
                "workbook_info": {
                    "total_sheets": len(sheet_names),
                    "sheet_names": sheet_names
                },
                "sheet_summaries": {},
                "workbook_visualizations": {}
            }
            
            # Analyze each sheet
            for sheet_name in sheet_names:
                try:
                    df = read_excel_file(input_data.data_path, sheet_name)
                    sheet_tool = SheetSummaryTool()
                    sheet_input = SheetSummaryInput(
                        data=None,
                        data_path=None,
                        include_visualizations=input_data.include_visualizations
                    )
                    # Manually set the data for the sheet tool
                    sheet_summary = sheet_tool._generate_sheet_summary(df, input_data.include_visualizations)
                    workbook_summary["sheet_summaries"][sheet_name] = sheet_summary
                    
                except Exception as e:
                    workbook_summary["sheet_summaries"][sheet_name] = {
                        "error": f"Failed to analyze sheet: {str(e)}"
                    }
            
            # Generate workbook-level visualizations
            if input_data.include_visualizations:
                workbook_summary["workbook_visualizations"] = self._generate_workbook_visualizations(workbook_summary)
            
            # Return results
            if input_data.output_path:
                html_path = input_data.output_path.replace('.csv', '.html')
                html_content = self._create_workbook_html(workbook_summary)
                save_to_s3(html_content, html_path, format='html')
                
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=html_path,
                    metadata={"summary_type": "workbook", "total_sheets": len(sheet_names)}
                )
            else:
                return ToolOutput(
                    success=True,
                    data=workbook_summary,
                    metadata={"summary_type": "workbook", "total_sheets": len(sheet_names)}
                )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Workbook summary failed: {str(e)}"
            )
    
    def _generate_workbook_visualizations(self, workbook_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate workbook-level visualizations"""
        visualizations = {}
        
        # Sheet size comparison
        sheet_data = []
        for sheet_name, summary in workbook_summary["sheet_summaries"].items():
            if "error" not in summary:
                sheet_data.append({
                    "Sheet": sheet_name,
                    "Rows": summary["basic_info"]["rows"],
                    "Columns": summary["basic_info"]["columns"]
                })
        
        if sheet_data:
            df_sheets = pd.DataFrame(sheet_data)
            
            # Sheet size comparison bar chart
            fig_sizes = px.bar(df_sheets, x="Sheet", y=["Rows", "Columns"], 
                              title="Sheet Size Comparison", barmode="group")
            visualizations["sheet_sizes"] = fig_sizes.to_html(include_plotlyjs=True)
        
        return visualizations
    
    def _create_workbook_html(self, workbook_summary: Dict[str, Any]) -> str:
        """Create HTML representation of workbook summary"""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Workbook Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .sheet-section { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Workbook Summary Report</h1>
            
            <div class="section">
                <h2>Workbook Overview</h2>
                <div class="metric">Total Sheets: {{ workbook_summary.workbook_info.total_sheets }}</div>
                <div class="metric">Sheet Names: {{ ", ".join(workbook_summary.workbook_info.sheet_names) }}</div>
            </div>
            
            {% if workbook_summary.workbook_visualizations %}
            <div class="section">
                <h2>Workbook Visualizations</h2>
                {% for viz_name, viz_html in workbook_summary.workbook_visualizations.items() %}
                    {{ viz_html|safe }}
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Sheet Details</h2>
                {% for sheet_name, sheet_summary in workbook_summary.sheet_summaries.items() %}
                    <div class="sheet-section">
                        <h3>{{ sheet_name }}</h3>
                        {% if "error" in sheet_summary %}
                            <p style="color: red;">Error: {{ sheet_summary.error }}</p>
                        {% else %}
                            <div class="metric">Rows: {{ sheet_summary.basic_info.rows }}</div>
                            <div class="metric">Columns: {{ sheet_summary.basic_info.columns }}</div>
                            <div class="metric">Null %: {{ "%.2f"|format(sheet_summary.data_quality.null_percentage) }}%</div>
                        {% endif %}
                    </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """)
        
        return template.render(workbook_summary=workbook_summary) 