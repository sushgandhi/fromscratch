"""
Summary tools for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolInput, ToolOutput
from src.utils.data_utils import get_workbook_sheets, read_excel_file, save_to_s3, convert_numpy_types
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template


class ColumnSummaryInput(ToolInput):
    """Input model for column summary tool"""
    columns: Optional[List[str]] = Field(None, description="Specific columns to summarize")


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
            
            summary = self._generate_column_summary(df_to_summarize)
            
            if input_data.output_path:
                save_to_s3(summary, input_data.output_path, format='json')
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=input_data.output_path,
                    metadata={"columns_analyzed": len(df_to_summarize.columns)}
                )
            else:
                return ToolOutput(
                    success=True,
                    data=summary,
                    metadata={"columns_analyzed": len(df_to_summarize.columns)}
                )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Column summary failed: {str(e)}"
            )
    
    def _generate_column_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed column summary"""
        summary = {}
        
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
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                text_data = col_data.dropna().astype(str)
                if len(text_data) > 0:
                    col_summary.update({
                        "avg_length": text_data.str.len().mean(),
                        "min_length": text_data.str.len().min(),
                        "max_length": text_data.str.len().max(),
                        "most_common": text_data.value_counts().head(5).to_dict()
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
        
        return summary


class RowSummaryInput(ToolInput):
    """Input model for row summary tool"""
    sample_size: int = Field(default=10, description="Number of sample rows to include")


class RowSummaryTool(BaseTool):
    """Tool for generating row summaries"""
    
    def __init__(self):
        super().__init__()
        self.description = "Generate row-level summaries and samples"
    
    def execute(self, input_data: RowSummaryInput) -> ToolOutput:
        """Generate row summary"""
        try:
            df = self.load_data(input_data)
            
            # Generate summary
            summary = self._generate_row_summary(df, input_data.sample_size)
            
            # Return results
            if input_data.output_path:
                save_to_s3(summary, input_data.output_path, format='json')
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=input_data.output_path,
                    metadata={"total_rows": len(df)}
                )
            else:
                return ToolOutput(
                    success=True,
                    data=summary,
                    metadata={"total_rows": len(df)}
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


class SheetSummaryInput(ToolInput):
    """Input model for sheet summary tool"""
    include_visualizations: bool = Field(default=True, description="Whether to include basic visualizations")


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
            summary = self._generate_sheet_summary(df, input_data.include_visualizations)
            
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
    
    def _generate_sheet_summary(self, df: pd.DataFrame, include_viz: bool) -> Dict[str, Any]:
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