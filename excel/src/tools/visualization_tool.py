"""
Visualization tool for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import json
from .base_tool import BaseTool, ToolInput, ToolOutput
from src.utils.data_utils import save_to_s3


class VisualizationInput(ToolInput):
    """Input model for visualization tool"""
    chart_type: str = Field(description="Type of chart: bar, line, scatter, pie, histogram, box, heatmap")
    x_column: Optional[str] = Field(None, description="Column for x-axis")
    y_column: Optional[str] = Field(None, description="Column for y-axis")
    color_column: Optional[str] = Field(None, description="Column for color grouping")
    size_column: Optional[str] = Field(None, description="Column for size (scatter plots)")
    title: Optional[str] = Field(None, description="Chart title")
    x_title: Optional[str] = Field(None, description="X-axis title")
    y_title: Optional[str] = Field(None, description="Y-axis title")
    width: int = Field(default=800, description="Chart width")
    height: int = Field(default=600, description="Chart height")
    in_memory_only: bool = Field(default=False, description="If True, don't save files, only return content in metadata")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class VisualizationTool(BaseTool):
    """Tool for creating visualizations"""
    
    def __init__(self):
        super().__init__()
        self.description = "Create interactive visualizations using Plotly"
        self.valid_chart_types = ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap"]
    
    def execute(self, input_data: VisualizationInput) -> ToolOutput:
        """
        Execute visualization operation
        
        Args:
            input_data: Visualization parameters
            
        Returns:
            Visualization output (HTML and JSON)
        """
        try:
            # Load data
            df = self.load_data(input_data)
            
            # Validate chart type
            if input_data.chart_type not in self.valid_chart_types:
                return ToolOutput(
                    success=False,
                    error_message=f"Invalid chart type '{input_data.chart_type}'. Valid types: {self.valid_chart_types}"
                )
            
            # Validate and correct column names
            required_columns = []
            if input_data.x_column:
                required_columns.append(input_data.x_column)
            if input_data.y_column:
                required_columns.append(input_data.y_column)
            if input_data.color_column:
                required_columns.append(input_data.color_column)
            if input_data.size_column:
                required_columns.append(input_data.size_column)
            
            if required_columns:
                corrected_columns, error_message = self.validate_and_correct_columns(required_columns, df)
                if error_message:
                    return ToolOutput(
                        success=False,
                        error_message=f"Visualization columns error: {error_message}"
                    )
                
                # Update input_data with corrected columns if any corrections were made
                if corrected_columns != required_columns:
                    print(f"Auto-corrected visualization columns: {required_columns} -> {corrected_columns}")
                    
                    # Map back to specific columns
                    idx = 0
                    if input_data.x_column:
                        input_data.x_column = corrected_columns[idx]
                        idx += 1
                    if input_data.y_column:
                        input_data.y_column = corrected_columns[idx]
                        idx += 1
                    if input_data.color_column:
                        input_data.color_column = corrected_columns[idx]
                        idx += 1
                    if input_data.size_column:
                        input_data.size_column = corrected_columns[idx]
            
            # Log data info before preprocessing
            print(f"🔍 Visualization Debug - Before preprocessing:")
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Data types: {dict(df.dtypes)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data: {df.head(2).to_dict('records')}")
            
            # Preprocess data for visualization
            df = self._preprocess_data_for_visualization(df, input_data)
            
            # Log data info after preprocessing
            print(f"🔍 Visualization Debug - After preprocessing:")
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Data types: {dict(df.dtypes)}")
            print(f"   Sample data: {df.head(2).to_dict('records')}")
            print(f"   Chart params: x={input_data.x_column}, y={input_data.y_column}, type={input_data.chart_type}")
            
            # Create visualization
            fig = self._create_visualization(df, input_data)
            
            # Convert to HTML and JSON
            html_content = fig.to_html(include_plotlyjs=True)
            json_content = fig.to_json()
            
            # Save or return results
            if input_data.output_path:
                # Save HTML to S3
                html_path = input_data.output_path.replace('.csv', '.html')
                save_to_s3(html_content, html_path, format='html')
                
                # Save JSON to S3
                json_path = input_data.output_path.replace('.csv', '.json')
                save_to_s3(json_content, json_path, format='json')
                
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=html_path,
                    metadata={
                        "chart_type": input_data.chart_type,
                        "html_path": html_path,
                        "json_path": json_path,
                        "data_points": len(df)
                    }
                )
            else:
                # Check if in-memory only mode
                if input_data.in_memory_only:
                    print(f"🧠 Chart kept in-memory only (no file saved)")
                    
                    return ToolOutput(
                        success=True,
                        data=None,  # Visualization doesn't return tabular data  
                        output_path=None,  # No file saved
                        metadata={
                            "chart_type": input_data.chart_type,
                            "data_points": len(df),
                            "html_content": html_content,  # Full HTML content for API clients
                            "json_content": json_content,  # Full JSON content for API clients
                            "visualization_created": True,
                            "in_memory_only": True
                        }
                    )
                else:
                    # Save locally when no S3 path specified
                    import os
                    from datetime import datetime
                    
                    # Create outputs directory if it doesn't exist
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    chart_title = input_data.title or f"{input_data.chart_type}_chart"
                    safe_title = "".join(c for c in chart_title if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
                    
                    html_filename = f"{safe_title}_{timestamp}.html"
                    html_path = os.path.join(output_dir, html_filename)
                    
                    # Save HTML file
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    print(f"📁 Chart saved locally: {html_path}")
                    
                    return ToolOutput(
                        success=True,
                        data=None,  # Visualization doesn't return tabular data
                        output_path=html_path,
                        metadata={
                            "chart_type": input_data.chart_type,
                            "data_points": len(df),
                            "local_html_path": html_path,
                            "html_content": html_content,  # Include HTML for API clients
                            "json_content": json_content,  # Include JSON for API clients
                            "visualization_created": True
                        }
                    )
                
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Visualization creation failed: {str(e)}"
            )
    
    def _preprocess_data_for_visualization(self, df: pd.DataFrame, input_data: VisualizationInput) -> pd.DataFrame:
        """
        Preprocess data to handle Plotly data type requirements
        
        Args:
            df: Input DataFrame
            input_data: Visualization parameters
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"🔧 Starting data preprocessing for {input_data.chart_type} chart...")
        df_processed = df.copy()
        
        # Get columns that will be used
        columns_to_check = []
        if input_data.x_column:
            columns_to_check.append(input_data.x_column)
        if input_data.y_column:
            columns_to_check.append(input_data.y_column)
        if input_data.color_column:
            columns_to_check.append(input_data.color_column)
        if input_data.size_column:
            columns_to_check.append(input_data.size_column)
        
        # Handle data type conversions for each column
        for col in columns_to_check:
            if col in df_processed.columns:
                original_dtype = df_processed[col].dtype
                print(f"   Processing column '{col}' (original type: {original_dtype})")
                
                # Convert object/mixed types to string for categorical data
                if df_processed[col].dtype == 'object':
                    # Check if it looks like numeric data
                    try:
                        # Try to convert to numeric, if it works, keep as numeric
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='ignore')
                        if df_processed[col].dtype == 'object':
                            # If still object, convert to string
                            df_processed[col] = df_processed[col].astype(str)
                            print(f"   → Converted to string")
                        else:
                            print(f"   → Converted to numeric: {df_processed[col].dtype}")
                    except Exception as e:
                        print(f"   → Error converting, defaulting to string: {e}")
                        df_processed[col] = df_processed[col].astype(str)
                else:
                    print(f"   → Keeping original type: {df_processed[col].dtype}")
                
                # Handle datetime columns for x-axis (common for time series)
                if input_data.x_column == col and input_data.chart_type in ['line', 'bar']:
                    try:
                        # Try to parse as datetime if it looks like dates
                        if df_processed[col].dtype == 'object':
                            sample_val = str(df_processed[col].iloc[0])
                            if any(char in sample_val for char in ['-', '/', ':']):
                                df_processed[col] = pd.to_datetime(df_processed[col], errors='ignore')
                    except:
                        pass
        
        # Ensure numeric columns for y-axis in charts that require it
        if input_data.y_column and input_data.chart_type in ['bar', 'line', 'scatter', 'box']:
            try:
                df_processed[input_data.y_column] = pd.to_numeric(
                    df_processed[input_data.y_column], 
                    errors='coerce'
                )
                # Drop rows with NaN values in y column
                df_processed = df_processed.dropna(subset=[input_data.y_column])
            except:
                pass
        
        # Handle size column for scatter plots
        if input_data.size_column and input_data.chart_type == 'scatter':
            try:
                df_processed[input_data.size_column] = pd.to_numeric(
                    df_processed[input_data.size_column], 
                    errors='coerce'
                )
                # Fill NaN with 1 for size
                df_processed[input_data.size_column] = df_processed[input_data.size_column].fillna(1)
            except:
                pass
        
        print(f"🎯 Preprocessing complete. Final DataFrame:")
        print(f"   Shape: {df_processed.shape}")
        print(f"   Dtypes: {dict(df_processed.dtypes)}")
        if not df_processed.empty:
            print(f"   First row: {df_processed.iloc[0].to_dict()}")
        
        return df_processed
    
    def _create_visualization(self, df: pd.DataFrame, input_data: VisualizationInput):
        """Create the appropriate visualization based on chart type"""
        
        # Set default title if not provided
        title = input_data.title or f"{input_data.chart_type.title()} Chart"
        
        if input_data.chart_type == "bar":
            print(f"📊 Creating bar chart with x='{input_data.x_column}', y='{input_data.y_column}'")
            try:
                fig = px.bar(
                    df, 
                    x=input_data.x_column, 
                    y=input_data.y_column,
                    color=input_data.color_column,
                    title=title,
                    width=input_data.width,
                    height=input_data.height
                )
                print(f"✅ Bar chart created successfully")
            except Exception as e:
                print(f"❌ Bar chart creation failed: {e}")
                print(f"   Data types for chart: x={df[input_data.x_column].dtype}, y={df[input_data.y_column].dtype}")
                print(f"   Sample x values: {df[input_data.x_column].head(3).tolist()}")
                print(f"   Sample y values: {df[input_data.y_column].head(3).tolist()}")
                raise
        
        elif input_data.chart_type == "line":
            fig = px.line(
                df, 
                x=input_data.x_column, 
                y=input_data.y_column,
                color=input_data.color_column,
                title=title,
                width=input_data.width,
                height=input_data.height
            )
        
        elif input_data.chart_type == "scatter":
            fig = px.scatter(
                df, 
                x=input_data.x_column, 
                y=input_data.y_column,
                color=input_data.color_column,
                size=input_data.size_column,
                title=title,
                width=input_data.width,
                height=input_data.height
            )
        
        elif input_data.chart_type == "pie":
            fig = px.pie(
                df, 
                values=input_data.y_column, 
                names=input_data.x_column,
                title=title,
                width=input_data.width,
                height=input_data.height
            )
        
        elif input_data.chart_type == "histogram":
            fig = px.histogram(
                df, 
                x=input_data.x_column,
                color=input_data.color_column,
                title=title,
                width=input_data.width,
                height=input_data.height
            )
        
        elif input_data.chart_type == "box":
            fig = px.box(
                df, 
                x=input_data.x_column, 
                y=input_data.y_column,
                color=input_data.color_column,
                title=title,
                width=input_data.width,
                height=input_data.height
            )
        
        elif input_data.chart_type == "heatmap":
            # For heatmap, we need numeric data
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                raise ValueError("No numeric columns found for heatmap")
            
            correlation_matrix = numeric_df.corr()
            fig = px.imshow(
                correlation_matrix,
                title=title or "Correlation Heatmap",
                width=input_data.width,
                height=input_data.height
            )
        
        # Update axis labels if provided
        if input_data.x_title:
            fig.update_xaxes(title_text=input_data.x_title)
        if input_data.y_title:
            fig.update_yaxes(title_text=input_data.y_title)
        
        return fig 