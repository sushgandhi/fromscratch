"""
Filter tool for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
from .base_tool import BaseTool, ToolInput, ToolOutput
from ..utils.column_matcher import auto_correct_column_name, suggest_column_correction


class FilterInput(ToolInput):
    """Input model for filter tool"""
    column: str = Field(description="Column name to filter on")
    value: Union[str, int, float, bool] = Field(description="Value to filter for")
    operator: str = Field(default="==", description="Filter operator: ==, !=, >, <, >=, <=, contains, startswith, endswith")
    case_sensitive: bool = Field(default=False, description="Whether string comparisons should be case sensitive")
    operation_id: Optional[str] = Field(None, description="Unique operation ID for file naming")


class FilterTool(BaseTool):
    """Tool for filtering data based on column conditions"""
    
    def __init__(self):
        super().__init__()
        self.description = "Filter data based on column conditions"
        self.valid_operators = ["==", "!=", ">", "<", ">=", "<=", "contains", "startswith", "endswith"]
    
    def execute(self, input_data: FilterInput) -> ToolOutput:
        """
        Execute filter operation
        
        Args:
            input_data: Filter parameters
            
        Returns:
            Filtered data
        """
        try:
            # Load data
            df = self.load_data(input_data)
            
            # Validate column exists and auto-correct if possible
            original_column = input_data.column
            corrected_column, error_message = self.validate_and_correct_column(input_data.column, df)
            
            if error_message:
                return ToolOutput(
                    success=False,
                    error_message=error_message
                )
            
            if corrected_column != original_column:
                input_data.column = corrected_column
                print(f"Auto-corrected column name from '{original_column}' to '{corrected_column}'")
            
            # Validate operator
            if input_data.operator not in self.valid_operators:
                return ToolOutput(
                    success=False,
                    error_message=f"Invalid operator '{input_data.operator}'. Valid operators: {self.valid_operators}"
                )
            
            # Apply filter
            filtered_df = self._apply_filter(df, input_data)
            
            # Save results with summary to prevent context overflow
            if input_data.operation_id:
                # Use new save_with_summary method for large datasets
                result = self.save_with_summary(filtered_df, input_data.operation_id)
            else:
                # Fallback to old method for backward compatibility
                result = self.save_or_return_data(filtered_df, input_data.output_path)
            
            # Add filter-specific metadata
            result.metadata.update({
                "filter_condition": f"{input_data.column} {input_data.operator} {input_data.value}",
                "original_rows": len(df),
                "filtered_rows": len(filtered_df),
                "tool_type": "filter"
            })
            
            return result
            
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Filter operation failed: {str(e)}"
            )
    
    def _apply_filter(self, df: pd.DataFrame, input_data: FilterInput) -> pd.DataFrame:
        """Apply the filter operation"""
        column_data = df[input_data.column]
        
        # Handle string operations
        if input_data.operator in ["contains", "startswith", "endswith"]:
            if not input_data.case_sensitive:
                column_data = column_data.astype(str).str.lower()
                value = str(input_data.value).lower()
            else:
                column_data = column_data.astype(str)
                value = str(input_data.value)
            
            if input_data.operator == "contains":
                mask = column_data.str.contains(value, na=False)
            elif input_data.operator == "startswith":
                mask = column_data.str.startswith(value, na=False)
            elif input_data.operator == "endswith":
                mask = column_data.str.endswith(value, na=False)
        
        # Handle comparison operations
        else:
            if input_data.operator == "==":
                mask = column_data == input_data.value
            elif input_data.operator == "!=":
                mask = column_data != input_data.value
            elif input_data.operator == ">":
                mask = column_data > input_data.value
            elif input_data.operator == "<":
                mask = column_data < input_data.value
            elif input_data.operator == ">=":
                mask = column_data >= input_data.value
            elif input_data.operator == "<=":
                mask = column_data <= input_data.value
        
        return df[mask] 