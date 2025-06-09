"""
Pivot tool for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
from .base_tool import BaseTool, ToolInput, ToolOutput


class PivotInput(ToolInput):
    """Input model for pivot tool"""
    index: Union[str, List[str]] = Field(description="Column(s) to use as index/rows")
    columns: Optional[Union[str, List[str]]] = Field(None, description="Column(s) to use as columns")
    values: Union[str, List[str]] = Field(description="Column(s) to use as values")
    aggfunc: str = Field(default="sum", description="Aggregation function: sum, mean, count, min, max, std, var")
    fill_value: Optional[Union[int, float, str]] = Field(None, description="Value to fill missing entries")
    margins: bool = Field(default=False, description="Whether to add row/column totals")


class PivotTool(BaseTool):
    """Tool for creating pivot tables"""
    
    def __init__(self):
        super().__init__()
        self.description = "Create pivot tables from data"
        self.valid_aggfuncs = ["sum", "mean", "count", "min", "max", "std", "var", "median"]
    
    def execute(self, input_data: PivotInput) -> ToolOutput:
        """
        Execute pivot operation
        
        Args:
            input_data: Pivot parameters
            
        Returns:
            Pivot table data
        """
        try:
            # Load data
            df = self.load_data(input_data)
            
            # Validate and correct columns
            all_columns = []
            
            # Collect index columns
            index_columns = input_data.index if isinstance(input_data.index, list) else [input_data.index]
            all_columns.extend(index_columns)
            
            # Collect column columns
            if input_data.columns:
                column_columns = input_data.columns if isinstance(input_data.columns, list) else [input_data.columns]
                all_columns.extend(column_columns)
            
            # Collect value columns
            value_columns = input_data.values if isinstance(input_data.values, list) else [input_data.values]
            all_columns.extend(value_columns)
            
            # Validate and correct all columns
            corrected_columns, error_message = self.validate_and_correct_columns(all_columns, df)
            if error_message:
                return ToolOutput(
                    success=False,
                    error_message=f"Pivot columns error: {error_message}"
                )
            
            # Update input_data with corrected columns if any corrections were made
            if corrected_columns != all_columns:
                print(f"Auto-corrected pivot columns: {all_columns} -> {corrected_columns}")
                
                # Map corrected columns back to their respective parameters
                corrected_index = corrected_columns[:len(index_columns)]
                corrected_col_start = len(index_columns)
                
                if input_data.columns:
                    column_count = len(column_columns)
                    corrected_pivot_columns = corrected_columns[corrected_col_start:corrected_col_start + column_count]
                    corrected_col_start += column_count
                else:
                    corrected_pivot_columns = None
                
                corrected_values = corrected_columns[corrected_col_start:]
                
                # Update the input_data
                input_data.index = corrected_index if isinstance(input_data.index, list) else corrected_index[0]
                if input_data.columns:
                    input_data.columns = corrected_pivot_columns if isinstance(input_data.columns, list) else corrected_pivot_columns[0]
                input_data.values = corrected_values if isinstance(input_data.values, list) else corrected_values[0]
            
            # Validate aggregation function
            if input_data.aggfunc not in self.valid_aggfuncs:
                return ToolOutput(
                    success=False,
                    error_message=f"Invalid aggregation function '{input_data.aggfunc}'. Valid functions: {self.valid_aggfuncs}"
                )
            
            # Create pivot table
            pivot_df = self._create_pivot(df, input_data)
            
            # Reset index to make it a regular DataFrame
            pivot_df = pivot_df.reset_index()
            
            # Return results
            result = self.save_or_return_data(pivot_df, input_data.output_path)
            result.metadata.update({
                "pivot_index": input_data.index,
                "pivot_columns": input_data.columns,
                "pivot_values": input_data.values,
                "aggregation_function": input_data.aggfunc,
                "original_rows": len(df),
                "pivot_rows": len(pivot_df)
            })
            
            return result
            
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Pivot operation failed: {str(e)}"
            )
    
    def _create_pivot(self, df: pd.DataFrame, input_data: PivotInput) -> pd.DataFrame:
        """Create the pivot table"""
        
        # Handle different aggregation functions
        if input_data.aggfunc == "count":
            aggfunc = "count"
        else:
            aggfunc = input_data.aggfunc
        
        # Create pivot table
        pivot_df = pd.pivot_table(
            df,
            index=input_data.index,
            columns=input_data.columns,
            values=input_data.values,
            aggfunc=aggfunc,
            fill_value=input_data.fill_value,
            margins=input_data.margins
        )
        
        return pivot_df 