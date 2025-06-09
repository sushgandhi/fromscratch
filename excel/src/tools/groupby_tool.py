"""
Group by tool for Excel Agent
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
from .base_tool import BaseTool, ToolInput, ToolOutput


class GroupByInput(ToolInput):
    """Input model for group by tool"""
    group_by: Union[str, List[str]] = Field(description="Column(s) to group by")
    aggregations: Dict[str, Union[str, List[str]]] = Field(
        description="Dictionary mapping column names to aggregation functions"
    )
    sort_by: Optional[str] = Field(None, description="Column to sort results by")
    sort_ascending: bool = Field(default=True, description="Sort order")


class GroupByTool(BaseTool):
    """Tool for grouping and aggregating data"""
    
    def __init__(self):
        super().__init__()
        self.description = "Group data and apply aggregation functions"
        self.valid_aggfuncs = ["sum", "mean", "count", "min", "max", "std", "var", "median", "first", "last"]
    
    def execute(self, input_data: GroupByInput) -> ToolOutput:
        """
        Execute group by operation
        
        Args:
            input_data: Group by parameters
            
        Returns:
            Grouped and aggregated data
        """
        try:
            # Load data
            df = self.load_data(input_data)
            
            # Validate and correct group by columns
            group_columns = input_data.group_by if isinstance(input_data.group_by, list) else [input_data.group_by]
            corrected_group_columns, error_message = self.validate_and_correct_columns(group_columns, df)
            if error_message:
                return ToolOutput(
                    success=False,
                    error_message=f"Group by columns error: {error_message}"
                )
            
            # Update group_by with corrected columns
            if corrected_group_columns != group_columns:
                input_data.group_by = corrected_group_columns if isinstance(input_data.group_by, list) else corrected_group_columns[0]
                print(f"Auto-corrected group by columns: {group_columns} -> {corrected_group_columns}")
            
            # Validate and correct aggregation columns
            agg_columns = list(input_data.aggregations.keys())
            corrected_agg_columns, error_message = self.validate_and_correct_columns(agg_columns, df)
            if error_message:
                return ToolOutput(
                    success=False,
                    error_message=f"Aggregation columns error: {error_message}"
                )
            
            # Update aggregations with corrected columns
            if corrected_agg_columns != agg_columns:
                new_aggregations = {}
                for original_col, corrected_col in zip(agg_columns, corrected_agg_columns):
                    new_aggregations[corrected_col] = input_data.aggregations[original_col]
                input_data.aggregations = new_aggregations
                print(f"Auto-corrected aggregation columns: {agg_columns} -> {corrected_agg_columns}")
            
            # Validate aggregation functions
            for col, funcs in input_data.aggregations.items():
                func_list = funcs if isinstance(funcs, list) else [funcs]
                invalid_funcs = [f for f in func_list if f not in self.valid_aggfuncs]
                if invalid_funcs:
                    return ToolOutput(
                        success=False,
                        error_message=f"Invalid aggregation functions for column '{col}': {invalid_funcs}. Valid functions: {self.valid_aggfuncs}"
                    )
            
            # Perform group by operation
            grouped_df = self._perform_groupby(df, input_data)
            
            # Sort if requested
            if input_data.sort_by:
                if input_data.sort_by in grouped_df.columns:
                    grouped_df = grouped_df.sort_values(
                        input_data.sort_by, 
                        ascending=input_data.sort_ascending
                    )
                else:
                    return ToolOutput(
                        success=False,
                        error_message=f"Sort column '{input_data.sort_by}' not found in results"
                    )
            
            # Reset index to make it a regular DataFrame
            grouped_df = grouped_df.reset_index()
            
            # Return results
            result = self.save_or_return_data(grouped_df, input_data.output_path)
            result.metadata.update({
                "group_by_columns": input_data.group_by,
                "aggregations": input_data.aggregations,
                "original_rows": len(df),
                "grouped_rows": len(grouped_df)
            })
            
            return result
            
        except Exception as e:
            return ToolOutput(
                success=False,
                error_message=f"Group by operation failed: {str(e)}"
            )
    
    def _perform_groupby(self, df: pd.DataFrame, input_data: GroupByInput) -> pd.DataFrame:
        """Perform the group by operation"""
        
        # Group the data
        grouped = df.groupby(input_data.group_by)
        
        # Apply aggregations
        result = grouped.agg(input_data.aggregations)
        
        # Flatten column names if multi-level
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
        
        return result 