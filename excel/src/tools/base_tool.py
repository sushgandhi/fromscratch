"""
Base tool class for Excel Agent tools
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import pandas as pd
from src.utils.data_utils import (
    read_excel_file, read_csv_file, save_to_s3, 
    data_to_records, records_to_dataframe, convert_numpy_types
)
from src.utils.column_matcher import auto_correct_column_name, suggest_column_correction


class ToolInput(BaseModel):
    """Base input model for all tools"""
    data_path: Optional[str] = Field(None, description="Path to data file (local or S3)")
    data: Optional[List[Dict]] = Field(None, description="Data as list of dictionaries")
    output_path: Optional[str] = Field(None, description="S3 path to save results (optional)")
    sheet_name: Optional[str] = Field(None, description="Sheet name for Excel files")
    
    def model_validate_input(self):
        """Validate that either data_path OR data is provided, not both"""
        if self.data_path and self.data:
            raise ValueError("Provide either data_path OR data, not both")
        if not self.data_path and not self.data:
            raise ValueError("Must provide either data_path OR data")


class ToolOutput(BaseModel):
    """Base output model for all tools"""
    success: bool = Field(description="Whether the operation was successful")
    data: Optional[List[Dict]] = Field(None, description="Result data as list of dictionaries")
    output_path: Optional[str] = Field(None, description="S3 path where results were saved")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


class BaseTool(ABC):
    """Base class for all Excel operation tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    def load_data(self, input_data: ToolInput) -> pd.DataFrame:
        """
        Load data from either data_path or data field
        
        Args:
            input_data: Tool input containing data_path or data
            
        Returns:
            DataFrame with the loaded data
        """
        input_data.model_validate_input()
        
        if input_data.data_path:
            # Load from file path
            if input_data.data_path.endswith('.csv'):
                return read_csv_file(input_data.data_path)
            else:
                return read_excel_file(input_data.data_path, input_data.sheet_name)
        else:
            # Convert from list of dicts to DataFrame
            return records_to_dataframe(input_data.data)
    
    def save_or_return_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> ToolOutput:
        """
        Save data to S3 or return as list of dicts
        
        Args:
            df: DataFrame to save or return
            output_path: Optional S3 path to save to
            
        Returns:
            ToolOutput with results
        """
        try:
            if output_path:
                # Save to S3
                saved_path = save_to_s3(df, output_path, format='csv')
                return ToolOutput(
                    success=True,
                    data=None,
                    output_path=saved_path,
                    metadata={"rows": len(df), "columns": len(df.columns)}
                )
            else:
                # Return as list of dicts
                return ToolOutput(
                    success=True,
                    data=data_to_records(df),
                    output_path=None,
                    metadata={"rows": len(df), "columns": len(df.columns)}
                )
        except Exception as e:
            return ToolOutput(
                success=False,
                data=None,
                output_path=None,
                error_message=str(e)
            )
    
    def validate_and_correct_column(self, column_name: str, df: pd.DataFrame) -> tuple[str, Optional[str]]:
        """
        Validate and potentially auto-correct a column name
        
        Args:
            column_name: The column name to validate
            df: DataFrame containing the available columns
            
        Returns:
            Tuple of (corrected_column_name, error_message_if_failed)
        """
        if column_name in df.columns:
            return column_name, None
        
        # Try to auto-correct the column name
        corrected_column = auto_correct_column_name(column_name, list(df.columns))
        
        if corrected_column != column_name and corrected_column in df.columns:
            return corrected_column, None
        
        # No good match found, generate error message
        error_message = suggest_column_correction(column_name, list(df.columns))
        return column_name, error_message
    
    def validate_and_correct_columns(self, column_names: List[str], df: pd.DataFrame) -> tuple[List[str], Optional[str]]:
        """
        Validate and potentially auto-correct multiple column names
        
        Args:
            column_names: List of column names to validate
            df: DataFrame containing the available columns
            
        Returns:
            Tuple of (corrected_column_names, error_message_if_any_failed)
        """
        corrected_columns = []
        
        for column_name in column_names:
            corrected_column, error_message = self.validate_and_correct_column(column_name, df)
            
            if error_message:
                return column_names, error_message
            
            corrected_columns.append(corrected_column)
        
        return corrected_columns, None
    
    @abstractmethod
    def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Execute the tool operation
        
        Args:
            input_data: Tool input parameters
            
        Returns:
            Tool output with results
        """
        pass 