"""
Data utilities for Excel Agent
"""
import pandas as pd
import numpy as np
import boto3
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import json
from io import BytesIO


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy and pandas data types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def read_excel_file(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read Excel file from local path or S3.
    
    Args:
        file_path: Path to Excel file (local or S3)
        sheet_name: Name of sheet to read (optional)
        
    Returns:
        DataFrame containing the Excel data
    """
    if file_path.startswith('s3://'):
        return read_from_s3(file_path, sheet_name=sheet_name)
    else:
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return pd.read_excel(file_path)


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read CSV file from local path or S3.
    
    Args:
        file_path: Path to CSV file (local or S3)
        
    Returns:
        DataFrame containing the CSV data
    """
    if file_path.startswith('s3://'):
        return read_csv_from_s3(file_path)
    else:
        return pd.read_csv(file_path)


def read_from_s3(s3_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read Excel file from S3.
    
    Args:
        s3_path: S3 path (s3://bucket/key)
        sheet_name: Name of sheet to read (optional)
        
    Returns:
        DataFrame containing the data
    """
    # Parse S3 path
    path_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1]
    
    # Download from S3
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    
    # Read based on file extension
    if key.endswith('.xlsx') or key.endswith('.xls'):
        if sheet_name:
            return pd.read_excel(BytesIO(obj['Body'].read()), sheet_name=sheet_name)
        else:
            return pd.read_excel(BytesIO(obj['Body'].read()))
    elif key.endswith('.csv'):
        return pd.read_csv(BytesIO(obj['Body'].read()))
    else:
        raise ValueError(f"Unsupported file format: {key}")


def read_csv_from_s3(s3_path: str) -> pd.DataFrame:
    """
    Read CSV file from S3.
    
    Args:
        s3_path: S3 path (s3://bucket/key)
        
    Returns:
        DataFrame containing the CSV data
    """
    # Parse S3 path
    path_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1]
    
    # Download from S3
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    
    return pd.read_csv(BytesIO(obj['Body'].read()))


def save_to_s3(data: Union[pd.DataFrame, Dict, str], s3_path: str, format: str = 'csv') -> str:
    """
    Save data to S3.
    
    Args:
        data: Data to save (DataFrame, dict, or string)
        s3_path: S3 path (s3://bucket/key)
        format: Output format ('csv', 'json', 'html')
        
    Returns:
        S3 path where data was saved
    """
    # Parse S3 path
    path_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1]
    
    s3_client = boto3.client('s3')
    
    if format == 'csv' and isinstance(data, pd.DataFrame):
        csv_buffer = BytesIO()
        data.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    elif format == 'json':
        if isinstance(data, pd.DataFrame):
            json_data = data.to_json(orient='records')
        else:
            json_data = json.dumps(convert_numpy_types(data))
        s3_client.put_object(Bucket=bucket, Key=key, Body=json_data)
    elif format == 'html':
        if isinstance(data, pd.DataFrame):
            html_data = data.to_html()
        else:
            html_data = str(data)
        s3_client.put_object(Bucket=bucket, Key=key, Body=html_data)
    
    return s3_path


def data_to_records(data: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
    """
    Convert data to list of dictionaries format.
    
    Args:
        data: Input data (DataFrame or list of dicts)
        
    Returns:
        List of dictionaries
    """
    if isinstance(data, pd.DataFrame):
        return convert_numpy_types(data.to_dict('records'))
    elif isinstance(data, list):
        return convert_numpy_types(data)
    else:
        raise ValueError("Data must be DataFrame or list of dictionaries")


def records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    """
    Convert list of dictionaries to DataFrame.
    
    Args:
        records: List of dictionaries
        
    Returns:
        DataFrame
    """
    return pd.DataFrame(records)


def get_workbook_sheets(file_path: str) -> List[str]:
    """
    Get list of sheet names from Excel workbook.
    
    Args:
        file_path: Path to Excel file (local or S3)
        
    Returns:
        List of sheet names
    """
    if file_path.startswith('s3://'):
        # Parse S3 path and download
        path_parts = file_path.replace('s3://', '').split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1]
        
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        
        excel_file = pd.ExcelFile(BytesIO(obj['Body'].read()))
        return excel_file.sheet_names
    else:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names 