"""
Excel Output Management for Excel Agent

This module handles creating multi-sheet Excel files with intermediate and final outputs.
"""
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
from io import BytesIO
import boto3
from botocore.exceptions import ClientError

from .data_utils import save_to_s3, convert_numpy_types


class ExcelWorkbookManager:
    """Manages creation of multi-sheet Excel workbooks with operation results"""
    
    def __init__(self, goal: str, session_id: Optional[str] = None):
        """
        Initialize workbook manager
        
        Args:
            goal: User's goal description (used for filename)
            session_id: Optional session identifier
        """
        self.goal = goal
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.sheets_data: Dict[str, pd.DataFrame] = {}
        self.sheets_metadata: Dict[str, Dict[str, Any]] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create safe filename from goal
        safe_goal = "".join(c for c in goal if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_goal = safe_goal.replace(' ', '_')[:50]  # Limit length
        self.filename = f"{safe_goal}_{self.timestamp}_{self.session_id}.xlsx"
        
    def add_operation_result(self, operation_id: str, operation_name: str, 
                           data: Union[pd.DataFrame, List[Dict]], 
                           metadata: Optional[Dict[str, Any]] = None,
                           description: Optional[str] = None):
        """
        Add operation result as a new sheet
        
        Args:
            operation_id: Unique operation identifier
            operation_name: Human-readable operation name
            data: Operation result data
            metadata: Additional operation metadata
            description: Operation description
        """
        # Convert data to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            # For non-tabular data, create a simple DataFrame
            df = pd.DataFrame({'result': [str(data)]})
        
        # Create sheet name (Excel sheet names have 31 char limit)
        sheet_name = f"{operation_id}_{operation_name}"[:31]
        
        # Store data and metadata
        self.sheets_data[sheet_name] = df
        self.sheets_metadata[sheet_name] = {
            'operation_id': operation_id,
            'operation_name': operation_name,
            'description': description or '',
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'row_count': len(df),
            'column_count': len(df.columns) if hasattr(df, 'columns') else 0
        }
        
        print(f"📊 Added sheet '{sheet_name}' with {len(df)} rows")
    
    def add_final_result(self, final_data: Union[pd.DataFrame, List[Dict]], 
                        summary: Dict[str, Any]):
        """
        Add final result as the main output sheet
        
        Args:
            final_data: Final processed data
            summary: Summary of all operations performed
        """
        # Convert data to DataFrame if needed
        if isinstance(final_data, list):
            df = pd.DataFrame(final_data)
        elif isinstance(final_data, pd.DataFrame):
            df = final_data.copy()
        else:
            df = pd.DataFrame({'final_result': [str(final_data)]})
        
        # Add as 'Final_Result' sheet
        self.sheets_data['Final_Result'] = df
        self.sheets_metadata['Final_Result'] = {
            'operation_id': 'final',
            'operation_name': 'Final Result',
            'description': f"Final result for: {self.goal}",
            'timestamp': datetime.now().isoformat(),
            'metadata': summary,
            'row_count': len(df),
            'column_count': len(df.columns) if hasattr(df, 'columns') else 0
        }
        
        print(f"🎯 Added final result sheet with {len(df)} rows")
    
    def add_summary_sheet(self, operations: List[Dict[str, Any]]):
        """
        Add a summary sheet with operation details
        
        Args:
            operations: List of operation dictionaries
        """
        summary_data = []
        
        # Add general information
        summary_data.append(['Goal', self.goal])
        summary_data.append(['Session ID', self.session_id])
        summary_data.append(['Timestamp', self.timestamp])
        summary_data.append(['Total Operations', len(operations)])
        summary_data.append(['', ''])  # Empty row
        
        # Add operation details
        summary_data.append(['Operation Details', ''])
        summary_data.append(['Operation ID', 'Tool', 'Description', 'Status', 'Rows', 'Columns'])
        
        for op in operations:
            op_id = op.get('operation_id', 'unknown')
            tool = op.get('tool', 'unknown')
            desc = op.get('description', '')
            result = op.get('result', {})
            status = 'Success' if result.get('success', False) else 'Failed'
            
            # Find corresponding sheet metadata
            sheet_meta = None
            for sheet_name, meta in self.sheets_metadata.items():
                if meta['operation_id'] == op_id:
                    sheet_meta = meta
                    break
            
            rows = sheet_meta['row_count'] if sheet_meta else 0
            cols = sheet_meta['column_count'] if sheet_meta else 0
            
            summary_data.append([op_id, tool, desc, status, rows, cols])
        
        # Create DataFrame with proper formatting
        df = pd.DataFrame(summary_data, columns=['Property', 'Value', 'Extra1', 'Extra2', 'Extra3', 'Extra4'])
        
        self.sheets_data['Summary'] = df
        self.sheets_metadata['Summary'] = {
            'operation_id': 'summary',
            'operation_name': 'Execution Summary',
            'description': 'Summary of all operations performed',
            'timestamp': datetime.now().isoformat(),
            'metadata': {'goal': self.goal, 'operations_count': len(operations)},
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        print(f"📋 Added summary sheet with {len(operations)} operations")
    
    def create_local_file(self, output_dir: str = "outputs") -> str:
        """
        Create local Excel file with all sheets
        
        Args:
            output_dir: Directory to save the file
            
        Returns:
            Path to created file
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, self.filename)
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Write Summary sheet first
            if 'Summary' in self.sheets_data:
                self.sheets_data['Summary'].to_excel(writer, sheet_name='Summary', index=False)
            
            # Write Final_Result sheet second
            if 'Final_Result' in self.sheets_data:
                self.sheets_data['Final_Result'].to_excel(writer, sheet_name='Final_Result', index=False)
            
            # Write other sheets
            for sheet_name, df in self.sheets_data.items():
                if sheet_name not in ['Summary', 'Final_Result']:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"💾 Excel file created: {file_path}")
        return file_path
    
    def upload_to_s3(self, s3_bucket: str, s3_prefix: str = "excel-agent-results") -> str:
        """
        Upload Excel workbook to S3
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 key prefix
            
        Returns:
            S3 URL of uploaded file
        """
        # Create Excel file in memory
        excel_buffer = BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Write Summary sheet first
            if 'Summary' in self.sheets_data:
                self.sheets_data['Summary'].to_excel(writer, sheet_name='Summary', index=False)
            
            # Write Final_Result sheet second
            if 'Final_Result' in self.sheets_data:
                self.sheets_data['Final_Result'].to_excel(writer, sheet_name='Final_Result', index=False)
            
            # Write other sheets
            for sheet_name, df in self.sheets_data.items():
                if sheet_name not in ['Summary', 'Final_Result']:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Upload to S3
        s3_key = f"{s3_prefix}/{self.filename}"
        s3_path = f"s3://{s3_bucket}/{s3_key}"
        
        try:
            s3_client = boto3.client('s3')
            excel_buffer.seek(0)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=excel_buffer.getvalue(),
                ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
            print(f"☁️ Excel file uploaded to S3: {s3_path}")
            return s3_path
        
        except ClientError as e:
            print(f"❌ Failed to upload to S3: {e}")
            # Fallback to local file
            local_path = self.create_local_file()
            print(f"💾 Saved locally instead: {local_path}")
            return local_path
    
    def get_summary_info(self) -> Dict[str, Any]:
        """
        Get summary information about the workbook
        
        Returns:
            Dictionary with workbook summary
        """
        return {
            'filename': self.filename,
            'goal': self.goal,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'sheets_count': len(self.sheets_data),
            'sheet_names': list(self.sheets_data.keys()),
            'total_rows': sum(meta['row_count'] for meta in self.sheets_metadata.values()),
            'sheets_metadata': self.sheets_metadata
        } 