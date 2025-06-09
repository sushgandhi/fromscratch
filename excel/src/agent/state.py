"""
State management for Excel Agent
"""
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel
import operator
from langgraph.graph import add_messages


class Operation(BaseModel):
    """Represents a single operation to be performed"""
    tool_name: str
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[str]] = None  # List of operation IDs this depends on
    operation_id: str


class FileInfo(BaseModel):
    """Information about a file to be processed"""
    file_id: str
    data_path: Optional[str] = None
    data: Optional[List[Dict]] = None
    sheet_name: Optional[str] = None
    filename: Optional[str] = None
    s3_bucket: Optional[str] = None


class AgentState(TypedDict):
    """State for the Excel Agent - Enhanced for multiple files"""
    messages: List[Dict[str, Any]]
    goal: str
    
    # Multi-file support
    files: List[FileInfo] = []  # Multiple files to process
    current_file_index: int = 0
    
    # Legacy single file support (for backwards compatibility)
    current_data: Optional[List[Dict]] = None
    data_path: Optional[str] = None
    sheet_name: Optional[str] = None
    
    # Planning and execution
    clarifications: Dict[str, str] = {}
    file_operations: Dict[str, List[Operation]] = {}  # Operations per file
    completed_operations: List[str] = []
    operation_results: Dict[str, Dict[str, Any]] = {}  # Results per operation
    file_results: Dict[str, Dict[str, Any]] = {}  # Results per file
    
    # Control flow
    current_operation_index: int = 0
    error_message: Optional[str] = None
    final_result: Optional[Dict[str, Any]] = None
    iteration_count: int = 0
    max_iterations: int = 10
    
    # Sheet clarification support
    sheet_analysis_results: Dict[str, Dict[str, Any]] = {}  # Sheet analysis per file
    pending_sheet_clarifications: List[str] = []  # Files needing sheet clarification
    sheet_clarification_responses: Dict[str, str] = {}  # User responses to sheet questions
    
    # Processing mode
    parallel_mode: bool = True  # Whether to process files in parallel
    
    # Excel output management
    excel_workbook_manager: Optional[Any] = None  # ExcelWorkbookManager instance
    s3_bucket: Optional[str] = None  # S3 bucket for uploads
    final_excel_s3_path: Optional[str] = None  # Final Excel file S3 path 