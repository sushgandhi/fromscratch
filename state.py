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
    depends_on: Optional[str] = None  # ID of operation this depends on
    operation_id: str


class AgentState(TypedDict):
    """State for the Excel Agent"""
    messages: List[Dict[str, Any]]
    goal: str
    current_data: Optional[List[Dict]] = None
    data_path: Optional[str] = None
    sheet_name: Optional[str] = None
    clarifications: Dict[str, str] = {}
    operations: List[Operation] = []
    completed_operations: List[str] = []
    operation_results: Dict[str, Dict[str, Any]] = {}
    current_operation_index: int = 0
    error_message: Optional[str] = None
    final_result: Optional[Dict[str, Any]] = None
    iteration_count: int = 0
    max_iterations: int = 10
    # Excel output management
    excel_workbook_manager: Optional[Any] = None  # ExcelWorkbookManager instance
    s3_bucket: Optional[str] = None  # S3 bucket for uploads
    final_excel_s3_path: Optional[str] = None  # Final Excel file S3 path 
