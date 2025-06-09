"""
FastAPI interface for Excel Agent
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import tempfile
import pandas as pd

from .agent.supervisor import ExcelAgentSupervisor
from .utils.data_utils import read_excel_file, data_to_records, get_workbook_sheets
from .utils.claude_client import get_claude_client

# Load environment variables
load_dotenv()

app = FastAPI(title="Excel Agent API", description="AI-powered Excel data analysis")
security = HTTPBearer()

# get_claude_client is now imported from utils.claude_client

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract API key from Bearer token"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required. Please provide your Anthropic API key as Bearer token."
        )
    return credentials.credentials

def get_supervisor(api_key: str = Depends(get_api_key)) -> ExcelAgentSupervisor:
    """Create supervisor instance with the provided API key"""
    try:
        claude_client = get_claude_client(api_key)
        return ExcelAgentSupervisor(claude_client)
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key or Claude client initialization failed: {str(e)}"
                 )


async def check_clarifications_needed(
    request: "ExcelAgentRequest", 
    supervisor: ExcelAgentSupervisor
) -> Optional["ExcelAgentResponse"]:
    """Check if clarifications are needed before processing"""
    
    # Check for sheet name clarification
    if request.data_path and not request.sheet_name:
        try:
            sheets = get_workbook_sheets(request.data_path)
            if len(sheets) > 1:
                # Multiple sheets - need clarification
                return ExcelAgentResponse(
                    success=False,
                    goal=request.goal,
                    clarification_needed=ClarificationRequest(
                        question=f"This Excel file has {len(sheets)} sheets. Which sheet would you like to analyze?",
                        options=sheets,
                        clarification_type="sheet_name"
                    ),
                    available_sheets=sheets
                )
        except Exception:
            # If we can't read sheets, continue with default behavior
            pass
    
    # Check for column name ambiguity using Claude
    if request.goal and (request.data or request.data_path):
        try:
            # Load a sample of the data to check columns
            if request.data:
                df = pd.DataFrame(request.data)
            else:
                df = read_excel_file(request.data_path, request.sheet_name)
            
            columns = df.columns.tolist()
            
            # Use Claude to check if goal has ambiguous column references
            ambiguity_check = await check_column_ambiguity(
                request.goal, columns, supervisor.llm
            )
            
            if ambiguity_check:
                return ExcelAgentResponse(
                    success=False,
                    goal=request.goal,
                    clarification_needed=ambiguity_check,
                    available_columns=columns
                )
                
        except Exception:
            # If clarification check fails, continue with normal processing
            pass
    
    return None


async def check_column_ambiguity(goal: str, columns: List[str], llm) -> Optional[ClarificationRequest]:
    """Use Claude to check for column name ambiguity"""
    
    prompt = f"""
    Analyze this user goal and available columns to check for ambiguity:
    
    Goal: {goal}
    Available columns: {columns}
    
    Check if the goal references column names that are:
    1. Ambiguous (could match multiple columns)
    2. Not clearly specified
    3. Need clarification
    
    If clarification is needed, respond with JSON:
    {{
        "needs_clarification": true,
        "question": "Which column did you mean?",
        "clarification_type": "column_name",
        "suggested_columns": ["col1", "col2"]
    }}
    
    If no clarification needed, respond with:
    {{"needs_clarification": false}}
    """
    
    try:
        import json
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        
        if result.get("needs_clarification"):
            return ClarificationRequest(
                question=result["question"],
                options=result.get("suggested_columns"),
                clarification_type=result["clarification_type"]
            )
    except Exception:
        pass
    
    return None


def extract_visualizations(operation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract visualization data from operation results for API clients"""
    visualizations = []
    
    for op_id, result in operation_results.items():
        if result.get("success") and result.get("metadata", {}).get("visualization_created"):
            metadata = result["metadata"]
            
            viz_data = {
                "operation_id": op_id,
                "chart_type": metadata.get("chart_type"),
                "data_points": metadata.get("data_points"),
                "html_content": metadata.get("html_content"),
                "json_content": metadata.get("json_content"),
                "local_path": metadata.get("local_html_path"),  # Only useful for local development
                "s3_path": result.get("output_path") if result.get("output_path", "").startswith("s3://") else None
            }
            
            # Remove None values for cleaner response
            viz_data = {k: v for k, v in viz_data.items() if v is not None}
            visualizations.append(viz_data)
    
    return visualizations if visualizations else None


class ClarificationRequest(BaseModel):
    """Model for clarification questions"""
    question: str
    options: Optional[List[str]] = None
    clarification_type: str  # "sheet_name", "column_name", "parameter", etc.


class ExcelAgentRequest(BaseModel):
    """Request model for Excel Agent"""
    goal: str
    data_path: Optional[str] = None
    data: Optional[List[Dict]] = None
    sheet_name: Optional[str] = None  # User can specify sheet name
    clarifications: Optional[Dict[str, str]] = None  # Previous clarifications
    s3_bucket: Optional[str] = None  # S3 bucket for results upload


class ExcelAgentResponse(BaseModel):
    """Response model for Excel Agent"""
    success: bool
    goal: str
    completed_operations: Optional[List[str]] = None
    operation_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    clarification_needed: Optional[ClarificationRequest] = None
    available_sheets: Optional[List[str]] = None  # For sheet selection
    available_columns: Optional[List[str]] = None  # For column selection
    visualizations: Optional[List[Dict[str, Any]]] = None  # Extracted visualization info
    # Excel workbook results
    excel_workbook_s3_path: Optional[str] = None  # S3 path to complete Excel workbook
    excel_workbook_local_path: Optional[str] = None  # Local path (fallback)
    workbook_summary: Optional[Dict[str, Any]] = None  # Workbook metadata summary
    # HTML summary results
    html_summaries: Optional[List[Dict[str, Any]]] = None  # List of HTML summary reports


@app.post("/analyze", response_model=ExcelAgentResponse)
async def analyze_excel(
    request: ExcelAgentRequest,
    supervisor: ExcelAgentSupervisor = Depends(get_supervisor)
):
    """
    Analyze Excel data based on the provided goal
    Requires Bearer token with Anthropic API key in Authorization header
    """
    try:
        # Validate input
        if not request.data_path and not request.data:
            raise HTTPException(
                status_code=400, 
                detail="Either data_path or data must be provided"
            )
        
        if request.data_path and request.data:
            raise HTTPException(
                status_code=400, 
                detail="Provide either data_path OR data, not both"
            )
        
        # Check if we need clarifications first
        clarification = await check_clarifications_needed(
            request, supervisor
        )
        
        if clarification:
            return clarification
        
        # Run the agent
        result = supervisor.run(
            goal=request.goal,
            data_path=request.data_path,
            data=request.data,
            sheet_name=request.sheet_name,
            clarifications=request.clarifications,
            s3_bucket=request.s3_bucket
        )
        
        if result["success"]:
            # Extract visualization data for easier API consumption
            visualizations = extract_visualizations(result.get("operation_results", {}))
            
            # Extract Excel workbook information
            excel_path = result.get("excel_workbook_path")
            excel_s3_path = None
            excel_local_path = None
            
            if excel_path:
                if excel_path.startswith("s3://"):
                    excel_s3_path = excel_path
                else:
                    excel_local_path = excel_path
            
            return ExcelAgentResponse(
                success=True,
                goal=result["goal"],
                completed_operations=result["completed_operations"],
                operation_results=result["operation_results"],
                visualizations=visualizations,
                excel_workbook_s3_path=excel_s3_path,
                excel_workbook_local_path=excel_local_path,
                workbook_summary=result.get("workbook_summary"),
                html_summaries=result.get("html_summaries")
            )
        else:
            return ExcelAgentResponse(
                success=False,
                goal=result["goal"],
                error=result.get("error", "Unknown error")
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    goal: str = None,
    s3_bucket: str = None,
    supervisor: ExcelAgentSupervisor = Depends(get_supervisor)
):
    """
    Upload an Excel file and analyze it
    Requires Bearer token with Anthropic API key in Authorization header
    """
    if not goal:
        raise HTTPException(status_code=400, detail="Goal parameter is required")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Read the Excel file and convert to data format
        try:
            df = read_excel_file(tmp_file_path)
            data = data_to_records(df)
            
            # Run the agent (no clarification check for uploaded files - they're single sheet)
            result = supervisor.run(goal=goal, data=data, s3_bucket=s3_bucket)
            
            if result["success"]:
                # Extract visualization data for easier API consumption
                visualizations = extract_visualizations(result.get("operation_results", {}))
                
                # Extract Excel workbook information
                excel_path = result.get("excel_workbook_path")
                excel_s3_path = None
                excel_local_path = None
                
                if excel_path:
                    if excel_path.startswith("s3://"):
                        excel_s3_path = excel_path
                    else:
                        excel_local_path = excel_path
                
                return ExcelAgentResponse(
                    success=True,
                    goal=result["goal"],
                    completed_operations=result["completed_operations"],
                    operation_results=result["operation_results"],
                    visualizations=visualizations,
                    excel_workbook_s3_path=excel_s3_path,
                    excel_workbook_local_path=excel_local_path,
                    workbook_summary=result.get("workbook_summary"),
                    html_summaries=result.get("html_summaries")
                )
            else:
                return ExcelAgentResponse(
                    success=False,
                    goal=result["goal"],
                    error=result.get("error", "Unknown error")
                )
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Excel Agent API"}


@app.get("/tools")
async def get_available_tools():
    """Get list of available tools and their descriptions"""
    return {
        "tools": [
            {
                "name": "filter",
                "description": "Filter data based on column conditions",
                "parameters": [
                    "column", "value", "operator", "case_sensitive"
                ]
            },
            {
                "name": "pivot",
                "description": "Create pivot tables from data",
                "parameters": [
                    "index", "columns", "values", "aggfunc", "fill_value", "margins"
                ]
            },
            {
                "name": "groupby",
                "description": "Group data and apply aggregation functions",
                "parameters": [
                    "group_by", "aggregations", "sort_by", "sort_ascending"
                ]
            },
            {
                "name": "visualization",
                "description": "Create interactive visualizations using Plotly",
                "parameters": [
                    "chart_type", "x_column", "y_column", "color_column", 
                    "size_column", "title", "width", "height"
                ]
            },
            {
                "name": "column_summary",
                "description": "Generate statistical summaries for columns",
                "parameters": ["columns"]
            },
            {
                "name": "sheet_summary",
                "description": "Generate comprehensive sheet summaries",
                "parameters": ["include_visualizations"]
            },
            {
                "name": "workbook_summary",
                "description": "Generate analysis of entire workbook",
                "parameters": ["include_visualizations"]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 