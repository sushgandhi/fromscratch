"""
Redesigned Workflow Integration
- analyze_goal: Plans what Excel tools to use (planning logic)
- execute_goal: Conditionally executes planned operations (executor logic)
- prepare_output: Consolidates and formats results
- finish_excel: Final delivery and notifications
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
import asyncio
import uuid
from datetime import datetime

from ..agent.supervisor import ExcelAgentSupervisor
from ..utils.claude_client import get_claude_client
from .grpc_queue_client import GRPCQueueClient, QueueConfig


class WorkflowState(TypedDict):
    """Enhanced state for redesigned workflow"""
    session_id: str
    uploaded_files: List[Dict[str, Any]]
    user_goals: List[str]  # Original user goals
    
    # Planning outputs (from analyze_goal)
    planned_operations: List[Dict[str, Any]]  # Per-file operation plans
    execution_strategy: Dict[str, Any]  # Parallel vs sequential, dependencies
    
    # Execution outputs (from execute_goal) 
    execution_results: List[Dict[str, Any]]  # Results per file
    operation_outputs: Dict[str, Any]  # Detailed operation results
    
    # Processing outputs
    prepared_outputs: List[Dict[str, Any]]
    final_deliverables: List[str]
    
    # Status tracking
    errors: List[str]
    current_stage: str
    queue_client: Optional[GRPCQueueClient]


class RedesignedWorkflow:
    """Redesigned workflow with proper separation of concerns"""
    
    def __init__(self, queue_endpoint: str = "localhost:50051"):
        self.queue_config = QueueConfig(
            queue_endpoint=queue_endpoint,
            enable_intermediate_results=True,
            enable_status_updates=True
        )
        self.claude_client = None
        self.available_tools = [
            "filter", "groupby", "pivot", "visualization", 
            "column_summary", "sheet_summary", "workbook_summary"
        ]
    
    async def initialize(self):
        """Initialize Claude client"""
        self.claude_client = get_claude_client()
    
    def create_workflow(self) -> StateGraph:
        """Create the redesigned workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes with clear responsibilities
        workflow.add_node("analyze_goal", self._analyze_goal_node)      # PLANNER
        workflow.add_node("execute_goal", self._execute_goal_node)      # EXECUTOR  
        workflow.add_node("prepare_output", self._prepare_output_node)  # FORMATTER
        workflow.add_node("finish_excel", self._finish_excel_node)      # DELIVERER
        
        # Linear flow
        workflow.add_edge("analyze_goal", "execute_goal")
        workflow.add_edge("execute_goal", "prepare_output") 
        workflow.add_edge("prepare_output", "finish_excel")
        workflow.add_edge("finish_excel", END)
        
        workflow.set_entry_point("analyze_goal")
        return workflow.compile()
    
    async def _analyze_goal_node(self, state: WorkflowState) -> WorkflowState:
        """
        PLANNER: Analyze user goals and plan Excel operations
        This replaces the Excel Agent's internal planner
        """
        print(f"📋 Planning Phase - Analyzing goals for {len(state['uploaded_files'])} files")
        
        try:
            # Initialize queue client
            queue_client = GRPCQueueClient(
                queue_endpoint=self.queue_config.queue_endpoint,
                queue_name=self.queue_config.queue_name
            )
            await queue_client.connect()
            state["queue_client"] = queue_client
            
            # Send planning status
            await queue_client.send_status_update(
                session_id=state["session_id"],
                status="planning", 
                progress=0.1
            )
            
            planned_operations = []
            
            # Plan operations for each file
            for i, (file_info, user_goal) in enumerate(zip(state["uploaded_files"], state["user_goals"])):
                print(f"📊 Planning for file {i+1}: {file_info.get('filename')}")
                
                # Use Claude to analyze goal and plan operations
                file_plan = await self._plan_operations_for_file(file_info, user_goal, i)
                planned_operations.append(file_plan)
            
            # Determine execution strategy
            execution_strategy = self._determine_execution_strategy(planned_operations)
            
            state["planned_operations"] = planned_operations
            state["execution_strategy"] = execution_strategy
            state["current_stage"] = "planned"
            
            # Send planning complete status
            await queue_client.send_status_update(
                session_id=state["session_id"],
                status="planning_complete",
                progress=0.2
            )
            
            print(f"✅ Planning complete: {len(planned_operations)} file plans created")
            print(f"📋 Execution strategy: {execution_strategy['mode']} mode")
            
            return state
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            state["errors"].append(error_msg)
            if state.get("queue_client"):
                await state["queue_client"].send_error(
                    session_id=state["session_id"],
                    error_message=error_msg
                )
            return state
    
    async def _execute_goal_node(self, state: WorkflowState) -> WorkflowState:
        """
        EXECUTOR: Conditionally execute planned operations
        This replaces the Excel Agent's internal executor
        """
        print(f"⚡ Execution Phase - Running planned operations")
        
        try:
            # Send execution status
            if state.get("queue_client"):
                await state["queue_client"].send_status_update(
                    session_id=state["session_id"],
                    status="executing",
                    progress=0.3
                )
            
            execution_strategy = state["execution_strategy"]
            planned_operations = state["planned_operations"]
            
            if execution_strategy["mode"] == "parallel" and len(planned_operations) > 1:
                # Parallel execution for multiple files
                print(f"🔄 Running parallel execution for {len(planned_operations)} files")
                
                tasks = []
                for file_plan in planned_operations:
                    task = self._execute_file_operations(state["session_id"], file_plan)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
            else:
                # Sequential execution
                print(f"📝 Running sequential execution")
                results = []
                
                for i, file_plan in enumerate(planned_operations):
                    # Progress update
                    progress = 0.3 + (0.4 * (i + 1) / len(planned_operations))
                    if state.get("queue_client"):
                        await state["queue_client"].send_status_update(
                            session_id=state["session_id"],
                            status=f"executing_file_{i+1}",
                            progress=progress
                        )
                    
                    result = await self._execute_file_operations(state["session_id"], file_plan)
                    results.append(result)
            
            # Process results and send intermediate updates
            valid_results = []
            operation_outputs = {}
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"File {i+1} execution failed: {str(result)}"
                    state["errors"].append(error_msg)
                else:
                    valid_results.append(result)
                    
                    # Send intermediate result to queue
                    if state.get("queue_client"):
                        await state["queue_client"].send_intermediate_result(
                            session_id=state["session_id"],
                            operation_id=f"file_{i}",
                            operation_name="excel_operations",
                            result_data=result
                        )
                    
                    # Collect operation outputs
                    file_operations = result.get("operation_results", {})
                    operation_outputs[f"file_{i}"] = file_operations
            
            state["execution_results"] = valid_results
            state["operation_outputs"] = operation_outputs
            state["current_stage"] = "executed"
            
            print(f"✅ Execution complete: {len(valid_results)} successful, {len(results) - len(valid_results)} failed")
            
            return state
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            state["errors"].append(error_msg)
            if state.get("queue_client"):
                await state["queue_client"].send_error(
                    session_id=state["session_id"],
                    error_message=error_msg
                )
            return state
    
    async def _prepare_output_node(self, state: WorkflowState) -> WorkflowState:
        """
        FORMATTER: Consolidate and format execution results
        """
        print(f"🎯 Preparation Phase - Formatting outputs")
        
        try:
            # Send preparation status
            if state.get("queue_client"):
                await state["queue_client"].send_status_update(
                    session_id=state["session_id"],
                    status="preparing",
                    progress=0.8
                )
            
            prepared_outputs = []
            
            for result in state["execution_results"]:
                if result.get("success"):
                    # Format each successful result
                    prepared_output = {
                        "source_file": result.get("source_file", {}),
                        "workbook_path": result.get("excel_workbook_path"),
                        "workbook_summary": result.get("workbook_summary"),
                        "visualizations": result.get("visualizations", []),
                        "html_summaries": result.get("html_summaries", []),
                        "operations_performed": list(result.get("operation_results", {}).keys()),
                        "preparation_timestamp": datetime.now().isoformat()
                    }
                    
                    # Apply your custom formatting logic
                    prepared_output = self._apply_custom_formatting(prepared_output)
                    prepared_outputs.append(prepared_output)
            
            state["prepared_outputs"] = prepared_outputs
            state["current_stage"] = "prepared"
            
            print(f"✅ Preparation complete: {len(prepared_outputs)} outputs formatted")
            
            return state
            
        except Exception as e:
            error_msg = f"Preparation failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    async def _finish_excel_node(self, state: WorkflowState) -> WorkflowState:
        """
        DELIVERER: Final delivery and notifications
        """
        print(f"🏁 Finishing Phase - Final delivery")
        
        try:
            final_deliverables = []
            
            # Create final deliverables
            for prepared_output in state["prepared_outputs"]:
                deliverable = self._create_final_deliverable(prepared_output)
                final_deliverables.append(deliverable)
            
            # Send final result to queue
            if state.get("queue_client"):
                consolidated_result = {
                    "session_id": state["session_id"],
                    "total_files": len(state["uploaded_files"]),
                    "successful_files": len(final_deliverables),
                    "failed_files": len(state["errors"]),
                    "deliverables": final_deliverables,
                    "execution_summary": self._create_execution_summary(state)
                }
                
                await state["queue_client"].send_final_result(
                    session_id=state["session_id"],
                    final_result=consolidated_result
                )
                
                # Send completion status
                await state["queue_client"].send_status_update(
                    session_id=state["session_id"],
                    status="completed",
                    progress=1.0
                )
                
                # Disconnect
                await state["queue_client"].disconnect()
            
            state["final_deliverables"] = final_deliverables
            state["current_stage"] = "completed"
            
            print(f"✅ Workflow completed: {len(final_deliverables)} deliverables ready")
            
            return state
            
        except Exception as e:
            error_msg = f"Finishing failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    async def _plan_operations_for_file(self, file_info: Dict[str, Any], user_goal: str, file_index: int) -> Dict[str, Any]:
        """Plan what Excel operations to perform for a single file"""
        
        # Use simplified planning logic (in production, you'd use Claude here)
        operations_plan = self._simple_operation_planning(user_goal, file_info)
        
        return {
            "file_index": file_index,
            "file_info": file_info,
            "user_goal": user_goal,
            "operations": operations_plan["operations"],
            "dependencies": operations_plan["dependencies"],
            "parameters": operations_plan["parameters"]
        }
    
    def _simple_operation_planning(self, user_goal: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified operation planning logic"""
        goal_lower = user_goal.lower()
        operations = []
        dependencies = {}
        parameters = {}
        
        # Basic planning logic based on keywords
        if "filter" in goal_lower or "show only" in goal_lower:
            operations.append("filter")
            parameters["filter"] = {"extract_from_goal": True}
        
        if "group" in goal_lower or "aggregate" in goal_lower or "sum" in goal_lower:
            operations.append("groupby")
            parameters["groupby"] = {"extract_from_goal": True}
        
        if "pivot" in goal_lower or "cross-tab" in goal_lower:
            operations.append("pivot")
            parameters["pivot"] = {"extract_from_goal": True}
        
        if "chart" in goal_lower or "graph" in goal_lower or "visualiz" in goal_lower:
            operations.append("visualization")
            parameters["visualization"] = {"extract_from_goal": True}
        
        if "summary" in goal_lower or "report" in goal_lower or "analyze" in goal_lower:
            operations.append("sheet_summary")
            parameters["sheet_summary"] = {"include_visualizations": True}
        
        # Default to basic analysis if no specific operations identified
        if not operations:
            operations = ["column_summary", "sheet_summary"]
            parameters = {
                "column_summary": {},
                "sheet_summary": {"include_visualizations": True}
            }
        
        return {
            "operations": operations,
            "dependencies": dependencies,
            "parameters": parameters
        }
    
    def _determine_execution_strategy(self, planned_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine whether to run in parallel or sequential mode"""
        
        # Parallel if multiple files and no cross-file dependencies
        if len(planned_operations) > 1:
            return {
                "mode": "parallel",
                "max_concurrent": min(len(planned_operations), 3),  # Limit concurrency
                "reason": "Multiple files with independent operations"
            }
        else:
            return {
                "mode": "sequential", 
                "reason": "Single file or dependent operations"
            }
    
    async def _execute_file_operations(self, session_id: str, file_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planned operations for a single file"""
        
        # Create an Excel Agent instance for this file
        excel_agent = ExcelAgentSupervisor(self.claude_client)
        
        # Convert the plan into an Excel Agent goal
        enhanced_goal = self._convert_plan_to_goal(file_plan)
        
        # Execute using Excel Agent
        excel_input = {
            "goal": enhanced_goal,
            "data_path": file_plan["file_info"].get("file_path"),
            "data": file_plan["file_info"].get("data"),
            "sheet_name": file_plan["file_info"].get("sheet_name"),
            "s3_bucket": file_plan["file_info"].get("s3_bucket")
        }
        
        result = excel_agent.run(**excel_input)
        
        # Add file context
        result["source_file"] = file_plan["file_info"]
        result["file_index"] = file_plan["file_index"]
        result["planned_operations"] = file_plan["operations"]
        
        return result
    
    def _convert_plan_to_goal(self, file_plan: Dict[str, Any]) -> str:
        """Convert operation plan back to a goal string for Excel Agent"""
        
        operations = file_plan["operations"]
        user_goal = file_plan["user_goal"]
        
        # Create enhanced goal that specifies the operations
        enhanced_goal = f"{user_goal}\n\nSpecific operations to perform: {', '.join(operations)}"
        
        return enhanced_goal
    
    def _apply_custom_formatting(self, prepared_output: Dict[str, Any]) -> Dict[str, Any]:
        """Apply your custom formatting logic"""
        # Add your organization's specific formatting requirements
        prepared_output["format_version"] = "v2.0"
        prepared_output["formatted_at"] = datetime.now().isoformat()
        
        return prepared_output
    
    def _create_final_deliverable(self, prepared_output: Dict[str, Any]) -> str:
        """Create final deliverable path/identifier"""
        return prepared_output.get("workbook_path", "")
    
    def _create_execution_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Create execution summary for reporting"""
        return {
            "total_operations": sum(
                len(result.get("operations", [])) 
                for result in state["execution_results"]
            ),
            "total_visualizations": sum(
                len(result.get("visualizations", [])) 
                for result in state["execution_results"]
            ),
            "execution_time": "calculated_duration",
            "files_processed": len(state["execution_results"])
        }


# Helper function to create initial state
def create_workflow_state(
    session_id: str,
    uploaded_files: List[Dict[str, Any]], 
    user_goals: List[str]
) -> WorkflowState:
    """Create initial workflow state"""
    
    return WorkflowState(
        session_id=session_id,
        uploaded_files=uploaded_files,
        user_goals=user_goals,
        planned_operations=[],
        execution_strategy={},
        execution_results=[],
        operation_outputs={},
        prepared_outputs=[],
        final_deliverables=[],
        errors=[],
        current_stage="initialized",
        queue_client=None
    ) 