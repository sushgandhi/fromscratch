"""
Subgraph Adapter for Excel Agent Integration
Adapts Excel Agent to work as a subgraph in larger LangGraph workflows
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
import asyncio
import uuid
from datetime import datetime

from ..agent.supervisor import ExcelAgentSupervisor
from ..utils.claude_client import get_claude_client
from .grpc_queue_client import GRPCQueueClient, QueueConfig


class SubgraphState(TypedDict):
    """State structure for the larger workflow system"""
    session_id: str
    files: List[Dict[str, Any]]  # List of uploaded files
    analysis_goals: List[str]  # Goals for each file or combined
    excel_results: List[Dict[str, Any]]  # Results from Excel Agent
    current_file_index: int
    total_files: int
    parallel_mode: bool
    queue_client: Optional[GRPCQueueClient]
    error_messages: List[str]
    status: str  # "analyzing", "executing", "preparing", "finished"


class ExcelAgentSubgraph:
    """Excel Agent integrated as a subgraph"""
    
    def __init__(self, 
                 queue_config: Optional[QueueConfig] = None,
                 enable_parallel: bool = True):
        """
        Initialize Excel Agent subgraph
        
        Args:
            queue_config: Configuration for gRPC queue integration
            enable_parallel: Whether to enable parallel processing
        """
        self.queue_config = queue_config
        self.enable_parallel = enable_parallel
        self.excel_agent = None
        self.queue_client = None
        
    async def initialize(self):
        """Initialize the Excel Agent and queue client"""
        # Initialize Excel Agent
        claude_client = get_claude_client()
        self.excel_agent = ExcelAgentSupervisor(claude_client)
        
        # Initialize queue client if configured
        if self.queue_config:
            self.queue_client = GRPCQueueClient(
                queue_endpoint=self.queue_config.queue_endpoint,
                queue_name=self.queue_config.queue_name
            )
            await self.queue_client.connect()
    
    def create_subgraph(self) -> StateGraph:
        """Create the Excel Agent subgraph"""
        
        # Create subgraph
        subgraph = StateGraph(SubgraphState)
        
        # Add nodes
        subgraph.add_node("excel_planner", self._excel_planner_node)
        subgraph.add_node("excel_executor", self._excel_executor_node)
        subgraph.add_node("excel_finalizer", self._excel_finalizer_node)
        subgraph.add_node("send_intermediate", self._send_intermediate_node)
        subgraph.add_node("send_final", self._send_final_node)
        
        # Add edges
        subgraph.add_edge("excel_planner", "excel_executor")
        subgraph.add_edge("excel_executor", "send_intermediate")
        subgraph.add_edge("send_intermediate", "excel_finalizer")
        subgraph.add_edge("excel_finalizer", "send_final")
        subgraph.add_edge("send_final", END)
        
        # Set entry point
        subgraph.set_entry_point("excel_planner")
        
        return subgraph.compile()
    
    async def _excel_planner_node(self, state: SubgraphState) -> SubgraphState:
        """Planning phase - analyze goals and prepare execution"""
        print(f"📋 Excel Agent Planning Phase - Session: {state['session_id']}")
        
        try:
            # Send status update
            if self.queue_client:
                await self.queue_client.send_status_update(
                    session_id=state["session_id"],
                    status="planning",
                    progress=0.1
                )
            
            # For each file, prepare analysis goals
            enriched_goals = []
            for i, (file_info, goal) in enumerate(zip(state["files"], state["analysis_goals"])):
                enriched_goal = {
                    "file_index": i,
                    "file_info": file_info,
                    "original_goal": goal,
                    "enhanced_goal": self._enhance_goal_with_context(goal, file_info)
                }
                enriched_goals.append(enriched_goal)
            
            state["analysis_goals"] = enriched_goals
            state["status"] = "planned"
            
            print(f"✅ Planning complete for {len(enriched_goals)} files")
            return state
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            state["error_messages"].append(error_msg)
            if self.queue_client:
                await self.queue_client.send_error(
                    session_id=state["session_id"],
                    error_message=error_msg
                )
            return state
    
    async def _excel_executor_node(self, state: SubgraphState) -> SubgraphState:
        """Execution phase - run Excel analysis for each file"""
        print(f"⚡ Excel Agent Execution Phase - Session: {state['session_id']}")
        
        try:
            # Send status update
            if self.queue_client:
                await self.queue_client.send_status_update(
                    session_id=state["session_id"],
                    status="executing",
                    progress=0.3
                )
            
            results = []
            
            if state["parallel_mode"] and len(state["analysis_goals"]) > 1:
                # Parallel execution for multiple files
                print(f"🔄 Running parallel execution for {len(state['analysis_goals'])} files")
                tasks = []
                for goal_info in state["analysis_goals"]:
                    task = self._execute_single_file(state["session_id"], goal_info)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
            else:
                # Sequential execution
                print(f"📝 Running sequential execution for {len(state['analysis_goals'])} files")
                for i, goal_info in enumerate(state["analysis_goals"]):
                    # Update progress
                    progress = 0.3 + (0.4 * (i + 1) / len(state["analysis_goals"]))
                    if self.queue_client:
                        await self.queue_client.send_status_update(
                            session_id=state["session_id"],
                            status=f"executing_file_{i+1}",
                            progress=progress
                        )
                    
                    result = await self._execute_single_file(state["session_id"], goal_info)
                    results.append(result)
            
            # Process results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"File {i+1} execution failed: {str(result)}"
                    state["error_messages"].append(error_msg)
                else:
                    valid_results.append(result)
            
            state["excel_results"] = valid_results
            state["status"] = "executed"
            
            print(f"✅ Execution complete: {len(valid_results)} successful, {len(results) - len(valid_results)} failed")
            return state
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            state["error_messages"].append(error_msg)
            if self.queue_client:
                await self.queue_client.send_error(
                    session_id=state["session_id"],
                    error_message=error_msg
                )
            return state
    
    async def _execute_single_file(self, session_id: str, goal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Excel analysis for a single file"""
        file_info = goal_info["file_info"]
        enhanced_goal = goal_info["enhanced_goal"]
        
        print(f"📊 Processing file: {file_info.get('filename', 'unknown')}")
        
        # Prepare Excel Agent input
        excel_input = {
            "goal": enhanced_goal,
            "data_path": file_info.get("file_path"),
            "data": file_info.get("data"),  # If data is already loaded
            "sheet_name": file_info.get("sheet_name"),
            "s3_bucket": file_info.get("s3_bucket")
        }
        
        # Run Excel Agent
        result = self.excel_agent.run(**excel_input)
        
        # Add file context to result
        result["source_file"] = file_info
        result["file_index"] = goal_info["file_index"]
        result["execution_timestamp"] = datetime.now().isoformat()
        
        return result
    
    async def _excel_finalizer_node(self, state: SubgraphState) -> SubgraphState:
        """Finalization phase - prepare consolidated results"""
        print(f"🎯 Excel Agent Finalization Phase - Session: {state['session_id']}")
        
        try:
            # Send status update
            if self.queue_client:
                await self.queue_client.send_status_update(
                    session_id=state["session_id"],
                    status="finalizing",
                    progress=0.8
                )
            
            # Consolidate results from multiple files
            consolidated_result = {
                "session_id": state["session_id"],
                "total_files_processed": len(state["excel_results"]),
                "successful_files": len([r for r in state["excel_results"] if r.get("success")]),
                "failed_files": len(state["error_messages"]),
                "individual_results": state["excel_results"],
                "error_messages": state["error_messages"],
                "consolidated_workbooks": [],
                "summary_insights": self._generate_multi_file_insights(state["excel_results"])
            }
            
            # Collect all workbook paths
            for result in state["excel_results"]:
                if result.get("success") and result.get("excel_workbook_path"):
                    consolidated_result["consolidated_workbooks"].append({
                        "file_index": result.get("file_index"),
                        "source_file": result.get("source_file", {}).get("filename"),
                        "workbook_path": result["excel_workbook_path"],
                        "workbook_summary": result.get("workbook_summary")
                    })
            
            state["final_consolidated_result"] = consolidated_result
            state["status"] = "finalized"
            
            print(f"✅ Finalization complete")
            return state
            
        except Exception as e:
            error_msg = f"Finalization failed: {str(e)}"
            state["error_messages"].append(error_msg)
            if self.queue_client:
                await self.queue_client.send_error(
                    session_id=state["session_id"],
                    error_message=error_msg
                )
            return state
    
    async def _send_intermediate_node(self, state: SubgraphState) -> SubgraphState:
        """Send intermediate results to gRPC queue"""
        if not self.queue_client:
            return state
        
        print(f"📤 Sending intermediate results to queue")
        
        try:
            # Send each file's result as intermediate
            for i, result in enumerate(state["excel_results"]):
                await self.queue_client.send_intermediate_result(
                    session_id=state["session_id"],
                    operation_id=f"file_{i}",
                    operation_name="excel_analysis",
                    result_data=result
                )
            
            return state
            
        except Exception as e:
            print(f"⚠️ Failed to send intermediate results: {e}")
            return state
    
    async def _send_final_node(self, state: SubgraphState) -> SubgraphState:
        """Send final consolidated result to gRPC queue"""
        if not self.queue_client:
            return state
        
        print(f"📤 Sending final result to queue")
        
        try:
            await self.queue_client.send_final_result(
                session_id=state["session_id"],
                final_result=state["final_consolidated_result"]
            )
            
            # Send completion status
            await self.queue_client.send_status_update(
                session_id=state["session_id"],
                status="completed",
                progress=1.0
            )
            
            return state
            
        except Exception as e:
            print(f"⚠️ Failed to send final results: {e}")
            await self.queue_client.send_error(
                session_id=state["session_id"],
                error_message=f"Failed to send results: {str(e)}"
            )
            return state
    
    def _enhance_goal_with_context(self, goal: str, file_info: Dict[str, Any]) -> str:
        """Enhance analysis goal with file context"""
        enhanced_goal = goal
        
        # Add file-specific context
        if file_info.get("filename"):
            enhanced_goal += f"\n\nFile Context: Analyzing data from {file_info['filename']}"
        
        if file_info.get("sheet_name"):
            enhanced_goal += f"\nSheet: {file_info['sheet_name']}"
        
        # Add any file metadata
        if file_info.get("metadata"):
            enhanced_goal += f"\nFile Metadata: {file_info['metadata']}"
        
        return enhanced_goal
    
    def _generate_multi_file_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights across multiple file analyses"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("success")]
        
        insights = {
            "total_files_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "total_operations_performed": sum(
                len(r.get("operations", [])) for r in successful_results
            ),
            "total_visualizations_created": sum(
                len(r.get("visualizations", [])) for r in successful_results
            ),
            "total_html_summaries": sum(
                len(r.get("html_summaries", [])) for r in successful_results
            )
        }
        
        return insights


# Integration helper functions
def create_excel_subgraph_state(
    session_id: str,
    files: List[Dict[str, Any]],
    analysis_goals: List[str],
    parallel_mode: bool = True,
    queue_config: Optional[QueueConfig] = None
) -> SubgraphState:
    """Create initial state for Excel Agent subgraph"""
    
    return SubgraphState(
        session_id=session_id,
        files=files,
        analysis_goals=analysis_goals,
        excel_results=[],
        current_file_index=0,
        total_files=len(files),
        parallel_mode=parallel_mode,
        queue_client=None,
        error_messages=[],
        status="initialized"
    ) 