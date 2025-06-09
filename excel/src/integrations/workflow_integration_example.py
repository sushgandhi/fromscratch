"""
Example: Integrating Excel Agent into Existing LangGraph Workflow
Shows how to modify your current 4-node workflow to include Excel Agent as a subgraph
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
import uuid
from datetime import datetime

from .subgraph_adapter import ExcelAgentSubgraph, QueueConfig
from .grpc_queue_client import GRPCQueueClient


class MainWorkflowState(TypedDict):
    """State for the main workflow system"""
    session_id: str
    uploaded_files: List[Dict[str, Any]]
    analyzed_goals: List[str]
    execution_results: Dict[str, Any]
    prepared_outputs: List[Dict[str, Any]]
    final_excel_outputs: List[str]
    errors: List[str]
    current_stage: str


class IntegratedWorkflow:
    """Main workflow with Excel Agent integrated as subgraph"""
    
    def __init__(self, 
                 queue_endpoint: str = "localhost:50051",
                 enable_parallel: bool = True):
        """Initialize the integrated workflow"""
        self.queue_config = QueueConfig(
            queue_endpoint=queue_endpoint,
            enable_intermediate_results=True,
            enable_status_updates=True
        )
        self.enable_parallel = enable_parallel
        self.excel_subgraph = ExcelAgentSubgraph(
            queue_config=self.queue_config,
            enable_parallel=enable_parallel
        )
    
    def create_main_workflow(self) -> StateGraph:
        """Create the main workflow graph"""
        
        # Create main workflow
        workflow = StateGraph(MainWorkflowState)
        
        # Add your existing nodes
        workflow.add_node("analyze_goal", self._analyze_goal_node)
        workflow.add_node("execute_goal", self._execute_goal_node)  # This will call Excel subgraph
        workflow.add_node("prepare_output", self._prepare_output_node)
        workflow.add_node("finish_excel", self._finish_excel_node)
        
        # Add edges (your existing flow)
        workflow.add_edge("analyze_goal", "execute_goal")
        workflow.add_edge("execute_goal", "prepare_output")
        workflow.add_edge("prepare_output", "finish_excel")
        workflow.add_edge("finish_excel", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_goal")
        
        return workflow.compile()
    
    async def _analyze_goal_node(self, state: MainWorkflowState) -> MainWorkflowState:
        """
        Analyze Goal Node - Your existing logic
        Enhanced to support multiple files and parallel processing
        """
        print(f"📋 Analyzing goals for session: {state['session_id']}")
        
        try:
            # Your existing goal analysis logic here
            analyzed_goals = []
            
            for file_info in state["uploaded_files"]:
                # Analyze each file and determine what analysis is needed
                goal = self._determine_analysis_goal(file_info)
                analyzed_goals.append(goal)
            
            state["analyzed_goals"] = analyzed_goals
            state["current_stage"] = "goals_analyzed"
            
            print(f"✅ Goal analysis complete for {len(analyzed_goals)} files")
            return state
            
        except Exception as e:
            error_msg = f"Goal analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    async def _execute_goal_node(self, state: MainWorkflowState) -> MainWorkflowState:
        """
        Execute Goal Node - NOW CALLS EXCEL AGENT SUBGRAPH
        This is where the Excel Agent integration happens
        """
        print(f"⚡ Executing goals via Excel Agent subgraph")
        
        try:
            # Initialize Excel Agent subgraph
            await self.excel_subgraph.initialize()
            excel_graph = self.excel_subgraph.create_subgraph()
            
            # Prepare input for Excel Agent subgraph
            from .subgraph_adapter import create_excel_subgraph_state
            
            excel_state = create_excel_subgraph_state(
                session_id=state["session_id"],
                files=state["uploaded_files"],
                analysis_goals=state["analyzed_goals"],
                parallel_mode=self.enable_parallel,
                queue_config=self.queue_config
            )
            
            # Execute Excel Agent subgraph
            excel_result = await excel_graph.ainvoke(excel_state)
            
            # Extract results from Excel subgraph
            state["execution_results"] = {
                "excel_results": excel_result.get("excel_results", []),
                "consolidated_result": excel_result.get("final_consolidated_result", {}),
                "errors": excel_result.get("error_messages", [])
            }
            
            state["current_stage"] = "goals_executed"
            
            print(f"✅ Goal execution complete via Excel Agent")
            return state
            
        except Exception as e:
            error_msg = f"Goal execution failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    async def _prepare_output_node(self, state: MainWorkflowState) -> MainWorkflowState:
        """
        Prepare Output Node - Your existing logic
        Enhanced to handle Excel Agent results
        """
        print(f"🎯 Preparing outputs for session: {state['session_id']}")
        
        try:
            prepared_outputs = []
            
            execution_results = state["execution_results"]
            excel_results = execution_results.get("excel_results", [])
            
            for result in excel_results:
                if result.get("success"):
                    # Prepare output for each successful analysis
                    prepared_output = {
                        "source_file": result.get("source_file", {}),
                        "workbook_path": result.get("excel_workbook_path"),
                        "workbook_summary": result.get("workbook_summary"),
                        "visualizations": result.get("visualizations", []),
                        "html_summaries": result.get("html_summaries", []),
                        "preparation_timestamp": datetime.now().isoformat()
                    }
                    
                    # Your additional output preparation logic here
                    prepared_output = self._enhance_output(prepared_output)
                    prepared_outputs.append(prepared_output)
            
            state["prepared_outputs"] = prepared_outputs
            state["current_stage"] = "outputs_prepared"
            
            print(f"✅ Output preparation complete for {len(prepared_outputs)} results")
            return state
            
        except Exception as e:
            error_msg = f"Output preparation failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    async def _finish_excel_node(self, state: MainWorkflowState) -> MainWorkflowState:
        """
        Finish Excel Node - Your existing logic
        Final processing and delivery
        """
        print(f"🏁 Finishing Excel processing for session: {state['session_id']}")
        
        try:
            final_outputs = []
            
            for prepared_output in state["prepared_outputs"]:
                # Your final processing logic here
                final_output = self._finalize_output(prepared_output)
                final_outputs.append(final_output)
            
            state["final_excel_outputs"] = final_outputs
            state["current_stage"] = "completed"
            
            print(f"✅ Excel processing finished - {len(final_outputs)} final outputs")
            return state
            
        except Exception as e:
            error_msg = f"Excel finishing failed: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _determine_analysis_goal(self, file_info: Dict[str, Any]) -> str:
        """
        Determine what analysis goal to use for a file
        Replace with your actual goal determination logic
        """
        filename = file_info.get("filename", "")
        
        # Example logic - replace with your business rules
        if "sales" in filename.lower():
            return "Analyze sales data: create summary statistics, trends, and regional breakdown"
        elif "financial" in filename.lower():
            return "Perform financial analysis: calculate key metrics, ratios, and create charts"
        elif "customer" in filename.lower():
            return "Analyze customer data: segment analysis, behavior patterns, and insights"
        else:
            return "Perform comprehensive data analysis: statistics, visualizations, and insights"
    
    def _enhance_output(self, prepared_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the prepared output with additional information
        Replace with your actual enhancement logic
        """
        # Add your custom enhancements here
        prepared_output["enhanced_at"] = datetime.now().isoformat()
        prepared_output["output_format"] = "enhanced_excel_analysis"
        
        return prepared_output
    
    def _finalize_output(self, prepared_output: Dict[str, Any]) -> str:
        """
        Finalize the output (e.g., generate final paths, notifications)
        Replace with your actual finalization logic
        """
        # Your finalization logic here
        return prepared_output.get("workbook_path", "")


# Example usage function
async def run_integrated_workflow_example():
    """Example of how to run the integrated workflow"""
    
    # Sample input data (replace with your actual data structure)
    sample_files = [
        {
            "filename": "sales_q4.xlsx",
            "file_path": "s3://bucket/sales_q4.xlsx",
            "sheet_name": "Q4_Data",
            "metadata": {"quarter": "Q4", "year": "2024"}
        },
        {
            "filename": "customer_analysis.xlsx", 
            "file_path": "s3://bucket/customer_analysis.xlsx",
            "sheet_name": "Customer_Data",
            "metadata": {"type": "customer_data", "region": "all"}
        }
    ]
    
    # Create workflow
    workflow = IntegratedWorkflow(
        queue_endpoint="localhost:50051",
        enable_parallel=True
    )
    
    # Create workflow graph
    graph = workflow.create_main_workflow()
    
    # Create initial state
    initial_state = MainWorkflowState(
        session_id=str(uuid.uuid4()),
        uploaded_files=sample_files,
        analyzed_goals=[],
        execution_results={},
        prepared_outputs=[],
        final_excel_outputs=[],
        errors=[],
        current_stage="initialized"
    )
    
    # Run workflow
    print("🚀 Starting integrated workflow...")
    final_state = await graph.ainvoke(initial_state)
    
    # Print results
    print(f"\n📊 Workflow Results:")
    print(f"   Session ID: {final_state['session_id']}")
    print(f"   Stage: {final_state['current_stage']}")
    print(f"   Files processed: {len(final_state['uploaded_files'])}")
    print(f"   Final outputs: {len(final_state['final_excel_outputs'])}")
    print(f"   Errors: {len(final_state['errors'])}")
    
    return final_state


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_integrated_workflow_example()) 