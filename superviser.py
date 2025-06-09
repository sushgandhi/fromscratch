"""
Supervisor Agent for Excel Agent using LangGraph

This implementation uses a custom tool execution pattern rather than ToolNode
since we have deterministic operation planning and custom tool interfaces.
ToolNode is more suitable for AI-driven tool calling scenarios.
"""
from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import uuid

from .state import AgentState, Operation
from ..tools.filter_tool import FilterTool, FilterInput
from ..tools.pivot_tool import PivotTool, PivotInput
from ..tools.groupby_tool import GroupByTool, GroupByInput
from ..tools.visualization_tool import VisualizationTool, VisualizationInput
from ..tools.summary_tools import (
    ColumnSummaryTool, ColumnSummaryInput,
    SheetSummaryTool, SheetSummaryInput,
    WorkbookSummaryTool, WorkbookSummaryInput
)
from ..utils.claude_client import get_claude_client
from ..utils.excel_output import ExcelWorkbookManager
import os


class ExcelAgentSupervisor:
    """Supervisor agent that manages Excel operations"""
    
    def __init__(self, claude_client):
        """
        Initialize the supervisor with a Claude client
        
        Args:
            claude_client: Pre-configured Claude client instance
        """
        self.llm = claude_client
        
        # Initialize tools
        self.tools = {
            "filter": FilterTool(),
            "pivot": PivotTool(),
            "groupby": GroupByTool(),
            "visualization": VisualizationTool(),
            "column_summary": ColumnSummaryTool(),
            "sheet_summary": SheetSummaryTool(),
            "workbook_summary": WorkbookSummaryTool()
        }
        
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor",
            self._should_continue,
            {
                "continue": "executor",
                "review": "reviewer",
                "finalize": "finalizer",
                "end": END
            }
        )
        
        # Finalizer always goes to END
        workflow.add_edge("finalizer", END)
        workflow.add_conditional_edges(
            "reviewer",
            self._review_decision,
            {
                "replan": "planner",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Plan the operations needed to achieve the goal"""
        
        # Check if we already have operations planned
        if state["operations"] and state["iteration_count"] == 0:
            return state
        
        # Increment iteration count
        state["iteration_count"] += 1
        
        # Check max iterations
        if state["iteration_count"] > state["max_iterations"]:
            state["error_message"] = "Maximum iterations reached"
            return state
        
        planner_prompt = ChatPromptTemplate.from_template("""
        You are an Excel data analysis expert. Given a user goal, break it down into a sequence of operations.

        Available tools:
        - filter: Filter data based on column conditions (params: column, value, operator)
        - pivot: Create pivot tables (params: index, columns, values, aggfunc)
        - groupby: Group data and apply aggregations (params: group_by, aggregations)
        - visualization: Create charts and graphs (params: chart_type, x_column, y_column, color_column, title)
        - column_summary: Get statistical summaries of columns (params: column)
        - sheet_summary: Get comprehensive sheet analysis
        - workbook_summary: Get analysis of entire workbook

        User Goal: {goal}
        Current Data Available: {has_data}
        Data Path: {data_path}
        Sheet Name: {sheet_name}
        User Clarifications: {clarifications}
        Previous Operations: {completed_operations}
        Previous Results: {operation_results}

        Create a list of operations in JSON format. Each operation should have:
        - operation_id: unique identifier
        - tool_name: one of the available tools
        - description: what this operation does
        - parameters: tool-specific parameters
        - depends_on: operation_id this depends on (null if none)

        Example format:
        [
            {{
                "operation_id": "op1",
                "tool_name": "filter",
                "description": "Filter sales data for product A",
                "parameters": {{
                    "column": "product",
                    "value": "A",
                    "operator": "=="
                }},
                "depends_on": null
            }},
            {{
                "operation_id": "op2",
                "tool_name": "visualization",
                "description": "Create bar chart showing sales by date",
                "parameters": {{
                    "chart_type": "bar",
                    "x_column": "date",
                    "y_column": "sales",
                    "title": "Sales by Date"
                }},
                "depends_on": "op1"
            }}
        ]
        
        Important for visualization:
        - Always specify x_column and y_column explicitly
        - chart_type can be: bar, line, scatter, pie, histogram, box, heatmap
        - For "showing X by Y", X is usually y_column and Y is usually x_column

        Only return the JSON array, no other text.
        """)
        
        has_data = "Yes" if state["current_data"] else "No"
        
        response = self.llm.invoke(
            planner_prompt.format(
                goal=state["goal"],
                has_data=has_data,
                data_path=state["data_path"],
                sheet_name=state.get("sheet_name", "None"),
                clarifications=state.get("clarifications", {}),
                completed_operations=state["completed_operations"],
                operation_results=list(state["operation_results"].keys())
            )
        )
        
        try:
            print(f"🧠 Planner response: {response.content}")
            operations_data = json.loads(response.content)
            print(f"📋 Parsed operations: {operations_data}")
            operations = [Operation(**op) for op in operations_data]
            state["operations"] = operations
            state["current_operation_index"] = 0
        except Exception as e:
            print(f"❌ Failed to parse planner response: {e}")
            print(f"   Raw response: {response.content}")
            state["error_message"] = f"Failed to parse operations: {str(e)}"
        
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Execute the current operation"""
        
        if state["error_message"]:
            return state
        
        operations = state["operations"]
        current_index = state["current_operation_index"]
        
        if current_index >= len(operations):
            return state
        
        current_op = operations[current_index]
        
        # Check dependencies
        if current_op.depends_on and current_op.depends_on not in state["completed_operations"]:
            state["error_message"] = f"Dependency {current_op.depends_on} not completed"
            return state
        
        # Get the tool
        tool = self.tools.get(current_op.tool_name)
        if not tool:
            state["error_message"] = f"Unknown tool: {current_op.tool_name}"
            return state
        
        # Prepare input data
        input_params = current_op.parameters.copy()
        
        # If this operation depends on another, use its result as input data
        if current_op.depends_on:
            prev_result = state["operation_results"][current_op.depends_on]
            if prev_result.get("data"):
                input_params["data"] = prev_result["data"]
                input_params["data_path"] = None
        else:
            # Use initial data
            if state["current_data"]:
                input_params["data"] = state["current_data"]
                input_params["data_path"] = None
            else:
                input_params["data_path"] = state["data_path"]
                input_params["data"] = None
                # Include sheet name if available
                if state.get("sheet_name"):
                    input_params["sheet_name"] = state["sheet_name"]
        
        # Create appropriate input object
        try:
            print(f"🔧 Executing {current_op.tool_name} with params: {input_params}")
            
            if current_op.tool_name == "filter":
                tool_input = FilterInput(**input_params)
            elif current_op.tool_name == "pivot":
                tool_input = PivotInput(**input_params)
            elif current_op.tool_name == "groupby":
                tool_input = GroupByInput(**input_params)
            elif current_op.tool_name == "visualization":
                tool_input = VisualizationInput(**input_params)
                print(f"📊 Visualization input created: chart_type={tool_input.chart_type}, x_column={tool_input.x_column}, y_column={tool_input.y_column}")
            elif current_op.tool_name == "column_summary":
                tool_input = ColumnSummaryInput(**input_params)
            elif current_op.tool_name == "sheet_summary":
                tool_input = SheetSummaryInput(**input_params)
            elif current_op.tool_name == "workbook_summary":
                tool_input = WorkbookSummaryInput(**input_params)
            else:
                state["error_message"] = f"No input class for tool: {current_op.tool_name}"
                return state
            
            # Execute the tool
            result = tool.execute(tool_input)
            
            # Store result
            state["operation_results"][current_op.operation_id] = {
                "success": result.success,
                "data": result.data,
                "output_path": result.output_path,
                "metadata": result.metadata,
                "error_message": result.error_message
            }
            
            # Add to Excel workbook if successful and has data
            if result.success and result.data and state.get("excel_workbook_manager"):
                workbook = state["excel_workbook_manager"]
                workbook.add_operation_result(
                    operation_id=current_op.operation_id,
                    operation_name=current_op.tool_name,
                    data=result.data,
                    metadata=result.metadata,
                    description=current_op.description
                )
            
            if result.success:
                state["completed_operations"].append(current_op.operation_id)
                state["current_operation_index"] += 1
                
                # Update current data if this operation produces data
                if result.data:
                    state["current_data"] = result.data
            else:
                state["error_message"] = f"Operation {current_op.operation_id} failed: {result.error_message}"
            
        except Exception as e:
            state["error_message"] = f"Failed to execute operation {current_op.operation_id}: {str(e)}"
        
        return state
    
    def _reviewer_node(self, state: AgentState) -> AgentState:
        """Review the results and determine if goal is achieved"""
        
        reviewer_prompt = ChatPromptTemplate.from_template("""
        Review the execution results and determine if the user's goal has been achieved.

        Original Goal: {goal}
        Completed Operations: {completed_operations}
        Operation Results: {operation_results}
        Error Message: {error_message}

        Based on the results, determine if:
        1. The goal has been fully achieved (respond: "achieved")
        2. More operations are needed (respond: "replan")
        3. There's an error that can't be recovered (respond: "error")

        Only respond with one word: "achieved", "replan", or "error"
        """)
        
        response = self.llm.invoke(
            reviewer_prompt.format(
                goal=state["goal"],
                completed_operations=state["completed_operations"],
                operation_results=state["operation_results"],
                error_message=state["error_message"]
            )
        )
        
        decision = response.content.strip().lower()
        
        if decision == "achieved":
            # Compile final result
            state["final_result"] = {
                "goal": state["goal"],
                "completed_operations": state["completed_operations"],
                "operation_results": state["operation_results"],
                "success": True
            }
        elif decision == "error":
            state["final_result"] = {
                "goal": state["goal"],
                "error": state["error_message"],
                "success": False
            }
        
        return state
    
    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Create the final result with Excel workbook and S3 upload"""
        print(f"🎯 All operations completed! Creating final result...")
        print(f"   Operations: {len(state['operations'])}")
        print(f"   Completed: {state['completed_operations']}")
        print(f"   Results: {list(state['operation_results'].keys())}")
        
        # Create Excel workbook with all results
        excel_s3_path = None
        if state.get("excel_workbook_manager"):
            workbook = state["excel_workbook_manager"]
            
            # Add final result sheet
            if state.get("current_data"):
                summary = {
                    "goal": state["goal"],
                    "operations_count": len(state["operations"]),
                    "completed_operations": state["completed_operations"],
                    "success": True
                }
                workbook.add_final_result(state["current_data"], summary)
            
            # Add summary sheet with operation details
            operations_summary = []
            for op in state["operations"]:
                operations_summary.append({
                    "operation_id": op.operation_id,
                    "tool": op.tool_name,
                    "description": op.description,
                    "result": state["operation_results"].get(op.operation_id, {})
                })
            workbook.add_summary_sheet(operations_summary)
            
            # Upload to S3 or save locally
            if state.get("s3_bucket"):
                try:
                    excel_s3_path = workbook.upload_to_s3(state["s3_bucket"])
                    state["final_excel_s3_path"] = excel_s3_path
                    print(f"📤 Excel workbook uploaded to: {excel_s3_path}")
                except Exception as e:
                    print(f"❌ Failed to upload to S3: {e}")
                    # Fallback to local save
                    local_path = workbook.create_local_file()
                    excel_s3_path = local_path
                    print(f"💾 Excel workbook saved locally: {local_path}")
            else:
                # Save locally
                local_path = workbook.create_local_file()
                excel_s3_path = local_path
                print(f"💾 Excel workbook saved locally: {local_path}")
        
        state["final_result"] = {
            "goal": state["goal"],
            "operations": [{"operation_id": op.operation_id, "tool": op.tool_name, "description": op.description, "result": state["operation_results"].get(op.operation_id, {})} for op in state["operations"]],
            "results": state["operation_results"],
            "excel_workbook_path": excel_s3_path,
            "workbook_summary": state["excel_workbook_manager"].get_summary_info() if state.get("excel_workbook_manager") else None,
            "success": True
        }
        print(f"✅ Final result created successfully!")
        if excel_s3_path:
            print(f"📊 Complete Excel workbook: {excel_s3_path}")
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if execution should continue (reviewer temporarily disabled)"""
        if state["error_message"]:
            return "end"
        
        if state["current_operation_index"] >= len(state["operations"]):
            # All operations completed, go to finalizer
            return "finalize"
        
        return "continue"
    
    def _review_decision(self, state: AgentState) -> str:
        """Determine next step based on review"""
        if state["final_result"]:
            return "end"
        
        return "replan"
    
    def run(self, goal: str, data_path: Optional[str] = None, data: Optional[List[Dict]] = None, sheet_name: Optional[str] = None, clarifications: Optional[Dict[str, str]] = None, s3_bucket: Optional[str] = None) -> Dict[str, Any]:
        """Run the Excel Agent with the given goal"""
        
        # Initialize Excel workbook manager
        excel_workbook_manager = ExcelWorkbookManager(goal=goal)
        
        # Get S3 bucket from environment if not provided
        if not s3_bucket:
            s3_bucket = os.getenv('AWS_S3_BUCKET')
        
        initial_state = {
            "messages": [],
            "goal": goal,
            "current_data": data,
            "data_path": data_path,
            "sheet_name": sheet_name,
            "clarifications": clarifications or {},
            "operations": [],
            "completed_operations": [],
            "operation_results": {},
            "current_operation_index": 0,
            "error_message": None,
            "final_result": None,
            "iteration_count": 0,
            "max_iterations": 10,
            "excel_workbook_manager": excel_workbook_manager,
            "s3_bucket": s3_bucket,
            "final_excel_s3_path": None
        }
        
        final_state = self.graph.invoke(initial_state)
        
        print(f"🏁 Final state keys: {list(final_state.keys())}")
        print(f"🏁 Final result: {final_state.get('final_result')}")
        print(f"🏁 Error message: {final_state.get('error_message')}")
        
        return final_state["final_result"] or {
            "goal": goal,
            "error": final_state.get("error_message", "Unknown error"),
            "success": False
        } 
