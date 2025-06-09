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
    RowSummaryTool, RowSummaryInput,
    SheetSummaryTool, SheetSummaryInput,
    WorkbookSummaryTool, WorkbookSummaryInput
)
from ..utils.claude_client import get_claude_client
from ..utils.excel_output import ExcelWorkbookManager
from ..utils.sheet_intelligence import (
    analyze_sheet_selection, 
    create_clarification_response,
    extract_potential_sheet_names
)
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
            "row_summary": RowSummaryTool(),
            "sheet_summary": SheetSummaryTool(),
            "workbook_summary": WorkbookSummaryTool()
        }
        
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("sheet_analyzer", self._sheet_analyzer_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Add edges
        workflow.set_entry_point("sheet_analyzer")
        workflow.add_conditional_edges(
            "sheet_analyzer",
            self._sheet_analysis_decision,
            {
                "proceed": "planner",
                "clarify": END  # Return to user for clarification
            }
        )
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
    
    def _sheet_analyzer_node(self, state: AgentState) -> AgentState:
        """Analyze sheet selection for all files and handle clarifications"""
        
        # Skip if we've already processed sheet analysis or have clarification responses
        if state["sheet_analysis_results"] or state["sheet_clarification_responses"]:
            return state
        
        files_needing_clarification = []
        
        for file_info in state["files"]:
            # Skip if file doesn't have a data path (e.g., in-memory data)
            if not file_info.data_path:
                continue
                
            # Skip if sheet name is already explicitly set and valid
            if file_info.sheet_name:
                continue
            
            # Analyze sheet selection for this file
            try:
                analysis_result = analyze_sheet_selection(
                    user_query=state["goal"],
                    file_path=file_info.data_path,
                    provided_sheet_name=file_info.sheet_name
                )
                
                state["sheet_analysis_results"][file_info.file_id] = analysis_result
                
                if analysis_result.get("needs_clarification"):
                    files_needing_clarification.append(file_info.file_id)
                    state["pending_sheet_clarifications"].append(file_info.file_id)
                else:
                    # Update file info with selected sheet
                    selected_sheet = analysis_result.get("selected_sheet")
                    if selected_sheet:
                        # Update the file info in place
                        for i, f in enumerate(state["files"]):
                            if f.file_id == file_info.file_id:
                                state["files"][i].sheet_name = selected_sheet
                                break
                        
                        print(f"🎯 Auto-selected sheet '{selected_sheet}' for file {file_info.file_id}")
                        print(f"   Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}")
                
            except Exception as e:
                print(f"❌ Sheet analysis failed for {file_info.file_id}: {e}")
                # Don't block execution for sheet analysis failures
                continue
        
        # If any files need clarification, prepare the clarification message
        if files_needing_clarification:
            clarification_messages = []
            
            for file_id in files_needing_clarification:
                analysis = state["sheet_analysis_results"].get(file_id, {})
                clarification = create_clarification_response(analysis)
                
                file_info = next((f for f in state["files"] if f.file_id == file_id), None)
                filename = file_info.filename if file_info else file_id
                
                clarification_messages.append(f"""
📁 **File: {filename}**
❓ {clarification.get('clarification_question', 'Please specify which sheet to use.')}

Available sheets: {', '.join(clarification.get('available_options', []))}
Example response: "{clarification.get('example_response', 'Use sheet Sheet1')}"
""")
            
            # Store clarification message for the user
            state["clarifications"]["sheet_selection"] = "\n".join(clarification_messages)
            
            print("📋 Sheet clarifications needed:")
            print(state["clarifications"]["sheet_selection"])
        
        return state
    
    def _sheet_analysis_decision(self, state: AgentState) -> str:
        """Decide whether to proceed with planning or request clarification"""
        if state["pending_sheet_clarifications"]:
            return "clarify"
        return "proceed"
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Plan the operations needed to achieve the goal for all files"""
        
        # Check if we already have operations planned
        if state["file_operations"] and state["iteration_count"] == 0:
            return state
        
        # Increment iteration count
        state["iteration_count"] += 1
        
        # Check max iterations
        if state["iteration_count"] > state["max_iterations"]:
            state["error_message"] = "Maximum iterations reached"
            return state
        
        # Plan operations for each file
        file_operations = {}
        
        for file_info in state["files"]:
            planner_prompt = ChatPromptTemplate.from_template("""
            You are an Excel data analysis expert. Given a user goal, break it down into a sequence of operations for a specific file.

            Available tools:
            - filter: Filter data based on column conditions (params: column, value, operator)
            - pivot: Create pivot tables (params: index, columns, values, aggfunc)
            - groupby: Group data and apply aggregations (params: group_by, aggregations)
            - visualization: Create charts and graphs (params: chart_type, x_column, y_column, color_column, title)
            - column_summary: Get statistical summaries of columns with LLM text analysis (params: columns, use_llm_for_text, text_sample_size)
            - row_summary: Get row-level summaries and optionally create LLM summary column (params: sample_size, create_summary_column, summary_column_name, max_rows_for_llm)
            - sheet_summary: Get comprehensive sheet analysis with LLM insights (params: include_visualizations, use_llm_analysis, sample_rows)
            - workbook_summary: Get analysis of entire workbook

            User Goal: {goal}
            File ID: {file_id}
            Filename: {filename}
            Data Path: {data_path}
            Sheet Name: {sheet_name}
            User Clarifications: {clarifications}

            Create a list of operations for this specific file in JSON format. Each operation should have:
            - operation_id: unique identifier (prefix with file_id: e.g., "file_0_op1")
            - tool_name: one of the available tools
            - description: what this operation does
            - parameters: tool-specific parameters
            - depends_on: list of operation_ids this depends on (null if none, ["file_0_op1"] for single dependency)

            Example format:
            [
                {{
                    "operation_id": "{file_id}_op1",
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
                    "operation_id": "{file_id}_op2",
                    "tool_name": "visualization",
                    "description": "Create bar chart showing sales by date",
                    "parameters": {{
                        "chart_type": "bar",
                        "x_column": "date",
                        "y_column": "sales",
                        "title": "Sales by Date - {filename}"
                    }},
                    "depends_on": ["{file_id}_op1"]
                }}
            ]
            
            Important guidelines:
            - Always prefix operation_id with {file_id}_
            - For visualization: Always specify x_column and y_column explicitly  
            - For groupby: Map column names to aggregation functions. For multiple aggregations on same column, use list: {{"sales": ["sum", "mean"]}}
            - chart_type can be: bar, line, scatter, pie, histogram, box, heatmap
            - For "showing X by Y", X is usually y_column and Y is usually x_column
            - Use depends_on as a list: null, ["{file_id}_op1"], or ["{file_id}_op1", "{file_id}_op2"]
            - Include filename in chart titles for clarity

            Only return the JSON array, no other text.
            """)
            
            response = self.llm.invoke(
                planner_prompt.format(
                    goal=state["goal"],
                    file_id=file_info.file_id,
                    filename=file_info.filename,
                    data_path=file_info.data_path,
                    sheet_name=file_info.sheet_name or "None",
                    clarifications=state.get("clarifications", {})
                )
            )
            
            try:
                print(f"🧠 Planner response for {file_info.file_id}: {response.content}")
                operations_data = json.loads(response.content)
                print(f"📋 Parsed operations for {file_info.file_id}: {operations_data}")
                operations = [Operation(**op) for op in operations_data]
                file_operations[file_info.file_id] = operations
            except Exception as e:
                print(f"❌ Failed to parse planner response for {file_info.file_id}: {e}")
                print(f"   Raw response: {response.content}")
                state["error_message"] = f"Failed to parse operations for {file_info.file_id}: {str(e)}"
                return state
        
        state["file_operations"] = file_operations
        state["current_operation_index"] = 0

        
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Execute operations for all files in parallel"""
        
        if state["error_message"]:
            return state
        
        if not state["file_operations"]:
            state["error_message"] = "No operations planned"
            return state
        
        # Check if parallel mode is enabled
        if state.get("parallel_mode", True) and len(state["files"]) > 1:
            return self._execute_parallel(state)
        else:
            return self._execute_sequential(state)
    
    def _execute_parallel(self, state: AgentState) -> AgentState:
        """Execute operations for all files in parallel"""
        import asyncio
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        print(f"🚀 Starting parallel execution for {len(state['files'])} files...")
        
        def execute_file_operations(file_info):
            """Execute all operations for a single file"""
            file_id = file_info.file_id
            operations = state["file_operations"].get(file_id, [])
            
            if not operations:
                return file_id, {"error": f"No operations for {file_id}"}
            
            print(f"📁 Processing {file_id} with {len(operations)} operations...")
            
            # File-specific state
            file_state = {
                "current_data": file_info.data,
                "data_path": file_info.data_path,
                "sheet_name": file_info.sheet_name,
                "completed_operations": [],
                "operation_results": {},
                "error_message": None
            }
            
            # Execute operations sequentially within the file
            for op in operations:
                if file_state["error_message"]:
                    break
                
                # Check dependencies
                if op.depends_on:
                    missing_deps = [dep for dep in op.depends_on if dep not in file_state["completed_operations"]]
                    if missing_deps:
                        file_state["error_message"] = f"Dependencies {missing_deps} not completed"
                        break
                
                # Execute operation
                result = self._execute_single_operation(op, file_state, file_info)
                
                # Store result
                file_state["operation_results"][op.operation_id] = result
                
                if result["success"]:
                    file_state["completed_operations"].append(op.operation_id)
                    # Update current data if this operation produces data
                    if result["data"]:
                        file_state["current_data"] = result["data"]
                else:
                    file_state["error_message"] = f"Operation {op.operation_id} failed: {result['error_message']}"
                    break
            
            return file_id, file_state
        
        # Execute all files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(state["files"]), 4)) as executor:
            future_to_file = {executor.submit(execute_file_operations, file_info): file_info.file_id 
                             for file_info in state["files"]}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_id = future_to_file[future]
                try:
                    file_id, file_result = future.result()
                    state["file_results"][file_id] = file_result
                    
                    # Merge operation results into main state
                    state["operation_results"].update(file_result["operation_results"])
                    state["completed_operations"].extend(file_result["completed_operations"])
                    
                    print(f"✅ Completed processing {file_id}")
                    
                except Exception as e:
                    print(f"❌ Error processing {file_id}: {e}")
                    state["file_results"][file_id] = {"error": str(e)}
        
        print(f"🎉 Parallel execution completed!")
        return state
    
    def _execute_sequential(self, state: AgentState) -> AgentState:
        """Execute operations for all files sequentially"""
        print(f"🔄 Starting sequential execution for {len(state['files'])} files...")
        
        for file_info in state["files"]:
            file_id = file_info.file_id
            operations = state["file_operations"].get(file_id, [])
            
            if not operations:
                state["file_results"][file_id] = {"error": f"No operations for {file_id}"}
                continue
            
            print(f"📁 Processing {file_id} with {len(operations)} operations...")
            
            # File-specific state
            file_state = {
                "current_data": file_info.data,
                "data_path": file_info.data_path,
                "sheet_name": file_info.sheet_name,
                "completed_operations": [],
                "operation_results": {},
                "error_message": None
            }
            
            # Execute operations sequentially
            for op in operations:
                if file_state["error_message"]:
                    break
                
                # Check dependencies
                if op.depends_on:
                    missing_deps = [dep for dep in op.depends_on if dep not in file_state["completed_operations"]]
                    if missing_deps:
                        file_state["error_message"] = f"Dependencies {missing_deps} not completed"
                        break
                
                # Execute operation
                result = self._execute_single_operation(op, file_state, file_info)
                
                # Store result
                file_state["operation_results"][op.operation_id] = result
                
                if result["success"]:
                    file_state["completed_operations"].append(op.operation_id)
                    # Update current data if this operation produces data
                    if result["data"]:
                        file_state["current_data"] = result["data"]
                else:
                    file_state["error_message"] = f"Operation {op.operation_id} failed: {result['error_message']}"
                    break
            
            state["file_results"][file_id] = file_state
            
            # Merge operation results into main state
            state["operation_results"].update(file_state["operation_results"])
            state["completed_operations"].extend(file_state["completed_operations"])
            
            print(f"✅ Completed processing {file_id}")
        
        print(f"🎉 Sequential execution completed!")
        return state
    
    def _execute_single_operation(self, operation, file_state, file_info):
        """Execute a single operation"""
        # Get the tool
        tool = self.tools.get(operation.tool_name)
        if not tool:
            return {
                "success": False,
                "data": None,
                "output_path": None,
                "metadata": {},
                "error_message": f"Unknown tool: {operation.tool_name}"
            }
        
        # Prepare input data
        input_params = operation.parameters.copy()
        
        # If this operation depends on others, use the most recent result as input data
        if operation.depends_on:
            # Use the last dependency's data as the primary input
            last_dep = operation.depends_on[-1]
            prev_result = file_state["operation_results"][last_dep]
            if prev_result.get("data"):
                input_params["data"] = prev_result["data"]
                input_params["data_path"] = None
        else:
            # Use initial file data
            if file_state["current_data"]:
                input_params["data"] = file_state["current_data"]
                input_params["data_path"] = None
            else:
                input_params["data_path"] = file_state["data_path"]
                input_params["data"] = None
                # Include sheet name if available
                if file_state.get("sheet_name"):
                    input_params["sheet_name"] = file_state["sheet_name"]
        
        # Create appropriate input object
        try:
            print(f"🔧 Executing {operation.tool_name} for {file_info.file_id} with params: {input_params}")
            
            if operation.tool_name == "filter":
                tool_input = FilterInput(**input_params)
            elif operation.tool_name == "pivot":
                tool_input = PivotInput(**input_params)
            elif operation.tool_name == "groupby":
                tool_input = GroupByInput(**input_params)
            elif operation.tool_name == "visualization":
                # Keep visualizations in-memory by default for API efficiency
                input_params["in_memory_only"] = True
                tool_input = VisualizationInput(**input_params)
                print(f"📊 Visualization input created: chart_type={tool_input.chart_type}, x_column={tool_input.x_column}, y_column={tool_input.y_column} (in-memory only)")
            elif operation.tool_name == "column_summary":
                tool_input = ColumnSummaryInput(**input_params)
            elif operation.tool_name == "row_summary":
                tool_input = RowSummaryInput(**input_params)
            elif operation.tool_name == "sheet_summary":
                tool_input = SheetSummaryInput(**input_params)
            elif operation.tool_name == "workbook_summary":
                tool_input = WorkbookSummaryInput(**input_params)
            else:
                return {
                    "success": False,
                    "data": None,
                    "output_path": None,
                    "metadata": {},
                    "error_message": f"No input class for tool: {operation.tool_name}"
                }
            
            # Execute the tool
            result = tool.execute(tool_input)
            
            # Convert ToolResult to dict
            result_dict = {
                "success": result.success,
                "data": result.data,
                "output_path": result.output_path,
                "metadata": result.metadata,
                "error_message": result.error_message
            }
            
            # Add to Excel workbook if successful and has data (only for main state)
            if result.success and result.data and hasattr(self, 'excel_workbook_manager'):
                workbook = getattr(self, 'excel_workbook_manager', None)
                if workbook:
                    workbook.add_operation_result(
                        operation_id=operation.operation_id,
                        operation_name=operation.tool_name,
                        data=result.data,
                        metadata=result.metadata,
                        description=operation.description
                    )
            
            return result_dict
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "output_path": None,
                "metadata": {},
                "error_message": f"Failed to execute operation {operation.operation_id}: {str(e)}"
            }
    
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
        html_summary_paths = []
        visualization_content = []
        
        if state.get("excel_workbook_manager"):
            workbook = state["excel_workbook_manager"]
            
            # Add final result sheet for each file
            for file_info in state["files"]:
                file_result = state["file_results"].get(file_info.file_id, {})
                if file_result.get("current_data"):
                    summary = {
                        "goal": state["goal"],
                        "file_id": file_info.file_id,
                        "filename": file_info.filename,
                        "operations_count": len(state["file_operations"].get(file_info.file_id, [])),
                        "completed_operations": file_result.get("completed_operations", []),
                        "success": not file_result.get("error_message")
                    }
                    workbook.add_final_result(file_result["current_data"], summary)
            
            # Add summary sheet with operation details for all files
            operations_summary = []
            for file_id, ops in state["file_operations"].items():
                for op in ops:
                    operations_summary.append({
                        "file_id": file_id,
                        "operation_id": op.operation_id,
                        "tool": op.tool_name,
                        "description": op.description,
                        "result": state["operation_results"].get(op.operation_id, {})
                    })
            workbook.add_summary_sheet(operations_summary)
            
            # Check for HTML summary operations and visualization content
            for op_id, result in state["operation_results"].items():
                if result.get("success"):
                    metadata = result.get("metadata", {})
                    
                    # Collect HTML summary files
                    if result.get("output_path") and result["output_path"].endswith('.html'):
                        if metadata.get("summary_type") in ["sheet", "workbook"]:
                            html_summary_paths.append({
                                "operation_id": op_id,
                                "summary_type": metadata.get("summary_type"),
                                "html_path": result["output_path"]
                            })
                    
                    # Collect in-memory visualization content
                    if metadata.get("visualization_created") and metadata.get("in_memory_only"):
                        visualization_content.append({
                            "operation_id": op_id,
                            "chart_type": metadata.get("chart_type"),
                            "html_content": metadata.get("html_content"),
                            "data_points": metadata.get("data_points")
                        })
            
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
        
        # Prepare final result with both Excel and HTML outputs
        all_operations = []
        for file_id, ops in state["file_operations"].items():
            for op in ops:
                all_operations.append({
                    "file_id": file_id,
                    "operation_id": op.operation_id, 
                    "tool": op.tool_name, 
                    "description": op.description, 
                    "result": state["operation_results"].get(op.operation_id, {})
                })
        
        final_result = {
            "goal": state["goal"],
            "files_processed": len(state["files"]),
            "parallel_mode": state.get("parallel_mode", True),
            "operations": all_operations,
            "results": state["operation_results"],
            "file_results": state["file_results"],
            "excel_workbook_path": excel_s3_path,
            "workbook_summary": state["excel_workbook_manager"].get_summary_info() if state.get("excel_workbook_manager") else None,
            "success": True
        }
        
        # Add HTML summary information if any were generated
        if html_summary_paths:
            final_result["html_summaries"] = html_summary_paths
            print(f"📋 HTML summaries generated: {len(html_summary_paths)}")
            for summary in html_summary_paths:
                print(f"   - {summary['summary_type']}: {summary['html_path']}")
        
        # Add in-memory visualization content
        if visualization_content:
            final_result["visualizations"] = visualization_content
            print(f"📊 In-memory visualizations: {len(visualization_content)}")
            for viz in visualization_content:
                print(f"   - {viz['chart_type']} chart ({viz['data_points']} data points)")
        
        state["final_result"] = final_result
        
        print(f"✅ Final result created successfully!")
        if excel_s3_path:
            print(f"📊 Complete Excel workbook: {excel_s3_path}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if execution should continue (reviewer temporarily disabled)"""
        if state["error_message"]:
            return "end"
        
        # Check if all files have been processed
        if not state["file_operations"]:
            return "end"
        
        # Check if all operations are completed for all files
        total_operations = sum(len(ops) for ops in state["file_operations"].values())
        if len(state["completed_operations"]) >= total_operations:
            return "finalize"
        
        return "continue"
    
    def _review_decision(self, state: AgentState) -> str:
        """Determine next step based on review"""
        if state["final_result"]:
            return "end"
        
        return "replan"
    
    def handle_sheet_clarifications(self, clarification_responses: Dict[str, str]) -> None:
        """
        Process user responses to sheet clarification questions.
        
        Args:
            clarification_responses: Dictionary mapping file_id to selected sheet name
        """
        for file_id, sheet_response in clarification_responses.items():
            # Extract sheet name from response (handle formats like "Use sheet 'Sales'" or just "Sales")
            sheet_name = sheet_response
            
            # Clean up common response formats
            if sheet_response.lower().startswith('use sheet'):
                # Extract from "use sheet 'Sales'" or "use sheet Sales"
                import re
                match = re.search(r'use sheet[s]?\s+["\']?([^"\'\n]+)["\']?', sheet_response, re.IGNORECASE)
                if match:
                    sheet_name = match.group(1).strip()
            elif sheet_response.lower().startswith('sheet'):
                # Extract from "sheet Sales" or "Sheet 'Sales'"
                import re
                match = re.search(r'sheet[s]?\s+["\']?([^"\'\n]+)["\']?', sheet_response, re.IGNORECASE)
                if match:
                    sheet_name = match.group(1).strip()
            
            # Remove quotes if present
            sheet_name = sheet_name.strip('\'"')
            
            print(f"🎯 User selected sheet '{sheet_name}' for file {file_id}")
    
    def run_with_clarifications(self, 
                               previous_result: Dict[str, Any],
                               sheet_clarifications: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Re-run the agent with clarifications provided by the user.
        
        Args:
            previous_result: Previous result that ended with clarifications needed
            sheet_clarifications: User responses to sheet selection questions
            
        Returns:
            Updated result with clarifications applied
        """
        if not previous_result.get("needs_clarification"):
            return previous_result
        
        # Extract necessary information from previous result
        goal = previous_result.get("goal", "")
        files = previous_result.get("files", [])
        
        # Process sheet clarifications
        if sheet_clarifications:
            for file_id, sheet_name in sheet_clarifications.items():
                # Update file info with selected sheet
                for file_info in files:
                    if file_info.get("file_id") == file_id:
                        file_info["sheet_name"] = sheet_name
                        break
        
        # Create new state with updated clarifications
        clarification_dict = previous_result.get("clarifications", {})
        if sheet_clarifications:
            clarification_dict.update(sheet_clarifications)
        
        # Re-run with updated information
        return self.run(
            goal=goal,
            files=files,
            clarifications=clarification_dict,
            parallel_mode=previous_result.get("parallel_mode", True)
        )

    def run(self, 
            goal: str, 
            data_path: Optional[str] = None, 
            data: Optional[List[Dict]] = None, 
            sheet_name: Optional[str] = None, 
            clarifications: Optional[Dict[str, str]] = None, 
            s3_bucket: Optional[str] = None,
            files: Optional[List[Dict[str, Any]]] = None,
            parallel_mode: bool = True) -> Dict[str, Any]:
        """Run the Excel Agent with the given goal"""
        
        # Initialize Excel workbook manager
        excel_workbook_manager = ExcelWorkbookManager(goal=goal)
        
        # Get S3 bucket from environment if not provided
        if not s3_bucket:
            s3_bucket = os.getenv('AWS_S3_BUCKET')
        
        # Handle multiple files input
        file_list = []
        if files:
            # Multiple files provided
            from .state import FileInfo
            for i, file_info in enumerate(files):
                file_obj = FileInfo(
                    file_id=f"file_{i}",
                    data_path=file_info.get("data_path"),
                    data=file_info.get("data"),
                    sheet_name=file_info.get("sheet_name"),
                    filename=file_info.get("filename"),
                    s3_bucket=file_info.get("s3_bucket", s3_bucket)
                )
                file_list.append(file_obj)
        elif data_path or data:
            # Single file (backwards compatibility)
            from .state import FileInfo
            file_obj = FileInfo(
                file_id="file_0",
                data_path=data_path,
                data=data,
                sheet_name=sheet_name,
                filename=data_path.split('/')[-1] if data_path else "data",
                s3_bucket=s3_bucket
            )
            file_list.append(file_obj)
        
        initial_state = {
            "messages": [],
            "goal": goal,
            "files": file_list,
            "current_file_index": 0,
            "current_data": data,  # Keep for backwards compatibility
            "data_path": data_path,  # Keep for backwards compatibility
            "sheet_name": sheet_name,  # Keep for backwards compatibility
            "clarifications": clarifications or {},
            "file_operations": {},
            "completed_operations": [],
            "operation_results": {},
            "file_results": {},
            "current_operation_index": 0,
            "error_message": None,
            "final_result": None,
            "iteration_count": 0,
            "max_iterations": 10,
            "parallel_mode": parallel_mode,
            "excel_workbook_manager": excel_workbook_manager,
            "s3_bucket": s3_bucket,
            "final_excel_s3_path": None,
            # Sheet clarification support
            "sheet_analysis_results": {},
            "pending_sheet_clarifications": [],
            "sheet_clarification_responses": {}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Check if sheet clarifications are needed
        if final_state.get("pending_sheet_clarifications"):
            clarification_message = final_state.get("clarifications", {}).get("sheet_selection", "")
            
            return {
                "goal": goal,
                "files": [
                    {
                        "file_id": f.file_id,
                        "data_path": f.data_path,
                        "filename": f.filename,
                        "sheet_name": f.sheet_name
                    } for f in file_list
                ],
                "needs_clarification": True,
                "clarification_type": "sheet_selection",
                "clarification_message": clarification_message,
                "sheet_analysis_results": final_state.get("sheet_analysis_results", {}),
                "parallel_mode": parallel_mode,
                "success": False
            }
        
        print(f"🏁 Final state keys: {list(final_state.keys())}")
        print(f"🏁 Final result: {final_state.get('final_result')}")
        print(f"🏁 Error message: {final_state.get('error_message')}")
        
        return final_state["final_result"] or {
            "goal": goal,
            "error": final_state.get("error_message", "Unknown error"),
            "success": False
        } 