"""
Alternative Excel Agent Implementation Using ToolNode

This example shows how the Excel Agent could be adapted to use LangGraph's ToolNode
for AI-driven tool calling instead of the current deterministic approach.

This approach would be suitable for more conversational interactions where the AI
decides which tools to call based on the conversation flow.
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
import pandas as pd
from src.utils.claude_client import get_claude_client


# Define tools using @tool decorator for ToolNode compatibility
@tool
def filter_data(data: List[Dict], column: str, value: Any, operator: str = "==") -> List[Dict]:
    """Filter data based on column conditions.
    
    Args:
        data: List of dictionaries representing the data
        column: Column name to filter on
        value: Value to filter by
        operator: Comparison operator (==, !=, >, <, >=, <=, contains)
    
    Returns:
        Filtered data as list of dictionaries
    """
    df = pd.DataFrame(data)
    
    if operator == "==":
        filtered_df = df[df[column] == value]
    elif operator == "!=":
        filtered_df = df[df[column] != value]
    elif operator == ">":
        filtered_df = df[df[column] > value]
    elif operator == "<":
        filtered_df = df[df[column] < value]
    elif operator == ">=":
        filtered_df = df[df[column] >= value]
    elif operator == "<=":
        filtered_df = df[df[column] <= value]
    elif operator == "contains":
        filtered_df = df[df[column].astype(str).str.contains(str(value), na=False)]
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    
    return filtered_df.to_dict('records')


@tool
def create_pivot_table(data: List[Dict], values: str, index: str, columns: Optional[str] = None, aggfunc: str = "sum") -> List[Dict]:
    """Create a pivot table from the data.
    
    Args:
        data: List of dictionaries representing the data
        values: Column to aggregate
        index: Column to use as pivot index
        columns: Column to use as pivot columns (optional)
        aggfunc: Aggregation function (sum, mean, count, min, max)
    
    Returns:
        Pivot table as list of dictionaries
    """
    df = pd.DataFrame(data)
    
    pivot = pd.pivot_table(
        df, 
        values=values, 
        index=index, 
        columns=columns, 
        aggfunc=aggfunc, 
        fill_value=0
    )
    
    # Reset index to make it a regular DataFrame
    pivot_df = pivot.reset_index()
    
    return pivot_df.to_dict('records')


@tool
def get_column_summary(data: List[Dict], column: str) -> Dict[str, Any]:
    """Get statistical summary of a column.
    
    Args:
        data: List of dictionaries representing the data
        column: Column name to analyze
    
    Returns:
        Dictionary with summary statistics
    """
    df = pd.DataFrame(data)
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    col_data = df[column]
    
    if pd.api.types.is_numeric_dtype(col_data):
        return {
            "column": column,
            "type": "numeric",
            "count": int(col_data.count()),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median())
        }
    else:
        return {
            "column": column,
            "type": "categorical",
            "count": int(col_data.count()),
            "unique": int(col_data.nunique()),
            "top": str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
            "freq": int(col_data.value_counts().iloc[0]) if not col_data.empty else 0
        }


class ExcelAgentWithToolNode:
    """Excel Agent using ToolNode for AI-driven tool calling"""
    
    def __init__(self, api_key: str):
        """Initialize the agent"""
        self.claude_client = get_claude_client(api_key)
        
        # Define tools
        self.tools = [filter_data, create_pivot_table, get_column_summary]
        
        # Bind tools to the model
        self.model_with_tools = self.claude_client.bind_tools(self.tools)
        
        # Create ToolNode
        self.tool_node = ToolNode(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        def agent_node(state: MessagesState):
            """AI agent that decides which tools to call"""
            response = self.model_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def should_continue(state: MessagesState):
            """Determine if we should continue to tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the last message has tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            else:
                return END
        
        # Create the graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # Set the entrypoint
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def analyze(self, goal: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze data with the given goal"""
        
        # Create initial message with context
        initial_message = HumanMessage(
            content=f"""
            Goal: {goal}
            
            I have the following data available:
            {data[:3]}... (showing first 3 rows)
            
            Available columns: {list(data[0].keys()) if data else []}
            Total rows: {len(data)}
            
            Please help me achieve my goal. You have access to tools for:
            - filter_data: Filter data based on conditions
            - create_pivot_table: Create pivot tables 
            - get_column_summary: Get statistical summaries
            
            Use the tools as needed and provide analysis and insights.
            """
        )
        
        # Run the graph
        result = self.graph.invoke({"messages": [initial_message]})
        
        # Extract the conversation and results
        messages = result["messages"]
        
        return {
            "goal": goal,
            "conversation": [
                {
                    "type": type(msg).__name__,
                    "content": msg.content,
                    "tool_calls": getattr(msg, 'tool_calls', None)
                }
                for msg in messages
            ],
            "success": True
        }


# Example usage
def example_usage():
    """Example of how to use the ToolNode-based Excel Agent"""
    
    # Sample data
    sample_data = [
        {"product": "A", "sales": 100, "region": "North", "month": "Jan"},
        {"product": "B", "sales": 150, "region": "South", "month": "Jan"},
        {"product": "A", "sales": 120, "region": "North", "month": "Feb"},
        {"product": "B", "sales": 180, "region": "South", "month": "Feb"},
        {"product": "C", "sales": 90, "region": "East", "month": "Jan"},
        {"product": "C", "sales": 110, "region": "East", "month": "Feb"}
    ]
    
    # Initialize agent
    agent = ExcelAgentWithToolNode("your-anthropic-api-key")
    
    # Analyze data
    result = agent.analyze(
        goal="Show me sales summary by product and create a pivot table by region and month",
        data=sample_data
    )
    
    print("Analysis Result:")
    print(result)


if __name__ == "__main__":
    # This would require an actual API key to run
    print("ToolNode Alternative Example")
    print("This example shows how to use ToolNode for AI-driven tool calling")
    print("Run with a valid Anthropic API key to see it in action") 