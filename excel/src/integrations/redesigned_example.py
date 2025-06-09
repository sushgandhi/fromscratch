"""
Example: Redesigned Workflow in Action
Shows how the new architecture separates concerns properly
"""

import asyncio
import uuid
from .redesigned_workflow import RedesignedWorkflow, create_workflow_state


async def run_redesigned_workflow_example():
    """Example of the redesigned workflow with proper separation"""
    
    print("🚀 Redesigned Workflow Example")
    print("=" * 50)
    
    # Sample input: 2 files uploaded
    sample_files = [
        {
            "filename": "sales_q4.xlsx",
            "file_path": "s3://bucket/sales_q4.xlsx", 
            "sheet_name": "Q4_Data",
            "metadata": {"type": "sales", "quarter": "Q4"}
        },
        {
            "filename": "customer_analysis.xlsx",
            "file_path": "s3://bucket/customer_analysis.xlsx",
            "sheet_name": "Customer_Data", 
            "metadata": {"type": "customer", "region": "all"}
        }
    ]
    
    # User goals for each file
    user_goals = [
        "Analyze Q4 sales data: filter top products, create visualizations, generate summary report",
        "Perform customer analysis: group by segment, create charts, provide insights"
    ]
    
    # Create workflow
    workflow = RedesignedWorkflow(queue_endpoint="localhost:50051")
    await workflow.initialize()
    
    # Create workflow graph
    graph = workflow.create_workflow()
    
    # Create initial state
    initial_state = create_workflow_state(
        session_id=str(uuid.uuid4()),
        uploaded_files=sample_files,
        user_goals=user_goals
    )
    
    print(f"📋 Starting workflow for session: {initial_state['session_id']}")
    print(f"📂 Files to process: {len(sample_files)}")
    
    # Run the workflow
    final_state = await graph.ainvoke(initial_state)
    
    # Display results
    print(f"\n📊 Workflow Results:")
    print(f"   Session ID: {final_state['session_id']}")
    print(f"   Final Stage: {final_state['current_stage']}")
    print(f"   Files Processed: {len(final_state['execution_results'])}")
    print(f"   Deliverables: {len(final_state['final_deliverables'])}")
    print(f"   Errors: {len(final_state['errors'])}")
    
    # Show detailed breakdown
    if final_state.get('planned_operations'):
        print(f"\n📋 Planning Results:")
        for i, plan in enumerate(final_state['planned_operations']):
            print(f"   File {i+1}: {plan['file_info']['filename']}")
            print(f"     Operations: {plan['operations']}")
    
    if final_state.get('execution_results'):
        print(f"\n⚡ Execution Results:")
        for i, result in enumerate(final_state['execution_results']):
            print(f"   File {i+1}: {result.get('source_file', {}).get('filename')}")
            print(f"     Success: {result.get('success')}")
            print(f"     Operations: {len(result.get('operation_results', {}))}")
            print(f"     Visualizations: {len(result.get('visualizations', []))}")
    
    return final_state


def show_workflow_comparison():
    """Show the difference between old and new architecture"""
    
    print("\n🔄 Workflow Architecture Comparison:")
    print("=" * 50)
    
    print("\n❌ OLD Architecture (Subgraph approach):")
    print("┌─────────────┐    ┌──────────────────────────────────┐")
    print("│analyze_goal │ -> │         execute_goal             │")
    print("│             │    │  ┌─────────────────────────────┐ │")
    print("│• Basic goal │    │  │     Excel Agent Subgraph    │ │")
    print("│  analysis   │    │  │ ┌─────────┐ ┌────────────┐  │ │")
    print("│             │    │  │ │ Planner │>│ Executor   │  │ │")
    print("│             │    │  │ └─────────┘ └────────────┘  │ │")
    print("│             │    │  │ ┌─────────┐ ┌────────────┐  │ │")
    print("│             │    │  │ │Reviewer │ │ Finalizer  │  │ │")
    print("│             │    │  │ └─────────┘ └────────────┘  │ │")
    print("│             │    │  └─────────────────────────────┘ │")
    print("└─────────────┘    └──────────────────────────────────┘")
    
    print("\n✅ NEW Architecture (Distributed responsibilities):")
    print("┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐")
    print("│analyze_goal │ -> │execute_goal  │ -> │prepare_output │ -> │finish_excel │")
    print("│             │    │              │    │               │    │             │")
    print("│• Parse goals│    │• Conditional │    │• Consolidate  │    │• Deliver    │")
    print("│• Plan tools │    │  execution   │    │• Format       │    │• Queue      │")
    print("│• Determine  │    │• Parallel    │    │• Enhance      │    │• Notify     │")
    print("│  strategy   │    │  processing  │    │               │    │             │")
    print("│• Queue init │    │• Queue msgs  │    │               │    │             │")
    print("└─────────────┘    └──────────────┘    └───────────────┘    └─────────────┘")
    
    print("\n🎯 Key Improvements:")
    print("✅ Clear separation of concerns")
    print("✅ Planning logic in analyze_goal")
    print("✅ Conditional execution in execute_goal")  
    print("✅ Parallel processing built-in")
    print("✅ Better queue integration")
    print("✅ Easier to maintain and extend")


async def demonstrate_parallel_execution():
    """Demonstrate parallel vs sequential execution"""
    
    print("\n🔄 Parallel Execution Demonstration:")
    print("=" * 50)
    
    # Multiple files scenario
    multiple_files = [
        {"filename": "file1.xlsx", "data": [{"sales": 100}]},
        {"filename": "file2.xlsx", "data": [{"revenue": 200}]},
        {"filename": "file3.xlsx", "data": [{"profit": 50}]}
    ]
    
    goals = [
        "Analyze sales data",
        "Analyze revenue data", 
        "Analyze profit data"
    ]
    
    print("📂 Input: 3 files uploaded")
    print("⚡ execute_goal will run operations in parallel:")
    print("   🔄 File 1: filter → groupby → visualization (async)")
    print("   🔄 File 2: pivot → chart → summary (async)")
    print("   🔄 File 3: analysis → report (async)")
    print("   ↓")
    print("   📤 Send intermediate results to queue")
    print("   ↓")
    print("   🎯 Consolidate all results")
    
    # Single file scenario
    print("\n📂 Single file scenario:")
    print("⚡ execute_goal will run operations sequentially:")
    print("   📝 File 1: operation1 → operation2 → operation3")
    print("   ↓")
    print("   📤 Send results to queue")


if __name__ == "__main__":
    # Run demonstrations
    show_workflow_comparison()
    
    # Run example
    print("\n" + "=" * 60)
    asyncio.run(run_redesigned_workflow_example())
    
    # Show parallel demo
    asyncio.run(demonstrate_parallel_execution()) 