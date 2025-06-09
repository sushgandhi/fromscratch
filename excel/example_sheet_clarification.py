#!/usr/bin/env python3
"""
Example: How to use Sheet Intelligence and Clarification
"""
from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client


def example_with_clarification():
    """Example showing how to handle sheet clarifications"""
    
    claude_client = get_claude_client()
    supervisor = ExcelAgentSupervisor(claude_client)
    
    # Example: User has a multi-sheet Excel file
    excel_file = "multi_sheet_data.xlsx"  # Assume this exists
    user_goal = "Show me the performance trends"  # Ambiguous request
    
    print("🎯 User Goal:", user_goal)
    print("📁 File:", excel_file)
    print("\n" + "="*50)
    
    # First attempt - likely to need clarification
    result = supervisor.run(
        goal=user_goal,
        files=[{
            "data_path": excel_file,
            "filename": excel_file
        }]
    )
    
    # Check if clarification is needed
    if result.get("needs_clarification"):
        print("🤔 The system needs clarification about which sheet to use:")
        print()
        print(result.get("clarification_message", "Please specify which sheet to analyze."))
        print()
        
        # Simulate user providing clarification
        print("👤 User responds: 'Use the Sales Data sheet'")
        print()
        
        # Re-run with clarification
        clarified_result = supervisor.run_with_clarifications(
            previous_result=result,
            sheet_clarifications={
                "file_0": "Sales Data"  # User's selection
            }
        )
        
        if clarified_result.get("success"):
            print("✅ Success! Analysis completed with clarified sheet.")
            print(f"📊 Generated Excel workbook: {clarified_result.get('excel_workbook_path')}")
            print(f"🔧 Operations performed: {len(clarified_result.get('operations', []))}")
        else:
            print("❌ Failed even after clarification.")
            print(f"Error: {clarified_result.get('error')}")
    
    else:
        print("✅ Sheet was automatically detected - no clarification needed!")
        print(f"📊 Generated Excel workbook: {result.get('excel_workbook_path')}")


def example_clear_sheet_reference():
    """Example where sheet is clearly specified in the query"""
    
    claude_client = get_claude_client()
    supervisor = ExcelAgentSupervisor(claude_client)
    
    # Clear sheet reference in the query
    user_goal = "Create a summary of the 'Employee Records' sheet"
    excel_file = "multi_sheet_data.xlsx"
    
    print("\n" + "="*50)
    print("🎯 User Goal:", user_goal)
    print("📁 File:", excel_file)
    print("\n" + "="*50)
    
    result = supervisor.run(
        goal=user_goal,
        files=[{
            "data_path": excel_file,
            "filename": excel_file
        }]
    )
    
    if result.get("needs_clarification"):
        print("🤔 Clarification needed (unexpected for this clear request):")
        print(result.get("clarification_message"))
    else:
        print("✅ Sheet was automatically detected from the query!")
        print(f"📊 Generated Excel workbook: {result.get('excel_workbook_path')}")


def example_multiple_files_with_clarifications():
    """Example with multiple files that may need different clarifications"""
    
    claude_client = get_claude_client()
    supervisor = ExcelAgentSupervisor(claude_client)
    
    user_goal = "Compare sales performance across regions"
    files = [
        {
            "data_path": "sales_q1.xlsx",
            "filename": "Q1 Sales"
        },
        {
            "data_path": "sales_q2.xlsx", 
            "filename": "Q2 Sales"
        }
    ]
    
    print("\n" + "="*50)
    print("🎯 User Goal:", user_goal)
    print("📁 Files:", [f["filename"] for f in files])
    print("\n" + "="*50)
    
    result = supervisor.run(
        goal=user_goal,
        files=files,
        parallel_mode=True
    )
    
    if result.get("needs_clarification"):
        print("🤔 Clarification needed for sheet selection:")
        print(result.get("clarification_message"))
        
        # Example: User clarifies sheets for both files
        print("\n👤 User responds:")
        print("  - For Q1 Sales: Use 'Regional Sales' sheet")
        print("  - For Q2 Sales: Use 'Sales Summary' sheet")
        
        clarified_result = supervisor.run_with_clarifications(
            previous_result=result,
            sheet_clarifications={
                "file_0": "Regional Sales",
                "file_1": "Sales Summary"
            }
        )
        
        if clarified_result.get("success"):
            print("✅ Multi-file analysis completed!")
            print(f"📊 Excel workbook: {clarified_result.get('excel_workbook_path')}")
        else:
            print("❌ Analysis failed:", clarified_result.get('error'))
    else:
        print("✅ All sheets automatically detected!")
        print(f"📊 Excel workbook: {result.get('excel_workbook_path')}")


if __name__ == "__main__":
    print("📋 Sheet Intelligence and Clarification Examples")
    print("="*60)
    
    print("\n1. Example with ambiguous request (needs clarification)")
    example_with_clarification()
    
    print("\n2. Example with clear sheet reference")
    example_clear_sheet_reference()
    
    print("\n3. Example with multiple files") 
    example_multiple_files_with_clarifications()
    
    print("\n✅ Examples completed!")
    
    print("""
💡 Key Points:
- System automatically detects sheet names from user queries
- Uses pattern matching + LLM intelligence for sheet selection
- Asks for clarification when sheet selection is ambiguous
- Supports multiple files with individual sheet clarifications
- User can respond with natural language ("Use the Sales sheet")
- System handles various response formats automatically
""") 