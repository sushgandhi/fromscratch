#!/usr/bin/env python3
"""
Test script for Sheet Intelligence and Clarification features
"""
import pandas as pd
import os
from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client


def create_multi_sheet_test_file():
    """Create a test Excel file with multiple sheets"""
    
    # Sales data
    sales_data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
        'sales_amount': [1200, 800, 950, 600, 1100],
        'region': ['North', 'South', 'East', 'West', 'North']
    }
    
    # Employee data
    employee_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['John Smith', 'Jane Doe', 'Mike Johnson', 'Sarah Wilson', 'Tom Brown'],
        'department': ['Sales', 'Marketing', 'Sales', 'HR', 'IT'],
        'salary': [50000, 55000, 48000, 60000, 65000],
        'hire_date': ['2020-01-15', '2019-03-22', '2021-06-10', '2018-11-05', '2022-02-28']
    }
    
    # Quarterly summary
    quarterly_data = {
        'quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
        'total_sales': [45000, 52000, 48000, 55000],
        'total_employees': [125, 130, 128, 135],
        'avg_satisfaction': [4.2, 4.5, 4.3, 4.6]
    }
    
    # Create Excel file with multiple sheets
    filename = 'test_multi_sheet_data.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        pd.DataFrame(sales_data).to_excel(writer, sheet_name='Sales Data', index=False)
        pd.DataFrame(employee_data).to_excel(writer, sheet_name='Employee Records', index=False)  
        pd.DataFrame(quarterly_data).to_excel(writer, sheet_name='Q2023 Summary', index=False)
        
        # Add some confusing sheet names
        pd.DataFrame(sales_data).to_excel(writer, sheet_name='Sheet1', index=False)  # Default name
        pd.DataFrame(employee_data).to_excel(writer, sheet_name='Data', index=False)  # Generic name
    
    print(f"📁 Created test file: {filename}")
    print("   Sheets: Sales Data, Employee Records, Q2023 Summary, Sheet1, Data")
    
    return filename


def test_sheet_intelligence():
    """Test sheet intelligence with various user queries"""
    
    filename = create_multi_sheet_test_file()
    claude_client = get_claude_client()
    supervisor = ExcelAgentSupervisor(claude_client)
    
    test_cases = [
        {
            "name": "Clear Sales Request",
            "goal": "Analyze sales trends and create a bar chart showing sales by product",
            "expected_sheet": "Sales Data"
        },
        {
            "name": "Employee Analysis",
            "goal": "Show me salary distribution by department for all employees",
            "expected_sheet": "Employee Records"
        },
        {
            "name": "Quarterly Report",
            "goal": "Create a summary of Q2023 performance metrics",
            "expected_sheet": "Q2023 Summary"
        },
        {
            "name": "Ambiguous Request",
            "goal": "Show me the data trends",
            "expected_sheet": None  # Should ask for clarification
        },
        {
            "name": "Explicit Sheet Reference",
            "goal": "Analyze the data in 'Employee Records' sheet",
            "expected_sheet": "Employee Records"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test_case['name']}")
        print(f"Goal: {test_case['goal']}")
        print(f"Expected Sheet: {test_case.get('expected_sheet', 'CLARIFICATION NEEDED')}")
        print('='*60)
        
        try:
            # Run the agent
            result = supervisor.run(
                goal=test_case["goal"],
                files=[{
                    "data_path": filename,
                    "filename": filename
                }],
                parallel_mode=False
            )
            
            if result.get("needs_clarification"):
                print("🤔 CLARIFICATION NEEDED:")
                print(result.get("clarification_message", "No message provided"))
                
                # Simulate user response if we know the expected sheet
                if test_case.get("expected_sheet"):
                    print(f"\n🎯 Simulating user response: Use sheet '{test_case['expected_sheet']}'")
                    
                    # Re-run with clarification
                    clarified_result = supervisor.run_with_clarifications(
                        previous_result=result,
                        sheet_clarifications={"file_0": test_case["expected_sheet"]}
                    )
                    
                    if clarified_result.get("success"):
                        print("✅ Successfully processed after clarification!")
                        print(f"📊 Operations completed: {len(clarified_result.get('operations', []))}")
                    else:
                        print("❌ Failed even after clarification")
                        print(f"Error: {clarified_result.get('error', 'Unknown error')}")
                else:
                    print("🔄 This case is expected to need clarification - test passed!")
                    
            elif result.get("success"):
                print("✅ SUCCESS - Sheet automatically selected!")
                print(f"📊 Operations completed: {len(result.get('operations', []))}")
                print(f"📁 Excel output: {result.get('excel_workbook_path', 'Not specified')}")
                
            else:
                print("❌ FAILED")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception during test: {e}")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
        print(f"\n🧹 Cleaned up test file: {filename}")


def test_sheet_extraction():
    """Test the sheet name extraction functionality independently"""
    
    from src.utils.sheet_intelligence import extract_potential_sheet_names
    
    test_queries = [
        "Analyze sales data from the 'Sales' sheet",
        "Show me trends in sheet named 'Q1 Report'",
        "Create a pivot table from Employee Records",
        "Summarize the dashboard tab",
        "Filter data in Sales sheet",
        "Group by product in the revenue sheet",
        "Show January trends",
        "Analyze Q1 performance"
    ]
    
    print("\n" + "="*50)
    print("SHEET NAME EXTRACTION TESTS")
    print("="*50)
    
    for query in test_queries:
        extracted = extract_potential_sheet_names(query)
        print(f"Query: {query}")
        print(f"Extracted: {extracted}")
        print("-" * 40)


if __name__ == "__main__":
    print("🧪 Testing Sheet Intelligence and Clarification System")
    print("="*60)
    
    # Test extraction functionality first
    test_sheet_extraction()
    
    # Test full sheet intelligence
    test_sheet_intelligence()
    
    print("\n✅ All tests completed!") 