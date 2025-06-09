#!/usr/bin/env python3
"""
Test parallel processing of multiple Excel files
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.supervisor import ExcelAgentSupervisor
from utils.claude_client import get_claude_client

def create_test_data():
    """Create sample data for testing"""
    
    # Sample sales data for file 1
    sales_data_1 = [
        {"product": "Widget A", "sales": 1000, "region": "North", "quarter": "Q1", "category": "Electronics", "cost": 600, "profit": 400},
        {"product": "Widget B", "sales": 1500, "region": "South", "quarter": "Q1", "category": "Electronics", "cost": 800, "profit": 700},
        {"product": "Widget A", "sales": 1200, "region": "East", "quarter": "Q2", "category": "Electronics", "cost": 700, "profit": 500},
        {"product": "Widget C", "sales": 800, "region": "West", "quarter": "Q2", "category": "Home", "cost": 400, "profit": 400},
        {"product": "Widget B", "sales": 1800, "region": "North", "quarter": "Q3", "category": "Electronics", "cost": 900, "profit": 900},
        {"product": "Widget A", "sales": 1400, "region": "South", "quarter": "Q3", "category": "Electronics", "cost": 800, "profit": 600},
    ]
    
    # Sample employee data for file 2
    employee_data = [
        {"name": "Alice", "department": "Engineering", "salary": 90000, "experience": 5, "performance": "Excellent", "location": "NYC"},
        {"name": "Bob", "department": "Sales", "salary": 75000, "experience": 3, "performance": "Good", "location": "LA"},
        {"name": "Charlie", "department": "Engineering", "salary": 85000, "experience": 4, "performance": "Good", "location": "NYC"},
        {"name": "Diana", "department": "Marketing", "salary": 70000, "experience": 2, "performance": "Excellent", "location": "Chicago"},
        {"name": "Eve", "department": "Sales", "salary": 80000, "experience": 6, "performance": "Good", "location": "LA"},
        {"name": "Frank", "department": "Engineering", "salary": 95000, "experience": 7, "performance": "Excellent", "location": "Seattle"},
    ]
    
    # Product inventory data for file 3
    inventory_data = [
        {"product_id": "P001", "product_name": "Widget A", "stock": 150, "reorder_level": 50, "supplier": "Supplier X", "price": 25.99},
        {"product_id": "P002", "product_name": "Widget B", "stock": 75, "reorder_level": 30, "supplier": "Supplier Y", "price": 35.50},
        {"product_id": "P003", "product_name": "Widget C", "stock": 200, "reorder_level": 80, "supplier": "Supplier X", "price": 18.75},
        {"product_id": "P004", "product_name": "Widget D", "stock": 25, "reorder_level": 40, "supplier": "Supplier Z", "price": 42.00},
        {"product_id": "P005", "product_name": "Widget E", "stock": 100, "reorder_level": 60, "supplier": "Supplier Y", "price": 28.99},
    ]
    
    return [
        {
            "data": sales_data_1,
            "filename": "sales_q1_q3.xlsx",
            "sheet_name": "Sales"
        },
        {
            "data": employee_data,
            "filename": "employees.xlsx", 
            "sheet_name": "Employees"
        },
        {
            "data": inventory_data,
            "filename": "inventory.xlsx",
            "sheet_name": "Inventory"
        }
    ]

def test_parallel_processing():
    """Test parallel processing with multiple files"""
    
    print("🧪 Testing Parallel Processing for Multiple Excel Files")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    try:
        # Initialize Claude client and supervisor
        claude_client = get_claude_client()
        supervisor = ExcelAgentSupervisor(claude_client)
        
        # Create test data
        test_files_data = create_test_data()
        
        # Test goal that applies to all files
        goal = "Analyze the data and create visualizations showing key trends and summaries for each dataset"
        
        print(f"📊 Goal: {goal}")
        print(f"📁 Files to process: {len(test_files_data)}")
        for i, file_data in enumerate(test_files_data):
            print(f"   {i+1}. {file_data['filename']} ({len(file_data['data'])} rows)")
        
        print("\n🚀 Starting parallel execution...")
        
        # Run the agent with multiple files in parallel mode
        result = supervisor.run(
            goal=goal,
            files=test_files_data,
            parallel_mode=True
        )
        
        print("\n✅ Parallel processing completed!")
        print(f"📊 Success: {result.get('success', False)}")
        print(f"📁 Files processed: {result.get('files_processed', 0)}")
        print(f"⚡ Parallel mode: {result.get('parallel_mode', False)}")
        print(f"🔧 Total operations: {len(result.get('operations', []))}")
        
        # Show operations by file
        operations_by_file = {}
        for op in result.get('operations', []):
            file_id = op.get('file_id')
            if file_id not in operations_by_file:
                operations_by_file[file_id] = []
            operations_by_file[file_id].append(op)
        
        print(f"\n📋 Operations by file:")
        for file_id, ops in operations_by_file.items():
            print(f"   {file_id}: {len(ops)} operations")
            for op in ops:
                status = "✅" if op['result'].get('success') else "❌"
                print(f"      {status} {op['tool']} - {op['description']}")
        
        # Show file results
        print(f"\n📁 File processing results:")
        for file_id, file_result in result.get('file_results', {}).items():
            error = file_result.get('error_message') or file_result.get('error')
            if error:
                print(f"   {file_id}: ❌ {error}")
            else:
                completed = len(file_result.get('completed_operations', []))
                print(f"   {file_id}: ✅ {completed} operations completed")
        
        # Show Excel workbook info
        if result.get('excel_workbook_path'):
            print(f"\n📊 Excel workbook created: {result['excel_workbook_path']}")
        
        # Show visualizations
        if result.get('visualizations'):
            print(f"\n📈 Visualizations created: {len(result['visualizations'])}")
            for viz in result['visualizations']:
                print(f"   - {viz['chart_type']} chart ({viz['data_points']} data points)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sequential_mode():
    """Test sequential processing for comparison"""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Sequential Processing (for comparison)")
    print("=" * 60)
    
    try:
        # Initialize Claude client and supervisor
        claude_client = get_claude_client()
        supervisor = ExcelAgentSupervisor(claude_client)
        
        # Create test data (just first 2 files for faster test)
        test_files_data = create_test_data()[:2]
        
        goal = "Create basic summaries and one visualization for each dataset"
        
        print(f"📊 Goal: {goal}")
        print(f"📁 Files to process: {len(test_files_data)}")
        
        print("\n🔄 Starting sequential execution...")
        
        # Run the agent with multiple files in sequential mode
        result = supervisor.run(
            goal=goal,
            files=test_files_data,
            parallel_mode=False  # Sequential mode
        )
        
        print("\n✅ Sequential processing completed!")
        print(f"📊 Success: {result.get('success', False)}")
        print(f"📁 Files processed: {result.get('files_processed', 0)}")
        print(f"⚡ Parallel mode: {result.get('parallel_mode', False)}")
        print(f"🔧 Total operations: {len(result.get('operations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sequential test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Excel Agent Parallel Processing Test Suite")
    print("=" * 60)
    
    # Test parallel processing
    parallel_success = test_parallel_processing()
    
    # Test sequential processing
    sequential_success = test_sequential_mode()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Parallel Processing: {'✅ PASSED' if parallel_success else '❌ FAILED'}")
    print(f"Sequential Processing: {'✅ PASSED' if sequential_success else '❌ FAILED'}")
    
    if parallel_success and sequential_success:
        print("\n🎉 All tests passed! Parallel processing is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1) 