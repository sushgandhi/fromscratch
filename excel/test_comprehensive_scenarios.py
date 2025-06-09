"""
Comprehensive End-to-End Test Cases for Excel Agent
Tests all tools individually and in various combinations
"""

import json
import pandas as pd
from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client
import os
from typing import Dict, Any

# Sample datasets for testing
SALES_DATA = [
    {"product": "Widget A", "sales": 1200, "region": "North", "quarter": "Q1", "category": "Electronics", "cost": 800, "profit": 400},
    {"product": "Widget B", "sales": 850, "region": "South", "quarter": "Q1", "category": "Electronics", "cost": 600, "profit": 250},
    {"product": "Widget C", "sales": 1500, "region": "East", "quarter": "Q1", "category": "Electronics", "cost": 900, "profit": 600},
    {"product": "Gadget X", "sales": 2200, "region": "West", "quarter": "Q2", "category": "Technology", "cost": 1400, "profit": 800},
    {"product": "Gadget Y", "sales": 1800, "region": "North", "quarter": "Q2", "category": "Technology", "cost": 1200, "profit": 600},
    {"product": "Tool A", "sales": 950, "region": "South", "quarter": "Q2", "category": "Hardware", "cost": 700, "profit": 250},
    {"product": "Tool B", "sales": 1100, "region": "East", "quarter": "Q3", "category": "Hardware", "cost": 800, "profit": 300},
    {"product": "Device 1", "sales": 3200, "region": "West", "quarter": "Q3", "category": "Technology", "cost": 2000, "profit": 1200},
    {"product": "Device 2", "sales": 2800, "region": "North", "quarter": "Q3", "category": "Technology", "cost": 1800, "profit": 1000},
    {"product": "Component X", "sales": 650, "region": "South", "quarter": "Q4", "category": "Hardware", "cost": 450, "profit": 200},
    {"product": "Component Y", "sales": 750, "region": "East", "quarter": "Q4", "category": "Hardware", "cost": 500, "profit": 250},
    {"product": "Premium A", "sales": 4500, "region": "West", "quarter": "Q4", "category": "Electronics", "cost": 2700, "profit": 1800},
]

EMPLOYEE_DATA = [
    {"name": "John Smith", "department": "Sales", "salary": 65000, "experience": 3, "performance": 4.2, "location": "NYC"},
    {"name": "Sarah Johnson", "department": "Marketing", "salary": 70000, "experience": 5, "performance": 4.5, "location": "LA"},
    {"name": "Mike Davis", "department": "Engineering", "salary": 85000, "experience": 7, "performance": 4.8, "location": "SF"},
    {"name": "Emily Brown", "department": "Sales", "salary": 60000, "experience": 2, "performance": 4.0, "location": "NYC"},
    {"name": "David Wilson", "department": "Engineering", "salary": 95000, "experience": 10, "performance": 4.7, "location": "SF"},
    {"name": "Lisa Garcia", "department": "Marketing", "salary": 75000, "experience": 6, "performance": 4.4, "location": "Chicago"},
    {"name": "Tom Anderson", "department": "Sales", "salary": 68000, "experience": 4, "performance": 4.3, "location": "NYC"},
    {"name": "Anna Lee", "department": "Engineering", "salary": 78000, "experience": 4, "performance": 4.6, "location": "SF"},
]

def setup_agent():
    """Setup the Excel Agent for testing"""
    try:
        claude_client = get_claude_client()
        return ExcelAgentSupervisor(claude_client)
    except (ValueError, ConnectionError) as e:
        print(f"⚠️  Warning: Could not create Claude client: {e}")
        print(f"💡 Hint: Set ANTHROPIC_API_KEY environment variable to run actual tests")
        print(f"📋 This will show test structure only...")
        return None

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*60}")

def print_test_result(result: Dict[str, Any]):
    """Print formatted test result"""
    print(f"\n📊 RESULT:")
    print(f"   ✅ Success: {result.get('success', False)}")
    
    if result.get('excel_workbook_path'):
        print(f"   📄 Excel: {result['excel_workbook_path']}")
    
    if result.get('visualizations'):
        print(f"   📈 Visualizations: {len(result['visualizations'])}")
        for viz in result['visualizations']:
            print(f"      - {viz['chart_type']} ({viz['data_points']} points)")
    
    if result.get('html_summaries'):
        print(f"   📋 HTML Summaries: {len(result['html_summaries'])}")
        for summary in result['html_summaries']:
            print(f"      - {summary['summary_type']}: {summary['html_path']}")
    
    if result.get('workbook_summary'):
        wb_summary = result['workbook_summary']
        print(f"   📈 Workbook: {wb_summary.get('sheets_count', 0)} sheets, {wb_summary.get('total_rows', 0)} rows")

# =============================================================================
# SINGLE TOOL TEST CASES
# =============================================================================

def test_filter_tool():
    """Test: Single Filter Operation"""
    print_test_header("Single Filter Tool")
    
    agent = setup_agent()
    if agent is None:
        return {"success": False, "error": "No Claude client available"}
    
    goal = "Filter the sales data to show only products with sales greater than 1000"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_groupby_tool():
    """Test: Single GroupBy Operation"""
    print_test_header("Single GroupBy Tool")
    
    agent = setup_agent()
    goal = "Group the sales data by region and calculate total sales and average profit for each region"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_visualization_tool():
    """Test: Single Visualization Tool"""
    print_test_header("Single Visualization Tool")
    
    agent = setup_agent()
    goal = "Create a bar chart showing sales by product from the sales data"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_pivot_tool():
    """Test: Single Pivot Operation"""
    print_test_header("Single Pivot Tool")
    
    agent = setup_agent()
    goal = "Create a pivot table with regions as rows, quarters as columns, and sum of sales as values"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_column_summary_tool():
    """Test: Single Column Summary"""
    print_test_header("Single Column Summary Tool")
    
    agent = setup_agent()
    goal = "Analyze the sales and profit columns and provide statistical summaries"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_sheet_summary_tool():
    """Test: Single Sheet Summary with HTML Report"""
    print_test_header("Single Sheet Summary Tool")
    
    agent = setup_agent()
    goal = "Generate a comprehensive summary report of the sales data with visualizations"
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

# =============================================================================
# MULTI-TOOL WORKFLOW TEST CASES
# =============================================================================

def test_filter_then_visualize():
    """Test: Filter → Visualization Workflow"""
    print_test_header("Filter → Visualization Workflow")
    
    agent = setup_agent()
    goal = """
    1. Filter sales data to show only Technology category products
    2. Create a bar chart showing sales by product for the filtered data
    """
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_groupby_then_visualize():
    """Test: GroupBy → Visualization Workflow"""
    print_test_header("GroupBy → Visualization Workflow")
    
    agent = setup_agent()
    goal = """
    1. Group sales data by category and calculate total sales for each category
    2. Create a pie chart showing the sales distribution by category
    """
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_complex_analytics_workflow():
    """Test: Complex Multi-Step Analytics"""
    print_test_header("Complex Analytics Workflow")
    
    agent = setup_agent()
    goal = """
    1. Filter employee data to show only employees with experience > 3 years
    2. Group the filtered data by department and calculate average salary and performance
    3. Create a scatter plot showing the relationship between average salary and average performance by department
    4. Generate a comprehensive summary report with insights
    """
    
    result = agent.run(
        goal=goal,
        data=EMPLOYEE_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_sales_analysis_workflow():
    """Test: Complete Sales Analysis Pipeline"""
    print_test_header("Complete Sales Analysis Workflow")
    
    agent = setup_agent()
    goal = """
    1. Filter sales data to show only Q3 and Q4 data
    2. Create a pivot table with regions as rows, quarters as columns, and sales as values
    3. Group the data by region and calculate total sales, average profit margin
    4. Create a bar chart showing total sales by region
    5. Generate a comprehensive analysis report with all findings
    """
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_comparison_analysis():
    """Test: Comparative Analysis Workflow"""
    print_test_header("Comparative Analysis Workflow")
    
    agent = setup_agent()
    goal = """
    1. Group sales data by category and calculate total sales, total profit, and profit margin
    2. Filter to show only categories with total sales > 5000
    3. Create visualizations showing:
       - Bar chart of total sales by category
       - Line chart of profit margin by category
    4. Generate summary insights comparing category performance
    """
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_department_analysis():
    """Test: HR Department Analysis"""
    print_test_header("Department Analysis Workflow")
    
    agent = setup_agent()
    goal = """
    1. Group employee data by department and calculate:
       - Average salary
       - Average experience
       - Average performance rating
       - Employee count
    2. Create visualizations:
       - Bar chart showing average salary by department
       - Scatter plot of experience vs performance by department
    3. Generate department comparison report with recommendations
    """
    
    result = agent.run(
        goal=goal,
        data=EMPLOYEE_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

def test_trend_analysis():
    """Test: Quarterly Trend Analysis"""
    print_test_header("Quarterly Trend Analysis")
    
    agent = setup_agent()
    goal = """
    1. Group sales data by quarter and calculate total sales, total profit
    2. Create a line chart showing sales and profit trends over quarters
    3. Calculate quarter-over-quarter growth rates
    4. Generate trend analysis report with forecasting insights
    """
    
    result = agent.run(
        goal=goal,
        data=SALES_DATA,
        s3_bucket="test-bucket"
    )
    
    print_test_result(result)
    return result

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all test cases"""
    print(f"🚀 Starting Comprehensive Excel Agent Tests")
    print(f"📋 Testing all tools individually and in workflows")
    
    # Single tool tests
    single_tool_tests = [
        test_filter_tool,
        test_groupby_tool,
        test_visualization_tool,
        test_pivot_tool,
        test_column_summary_tool,
        test_sheet_summary_tool,
    ]
    
    # Multi-tool workflow tests
    workflow_tests = [
        test_filter_then_visualize,
        test_groupby_then_visualize,
        test_complex_analytics_workflow,
        test_sales_analysis_workflow,
        test_comparison_analysis,
        test_department_analysis,
        test_trend_analysis,
    ]
    
    all_results = []
    
    print(f"\n🔧 SINGLE TOOL TESTS ({len(single_tool_tests)} tests)")
    for test_func in single_tool_tests:
        try:
            result = test_func()
            all_results.append({"test": test_func.__name__, "result": result, "status": "✅ PASSED"})
        except Exception as e:
            all_results.append({"test": test_func.__name__, "error": str(e), "status": "❌ FAILED"})
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print(f"\n🔄 WORKFLOW TESTS ({len(workflow_tests)} tests)")
    for test_func in workflow_tests:
        try:
            result = test_func()
            all_results.append({"test": test_func.__name__, "result": result, "status": "✅ PASSED"})
        except Exception as e:
            all_results.append({"test": test_func.__name__, "error": str(e), "status": "❌ FAILED"})
            print(f"❌ {test_func.__name__} failed: {e}")
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY REPORT")
    print(f"{'='*60}")
    
    passed = sum(1 for r in all_results if r["status"] == "✅ PASSED")
    failed = sum(1 for r in all_results if r["status"] == "❌ FAILED")
    
    print(f"📈 Total Tests: {len(all_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/len(all_results)*100):.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for result in all_results:
        print(f"   {result['status']} {result['test']}")
        if "error" in result:
            print(f"      Error: {result['error']}")
    
    return all_results

if __name__ == "__main__":
    # Run specific test or all tests
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_functions = {
            "filter": test_filter_tool,
            "groupby": test_groupby_tool,
            "viz": test_visualization_tool,
            "pivot": test_pivot_tool,
            "column_summary": test_column_summary_tool,
            "sheet_summary": test_sheet_summary_tool,
            "filter_viz": test_filter_then_visualize,
            "groupby_viz": test_groupby_then_visualize,
            "complex": test_complex_analytics_workflow,
            "sales": test_sales_analysis_workflow,
            "comparison": test_comparison_analysis,
            "department": test_department_analysis,
            "trend": test_trend_analysis,
        }
        
        if test_name in test_functions:
            test_functions[test_name]()
        else:
            print(f"❌ Unknown test: {test_name}")
            print(f"Available tests: {list(test_functions.keys())}")
    else:
        # Run all tests
        run_all_tests() 