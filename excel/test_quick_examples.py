"""
Quick Example Tests - Demonstrates individual tools and basic workflows
Run these to see expected behavior before running comprehensive tests
"""

from test_comprehensive_scenarios import SALES_DATA, setup_agent, print_test_result

def quick_filter_example():
    """Quick example: Simple filter operation"""
    print("🔍 Quick Filter Example")
    print("Goal: Show products with sales > 1500")
    
    data = SALES_DATA[:5]  # Use smaller dataset
    
    agent = setup_agent()
    if agent is None:
        print("❌ Cannot run test without Claude client")
        return {"success": False, "error": "No Claude client"}
    
    result = agent.run(
        goal="Filter to show products with sales greater than 1500",
        data=data
    )
    
    print(f"✅ Success: {result.get('success')}")
    print(f"📊 Workbook sheets: {result.get('workbook_summary', {}).get('sheets_count', 0)}")
    return result

def quick_visualization_example():
    """Quick example: Simple chart creation"""
    print("\n📊 Quick Visualization Example") 
    print("Goal: Create bar chart of sales by product")
    
    data = SALES_DATA[:4]  # Use smaller dataset
    
    agent = setup_agent()
    result = agent.run(
        goal="Create a bar chart showing sales by product",
        data=data
    )
    
    print(f"✅ Success: {result.get('success')}")
    print(f"📈 Visualizations: {len(result.get('visualizations', []))}")
    if result.get('visualizations'):
        viz = result['visualizations'][0]
        print(f"   - Chart type: {viz['chart_type']}")
        print(f"   - Data points: {viz['data_points']}")
        print(f"   - Has HTML: {bool(viz.get('html_content'))}")
    return result

def quick_groupby_example():
    """Quick example: GroupBy operation"""
    print("\n📊 Quick GroupBy Example")
    print("Goal: Group by region and sum sales")
    
    agent = setup_agent()
    result = agent.run(
        goal="Group the data by region and calculate total sales for each region",
        data=SALES_DATA
    )
    
    print(f"✅ Success: {result.get('success')}")
    print(f"📊 Workbook sheets: {result.get('workbook_summary', {}).get('sheets_count', 0)}")
    return result

def quick_workflow_example():
    """Quick example: Filter + Visualization workflow"""
    print("\n🔄 Quick Workflow Example")
    print("Goal: Filter data and create chart")
    
    agent = setup_agent()
    result = agent.run(
        goal="""
        1. Filter sales data to show only Technology category products
        2. Create a bar chart showing sales by product for the filtered data
        """,
        data=SALES_DATA
    )
    
    print(f"✅ Success: {result.get('success')}")
    print(f"📊 Workbook sheets: {result.get('workbook_summary', {}).get('sheets_count', 0)}")
    print(f"📈 Visualizations: {len(result.get('visualizations', []))}")
    return result

def quick_summary_example():
    """Quick example: HTML summary generation"""
    print("\n📋 Quick Summary Example")
    print("Goal: Generate comprehensive data summary")
    
    agent = setup_agent()
    result = agent.run(
        goal="Generate a comprehensive summary report of the sales data with key insights",
        data=SALES_DATA
    )
    
    print(f"✅ Success: {result.get('success')}")
    print(f"📋 HTML Summaries: {len(result.get('html_summaries', []))}")
    if result.get('html_summaries'):
        summary = result['html_summaries'][0]
        print(f"   - Summary type: {summary['summary_type']}")
        print(f"   - HTML file: {summary['html_path']}")
    return result

def run_quick_examples():
    """Run all quick examples to demonstrate functionality"""
    print("🚀 Excel Agent - Quick Examples")
    print("=" * 50)
    
    examples = [
        quick_filter_example,
        quick_visualization_example, 
        quick_groupby_example,
        quick_workflow_example,
        quick_summary_example
    ]
    
    results = []
    for example in examples:
        try:
            result = example()
            results.append({"test": example.__name__, "success": True})
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
            results.append({"test": example.__name__, "success": False, "error": str(e)})
    
    print(f"\n📊 Quick Examples Summary:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"   {status} {r['test']}")
    
    return results

if __name__ == "__main__":
    run_quick_examples() 