"""
Example: Dual Output - Excel Workbook + HTML Summary

This example demonstrates how the system provides both:
1. Excel workbook with all intermediate + final data sheets
2. HTML summary reports with analysis and visualizations

Perfect for when you need both raw data AND formatted analysis reports.
"""
import requests
import json
from typing import Dict, Any

# API endpoint (adjust as needed)
API_BASE_URL = "http://localhost:8000"

def example_with_dual_output():
    """Example showing both Excel workbook and HTML summary outputs"""
    
    # Sample data - sales across different products and regions
    sample_data = [
        {"date": "2024-01-01", "product": "Laptop", "sales": 1200, "region": "North", "quantity": 3},
        {"date": "2024-01-02", "product": "Phone", "sales": 800, "region": "South", "quantity": 2},
        {"date": "2024-01-03", "product": "Laptop", "sales": 1500, "region": "East", "quantity": 4},
        {"date": "2024-01-04", "product": "Tablet", "sales": 600, "region": "West", "quantity": 2},
        {"date": "2024-01-05", "product": "Phone", "sales": 900, "region": "North", "quantity": 3},
        {"date": "2024-01-06", "product": "Laptop", "sales": 1800, "region": "South", "quantity": 5},
        {"date": "2024-01-07", "product": "Tablet", "sales": 700, "region": "East", "quantity": 3},
        {"date": "2024-01-08", "product": "Phone", "sales": 1100, "region": "West", "quantity": 4},
    ]
    
    # Goal that includes both data processing AND summary generation
    goal = """
    Analyze the sales data:
    1. Filter data for Laptop sales
    2. Create a bar chart showing laptop sales by date
    3. Generate a comprehensive sheet summary with visualizations
    """
    
    # API request payload
    payload = {
        "goal": goal,
        "data": sample_data,
        "s3_bucket": "my-excel-agent-bucket"  # Optional: for S3 upload
    }
    
    headers = {
        "Authorization": "Bearer YOUR_ANTHROPIC_API_KEY",  # Replace with real key
        "Content-Type": "application/json"
    }
    
    print("🚀 Making API request for dual output (Excel + HTML)...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Analysis completed successfully!")
            print(f"Goal: {result['goal']}")
            print(f"Operations completed: {len(result.get('completed_operations', []))}")
            
            # Excel Workbook Output
            print("\n📊 EXCEL WORKBOOK OUTPUT:")
            if result.get('excel_workbook_s3_path'):
                print(f"   S3 Path: {result['excel_workbook_s3_path']}")
            elif result.get('excel_workbook_local_path'):
                print(f"   Local Path: {result['excel_workbook_local_path']}")
            
            workbook_summary = result.get('workbook_summary', {})
            if workbook_summary:
                print(f"   📋 Sheets: {workbook_summary.get('sheet_names', [])}")
                print(f"   📊 Total Sheets: {workbook_summary.get('sheets_count', 0)}")
                print(f"   📈 Total Rows: {workbook_summary.get('total_rows', 0)}")
            
            # HTML Summary Output
            print("\n📋 HTML SUMMARY REPORTS:")
            html_summaries = result.get('html_summaries', [])
            if html_summaries:
                for i, summary in enumerate(html_summaries, 1):
                    print(f"   {i}. {summary['summary_type'].title()} Summary:")
                    print(f"      🔗 HTML Path: {summary['html_path']}")
                    print(f"      🆔 Operation: {summary['operation_id']}")
            else:
                print("   No HTML summaries generated")
            
            # Visualizations
            print("\n📈 VISUALIZATIONS:")
            visualizations = result.get('visualizations', [])
            if visualizations:
                for viz in visualizations:
                    print(f"   📊 {viz.get('chart_type', 'Unknown').title()} Chart")
                    if viz.get('s3_path'):
                        print(f"      🔗 S3: {viz['s3_path']}")
                    elif viz.get('local_path'):
                        print(f"      💾 Local: {viz['local_path']}")
            else:
                print("   No visualizations generated")
            
            # Summary of what you get
            print("\n🎯 WHAT YOU GET:")
            print("   📊 Excel Workbook with:")
            if workbook_summary.get('sheet_names'):
                for sheet in workbook_summary['sheet_names']:
                    print(f"      - {sheet} sheet")
            
            print("   📋 HTML Reports with:")
            for summary in html_summaries:
                print(f"      - Rich {summary['summary_type']} analysis with charts & stats")
            
            print("   📈 Individual Visualization Files")
            
            return result
        
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def explain_dual_output_benefits():
    """Explain the benefits of dual output approach"""
    
    print("\n🎯 DUAL OUTPUT BENEFITS:")
    print("\n📊 EXCEL WORKBOOK:")
    print("   ✅ Raw data for further analysis")
    print("   ✅ Multiple sheets with intermediate results")
    print("   ✅ Machine-readable format")
    print("   ✅ Easy to import into other tools")
    print("   ✅ Preserves complete audit trail")
    
    print("\n📋 HTML SUMMARY REPORTS:")
    print("   ✅ Human-readable formatted analysis")
    print("   ✅ Interactive visualizations embedded")
    print("   ✅ Statistical insights and correlations")
    print("   ✅ Professional presentation format")
    print("   ✅ Shareable business reports")
    
    print("\n🎯 USE CASES:")
    print("   📊 Data Scientists: Use Excel for further analysis")
    print("   📋 Business Users: Use HTML for presentations")
    print("   🔄 APIs: Use Excel data programmatically")
    print("   👥 Stakeholders: Share HTML reports")
    print("   📈 Compliance: Full audit trail in Excel")


def api_response_example():
    """Show what the API response looks like with dual output"""
    
    example_response = {
        "success": True,
        "goal": "Analyze sales data and create summary",
        "completed_operations": ["op1", "op2", "op3"],
        
        # Excel Workbook Output
        "excel_workbook_s3_path": "s3://my-bucket/excel-agent-results/Analyze_sales_data_20241205_143022_abc123.xlsx",
        "workbook_summary": {
            "filename": "Analyze_sales_data_20241205_143022_abc123.xlsx",
            "sheets_count": 4,
            "sheet_names": ["Summary", "Final_Result", "op1_filter", "op3_sheet_summary"],
            "total_rows": 145
        },
        
        # HTML Summary Output
        "html_summaries": [
            {
                "operation_id": "op3",
                "summary_type": "sheet",
                "html_path": "s3://my-bucket/excel-agent-results/sheet_summary_20241205_143022.html"
            }
        ],
        
        # Individual Visualizations
        "visualizations": [
            {
                "operation_id": "op2",
                "chart_type": "bar",
                "s3_path": "s3://my-bucket/excel-agent-results/laptop_sales_chart.html"
            }
        ]
    }
    
    print("\n📋 EXAMPLE API RESPONSE:")
    print(json.dumps(example_response, indent=2))


if __name__ == "__main__":
    print("📊 Excel Agent: Dual Output Example")
    print("=" * 50)
    
    explain_dual_output_benefits()
    api_response_example()
    
    print("\n" + "=" * 50)
    print("🚀 To run the actual example:")
    print("   1. Start the API server")
    print("   2. Replace YOUR_ANTHROPIC_API_KEY with real key")
    print("   3. Configure S3 bucket (optional)")
    print("   4. Run: python example_dual_output.py")
    
    # Uncomment to run actual API call
    # result = example_with_dual_output() 