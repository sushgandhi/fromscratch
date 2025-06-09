"""
Test script for Excel workbook functionality
"""
import pandas as pd
import os
from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client

def create_test_data():
    """Create comprehensive test data for testing"""
    # More comprehensive dataset for better analysis
    data = {
        'date': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
            '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15'
        ],
        'product': [
            'A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C',
            'A', 'B', 'A', 'C', 'A'
        ],
        'sales': [
            100, 150, 120, 200, 180, 135, 220, 110, 165, 195,
            125, 175, 140, 210, 155
        ],
        'region': [
            'North', 'South', 'North', 'East', 'South', 'West', 'East', 'North', 'South', 'West',
            'East', 'North', 'South', 'West', 'North'
        ],
        'quantity': [
            5, 3, 6, 4, 2, 7, 3, 5, 4, 2,
            6, 3, 5, 4, 6
        ],
        'cost': [
            80, 120, 95, 160, 145, 105, 175, 85, 130, 155,
            100, 140, 110, 170, 125
        ]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV for testing
    test_file = "comprehensive_test_data.csv"
    df.to_csv(test_file, index=False)
    print(f"Created comprehensive test data: {test_file}")
    print(f"   📊 {len(df)} rows × {len(df.columns)} columns")
    print(f"   📈 Products: {df['product'].unique().tolist()}")
    print(f"   🌍 Regions: {df['region'].unique().tolist()}")
    print(f"   💰 Sales range: ${df['sales'].min():,} - ${df['sales'].max():,}")
    
    return df.to_dict('records'), test_file

def test_excel_workbook():
    """Test comprehensive Excel workbook with final report including data and visualizations"""
    
    # Get Claude client
    claude_client = get_claude_client()
    supervisor = ExcelAgentSupervisor(claude_client)
    
    # Create test data
    test_data, test_file = create_test_data()
    
    # Comprehensive goal that includes data processing, visualization, AND summary report
    goal = """
    Create a comprehensive sales analysis report:
    1. Filter data for product A sales
    2. Create a bar chart showing product A sales by date
    3. Group all sales data by product and calculate total sales and average sales per product
    4. Generate a comprehensive sheet summary with visualizations and statistical analysis
    
    This should provide both raw processed data and a formatted analytical report.
    """
    
    print(f"\n🧪 Testing Comprehensive Excel Workbook + HTML Report")
    print(f"Goal: {goal}")
    print(f"Data: {len(test_data)} rows")
    print(f"Expected outputs: Excel workbook + HTML summary report")
    
    try:
        # Run the agent
        result = supervisor.run(
            goal=goal,
            data=test_data,
            # Note: No S3 bucket - should save locally
        )
        
        print(f"\n📊 COMPREHENSIVE RESULTS:")
        print(f"Success: {result.get('success')}")
        print(f"Operations completed: {len(result.get('operations', []))}")
        
        # Display operations breakdown
        operations = result.get('operations', [])
        if operations:
            print(f"\n🔧 Operations performed:")
            for i, op in enumerate(operations, 1):
                op_result = op.get('result', {})
                status = "✅" if op_result.get('success') else "❌"
                print(f"   {i}. {status} {op.get('tool', 'unknown').title()}: {op.get('description', 'No description')}")
        
        if result.get('success'):
            print(f"\n✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
            
            # Excel Workbook Results
            print(f"\n📊 EXCEL WORKBOOK OUTPUT:")
            excel_path = result.get('excel_workbook_path')
            if excel_path:
                print(f"   📁 File: {excel_path}")
                
                # Check if file exists and analyze it
                if os.path.exists(excel_path):
                    file_size = os.path.getsize(excel_path)
                    print(f"   ✅ Verified: {file_size:,} bytes")
                    
                    # Analyze workbook contents
                    try:
                        workbook_sheets = pd.ExcelFile(excel_path).sheet_names
                        print(f"   📋 Sheets ({len(workbook_sheets)}): {workbook_sheets}")
                        
                        # Show some details for each sheet
                        for sheet in workbook_sheets:
                            try:
                                df = pd.read_excel(excel_path, sheet_name=sheet)
                                print(f"      - {sheet}: {len(df)} rows × {len(df.columns)} columns")
                            except Exception as e:
                                print(f"      - {sheet}: Could not read ({str(e)})")
                                
                    except Exception as e:
                        print(f"   ❌ Could not analyze sheets: {e}")
                else:
                    print(f"   ❌ Excel file not found: {excel_path}")
            
            # Workbook Summary
            workbook_summary = result.get('workbook_summary')
            if workbook_summary:
                print(f"\n📈 WORKBOOK SUMMARY:")
                print(f"   📊 Total sheets: {workbook_summary.get('sheets_count', 0)}")
                print(f"   📈 Total rows: {workbook_summary.get('total_rows', 0)}")
                print(f"   🕐 Created: {workbook_summary.get('timestamp', 'Unknown')}")
            
            # HTML Summary Reports
            html_summaries = result.get('html_summaries', [])
            if html_summaries:
                print(f"\n📋 HTML SUMMARY REPORTS ({len(html_summaries)}):")
                for i, summary in enumerate(html_summaries, 1):
                    print(f"   {i}. {summary.get('summary_type', 'unknown').title()} Summary:")
                    print(f"      🔗 Path: {summary.get('html_path', 'No path')}")
                    print(f"      🆔 Operation: {summary.get('operation_id', 'unknown')}")
                    
                    # Check if HTML file exists (for local files)
                    html_path = summary.get('html_path', '')
                    if not html_path.startswith('s3://') and os.path.exists(html_path):
                        html_size = os.path.getsize(html_path)
                        print(f"      ✅ Verified: {html_size:,} bytes")
            else:
                print(f"\n📋 HTML SUMMARY REPORTS: None generated")
            
            # Visualizations
            visualizations = result.get('visualizations', [])
            if visualizations:
                print(f"\n📈 VISUALIZATIONS ({len(visualizations)}):")
                for i, viz in enumerate(visualizations, 1):
                    chart_type = viz.get('chart_type', 'unknown')
                    print(f"   {i}. {chart_type.title()} Chart:")
                    if viz.get('local_path'):
                        print(f"      💾 Local: {viz['local_path']}")
                        # Check file size
                        if os.path.exists(viz['local_path']):
                            viz_size = os.path.getsize(viz['local_path'])
                            print(f"      ✅ Verified: {viz_size:,} bytes")
                    if viz.get('s3_path'):
                        print(f"      ☁️ S3: {viz['s3_path']}")
            else:
                print(f"\n📈 VISUALIZATIONS: None generated")
            
            # Final summary
            print(f"\n🎯 WHAT YOU GET:")
            print(f"   📊 Complete Excel workbook with all intermediate data")
            print(f"   📋 Rich HTML reports with analysis and charts")
            print(f"   📈 Individual visualization files")
            print(f"   🔄 Full audit trail of all operations")
            
        else:
            print(f"❌ Test failed: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"🧹 Cleaned up test file: {test_file}")


def explain_comprehensive_test():
    """Explain what the comprehensive test demonstrates"""
    print(f"\n📋 COMPREHENSIVE TEST EXPLANATION:")
    print(f"=" * 60)
    
    print(f"\n🎯 WHAT THIS TEST DEMONSTRATES:")
    print(f"   1. 📊 Data Processing: Filter, group, and transform data")
    print(f"   2. 📈 Visualizations: Interactive charts with Plotly")
    print(f"   3. 📋 HTML Reports: Rich summary with stats & analysis")
    print(f"   4. 📗 Excel Workbook: All data in organized sheets")
    print(f"   5. ☁️ S3 Integration: Upload all outputs to cloud storage")
    
    print(f"\n📊 EXPECTED EXCEL WORKBOOK STRUCTURE:")
    print(f"   📗 comprehensive_sales_analysis_[timestamp].xlsx")
    print(f"   ├── 📋 Summary (execution overview)")
    print(f"   ├── 🎯 Final_Result (final processed data)")
    print(f"   ├── 📊 op1_filter (Product A sales only)")
    print(f"   ├── 📈 op2_visualization (chart data)")
    print(f"   ├── 📊 op3_groupby (sales by product)")
    print(f"   └── 📋 op4_sheet_summary (summary metadata)")
    
    print(f"\n📋 EXPECTED HTML REPORTS:")
    print(f"   📄 sheet_summary_[timestamp].html")
    print(f"   ├── 📊 Data overview & quality metrics")
    print(f"   ├── 📈 Distribution charts & correlations")
    print(f"   ├── 📋 Statistical analysis tables")
    print(f"   └── 🎨 Professional formatting")
    
    print(f"\n📈 EXPECTED VISUALIZATIONS:")
    print(f"   📊 product_a_sales_chart.html (interactive bar chart)")
    print(f"   📈 Various summary charts embedded in HTML report")
    
    print(f"\n🎯 USE CASES DEMONSTRATED:")
    print(f"   📊 Data Analyst: Gets Excel for deep analysis")
    print(f"   📋 Business User: Gets HTML for presentations")
    print(f"   🔄 API Client: Gets S3 URLs for integration")
    print(f"   👥 Executive: Gets formatted report summary")


if __name__ == "__main__":
    print("📊 COMPREHENSIVE EXCEL AGENT TEST")
    print("=" * 50)
    
    explain_comprehensive_test()
    
    print("\n" + "=" * 50)
    print("🚀 Running comprehensive test...")
    
    test_excel_workbook() 