# Excel Agent - Comprehensive Test Guide

## 🎯 Overview

This guide covers all available test cases for the Excel Agent system, from individual tool testing to complex multi-step workflows.

## 📋 Test Structure

### 🔧 **Single Tool Tests**
Test each tool individually to verify core functionality:

| Tool | Test Function | Description |
|------|---------------|-------------|
| **Filter** | `test_filter_tool()` | Filter data based on conditions |
| **GroupBy** | `test_groupby_tool()` | Aggregate data by groups |
| **Visualization** | `test_visualization_tool()` | Create charts (in-memory) |
| **Pivot** | `test_pivot_tool()` | Create pivot tables |
| **Column Summary** | `test_column_summary_tool()` | Statistical analysis of columns |
| **Sheet Summary** | `test_sheet_summary_tool()` | Generate HTML reports |

### 🔄 **Multi-Tool Workflows**
Test realistic business scenarios combining multiple tools:

| Workflow | Test Function | Steps |
|----------|---------------|-------|
| **Filter → Viz** | `test_filter_then_visualize()` | Filter data → Create chart |
| **GroupBy → Viz** | `test_groupby_then_visualize()` | Aggregate → Visualize results |
| **Complex Analytics** | `test_complex_analytics_workflow()` | Filter → GroupBy → Viz → Summary |
| **Sales Analysis** | `test_sales_analysis_workflow()` | Filter → Pivot → GroupBy → Viz → Report |
| **Comparison** | `test_comparison_analysis()` | GroupBy → Filter → Multiple Viz → Summary |
| **Department Analysis** | `test_department_analysis()` | GroupBy → Multiple Viz → Report |
| **Trend Analysis** | `test_trend_analysis()` | GroupBy → Line chart → Growth calculation |

## 🚀 How to Run Tests

### **Quick Examples (Start Here)**
```bash
# Run simple examples to verify basic functionality
python test_quick_examples.py
```

Expected output:
```
🚀 Excel Agent - Quick Examples
==================================================
🔍 Quick Filter Example
Goal: Show products with sales > 1500
✅ Success: True
📊 Workbook sheets: 3

📊 Quick Visualization Example
Goal: Create bar chart of sales by product  
✅ Success: True
📈 Visualizations: 1
   - Chart type: bar
   - Data points: 4
   - Has HTML: True
```

### **Individual Tool Tests**
```bash
# Test specific tool
python test_comprehensive_scenarios.py filter
python test_comprehensive_scenarios.py groupby
python test_comprehensive_scenarios.py viz
python test_comprehensive_scenarios.py pivot
python test_comprehensive_scenarios.py sheet_summary
```

### **Workflow Tests**
```bash
# Test specific workflow
python test_comprehensive_scenarios.py filter_viz
python test_comprehensive_scenarios.py complex
python test_comprehensive_scenarios.py sales
python test_comprehensive_scenarios.py department
```

### **Full Test Suite**
```bash
# Run all tests (13 total)
python test_comprehensive_scenarios.py
```

## 📊 Expected Output Structure

### **Single Tool Test Result:**
```json
{
  "success": true,
  "excel_workbook_path": "s3://bucket/workbook.xlsx",
  "workbook_summary": {
    "sheets_count": 3,
    "total_rows": 156
  },
  "visualizations": [
    {
      "operation_id": "op1",
      "chart_type": "bar", 
      "html_content": "<html>...</html>",
      "data_points": 12
    }
  ]
}
```

### **Multi-Tool Workflow Result:**
```json
{
  "success": true,
  "excel_workbook_path": "s3://bucket/comprehensive_analysis.xlsx",
  "workbook_summary": {
    "sheets_count": 6,
    "total_rows": 278  
  },
  "visualizations": [
    {
      "operation_id": "op2",
      "chart_type": "bar",
      "data_points": 4
    },
    {
      "operation_id": "op4", 
      "chart_type": "scatter",
      "data_points": 3
    }
  ],
  "html_summaries": [
    {
      "operation_id": "op5",
      "summary_type": "sheet",
      "html_path": "outputs/summary_20241205_143022.html"
    }
  ]
}
```

## 🗃️ Test Datasets

### **Sales Data (12 records)**
- **Fields**: product, sales, region, quarter, category, cost, profit
- **Use Cases**: Sales analysis, regional comparisons, trend analysis
- **Categories**: Electronics, Technology, Hardware

### **Employee Data (8 records)**  
- **Fields**: name, department, salary, experience, performance, location
- **Use Cases**: HR analytics, department comparisons, performance analysis
- **Departments**: Sales, Marketing, Engineering

## 🎯 Test Scenarios Covered

### **Data Operations**
- ✅ Filtering with various conditions
- ✅ GroupBy with multiple aggregations  
- ✅ Pivot tables with different configurations
- ✅ Column statistics and summaries

### **Visualizations**
- ✅ Bar charts (sales by product/region)
- ✅ Pie charts (distribution analysis)
- ✅ Scatter plots (correlation analysis)
- ✅ Line charts (trend analysis)
- ✅ In-memory chart generation

### **Reports**
- ✅ HTML summary reports with statistics
- ✅ Multi-sheet Excel workbooks
- ✅ S3 upload and local fallback

### **Workflows**
- ✅ Sequential operations with dependencies
- ✅ Data flow between operations
- ✅ Error handling and recovery
- ✅ Complex multi-step analysis

## 🧪 Memory vs File Testing

### **In-Memory Components:**
- ✅ Intermediate operation results
- ✅ Visualization HTML content
- ✅ Operation metadata and statistics

### **File-Based Components:**
- ✅ Final Excel workbook (consolidated results)
- ✅ HTML summary reports (standalone documents)
- ✅ S3 upload with local fallback

## 📈 Success Criteria

Each test should demonstrate:

1. **✅ Successful Execution** - No errors during processing
2. **📊 Excel Workbook Creation** - Multi-sheet workbook with all operations
3. **🧠 In-Memory Visualizations** - Charts generated without file writes
4. **📋 HTML Reports** - Rich summary documents when requested
5. **🔄 Proper Data Flow** - Operations use results from previous steps

## 🔧 Troubleshooting

### **Common Issues:**
- **No Claude API Key**: Set `ANTHROPIC_API_KEY` environment variable
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **S3 Access**: Tests work without S3, will save locally

### **Debug Mode:**
```bash
# Enable verbose logging
export EXCEL_AGENT_DEBUG=1
python test_quick_examples.py
```

## 🎉 Expected Test Results

**Full Test Suite:**
- **📈 Total Tests**: 13
- **✅ Expected Success Rate**: 100%
- **⏱️ Estimated Runtime**: 2-5 minutes
- **📁 Output Files**: Excel workbooks + HTML summaries in `outputs/`

This comprehensive test suite validates all aspects of the Excel Agent system, from individual tool functionality to complex business analytics workflows. 