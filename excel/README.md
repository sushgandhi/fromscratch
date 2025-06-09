# Excel Agent

An end-to-end AI-powered Excel analysis agent built with LangGraph and Claude. Transform natural language goals into sophisticated Excel operations including filtering, pivot tables, visualizations, and comprehensive data analysis.

## 🚀 Features

- **Natural Language Processing**: Describe what you want to achieve in plain English
- **Intelligent Operation Planning**: Claude breaks down complex goals into executable operations
- **Tool Chaining**: Operations can be chained together automatically
- **Multiple Data Sources**: Support for Excel files, CSV files, S3 storage, and direct data input
- **Rich Visualizations**: Interactive Plotly charts and graphs
- **Comprehensive Analysis**: Column summaries, sheet analysis, and workbook-wide insights
- **Smart Column Matching**: Automatically handles column name variations (case, spacing, typos)
- **REST API**: Easy integration with web applications
- **S3 Integration**: Save and retrieve results from cloud storage

## 📋 Available Operations

| Tool | Purpose | Key Features |
|------|---------|--------------|
| **Filter** | Filter data based on conditions | Supports multiple operators, case sensitivity |
| **Pivot** | Create pivot tables | Flexible aggregations, margins, fill values |
| **Group By** | Group and aggregate data | Multiple aggregation functions, sorting |
| **Visualization** | Create interactive charts | Bar, line, scatter, pie, histogram, box, heatmap |
| **Column Summary** | Statistical analysis of columns | Numeric stats, text analysis, date ranges |
| **Sheet Summary** | Comprehensive sheet analysis | Data quality, correlations, visualizations |
| **Workbook Summary** | Multi-sheet analysis | Cross-sheet comparison, HTML reports |

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd excel-flow
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (Optional for Python usage)**
```bash
# Copy the template (only needed if using Python directly)
cp env_template.txt .env

# Edit .env with your API keys (optional - API accepts Bearer tokens)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. **Optional: Configure AWS for S3 support**
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## 🎯 Quick Start

### Python Usage

```python
from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client
import os

# Initialize the agent
api_key = os.getenv("ANTHROPIC_API_KEY")  # Your API key
claude_client = get_claude_client(api_key)
agent = ExcelAgentSupervisor(claude_client)

# Sample data
data = [
    {"date": "2023-01-01", "product": "A", "sales": 100, "region": "North"},
    {"date": "2023-01-02", "product": "B", "sales": 150, "region": "South"},
    {"date": "2023-01-03", "product": "A", "sales": 120, "region": "North"},
]

# Natural language goal
result = agent.run(
    goal="Filter data for product A and create a bar chart showing sales by date",
    data=data
)

print(result)
```

### API Usage

1. **Start the API server**
```bash
python -m src.api
```

2. **Make requests**
```bash
# Analyze data (Bearer token required)
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-anthropic-api-key-here" \
     -d '{
       "goal": "Create a pivot table showing sales by region and product",
       "data": [{"product": "A", "region": "North", "sales": 100}]
     }'

# Upload and analyze Excel file (Bearer token required)
curl -X POST "http://localhost:8000/upload-and-analyze" \
     -H "Authorization: Bearer your-anthropic-api-key-here" \
     -F "file=@data.xlsx" \
     -F "goal=Analyze sales trends and create visualizations"
```

## 📖 Example Use Cases

### 1. Handling Clarifications

When the agent needs clarification (e.g., multiple sheets or ambiguous column names):

```bash
# Initial request
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "goal": "Analyze sales data",
       "data_path": "s3://bucket/sales.xlsx"
     }'

# Response with clarification needed
{
  "success": false,
  "goal": "Analyze sales data",
  "clarification_needed": {
    "question": "This Excel file has 3 sheets. Which sheet would you like to analyze?",
    "options": ["Q1_Sales", "Q2_Sales", "Summary"],
    "clarification_type": "sheet_name"
  },
  "available_sheets": ["Q1_Sales", "Q2_Sales", "Summary"]
}

# Follow-up request with clarification
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "goal": "Analyze sales data", 
       "data_path": "s3://bucket/sales.xlsx",
       "sheet_name": "Q1_Sales"
     }'
```

### 2. Sales Data Analysis
```python
goal = """
1. Filter sales data for the last quarter
2. Group by product category and calculate total revenue  
3. Create a bar chart showing revenue by category
4. Generate a summary report with key insights
"""

result = agent.run(goal=goal, data_path="sales_data.xlsx")
```

### 2. Financial Reporting
```python
goal = """
Analyze the financial data:
- Create a pivot table showing expenses by department and month
- Filter out expenses below $1000
- Generate a line chart showing expense trends
- Provide a summary of spending patterns
"""

result = agent.run(goal=goal, data=financial_data)
```

### 3. Multi-Sheet Analysis
```python
goal = """
Analyze the entire workbook:
- Summarize each sheet
- Identify correlations between different datasets
- Create visualizations showing relationships
- Generate a comprehensive HTML report
"""

result = agent.run(goal=goal, data_path="s3://bucket/workbook.xlsx")
```

## 🏗️ Architecture

The Excel Agent uses a **supervisor pattern** with LangGraph to orchestrate operations:

```
User Goal → Planner → Executor → Reviewer
                ↓         ↓         ↓
              Operations  Tools   Results
```

### Key Components

- **Supervisor Agent**: Orchestrates the entire workflow using Claude
- **Planner**: Breaks down goals into executable operations  
- **Executor**: Runs individual tools and manages data flow
- **Reviewer**: Evaluates results and determines next actions
- **Tools**: Modular operations (filter, pivot, visualize, etc.)

## 🔧 Configuration

### Claude Client Setup

The system expects a `get_claude_client()` function. You can customize it:

```python
def get_claude_client():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        anthropic_api_key="your-key",
        model="claude-3-sonnet-20240229",  # or claude-3-opus-20240229
        temperature=0.1  # Adjust creativity
    )
```

### Adding Custom Tools

1. Create a new tool inheriting from `BaseTool`
2. Define input/output models  
3. Implement the `execute()` method
4. Register in the supervisor

```python
class CustomTool(BaseTool):
    def execute(self, input_data):
        # Your custom logic here
        return ToolOutput(success=True, data=result)
```

## 📊 Output Formats

- **JSON**: Structured data and metadata
- **CSV**: Tabular results  
- **HTML**: Rich reports with visualizations
- **Plotly**: Interactive charts

Results can be:
- Returned directly for programmatic use
- Saved to S3 for persistence
- Embedded in HTML reports

## 🚨 Error Handling

The system includes robust error handling:

- **Validation**: Input parameters and data types
- **Recovery**: Graceful degradation on failures  
- **Limits**: Maximum iterations to prevent infinite loops
- **Logging**: Detailed operation tracking

## 🧪 Testing

Run the example scripts to test functionality:

```bash
cd examples
python usage_examples.py
```

### Available Examples

- `usage_examples.py` - Main Excel Agent examples using the deterministic approach
- `clarification_examples.py` - API clarification workflow examples  
- `toolnode_alternative_example.py` - Alternative implementation using ToolNode for AI-driven tool calling

Choose from various example scenarios to see the agent in action.

## 📈 Performance

- **Memory Efficient**: Streaming data processing for large files
- **Parallel Processing**: Multiple operations can run concurrently  
- **Caching**: Results cached for repeated operations
- **Optimized**: Pandas and NumPy optimizations for data processing

## 🔐 Security

- API keys passed as Bearer tokens (not stored on server)
- Per-request authentication with user's own API key
- Input validation and sanitization
- S3 bucket access controls  
- File upload size limits
- No sensitive data in logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the [implementation documentation](implementation.md)
2. Review the troubleshooting section
3. Open an issue with detailed reproduction steps

## 🗺️ Roadmap

- [ ] Support for more file formats (JSON, Parquet)
- [ ] Database connectivity (PostgreSQL, MySQL)
- [ ] Real-time streaming data processing
- [ ] Custom visualization templates
- [ ] Advanced ML operations (clustering, forecasting)
- [ ] Web UI for non-technical users 