# Excel Agent Implementation Documentation

## Overview

The Excel Agent is an end-to-end AI-powered system built using LangGraph in Python that can analyze Excel files and perform various operations based on natural language goals. The system uses Claude (Anthropic) as the underlying LLM for intelligent operation planning and execution.

## Architecture

### Core Components

1. **Supervisor Agent** (`src/agent/supervisor.py`)
   - Main orchestrator using LangGraph
   - Plans and executes operations based on user goals
   - Manages state and operation chaining
   - Uses Claude for intelligent decision making

2. **Tools System** (`src/tools/`)
   - Modular tool architecture with base class
   - Each tool handles specific Excel operations
   - Support for both data input and S3 file paths
   - Standardized input/output formats

3. **State Management** (`src/agent/state.py`)
   - LangGraph state definitions
   - Operation tracking and dependencies
   - Error handling and iteration limits

4. **Utilities** (`src/utils/`)
   - Data type conversion utilities
   - Excel/CSV reading functions
   - S3 integration for file storage

5. **API Interface** (`src/api.py`)
   - FastAPI REST endpoints
   - File upload support
   - Health checks and tool listings

## Available Tools

### 1. Filter Tool (`filter_tool.py`)
**Purpose**: Filter data based on column conditions

**Input Parameters**:
- `column`: Column name to filter on
- `value`: Value to filter for
- `operator`: Filter operator (==, !=, >, <, >=, <=, contains, startswith, endswith)
- `case_sensitive`: Boolean for string comparisons
- `data_path` OR `data`: Input data source
- `output_path`: Optional S3 path to save results

**Example**:
```python
FilterInput(
    data=sample_data,
    column="product",
    value="A",
    operator="=="
)
```

### 2. Pivot Tool (`pivot_tool.py`)
**Purpose**: Create pivot tables from data

**Input Parameters**:
- `index`: Column(s) to use as index/rows
- `columns`: Column(s) to use as columns (optional)
- `values`: Column(s) to use as values
- `aggfunc`: Aggregation function (sum, mean, count, min, max, std, var, median)
- `fill_value`: Value to fill missing entries (optional)
- `margins`: Whether to add row/column totals

**Example**:
```python
PivotInput(
    data=sample_data,
    index="month",
    columns="product",
    values="sales",
    aggfunc="sum"
)
```

### 3. Group By Tool (`groupby_tool.py`)
**Purpose**: Group data and apply aggregation functions

**Input Parameters**:
- `group_by`: Column(s) to group by
- `aggregations`: Dictionary mapping column names to aggregation functions
- `sort_by`: Column to sort results by (optional)
- `sort_ascending`: Sort order

**Example**:
```python
GroupByInput(
    data=sample_data,
    group_by="region",
    aggregations={"sales": "sum", "quantity": "mean"}
)
```

### 4. Visualization Tool (`visualization_tool.py`)
**Purpose**: Create interactive Plotly visualizations

**Input Parameters**:
- `chart_type`: Type of chart (bar, line, scatter, pie, histogram, box, heatmap)
- `x_column`: Column for x-axis
- `y_column`: Column for y-axis
- `color_column`: Column for color grouping (optional)
- `size_column`: Column for size in scatter plots (optional)
- `title`: Chart title (optional)
- `width/height`: Chart dimensions

**Example**:
```python
VisualizationInput(
    data=sample_data,
    chart_type="bar",
    x_column="product",
    y_column="sales",
    title="Sales by Product"
)
```

### 5. Summary Tools (`summary_tools.py`)

#### Column Summary Tool
**Purpose**: Generate statistical summaries for columns
- Numeric statistics (mean, median, std, min, max, quartiles)
- Text statistics (length, most common values)
- Date statistics (range, earliest, latest)

#### Sheet Summary Tool
**Purpose**: Generate comprehensive sheet analysis
- Basic info (rows, columns, data types)
- Data quality metrics (nulls, duplicates)
- Correlation analysis for numeric columns
- Optional visualizations

#### Workbook Summary Tool
**Purpose**: Analyze entire Excel workbooks
- Multi-sheet analysis
- Sheet comparison visualizations
- Comprehensive HTML reports

## LangGraph Workflow

### State Management
The agent maintains state through the `AgentState` TypedDict:
- `goal`: User's natural language goal
- `operations`: List of planned operations
- `current_data`: Current dataset being processed
- `operation_results`: Results from completed operations
- `completed_operations`: Track of finished operations
- `error_message`: Error handling
- `iteration_count`: Prevent infinite loops

### Workflow Nodes

1. **Planner Node**
   - Analyzes user goal using Claude
   - Breaks down goal into sequence of operations
   - Handles operation dependencies
   - Supports re-planning based on results

2. **Executor Node**
   - Executes individual operations
   - Manages data flow between operations
   - Handles tool input/output conversion
   - Updates state with results

3. **Reviewer Node**
   - Evaluates if goal is achieved
   - Determines next actions (continue, replan, end)
   - Compiles final results

### ToolNode vs Custom Execution

This implementation uses a **custom tool execution pattern** rather than LangGraph's `ToolNode`. Here's why:

**Our Custom Approach:**
- Deterministic operation planning by Claude
- Custom tool interfaces with Pydantic input models
- Explicit dependency management between operations
- Sequential execution with state management

**ToolNode Alternative:**
- AI-driven tool calling based on tool_calls in messages
- Standard LangChain tool interfaces
- Automatic tool execution from AI model decisions
- Better for conversational agents with dynamic tool selection

Our approach provides more control over operation sequencing and data flow, which is ideal for complex Excel workflows.

### Operation Chaining

Operations can be chained by setting dependencies:
```python
[
    {
        "operation_id": "filter_op",
        "tool_name": "filter",
        "parameters": {...},
        "depends_on": null
    },
    {
        "operation_id": "viz_op", 
        "tool_name": "visualization",
        "parameters": {...},
        "depends_on": "filter_op"  # Uses output from filter_op
    }
]
```

## Data Flow

### Input Data Sources
1. **Direct Data**: List of dictionaries format
2. **File Paths**: Local file paths or S3 URLs
3. **Uploaded Files**: Via FastAPI file upload

### Data Processing
1. All data is normalized to pandas DataFrames internally
2. Results can be returned as:
   - List of dictionaries (for API responses)
   - Saved to S3 (for persistence)
   - HTML reports (for summaries and visualizations)

### Output Formats
- **CSV**: For tabular data
- **JSON**: For structured data and metadata
- **HTML**: For reports with visualizations
- **Plotly JSON**: For interactive charts

## Error Handling

### Validation
- Input parameter validation at tool level
- Column existence checks
- Data type compatibility verification
- S3 path validation

### Recovery
- Maximum iteration limits prevent infinite loops
- Graceful degradation on tool failures
- Detailed error messages for debugging
- Operation rollback capabilities

### Logging
- Operation tracking in state
- Result metadata for monitoring
- Error message propagation

## Configuration

### Authentication
The API uses Bearer token authentication. Each request must include the Anthropic API key in the Authorization header:
```
Authorization: Bearer your-anthropic-api-key-here
```

### Environment Variables (Optional)
```bash
# Optional (for S3 support)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Claude Client Setup
The system uses a centralized Claude client utility located in `src/utils/claude_client.py`:

```python
from src.utils.claude_client import get_claude_client

# Create client with API key
claude_client = get_claude_client(
    api_key="your-api-key",
    model="claude-3-sonnet-20240229",  # optional
    temperature=0  # optional
)

# Use with supervisor
supervisor = ExcelAgentSupervisor(claude_client)
```

This centralized approach eliminates code duplication and provides consistent client configuration across the application.

## Clarification System

The API supports a stateless clarification workflow for ambiguous requests:

### Types of Clarifications

1. **Sheet Name Clarification**: When Excel file has multiple sheets
2. **Column Name Clarification**: When column references are ambiguous  
3. **Parameter Clarification**: When operation parameters need specification

### Clarification Workflow

1. **Initial Request**: Client sends request with goal
2. **Clarification Check**: API checks if clarification needed
3. **Clarification Response**: API returns clarification question if needed
4. **Follow-up Request**: Client provides clarification and re-submits
5. **Execution**: API processes with clarifications provided

### Request/Response Models

```python
class ClarificationRequest(BaseModel):
    question: str
    options: Optional[List[str]] = None
    clarification_type: str  # "sheet_name", "column_name", "parameter"

class ExcelAgentRequest(BaseModel):
    goal: str
    data_path: Optional[str] = None
    data: Optional[List[Dict]] = None
    sheet_name: Optional[str] = None  # User-specified sheet
    clarifications: Optional[Dict[str, str]] = None  # Previous clarifications

class ExcelAgentResponse(BaseModel):
    success: bool
    goal: str
    # ... existing fields ...
    clarification_needed: Optional[ClarificationRequest] = None
    available_sheets: Optional[List[str]] = None
    available_columns: Optional[List[str]] = None
```

## API Endpoints

All API endpoints require authentication via Bearer token in the Authorization header:
```
Authorization: Bearer your-anthropic-api-key-here
```

### POST `/analyze`
Analyze data with natural language goal
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-anthropic-api-key-here" \
     -d '{
       "goal": "Filter sales data for 2023 and create a bar chart",
       "data": [...] // or "data_path": "s3://bucket/file.xlsx"
     }'
```

### POST `/upload-and-analyze`
Upload Excel file and analyze
```bash
curl -X POST "http://localhost:8000/upload-and-analyze" \
     -H "Authorization: Bearer your-anthropic-api-key-here" \
     -F "file=@data.xlsx" \
     -F "goal=Analyze sales trends and create visualizations"
```

### GET `/health`
Health check endpoint

### GET `/tools`
List available tools and their parameters

## Usage Examples

### Basic Filtering and Visualization
```python
from src.agent.supervisor import ExcelAgentSupervisor

agent = ExcelAgentSupervisor()
result = agent.run(
    goal="Filter data for product A and create a bar chart showing sales by date",
    data=sample_data
)
```

### Complex Multi-Step Analysis
```python
result = agent.run(
    goal="""
    1. Filter data for Electronics category
    2. Group by product and calculate total sales
    3. Create a bar chart visualization
    4. Generate a summary report
    """,
    data=sample_data
)
```

### Excel File Analysis
```python
result = agent.run(
    goal="Analyze sales trends and create pivot table by region",
    data_path="/path/to/file.xlsx"
)
```

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-key-here"

# Run API server
python -m src.api
```

### Production Deployment
- Use environment variables for API keys
- Configure S3 bucket permissions
- Set up proper logging
- Consider rate limiting for API endpoints
- Use HTTPS in production

## Extension Points

### Adding New Tools
1. Create new tool class inheriting from `BaseTool`
2. Define input/output models
3. Implement `execute()` method
4. Add to supervisor's tool registry
5. Update API documentation

### Custom Claude Integration
Override the `get_claude_client()` function to customize:
- Model selection
- Temperature settings
- Custom prompting
- Usage tracking

### Data Source Extensions
Extend `data_utils.py` to support:
- Database connections
- Other file formats
- Streaming data sources
- Real-time data feeds

## Testing

### Unit Tests
- Individual tool testing
- Data utility function tests
- State management validation

### Integration Tests
- Full workflow testing
- API endpoint testing
- File upload/download testing

### Performance Testing
- Large dataset handling
- Memory usage optimization
- Operation chaining efficiency

## Security Considerations

- API keys passed as Bearer tokens (not stored on server)
- Per-request authentication with client's own API key
- No API key persistence or logging
- S3 bucket access controls
- Input validation and sanitization
- File upload size limits
- Rate limiting for API endpoints

## Troubleshooting

### Common Issues
1. **Claude API Key Issues**: Ensure ANTHROPIC_API_KEY is set correctly
2. **S3 Access Errors**: Verify AWS credentials and bucket permissions
3. **Memory Issues**: Large Excel files may require optimization
4. **Tool Chain Failures**: Check operation dependencies and data formats

### Debug Mode
Enable detailed logging by setting environment variables:
```bash
export LOG_LEVEL=DEBUG
export LANGCHAIN_VERBOSE=true
``` 