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