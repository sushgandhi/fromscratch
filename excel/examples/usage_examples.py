"""
Example usage of the Excel Agent
"""
import os
import sys
sys.path.append('..')

from src.agent.supervisor import ExcelAgentSupervisor
from src.utils.claude_client import get_claude_client
import os


def example_1_filter_and_visualize():
    """Example: Filter sales data and create visualization"""
    
    # Sample data
    sample_data = [
        {"date": "2023-01-01", "product": "A", "sales": 100, "region": "North"},
        {"date": "2023-01-02", "product": "B", "sales": 150, "region": "South"},
        {"date": "2023-01-03", "product": "A", "sales": 120, "region": "North"},
        {"date": "2023-01-04", "product": "C", "sales": 200, "region": "East"},
        {"date": "2023-01-05", "product": "B", "sales": 180, "region": "South"},
    ]
    
    # Initialize agent  
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    claude_client = get_claude_client(api_key)
    agent = ExcelAgentSupervisor(claude_client)
    
    # Run analysis
    result = agent.run(
        goal="Filter data for product A and create a bar chart showing sales by date",
        data=sample_data
    )
    
    print("Result:", result)


def example_2_pivot_and_summary():
    """Example: Create pivot table and generate summary"""
    
    # Sample sales data
    sample_data = [
        {"month": "Jan", "product": "A", "region": "North", "sales": 1000},
        {"month": "Jan", "product": "B", "region": "South", "sales": 1500},
        {"month": "Feb", "product": "A", "region": "North", "sales": 1200},
        {"month": "Feb", "product": "B", "region": "South", "sales": 1800},
        {"month": "Jan", "product": "A", "region": "South", "sales": 800},
        {"month": "Feb", "product": "B", "region": "North", "sales": 1600},
    ]
    
    # Initialize agent
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    claude_client = get_claude_client(api_key)
    agent = ExcelAgentSupervisor(claude_client)
    
    # Run analysis
    result = agent.run(
        goal="Create a pivot table showing total sales by month and product, then generate a summary of the results",
        data=sample_data
    )
    
    print("Result:", result)


def example_3_excel_file_analysis():
    """Example: Analyze Excel file from path"""
    
    # Initialize agent
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    claude_client = get_claude_client(api_key)
    agent = ExcelAgentSupervisor(claude_client)
    
    # Run analysis
    result = agent.run(
        goal="Analyze the sales data, filter for sales above 1000, group by region, and create a visualization",
        data_path="path/to/your/excel/file.xlsx"
    )
    
    print("Result:", result)


def example_4_chained_operations():
    """Example: Complex chained operations"""
    
    # Sample data
    sample_data = [
        {"date": "2023-01-01", "product": "Laptop", "category": "Electronics", "sales": 1200, "region": "North"},
        {"date": "2023-01-02", "product": "Phone", "category": "Electronics", "sales": 800, "region": "South"},
        {"date": "2023-01-03", "product": "Tablet", "category": "Electronics", "sales": 600, "region": "East"},
        {"date": "2023-01-04", "product": "Laptop", "category": "Electronics", "sales": 1500, "region": "West"},
        {"date": "2023-01-05", "product": "Phone", "category": "Electronics", "sales": 900, "region": "North"},
        {"date": "2023-01-06", "product": "Book", "category": "Education", "sales": 50, "region": "South"},
        {"date": "2023-01-07", "product": "Pen", "category": "Education", "sales": 20, "region": "East"},
    ]
    
    # Initialize agent
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    claude_client = get_claude_client(api_key)
    agent = ExcelAgentSupervisor(claude_client)
    
    # Run complex analysis
    result = agent.run(
        goal="""
        1. Filter data for Electronics category
        2. Group by product and calculate total sales
        3. Create a bar chart visualization of sales by product
        4. Generate a summary of the electronics sales data
        """,
        data=sample_data
    )
    
    print("Result:", result)


if __name__ == "__main__":
    print("Excel Agent Examples")
    print("1. Filter and Visualize")
    print("2. Pivot and Summary") 
    print("3. Excel File Analysis")
    print("4. Chained Operations")
    
    choice = input("Choose example (1-4): ")
    
    if choice == "1":
        example_1_filter_and_visualize()
    elif choice == "2":
        example_2_pivot_and_summary()
    elif choice == "3":
        example_3_excel_file_analysis()
    elif choice == "4":
        example_4_chained_operations()
    else:
        print("Invalid choice") 