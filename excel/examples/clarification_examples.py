"""
Examples showing clarification workflow in Excel Agent
"""
import requests
import json

API_BASE = "http://localhost:8000"
API_KEY = "your-anthropic-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def example_sheet_clarification():
    """Example: Multiple sheets requiring clarification"""
    
    # Initial request with multi-sheet Excel file
    request_data = {
        "goal": "Create a summary of sales data",
        "data_path": "s3://bucket/multi_sheet_sales.xlsx"
    }
    
    response = requests.post(f"{API_BASE}/analyze", headers=headers, json=request_data)
    result = response.json()
    
    print("Initial Response:")
    print(json.dumps(result, indent=2))
    
    # Check if clarification is needed
    if not result["success"] and result.get("clarification_needed"):
        clarification = result["clarification_needed"]
        
        if clarification["clarification_type"] == "sheet_name":
            print(f"\nClarification needed: {clarification['question']}")
            print(f"Available options: {clarification['options']}")
            
            # User selects a sheet
            selected_sheet = "Q1_Sales"  # User's choice
            
            # Make follow-up request
            follow_up_request = {
                "goal": "Create a summary of sales data",
                "data_path": "s3://bucket/multi_sheet_sales.xlsx",
                "sheet_name": selected_sheet
            }
            
            response = requests.post(f"{API_BASE}/analyze", headers=headers, json=follow_up_request)
            final_result = response.json()
            
            print(f"\nFinal Response after selecting '{selected_sheet}':")
            print(json.dumps(final_result, indent=2))


def example_column_clarification():
    """Example: Ambiguous column names requiring clarification"""
    
    # Sample data with similar column names
    sample_data = [
        {"sales_q1": 100, "sales_q2": 150, "total_sales": 250, "region": "North"},
        {"sales_q1": 200, "sales_q2": 180, "total_sales": 380, "region": "South"},
        {"sales_q1": 150, "sales_q2": 220, "total_sales": 370, "region": "East"},
    ]
    
    # Ambiguous goal that could refer to multiple columns
    request_data = {
        "goal": "Filter data where sales is greater than 150",  # Which sales column?
        "data": sample_data
    }
    
    response = requests.post(f"{API_BASE}/analyze", headers=headers, json=request_data)
    result = response.json()
    
    print("Initial Response:")
    print(json.dumps(result, indent=2))
    
    # Check if clarification is needed
    if not result["success"] and result.get("clarification_needed"):
        clarification = result["clarification_needed"]
        
        if clarification["clarification_type"] == "column_name":
            print(f"\nClarification needed: {clarification['question']}")
            print(f"Suggested columns: {clarification['options']}")
            
            # User provides clarification
            clarifications = {
                "sales_column": "total_sales"  # User's clarification
            }
            
            # Make follow-up request
            follow_up_request = {
                "goal": "Filter data where sales is greater than 150",
                "data": sample_data,
                "clarifications": clarifications
            }
            
            response = requests.post(f"{API_BASE}/analyze", headers=headers, json=follow_up_request)
            final_result = response.json()
            
            print(f"\nFinal Response with clarification:")
            print(json.dumps(final_result, indent=2))


def example_parameter_clarification():
    """Example: Missing parameter clarification"""
    
    sample_data = [
        {"date": "2023-01-01", "product": "A", "amount": 100, "category": "Electronics"},
        {"date": "2023-01-02", "product": "B", "amount": 150, "category": "Books"},
        {"date": "2023-01-03", "product": "C", "amount": 200, "category": "Electronics"},
    ]
    
    # Goal that needs clarification on aggregation method
    request_data = {
        "goal": "Create a pivot table showing amount by category",  # How to aggregate?
        "data": sample_data
    }
    
    response = requests.post(f"{API_BASE}/analyze", headers=headers, json=request_data)
    result = response.json()
    
    print("Initial Response:")
    print(json.dumps(result, indent=2))
    
    # The agent should be smart enough to use defaults, but could ask for clarification
    if not result["success"] and result.get("clarification_needed"):
        clarification = result["clarification_needed"]
        print(f"\nClarification needed: {clarification['question']}")
        
        # User provides clarification
        clarifications = {
            "aggregation_method": "sum"  # User specifies aggregation
        }
        
        follow_up_request = {
            "goal": "Create a pivot table showing amount by category",
            "data": sample_data,
            "clarifications": clarifications
        }
        
        response = requests.post(f"{API_BASE}/analyze", headers=headers, json=follow_up_request)
        final_result = response.json()
        
        print(f"\nFinal Response with clarification:")
        print(json.dumps(final_result, indent=2))


def example_clarification_workflow():
    """Complete clarification workflow example"""
    
    print("=== Excel Agent Clarification Examples ===\n")
    
    print("1. Sheet Name Clarification:")
    print("-" * 40)
    try:
        example_sheet_clarification()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n\n2. Column Name Clarification:")
    print("-" * 40)
    try:
        example_column_clarification()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n\n3. Parameter Clarification:")
    print("-" * 40)
    try:
        example_parameter_clarification()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_clarification_workflow() 