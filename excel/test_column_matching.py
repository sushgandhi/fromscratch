#!/usr/bin/env python3
"""
Test script for column matching functionality in Excel Agent
"""
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append('src')

from tools.filter_tool import FilterTool, FilterInput


def create_test_data():
    """Create test data with specific column names"""
    data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'product': ['A', 'B', 'A', 'B'],
        'sales': [100, 200, 150, 250],
        'region': ['North', 'South', 'North', 'South']
    }
    return pd.DataFrame(data)


def test_filter_tool_column_matching():
    """Test the filter tool with column name variations"""
    print("=" * 60)
    print("Testing Filter Tool Column Matching")
    print("=" * 60)
    
    # Create test data and save to CSV
    df = create_test_data()
    test_file = 'test_data.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Test data columns: {list(df.columns)}")
    print(f"Test data:\n{df}")
    print()
    
    # Initialize filter tool
    filter_tool = FilterTool()
    
    test_cases = [
        {
            'name': 'Exact Match (should work)',
            'column': 'product',
            'value': 'A',
            'expected_success': True
        },
        {
            'name': 'Case Mismatch (should auto-correct)',
            'column': 'Product',  # Note the capital P
            'value': 'A',
            'expected_success': True
        },
        {
            'name': 'Case Mismatch 2 (should auto-correct)',
            'column': 'SALES',  # All caps
            'value': 100,
            'expected_success': True
        },
        {
            'name': 'Typo (should suggest)',
            'column': 'prodct',  # Missing 'u'
            'value': 'A',
            'expected_success': False
        },
        {
            'name': 'Completely Wrong (should list options)',
            'column': 'xyz',
            'value': 'A',
            'expected_success': False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Looking for column: '{test_case['column']}'")
        
        # Create filter input
        filter_input = FilterInput(
            data_path=test_file,
            column=test_case['column'],
            value=test_case['value'],
            operator='=='
        )
        
        # Execute filter
        result = filter_tool.execute(filter_input)
        
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Filtered rows: {len(result.data) if result.data else 0}")
            if result.metadata:
                print(f"  Filter condition: {result.metadata.get('filter_condition')}")
        else:
            print(f"  Error: {result.error_message}")
        
        # Check if result matches expectation
        if result.success == test_case['expected_success']:
            print(f"  ✅ Test passed")
        else:
            print(f"  ❌ Test failed - expected success={test_case['expected_success']}")
        
        print()
    
    # Clean up test file
    os.remove(test_file)


def test_column_matcher_directly():
    """Test the column matcher utilities directly"""
    print("=" * 60)
    print("Testing Column Matcher Utilities Directly")
    print("=" * 60)
    
    from utils.column_matcher import (
        find_best_column_match, 
        suggest_column_correction,
        auto_correct_column_name
    )
    
    available_columns = ['date', 'product', 'sales', 'region', 'customer_name', 'order_id']
    
    test_cases = [
        ('Product', 'Should match product'),
        ('SALES', 'Should match sales'),
        ('Date', 'Should match date'),
        ('customer name', 'Should match customer_name'),
        ('Customer_Name', 'Should match customer_name'),
        ('prodct', 'Should suggest product'),
        ('xyz', 'Should not match anything'),
    ]
    
    for target_column, description in test_cases:
        print(f"Testing: '{target_column}' ({description})")
        
        # Test best match
        best_match = find_best_column_match(target_column, available_columns)
        if best_match:
            print(f"  Best match: '{best_match[0]}' (score: {best_match[1]:.2f})")
        else:
            print("  No good match found")
        
        # Test auto-correction
        corrected = auto_correct_column_name(target_column, available_columns)
        if corrected != target_column:
            print(f"  Auto-corrected to: '{corrected}'")
        
        # Test error message
        error_msg = suggest_column_correction(target_column, available_columns)
        print(f"  Error message: {error_msg}")
        print()


if __name__ == "__main__":
    print("Testing Excel Agent Column Matching Functionality")
    print()
    
    try:
        # Test the column matcher utilities directly
        test_column_matcher_directly()
        
        # Test the filter tool with column matching
        test_filter_tool_column_matching()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 