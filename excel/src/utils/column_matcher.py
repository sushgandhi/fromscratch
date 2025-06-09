"""
Column matching utilities for handling fuzzy column name matching
"""
from typing import List, Optional, Tuple
import re
from difflib import SequenceMatcher


def normalize_column_name(column_name: str) -> str:
    """
    Normalize a column name for matching
    
    Args:
        column_name: The column name to normalize
        
    Returns:
        Normalized column name (lowercase, no extra spaces, underscores for spaces)
    """
    if not isinstance(column_name, str):
        return str(column_name).lower().strip()
    
    # Convert to lowercase and strip
    normalized = column_name.lower().strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Replace spaces with underscores for more consistent matching
    normalized = normalized.replace(' ', '_')
    
    return normalized


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, str1, str2).ratio()


def find_best_column_match(
    target_column: str, 
    available_columns: List[str], 
    threshold: float = 0.8
) -> Optional[Tuple[str, float]]:
    """
    Find the best matching column from available columns
    
    Args:
        target_column: The column name we're looking for
        available_columns: List of available column names
        threshold: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (best_match, similarity_score) or None if no good match found
    """
    if not target_column or not available_columns:
        return None
    
    # First try exact match (case insensitive)
    target_normalized = normalize_column_name(target_column)
    
    for col in available_columns:
        if normalize_column_name(col) == target_normalized:
            return (col, 1.0)
    
    # If no exact match, try fuzzy matching
    best_match = None
    best_score = 0.0
    
    for col in available_columns:
        col_normalized = normalize_column_name(col)
        
        # Calculate similarity between normalized names
        score = calculate_similarity(target_normalized, col_normalized)
        
        if score > best_score and score >= threshold:
            best_match = col
            best_score = score
    
    return (best_match, best_score) if best_match else None


def find_similar_columns(
    target_column: str, 
    available_columns: List[str], 
    max_suggestions: int = 3,
    threshold: float = 0.6
) -> List[Tuple[str, float]]:
    """
    Find multiple similar columns for suggestions
    
    Args:
        target_column: The column name we're looking for
        available_columns: List of available column names
        max_suggestions: Maximum number of suggestions to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of tuples (column_name, similarity_score) sorted by score
    """
    if not target_column or not available_columns:
        return []
    
    target_normalized = normalize_column_name(target_column)
    
    suggestions = []
    
    for col in available_columns:
        col_normalized = normalize_column_name(col)
        score = calculate_similarity(target_normalized, col_normalized)
        
        if score >= threshold:
            suggestions.append((col, score))
    
    # Sort by similarity score (descending) and return top suggestions
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:max_suggestions]


def suggest_column_correction(
    target_column: str, 
    available_columns: List[str]
) -> str:
    """
    Generate a helpful error message with column suggestions
    
    Args:
        target_column: The column name that wasn't found
        available_columns: List of available column names
        
    Returns:
        Formatted error message with suggestions
    """
    # Find the best match
    best_match = find_best_column_match(target_column, available_columns, threshold=0.6)
    
    if best_match:
        match_col, score = best_match
        if score > 0.8:
            return f"Column '{target_column}' not found. Did you mean '{match_col}'?"
    
    # Get multiple suggestions
    suggestions = find_similar_columns(target_column, available_columns)
    
    if suggestions:
        suggestion_names = [col for col, _ in suggestions]
        if len(suggestion_names) == 1:
            return f"Column '{target_column}' not found. Did you mean '{suggestion_names[0]}'?"
        else:
            suggestions_str = "', '".join(suggestion_names)
            return f"Column '{target_column}' not found. Did you mean one of: '{suggestions_str}'?"
    
    # Fallback to just listing available columns
    cols_str = "', '".join(available_columns)
    return f"Column '{target_column}' not found. Available columns: ['{cols_str}']"


def auto_correct_column_name(
    target_column: str, 
    available_columns: List[str],
    auto_correct_threshold: float = 0.9
) -> str:
    """
    Automatically correct a column name if there's a very close match
    
    Args:
        target_column: The column name to correct
        available_columns: List of available column names
        auto_correct_threshold: Threshold for automatic correction
        
    Returns:
        Corrected column name or original if no good match
    """
    best_match = find_best_column_match(target_column, available_columns, threshold=auto_correct_threshold)
    
    if best_match:
        return best_match[0]
    
    return target_column


# Example usage and test functions
def test_column_matching():
    """Test the column matching functions"""
    available_columns = ['date', 'product', 'sales', 'region', 'customer_name', 'order_id']
    
    test_cases = [
        'Product',  # Should match 'product'
        'SALES',    # Should match 'sales'
        'Date',     # Should match 'date'
        'customer name',  # Should match 'customer_name'
        'Customer_Name',  # Should match 'customer_name'
        'prodct',   # Should suggest 'product'
        'xyz',      # Should not match anything
    ]
    
    print("Column Matching Test Results:")
    print("=" * 50)
    
    for test_col in test_cases:
        print(f"\nSearching for: '{test_col}'")
        
        # Test best match
        best_match = find_best_column_match(test_col, available_columns)
        if best_match:
            print(f"  Best match: '{best_match[0]}' (score: {best_match[1]:.2f})")
        else:
            print("  No good match found")
        
        # Test auto-correction
        corrected = auto_correct_column_name(test_col, available_columns)
        if corrected != test_col:
            print(f"  Auto-corrected to: '{corrected}'")
        
        # Test suggestions
        suggestions = find_similar_columns(test_col, available_columns)
        if suggestions:
            print(f"  Suggestions: {[(col, f'{score:.2f}') for col, score in suggestions]}")
        
        # Test error message
        error_msg = suggest_column_correction(test_col, available_columns)
        print(f"  Error message: {error_msg}")


if __name__ == "__main__":
    test_column_matching() 