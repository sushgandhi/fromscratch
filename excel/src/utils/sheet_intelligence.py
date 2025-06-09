"""
Sheet name intelligence and clarification utilities for Excel Agent
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from .data_utils import get_workbook_sheets
from .claude_client import get_claude_client
from langchain.schema import HumanMessage


def extract_potential_sheet_names(user_query: str) -> List[str]:
    """
    Extract potential sheet names from user query using various patterns.
    
    Args:
        user_query: User's query string
        
    Returns:
        List of potential sheet names found in the query
    """
    potential_names = []
    
    # Common patterns for sheet references
    patterns = [
        # Direct sheet mentions
        r'sheet[s]?\s+["\']([^"\']+)["\']',  # sheet "Sales Data"
        r'sheet[s]?\s+named\s+["\']([^"\']+)["\']',  # sheet named "Sales"
        r'in\s+["\']([^"\']+)["\'](?:\s+sheet)?',  # in "Sales Data" sheet
        r'from\s+["\']([^"\']+)["\'](?:\s+sheet)?',  # from "Sales" sheet
        r'on\s+["\']([^"\']+)["\'](?:\s+sheet)?',  # on "Dashboard" sheet
        
        # Tab references
        r'tab[s]?\s+["\']([^"\']+)["\']',  # tab "Summary"
        r'tab[s]?\s+named\s+["\']([^"\']+)["\']',  # tab named "Data"
        
        # Without quotes but with keywords
        r'sheet[s]?\s+(\w+(?:\s+\w+){0,2})',  # sheet Sales or sheet Sales Data
        r'tab[s]?\s+(\w+(?:\s+\w+){0,2})',  # tab Summary
        r'(?:in|from|on)\s+(\w+(?:\s+\w+){0,2})(?:\s+sheet|\s+tab)',  # in Sales sheet
        
        # Common sheet name patterns (without explicit keywords)
        r'\b(sales|revenue|dashboard|summary|data|report|analysis|trends|metrics|kpi)(?:\s+\w+)?\b',
        r'\b(\w+(?:\s+(?:data|sheet|report|summary|analysis))\b)',
        
        # Time-based sheets
        r'\b(q[1-4]|quarter\s*[1-4]|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(monthly|weekly|daily|annual|yearly)\b'
    ]
    
    query_lower = user_query.lower()
    
    for pattern in patterns:
        matches = re.finditer(pattern, query_lower, re.IGNORECASE)
        for match in matches:
            # Get the captured group (sheet name)
            if match.groups():
                sheet_name = match.group(1).strip()
                if sheet_name and len(sheet_name) > 1:  # Avoid single characters
                    potential_names.append(sheet_name)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in potential_names:
        name_clean = name.lower().strip()
        if name_clean not in seen:
            seen.add(name_clean)
            unique_names.append(name)
    
    return unique_names


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_best_sheet_matches(potential_names: List[str], available_sheets: List[str], threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Find best matching sheets from available sheets based on potential names.
    
    Args:
        potential_names: Potential sheet names extracted from query
        available_sheets: Available sheet names in the workbook
        threshold: Minimum similarity threshold for matching
        
    Returns:
        List of match dictionaries with scores and sheet names
    """
    matches = []
    
    for potential in potential_names:
        for available in available_sheets:
            similarity = calculate_similarity(potential, available)
            
            # Also check if potential name is contained in available sheet name
            containment_score = 0
            if potential.lower() in available.lower():
                containment_score = len(potential) / len(available)
            elif available.lower() in potential.lower():
                containment_score = len(available) / len(potential)
            
            # Combined score
            final_score = max(similarity, containment_score)
            
            if final_score >= threshold:
                matches.append({
                    'potential_name': potential,
                    'sheet_name': available,
                    'similarity_score': similarity,
                    'containment_score': containment_score,
                    'final_score': final_score
                })
    
    # Sort by final score (highest first)
    matches.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Remove duplicate sheet matches (keep highest scored)
    seen_sheets = set()
    unique_matches = []
    for match in matches:
        if match['sheet_name'] not in seen_sheets:
            seen_sheets.add(match['sheet_name'])
            unique_matches.append(match)
    
    return unique_matches


def use_llm_for_sheet_selection(user_query: str, available_sheets: List[str], potential_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use LLM to intelligently select the most appropriate sheet based on context.
    
    Args:
        user_query: Original user query
        available_sheets: All available sheet names
        potential_matches: Potential matches found by pattern matching
        
    Returns:
        Dictionary with LLM's sheet selection and reasoning
    """
    try:
        llm = get_claude_client()
        
        prompt = f"""You are helping to select the most appropriate Excel sheet for a user's data analysis request.

User Query: "{user_query}"

Available Sheets in Workbook:
{json.dumps(available_sheets, indent=2)}

Pattern-Based Matches Found:
{json.dumps(potential_matches, indent=2)}

Please analyze the user's intent and select the most appropriate sheet. Consider:
1. What type of analysis the user wants to perform
2. What data they're likely looking for
3. Which sheet name best matches their intent
4. Context clues in their query

Respond in the following JSON format:
{{
    "selected_sheet": "exact_sheet_name_or_null",
    "confidence": 0.95,
    "reasoning": "Why this sheet was selected based on user intent",
    "alternatives": ["other_possible_sheets"],
    "needs_clarification": false,
    "clarification_question": "Optional question if multiple sheets are equally likely"
}}

Guidelines:
1. Only select a sheet if you're confident (>0.7) it matches user intent
2. Set needs_clarification to true if multiple sheets are equally likely
3. Consider the business context and typical data organization
4. Return null for selected_sheet if no clear match exists

Provide only the JSON response, no other text."""

        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        content = response.content
        
        # Parse JSON response
        try:
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                json_content = content.strip()
            
            llm_result = json.loads(json_content)
            return llm_result
            
        except json.JSONDecodeError as e:
            return {
                "selected_sheet": None,
                "confidence": 0.0,
                "reasoning": f"Failed to parse LLM response: {e}",
                "alternatives": [],
                "needs_clarification": True,
                "clarification_question": "Could you please specify which sheet you'd like to analyze?"
            }
            
    except Exception as e:
        return {
            "selected_sheet": None,
            "confidence": 0.0,
            "reasoning": f"LLM analysis failed: {e}",
            "alternatives": [],
            "needs_clarification": True,
            "clarification_question": "Could you please specify which sheet you'd like to analyze?"
        }


def analyze_sheet_selection(user_query: str, file_path: str, provided_sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive sheet name analysis and selection.
    
    Args:
        user_query: User's original query
        file_path: Path to the Excel file
        provided_sheet_name: Sheet name explicitly provided by user (if any)
        
    Returns:
        Dictionary with sheet selection results and any needed clarifications
    """
    try:
        # Get available sheets
        available_sheets = get_workbook_sheets(file_path)
        
        result = {
            "available_sheets": available_sheets,
            "selected_sheet": None,
            "confidence": 0.0,
            "needs_clarification": False,
            "clarification_question": None,
            "reasoning": "",
            "analysis_steps": []
        }
        
        # If no sheets, return error
        if not available_sheets:
            result["reasoning"] = "No sheets found in workbook"
            result["needs_clarification"] = True
            result["clarification_question"] = "This workbook appears to have no sheets. Please check the file."
            return result
        
        # If only one sheet, use it
        if len(available_sheets) == 1:
            result["selected_sheet"] = available_sheets[0]
            result["confidence"] = 1.0
            result["reasoning"] = f"Only one sheet available: {available_sheets[0]}"
            result["analysis_steps"].append("single_sheet_auto_selected")
            return result
        
        # If sheet name was explicitly provided, validate it
        if provided_sheet_name:
            # Exact match
            if provided_sheet_name in available_sheets:
                result["selected_sheet"] = provided_sheet_name
                result["confidence"] = 1.0
                result["reasoning"] = f"Exact match for provided sheet name: {provided_sheet_name}"
                result["analysis_steps"].append("exact_match_provided_name")
                return result
            
            # Fuzzy match for provided name
            matches = find_best_sheet_matches([provided_sheet_name], available_sheets, threshold=0.7)
            if matches:
                best_match = matches[0]
                if best_match['final_score'] > 0.8:
                    result["selected_sheet"] = best_match['sheet_name']
                    result["confidence"] = best_match['final_score']
                    result["reasoning"] = f"Close match for provided sheet name '{provided_sheet_name}' -> '{best_match['sheet_name']}'"
                    result["analysis_steps"].append("fuzzy_match_provided_name")
                    return result
            
            # Provided name doesn't match well
            result["needs_clarification"] = True
            result["clarification_question"] = f"Sheet '{provided_sheet_name}' not found. Available sheets: {', '.join(available_sheets)}. Which would you like to use?"
            result["reasoning"] = f"Provided sheet name '{provided_sheet_name}' not found in workbook"
            result["analysis_steps"].append("provided_name_not_found")
            return result
        
        # Extract potential sheet names from query
        potential_names = extract_potential_sheet_names(user_query)
        result["analysis_steps"].append(f"extracted_potential_names: {potential_names}")
        
        if potential_names:
            # Find matches
            matches = find_best_sheet_matches(potential_names, available_sheets, threshold=0.6)
            result["analysis_steps"].append(f"pattern_matches_found: {len(matches)}")
            
            # High confidence match
            if matches and matches[0]['final_score'] > 0.8:
                result["selected_sheet"] = matches[0]['sheet_name']
                result["confidence"] = matches[0]['final_score']
                result["reasoning"] = f"High confidence match: '{matches[0]['potential_name']}' -> '{matches[0]['sheet_name']}'"
                result["analysis_steps"].append("high_confidence_pattern_match")
                return result
            
            # Use LLM for intelligent selection
            llm_result = use_llm_for_sheet_selection(user_query, available_sheets, matches)
            result["analysis_steps"].append("llm_analysis_performed")
            
            if llm_result.get("selected_sheet") and llm_result.get("confidence", 0) > 0.7:
                result["selected_sheet"] = llm_result["selected_sheet"]
                result["confidence"] = llm_result["confidence"]
                result["reasoning"] = llm_result["reasoning"]
                result["analysis_steps"].append("llm_confident_selection")
                return result
            
            # LLM suggests clarification needed
            if llm_result.get("needs_clarification"):
                result["needs_clarification"] = True
                result["clarification_question"] = llm_result.get("clarification_question", 
                    f"Multiple sheets could match your request. Available: {', '.join(available_sheets)}. Which would you like to analyze?")
                result["reasoning"] = llm_result.get("reasoning", "Multiple possible matches found")
                result["analysis_steps"].append("llm_requested_clarification")
                return result
        
        # No clear matches found - ask for clarification
        result["needs_clarification"] = True
        result["clarification_question"] = f"I couldn't determine which sheet to analyze from your request. Available sheets: {', '.join(available_sheets)}. Which would you like to use?"
        result["reasoning"] = "No clear sheet matches found in query"
        result["analysis_steps"].append("no_matches_found")
        
        return result
        
    except Exception as e:
        return {
            "available_sheets": [],
            "selected_sheet": None,
            "confidence": 0.0,
            "needs_clarification": True,
            "clarification_question": f"Error analyzing sheets: {str(e)}. Please specify which sheet to use.",
            "reasoning": f"Sheet analysis failed: {str(e)}",
            "analysis_steps": ["error_occurred"]
        }


def create_clarification_response(sheet_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a structured clarification response for the user.
    
    Args:
        sheet_analysis: Result from analyze_sheet_selection
        
    Returns:
        Formatted clarification response
    """
    if not sheet_analysis.get("needs_clarification"):
        return {
            "needs_clarification": False,
            "selected_sheet": sheet_analysis.get("selected_sheet"),
            "message": f"Using sheet: {sheet_analysis.get('selected_sheet')}"
        }
    
    available_sheets = sheet_analysis.get("available_sheets", [])
    
    return {
        "needs_clarification": True,
        "clarification_question": sheet_analysis.get("clarification_question"),
        "available_options": available_sheets,
        "suggested_format": "Please respond with the exact sheet name you'd like to analyze",
        "example_response": f"Use sheet '{available_sheets[0]}'" if available_sheets else "Use sheet 'Sheet1'"
    } 