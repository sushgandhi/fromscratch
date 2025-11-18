import json
import re

# Load the dictionary once at startup
try:
    with open("sp500.json", "r") as f:
        SP500_DIRECTORY = json.load(f)
except FileNotFoundError:
    print("Warning: sp500.json not found. Lookup tool will be empty.")
    SP500_DIRECTORY = {}

@tool
def lookup_company_id(company_name: str) -> str:
    """
    Searches the S&P 500 directory for a company ticker using a regex match.
    Example: "Apple", "apple inc", "Alphabet" will all return matches.

    Args:
        company_name: The common name or partial name to search for.

    Returns:
        A JSON string containing the best matching tickers and names.
    """
    # Create a clean search pattern:
    # 1. Escape special regex chars in the user input
    # 2. Allow for flexible spacing
    cleaned_input = re.escape(company_name.strip())
    pattern = re.compile(cleaned_input, re.IGNORECASE)
    
    matches = []

    for ticker, name in SP500_DIRECTORY.items():
        # Match against Ticker OR Name
        if pattern.search(ticker) or pattern.search(name):
            matches.append({
                "ticker": ticker,
                "name": name,
                # Score match quality: exact match is better than partial
                "score": 100 if company_name.lower() == name.lower() or company_name.upper() == ticker else 50
            })
    
    # Sort by score (descending) then by name length (shorter name usually = more exact match)
    matches.sort(key=lambda x: (-x['score'], len(x['name'])))
    
    if not matches:
        return json.dumps({"error": f"No S&P 500 company found matching '{company_name}'."})
        
    # Return top 5 matches
    return json.dumps(matches[:5], indent=2)
