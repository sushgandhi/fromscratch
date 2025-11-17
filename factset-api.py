
#    #  FactSet credentials
#    export FACTSET_USERNAME="YOUR_USERNAME_SERIAL"
#    export FACTSET_API_KEY="YOUR_API_KEY_HERE"
#
#    #  corporate NTLM proxy details
#    export PROXY_URL="http://your-proxy-server:8080"
#    export PROXY_USER="YOUR_NTLM_USERNAME"
#    export PROXY_PASSWORD="YOUR_NTLM_PASSWORD"
#

import subprocess
import os
import json
import urllib.parse
from typing import Any, Dict, List, Literal, Optional


from fastmcp import tool


# --- Type Definitions ---
StatementType = Literal["IS", "BS", "CF"]
Periodicity = Literal["ANN", "QTR", "SEM", "LTM"]
SegmentType = Literal["BUS", "GEO"] # Business or Geographic


class FactSetAPIClient:
    """
    A client for making live calls to the FactSet API by executing
    'curl' commands via subprocess.
    - Handles NTLM proxy authentication.
    - Uses HTTP Basic Auth (Username/API Key) for FactSet.
    """
    def __init__(self, 
                 base_url="https://api.factset.com/content/factset-fundamentals/v2"):
        self.base_url = base_url
        
        # 1. FactSet API Authentication (Basic Auth)
        self.api_username = os.environ.get("FACTSET_USERNAME")
        self.api_key = os.environ.get("FACTSET_API_KEY")
        
        # 2. Corporate Proxy (NTLM) Authentication
        self.proxy_url = os.environ.get("PROXY_URL")
        self.proxy_user = os.environ.get("PROXY_USER")
        self.proxy_pass = os.environ.get("PROXY_PASSWORD")

        # Validation
        if not self.api_username or not self.api_key:
            print("Error: FACTSET_USERNAME or FACTSET_API_KEY environment variable not set.")
        if not all([self.proxy_url, self.proxy_user, self.proxy_pass]):
            print("Warning: Proxy variables (PROXY_URL, PROXY_USER, PROXY_PASSWORD) are not all set.")
            print("The curl command may fail if a proxy is required.")

    def make_request(self, 
                     method: str, 
                     endpoint: str, 
                     params: Optional[Dict] = None, 
                     json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Makes a generic request to the FactSet API using curl.

        Args:
            method: HTTP method ("GET", "POST").
            endpoint: API endpoint path (e.g., "/metrics").
            params: URL query parameters for GET requests.
            json_data: The JSON body for POST requests.

        Returns:
            The JSON response as a dictionary.
        """
        if not self.api_username or not self.api_key:
             return {"error": "FactSet API credentials not configured."}

       
        curl_command = [
            'curl',
            '-s',          # Silent mode
            '-S',          # Show errors
            '-X', method   # Set request method (GET, POST)
        ]

        #  Add Proxy & NTLM Authentication
        if self.proxy_url and self.proxy_user and self.proxy_pass:
            curl_command.extend([
                '--proxy', self.proxy_url,
                '--proxy-ntlm',
                '--proxy-user', f"{self.proxy_user}:{self.proxy_pass}"
            ])
        
        # Add FactSet Basic Authentication
 
        curl_command.extend([
            '-u', f"{self.api_username}:{self.api_key}"
        ])

        # Add common headers
        curl_command.extend([
            '-H', "Accept: application/json"
        ])

        # Construct Full URL with query params (for GET)
        full_url = f"{self.base_url}{endpoint}"
        if params:
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{full_url}?{query_string}"
        
        curl_command.append(full_url)

        # Add Data Body (for POST)
        if json_data:
            data_string = json.dumps(json_data)
            curl_command.extend([
                '-H', "Content-Type: application/json",
                '-d', data_string
            ])

        # Execute the command
        print(f"[Debug] Executing curl for: {method} {endpoint}")
        
        try:
            result = subprocess.run(
                curl_command, 
                capture_output=True, 
                text=True, 
                check=True,
                encoding='utf-8'
            )
            return json.loads(result.stdout)
        
        except subprocess.CalledProcessError as e:
            print(f"Error executing curl: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            try:
                return json.loads(e.stdout or e.stderr)
            except json.JSONDecodeError:
                return {"error": "Curl command failed", "stderr": e.stderr}
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from curl output: {e}")
            return {"error": "JSON decode error", "output_snippet": result.stdout[:200] + "..."}
        except FileNotFoundError:
            return {"error": "curl is not installed or not in your PATH."}

    
    def get_profile(self, ids: List[str]) -> Dict[str, Any]:
        """Calls GET /company-reports/profile"""
        return self.make_request("GET", "/company-reports/profile", params={"ids": ids})

    def get_fundamentals_report(self, ids: List[str], periodicity: str = "ANN", fiscal_year: Optional[int] = None) -> Dict[str, Any]:
        """Calls GET /company-reports/fundamentals"""
        params = {"ids": ids, "periodicity": periodicity}
        if fiscal_year:
            params["fiscalYear"] = fiscal_year
        return self.make_request("GET", "/company-reports/fundamentals", params=params)

    def get_statement(self, ids: List[str], statement_type: StatementType, periodicity: Periodicity, fiscal_year: Optional[int] = None) -> Dict[str, Any]:
        """Calls GET /company-reports/financial-statement"""
        params = {
            "ids": ids,
            "statementType": statement_type,
            "periodicity": periodicity,
        }
        if fiscal_year:
            params["fiscalYear"] = fiscal_year
        return self.make_request("GET", "/company-reports/financial-statement", params=params)

    def post_fundamentals(self, payload: Dict) -> Dict[str, Any]:
        """Calls POST /fundamentals"""
        return self.make_request("POST", "/fundamentals", json_data=payload)

    def post_point_in_time(self, payload: Dict) -> Dict[str, Any]:
        """Calls POST /point-in-time"""
        return self.make_request("POST", "/point-in-time", json_data=payload)
    
    def post_segments(self, payload: Dict) -> Dict[str, Any]:
        """Calls POST /segments"""
        return self.make_request("POST", "/segments", json_data=payload)

    def get_metrics_catalog(self, category: Optional[str] = None, subcategory: Optional[str] = None) -> Dict[str, Any]:
        """Calls GET /metrics"""
        params = {}
        if category:
            params["category"] = category
        if subcategory:
            params["subcategory"] = subcategory
        return self.make_request("GET", "/metrics", params=params)


# --- Tool Definitions ---

# Initialize the live client.
client = FactSetAPIClient()

@tool
def get_company_profile(company_ticker: str) -> str:
    """
    Gets a high-level profile and key fundamentals for a single company.
    Args:
        company_ticker: The stock ticker symbol (e.g., "AAPL", "MSFT").
    Returns:
        A JSON string containing the company's profile and key fundamental data.
    """
    try:
        profile_data = client.get_profile(ids=[company_ticker])
        fundamentals_data = client.get_fundamentals_report(ids=[company_ticker], periodicity="LTM")
        combined_data = {
            "profile": profile_data.get("data", [{}])[0],
            "fundamentals": fundamentals_data.get("data", [{}])[0]
        }
        return json.dumps(combined_data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get company profile: {e}"})

@tool
def get_financial_statement(
    company_ticker: str,
    statement_type: StatementType,
    periodicity: Periodicity = "ANN",
    fiscal_year: Optional[int] = None
) -> str:
    """
    Fetches a pre-formatted financial statement for a single company.
    Args:
        company_ticker: The stock ticker symbol (e.g., "AAPL").
        statement_type: Must be one of: "IS", "BS", "CF".
        periodicity: Must be one of: "ANN", "QTR", "SEM", "LTM". Defaults to "ANN".
        fiscal_year: The specific fiscal year (e.g., 2023). If None, retrieves the latest.
    Returns:
        A JSON string containing the requested financial statement data.
    """
    try:
        statement_data = client.get_statement(
            ids=[company_ticker],
            statement_type=statement_type,
            periodicity=periodicity,
            fiscal_year=fiscal_year
        )
        return json.dumps(statement_data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get financial statement: {e}"})

@tool
def get_specific_metrics(
    company_tickers: List[str],
    metrics_list: List[str],
    periodicity: Periodicity,
    start_period: str,
    end_period: str,
    currency: str = "USD"
) -> str:
    """
    Gets specific financial data points (metrics) for one or more companies.
    Args:
        company_tickers: A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        metrics_list: A list of FactSet metric codes (e.g., ["FF_SALES", "FF_NET_INC"]).
        periodicity: Must be one of: "ANN", "QTR", "SEM", "LTM".
        start_period: The start of the period in "YYYY-MM-DD" or "YYYY" format.
        end_period: The end of the period in "YYYY-MM-DD" or "YYYY" format.
        currency: The currency for the data. Defaults to "USD".
    Returns:
        A JSON string mapping each company to its requested metric data.
    """
    try:
        payload = {
            "data": {
                "ids": company_tickers,
                "periodicity": periodicity,
                "fiscalPeriod": {"start": start_period, "end": end_period},
                "metrics": metrics_list,
                "currency": currency,
                "updateType": "RP"
            }
        }
        data = client.post_fundamentals(payload)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get specific metrics: {e}"})

@tool
def get_point_in_time_metrics(
    company_tickers: List[str],
    metrics_list: List[str],
    as_of_date: str
) -> str:
    """
    Gets fundamental data as it was known on a specific past date (Point-in-Time).
    Args:
        company_tickers: A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        metrics_list: A list of FactSet metric codes (e.g., ["FF_PE"]).
        as_of_date: The date to check for data, in "YYYY-MM-DD" format.
    Returns:
        A JSON string with the metric values as they were reported on that date.
    """
    try:
        payload = {
            "data": {
                "ids": company_tickers,
                "metrics": metrics_list,
                "pointInTimeDate": as_of_date
            }
        }
        data = client.post_point_in_time(payload)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get point-in-time metrics: {e}"})

@tool
def get_segment_data(
    company_tickers: List[str],
    segment_type: SegmentType,
    metrics_list: List[str],
    fiscal_period: str,
    currency: str = "USD"
) -> str:
    """
    Fetches business or geographic segment data for one or more companies.
    Args:
        company_tickers: A list of stock ticker symbols (e.g., ["DIS"]).
        segment_type: Must be one of: "BUS" (Business), "GEO" (Geographic).
        metrics_list: The list of segment metrics (e.g., ["FF_SEG_SALES"]).
        fiscal_period: The fiscal period to query, in "YYYY" format (e.g., "2023").
        currency: The currency for the data. Defaults to "USD".
    Returns:
        A JSON string containing the company's segment data.
    """
    try:
        payload = {
            "data": {
                "ids": company_tickers,
                "segmentType": segment_type,
                "metrics": metrics_list,
                "fiscalPeriod": {"start": fiscal_period, "end": fiscal_period},
                "currency": currency
            }
        }
        data = client.post_segments(payload)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get segment data: {e}"})

@tool
def find_metrics(
    category: Optional[str] = None,
    subcategory: Optional[str] = None
) -> str:
    """
    A helper tool to find and discover available metrics.
    Args:
        category: Filter by a broad category (e.g., "INCOME_STATEMENT", "RATIOS").
        subcategory: Filter by a subcategory (e.g., "PROFITABILITY", "REVENUES").
    Returns:
        A JSON string containing a list of matching metrics.
    """
    try:
        metrics_data = client.get_metrics_catalog(category=category, subcategory=subcategory)
        return json.dumps(metrics_data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search for metrics: {e}"})


if __name__ == "__main__":
    print("--- Testing FactSet Live API Tools (Basic Auth + curl/subprocess) ---")
    
    if not os.environ.get("FACTSET_USERNAME") or not os.environ.get("FACTSET_API_KEY"):
        print("\n!! Skipping tests: FACTSET_USERNAME or FACTSET_API_KEY not set. !!")
    elif not os.environ.get("PROXY_URL"):
        print("\n!! Skipping tests: PROXY_URL not set. !!")
    else:
        print("\n[Test] find_metrics(category='RATIOS', subcategory='PROFITABILITY')")
        print(find_metrics(category="RATIOS", subcategory="PROFITABILITY"))
        
        print("\n[Test] get_company_profile('AAPL')")
        print(get_company_profile("AAPL"))

        print("\n[Test] get_specific_metrics(['F'], ['FF_SALES'], 'ANN', '2022', '2023')")
        print(get_specific_metrics(
            company_tickers=["F"],
            metrics_list=["FF_SALES"],
            periodicity="ANN",
            start_period="2022",
            end_period="2023"
        ))
