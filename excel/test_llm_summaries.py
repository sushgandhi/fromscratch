#!/usr/bin/env python3
"""
Test enhanced LLM-based summary tools
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.supervisor import ExcelAgentSupervisor
from utils.claude_client import get_claude_client

def create_test_data():
    """Create sample data for testing LLM summaries"""
    
    # Rich sample data with text content suitable for LLM analysis
    customer_data = [
        {
            "customer_id": "C001",
            "name": "John Smith",
            "email": "john.smith@email.com",
            "company": "Tech Innovations Inc",
            "industry": "Technology",
            "purchase_history": "Bought premium software license for team collaboration",
            "feedback": "Excellent product, very satisfied with performance and support",
            "region": "North America",
            "revenue": 15000,
            "satisfaction_score": 9.2
        },
        {
            "customer_id": "C002", 
            "name": "Maria Garcia",
            "email": "maria.garcia@company.com",
            "company": "Global Marketing Solutions",
            "industry": "Marketing",
            "purchase_history": "Multiple purchases of analytics tools and training services",
            "feedback": "Great value for money, helped improve our campaign effectiveness significantly",
            "region": "Europe",
            "revenue": 8500,
            "satisfaction_score": 8.7
        },
        {
            "customer_id": "C003",
            "name": "David Chen",
            "email": "d.chen@financetech.com", 
            "company": "FinanceTech Solutions",
            "industry": "Finance",
            "purchase_history": "Enterprise security package with premium support",
            "feedback": "Good product but setup was complex, needed more technical guidance",
            "region": "Asia Pacific",
            "revenue": 25000,
            "satisfaction_score": 7.8
        }
    ]
    
    return customer_data

def test_combined_llm_analysis():
    """Test multiple LLM-enhanced tools together"""
    
    print("🧪 Testing Combined LLM Analysis")
    print("=" * 50)
    
    try:
        # Initialize Claude client and supervisor
        claude_client = get_claude_client()
        supervisor = ExcelAgentSupervisor(claude_client)
        
        # Create test data
        test_data = create_test_data()
        
        # Test comprehensive analysis with all LLM features
        goal = "Analyze customer data: create column summaries with text analysis, generate sheet summary with business insights, and create row summaries"
        
        result = supervisor.run(
            goal=goal,
            data=test_data
        )
        
        print(f"✅ Combined Analysis Test:")
        print(f"📊 Success: {result.get('success', False)}")
        print(f"🔧 Operations completed: {len(result.get('operations', []))}")
        
        # Analyze all operations
        for op in result.get('operations', []):
            print(f"\n📋 {op['tool'].title()}: {op['description']}")
            op_result = op.get('result', {})
            
            if op_result.get('success'):
                print(f"   ✅ Success")
                metadata = op_result.get('metadata', {})
                
                if metadata.get('html_path'):
                    print(f"   📄 Report: {metadata['html_path']}")
                    
            else:
                print(f"   ❌ Failed: {op_result.get('error_message')}")
        
        # Show final results
        if result.get('excel_workbook_path'):
            print(f"📊 Excel workbook: {result['excel_workbook_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Combined analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Enhanced LLM Summary Tools Test")
    print("=" * 40)
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("   Please set your Anthropic API key to run LLM tests")
        sys.exit(1)
    
    # Run test
    success = test_combined_llm_analysis()
    
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    if success:
        print("✅ LLM enhancement test PASSED!")
        print("   ✨ Enhanced summary tools are working correctly")
    else:
        print("❌ LLM enhancement test FAILED!")
        sys.exit(1) 