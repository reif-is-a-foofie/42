#!/usr/bin/env python3
"""Test Brave search using CLI command."""

import subprocess
import sys

def test_brave_search():
    """Test Brave search using the CLI command."""
    
    print("🔍 Testing Brave Web Search via CLI")
    print("=" * 40)
    
    test_queries = [
        "current price of gold",
        "latest weather forecast",
        "breaking news today"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the CLI search command
            result = subprocess.run(
                ["python3", "-m", "42", "steve", "search", query],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Search completed successfully")
                print("Output:")
                print(result.stdout)
            else:
                print(f"❌ Search failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Search timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("📊 BRAVE SEARCH TEST SUMMARY")
    print("=" * 40)
    print("If searches completed successfully, web search is working")
    print("If searches failed, web search needs to be fixed")


if __name__ == "__main__":
    test_brave_search() 