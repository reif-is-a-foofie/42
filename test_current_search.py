#!/usr/bin/env python3
"""Test current information search functionality."""

import subprocess
import time

def test_current_search():
    """Test current information search with the new directory structure."""
    
    print("🔍 Testing Current Information Search")
    print("=" * 50)
    
    # Test queries that should trigger web search
    current_queries = [
        "What is the current price of gold?",
        "What's the latest weather forecast?",
        "What are the current stock market prices?",
        "What's the latest news today?",
        "What's the current Bitcoin price?"
    ]
    
    results = []
    
    for i, query in enumerate(current_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Use the chat command with a timeout
            process = subprocess.Popen(
                ["python3", "-m", "42", "chat"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send query and exit
            input_data = f"{query}\nexit\n"
            stdout, stderr = process.communicate(input=input_data, timeout=30)
            
            # Parse response
            response_lines = stdout.split('\n')
            response_text = ""
            tools_used = []
            
            for line in response_lines:
                if "42 " in line and not line.startswith("You:"):
                    response_text = line.replace("42 ", "").strip()
                elif "Tools used:" in line:
                    tools = line.split(":")[1].strip()
                    if tools and tools != "[]":
                        tools_used = [t.strip() for t in tools.strip("[]").split(",")]
            
            # Check if web search was used
            used_web_search = "web_search" in tools_used or "search" in tools_used
            
            print(f"Response: {response_text[:100]}...")
            print(f"Tools Used: {tools_used}")
            print(f"Web Search Used: {'✅' if used_web_search else '❌'}")
            
            # Evaluate if response contains current information
            current_indicators = ["current", "latest", "today", "now", "price", "forecast", "news"]
            has_current_info = any(indicator in response_text.lower() for indicator in current_indicators)
            
            print(f"Contains Current Info: {'✅' if has_current_info else '❌'}")
            
            results.append({
                "query": query,
                "response": response_text,
                "tools_used": tools_used,
                "used_web_search": used_web_search,
                "has_current_info": has_current_info
            })
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            results.append({
                "query": query,
                "error": "Timeout",
                "used_web_search": False,
                "has_current_info": False
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "used_web_search": False,
                "has_current_info": False
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 CURRENT SEARCH TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
    
    if successful_tests:
        web_search_used = sum(1 for r in successful_tests if r["used_web_search"])
        current_info_found = sum(1 for r in successful_tests if r["has_current_info"])
        
        print(f"Web Search Used: {web_search_used}/{len(successful_tests)} ({web_search_used/len(successful_tests)*100:.1f}%)")
        print(f"Current Info Found: {current_info_found}/{len(successful_tests)} ({current_info_found/len(successful_tests)*100:.1f}%)")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅" if "error" not in result else "❌"
        print(f"{status} Test {i}: {result['query']}")
        if "error" not in result:
            print(f"   Web Search: {'✅' if result['used_web_search'] else '❌'}")
            print(f"   Current Info: {'✅' if result['has_current_info'] else '❌'}")
            print(f"   Tools: {result['tools_used']}")
        else:
            print(f"   Error: {result['error']}")


if __name__ == "__main__":
    test_current_search() 