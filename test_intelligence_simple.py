#!/usr/bin/env python3
"""Simple intelligence test for 42 Chat."""

import sys
import os
sys.path.append('.')

def test_chat_intelligence():
    """Test chat intelligence with real queries."""
    
    print("🧠 Testing 42 Chat Intelligence")
    print("=" * 50)
    
    # Test the chat command directly
    import subprocess
    import time
    
    # Test queries
    test_queries = [
        "What is the current price of gold?",
        "Tell me about artificial intelligence",
        "What is the capital of France?",
        "How does machine learning work?",
        "Explain quantum computing",
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Use the chat command
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
            
            # Analyze response
            response_lines = stdout.split('\n')
            response_text = ""
            confidence = 0.0
            provider = "unknown"
            tools_used = []
            
            for line in response_lines:
                if "42 " in line and not line.startswith("You:"):
                    response_text = line.replace("42 ", "").strip()
                elif "Confidence:" in line:
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "AI Provider:" in line:
                    provider = line.split(":")[1].strip()
                elif "Tools used:" in line:
                    tools = line.split(":")[1].strip()
                    if tools and tools != "[]":
                        tools_used = [t.strip() for t in tools.strip("[]").split(",")]
            
            # Evaluate response
            evaluation = evaluate_response(query, response_text, confidence, tools_used)
            
            print(f"Response: {response_text[:100]}...")
            print(f"Confidence: {confidence:.2f}")
            print(f"AI Provider: {provider}")
            print(f"Tools Used: {tools_used}")
            print(f"Evaluation: {evaluation}")
            
            results.append({
                "query": query,
                "response": response_text,
                "confidence": confidence,
                "provider": provider,
                "tools": tools_used,
                "evaluation": evaluation
            })
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout - Test took too long")
            results.append({
                "query": query,
                "error": "Timeout",
                "evaluation": "FAILED"
            })
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "evaluation": "FAILED"
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INTELLIGENCE TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
    
    if successful_tests:
        avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        tool_usage = {}
        for result in successful_tests:
            for tool in result["tools"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        print(f"Tool Usage: {tool_usage}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅" if "error" not in result else "❌"
        print(f"{status} Test {i}: {result['query']}")
        if "error" not in result:
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Provider: {result['provider']}")
            print(f"   Tools: {result['tools']}")
            print(f"   Evaluation: {result['evaluation']}")
        else:
            print(f"   Error: {result['error']}")


def evaluate_response(query: str, response: str, confidence: float, tools: list) -> str:
    """Evaluate the quality of a response."""
    
    # Check if response is empty or error-like
    if not response or "error" in response.lower() or "trouble" in response.lower():
        return "POOR - No meaningful response"
    
    # Check confidence
    if confidence < 0.3:
        return "POOR - Low confidence"
    elif confidence < 0.6:
        return "FAIR - Moderate confidence"
    elif confidence < 0.8:
        return "GOOD - High confidence"
    else:
        return "EXCELLENT - Very high confidence"
    
    # Check if tools were used appropriately
    if "search" in query.lower() and "search" not in tools:
        return "POOR - Search tool not used when requested"
    
    return "GOOD - Appropriate response"


if __name__ == "__main__":
    test_chat_intelligence() 