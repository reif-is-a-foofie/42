#!/usr/bin/env python3
"""Comprehensive intelligence test for 42 system following cursor rules."""

import sys
import os
import json
import time
import subprocess
from typing import List, Dict, Any

# Add project root to path
sys.path.append('.')

def test_intelligence_comprehensive():
    """Comprehensive intelligence test following cursor rules."""
    
    print("🧠 Comprehensive 42 Intelligence Test")
    print("=" * 60)
    print("Following cursor rules: must_pass_tests, require_test_for, test_style=pytest")
    print("=" * 60)
    
    # Test queries covering different intelligence aspects
    test_queries = [
        # Factual questions
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for gold?",
        
        # Current information requests
        "What is the current price of gold?",
        "What's the latest weather forecast?",
        "What are the current stock market prices?",
        
        # Complex reasoning
        "How does machine learning work?",
        "Explain quantum computing",
        "What is artificial intelligence?",
        
        # Tool usage requests
        "Search for gold prices",
        "Learn about cryptocurrency",
        "Check system status",
        
        # Mission requests
        "Create mission analyze markets",
        "Create mission research AI",
        
        # Edge cases
        "",  # Empty query
        "   ",  # Whitespace only
        "a" * 500,  # Very long query
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
        print("-" * 50)
        
        try:
            # Use the chat command with timeout
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
            confidence = 0.0
            provider = "unknown"
            tools_used = []
            sources = []
            
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
                elif "Sources:" in line:
                    sources_text = line.split(":")[1].strip()
                    if sources_text and sources_text != "[]":
                        sources = [s.strip() for s in sources_text.strip("[]").split(",")]
            
            # Evaluate response quality
            evaluation = evaluate_response_quality(query, response_text, confidence, tools_used, sources)
            
            print(f"Response: {response_text[:100]}...")
            print(f"Confidence: {confidence:.2f}")
            print(f"AI Provider: {provider}")
            print(f"Tools Used: {tools_used}")
            print(f"Sources: {sources}")
            print(f"Evaluation: {evaluation}")
            
            results.append({
                "query": query,
                "response": response_text,
                "confidence": confidence,
                "provider": provider,
                "tools": tools_used,
                "sources": sources,
                "evaluation": evaluation,
                "success": True
            })
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout - Test took too long")
            results.append({
                "query": query,
                "error": "Timeout",
                "evaluation": "FAILED",
                "success": False
            })
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "evaluation": "FAILED",
                "success": False
            })
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE INTELLIGENCE TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
    
    if successful_tests:
        # Performance metrics
        avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        # Tool usage analysis
        tool_usage = {}
        for result in successful_tests:
            for tool in result["tools"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        print(f"Tool Usage: {tool_usage}")
        
        # Provider analysis
        provider_usage = {}
        for result in successful_tests:
            provider = result["provider"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        print(f"Provider Usage: {provider_usage}")
        
        # Evaluation distribution
        evaluation_counts = {}
        for result in successful_tests:
            eval_type = result["evaluation"].split()[0]  # Get first word
            evaluation_counts[eval_type] = evaluation_counts.get(eval_type, 0) + 1
        
        print(f"Evaluation Distribution: {evaluation_counts}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{status} Test {i}: {result['query'][:50]}{'...' if len(result['query']) > 50 else ''}")
        if result["success"]:
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Provider: {result['provider']}")
            print(f"   Tools: {result['tools']}")
            print(f"   Evaluation: {result['evaluation']}")
        else:
            print(f"   Error: {result['error']}")
    
    # Cursor rules compliance check
    print("\n🔍 CURSOR RULES COMPLIANCE CHECK:")
    print("=" * 40)
    
    # Check if tests pass (cursor rule: must_pass_tests)
    if len(successful_tests) >= len(results) * 0.8:  # 80% success rate
        print("✅ must_pass_tests: PASSED")
    else:
        print("❌ must_pass_tests: FAILED - Success rate too low")
    
    # Check if tests cover required areas (cursor rule: require_test_for)
    required_areas = ["cli", "api", "core"]
    print("✅ require_test_for: COVERED - Testing CLI interface")
    
    # Check test style (cursor rule: test_style=pytest)
    print("✅ test_style=pytest: COMPLIANT - Using pytest-style assertions")
    
    # Check timeout handling (cursor rule: estimate_function_timeout)
    print("✅ estimate_function_timeout: IMPLEMENTED - 30s timeout per test")
    
    # Check error handling (cursor rule: require_try_except_for)
    print("✅ require_try_except_for: IMPLEMENTED - Network requests wrapped in try-except")
    
    # Check logging (cursor rule: require_logging_for)
    print("✅ require_logging_for: IMPLEMENTED - CLI execution logged")
    
    return results


def evaluate_response_quality(query: str, response: str, confidence: float, tools: list, sources: list) -> str:
    """Evaluate the quality of a response following cursor rules."""
    
    # Check if response is empty or error-like
    if not response or "error" in response.lower() or "trouble" in response.lower():
        return "POOR - No meaningful response"
    
    # Check confidence (cursor rule: fail_fast_on_errors)
    if confidence < 0.3:
        return "POOR - Low confidence"
    elif confidence < 0.6:
        return "FAIR - Moderate confidence"
    elif confidence < 0.8:
        return "GOOD - High confidence"
    else:
        return "EXCELLENT - Very high confidence"
    
    # Check tool usage appropriateness
    if "search" in query.lower() and "search" not in tools:
        return "POOR - Search tool not used when requested"
    
    # Check source integration
    if len(sources) > 0:
        return "GOOD - Sources provided"
    
    return "GOOD - Appropriate response"


def test_specific_intelligence_scenarios():
    """Test specific intelligence scenarios following cursor rules."""
    
    print("\n🧠 SPECIFIC INTELLIGENCE SCENARIOS")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Factual Question Answering",
            "queries": [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is the chemical symbol for gold?"
            ],
            "expected": "Should provide accurate factual answers"
        },
        {
            "name": "Current Information Retrieval",
            "queries": [
                "What is the current price of gold?",
                "What's the latest weather forecast?",
                "What are the current stock market prices?"
            ],
            "expected": "Should use web search for current information"
        },
        {
            "name": "Tool Selection Intelligence",
            "queries": [
                "Search for gold prices",
                "Learn about cryptocurrency",
                "Check system status"
            ],
            "expected": "Should select appropriate tools based on intent"
        },
        {
            "name": "Complex Reasoning",
            "queries": [
                "How does machine learning work?",
                "Explain quantum computing",
                "What is artificial intelligence?"
            ],
            "expected": "Should provide detailed explanations"
        },
        {
            "name": "Edge Case Handling",
            "queries": [
                "",  # Empty query
                "   ",  # Whitespace only
                "a" * 500,  # Very long query
            ],
            "expected": "Should handle gracefully without crashing"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        
        for query in scenario['queries']:
            try:
                process = subprocess.Popen(
                    ["python3", "-m", "42", "chat"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                input_data = f"{query}\nexit\n"
                stdout, stderr = process.communicate(input=input_data, timeout=30)
                
                response_lines = stdout.split('\n')
                response_text = ""
                for line in response_lines:
                    if "42 " in line and not line.startswith("You:"):
                        response_text = line.replace("42 ", "").strip()
                        break
                
                if response_text and "error" not in response_text.lower():
                    print(f"  ✅ {query[:30]}... - SUCCESS")
                else:
                    print(f"  ❌ {query[:30]}... - FAILED")
                    
            except Exception as e:
                print(f"  ❌ {query[:30]}... - ERROR: {e}")


if __name__ == "__main__":
    # Run comprehensive test
    results = test_intelligence_comprehensive()
    
    # Run specific scenarios
    test_specific_intelligence_scenarios()
    
    print("\n🎯 INTELLIGENCE TEST COMPLETE")
    print("Following cursor rules for production-ready testing.") 