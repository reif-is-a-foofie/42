#!/usr/bin/env python3
"""Simple test runner for chat intelligence evaluation."""

import sys
import os
sys.path.append('.')

# Use importlib to handle the 42 package name
import importlib.util
spec = importlib.util.spec_from_file_location("moroni", "42/moroni/moroni.py")
moroni_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moroni_module)
Moroni = moroni_module.Moroni
ConversationContext = moroni_module.ConversationContext


def test_chat_intelligence():
    """Test chat intelligence with real queries."""
    
    print("🧠 Testing 42 Chat Intelligence")
    print("=" * 50)
    
    # Initialize Moroni
    try:
        moroni = Moroni()
        print("✅ Moroni initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Moroni: {e}")
        return
    
    # Create conversation context
    context = ConversationContext(
        user_id="test_user",
        conversation_history=[],
        current_mission=None,
        knowledge_base=[],
        tools_available=list(moroni.tools.keys())
    )
    
    # Test queries
    test_queries = [
        "What is the current price of gold?",
        "Tell me about artificial intelligence",
        "What is the capital of France?",
        "How does machine learning work?",
        "What is the weather like today?",
        "Explain quantum computing",
        "What are the benefits of renewable energy?",
        "Tell me about blockchain technology",
        "What is the meaning of life?",
        "How do neural networks function?",
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Process the query
            response = moroni.process_request(query, context)
            
            # Evaluate response
            evaluation = evaluate_response(query, response)
            
            print(f"Response: {response.response[:100]}...")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"AI Provider: {response.ai_provider}")
            print(f"Tools Used: {response.tools_used}")
            print(f"Sources Found: {len(response.sources)}")
            print(f"Evaluation: {evaluation}")
            
            results.append({
                "query": query,
                "response": response.response,
                "confidence": response.confidence,
                "provider": response.ai_provider,
                "tools": response.tools_used,
                "sources": len(response.sources),
                "evaluation": evaluation
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
        
        avg_sources = sum(r["sources"] for r in successful_tests) / len(successful_tests)
        print(f"Average Sources: {avg_sources:.1f}")
        
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
            print(f"   Sources: {result['sources']}")
            print(f"   Evaluation: {result['evaluation']}")
        else:
            print(f"   Error: {result['error']}")


def evaluate_response(query: str, response) -> str:
    """Evaluate the quality of a response."""
    
    # Check if response is empty or error-like
    if not response.response or "error" in response.response.lower() or "trouble" in response.response.lower():
        return "POOR - No meaningful response"
    
    # Check confidence
    if response.confidence < 0.3:
        return "POOR - Low confidence"
    elif response.confidence < 0.6:
        return "FAIR - Moderate confidence"
    elif response.confidence < 0.8:
        return "GOOD - High confidence"
    else:
        return "EXCELLENT - Very high confidence"
    
    # Check if tools were used appropriately
    if "search" in query.lower() and "search" not in response.tools_used:
        return "POOR - Search tool not used when requested"
    
    # Check if sources were found
    if len(response.sources) == 0:
        return "FAIR - No sources found"
    
    return "GOOD - Appropriate response with sources"


if __name__ == "__main__":
    test_chat_intelligence() 