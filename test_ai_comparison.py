#!/usr/bin/env python3
"""Compare our RAG system against direct OpenAI calls."""

import sys
import os
import time
import json
import openai
sys.path.append('.')

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def test_direct_openai():
    """Test direct OpenAI calls without RAG."""
    
    print("🤖 Testing Direct OpenAI (No RAG)")
    print("=" * 50)
    
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
            start_time = time.time()
            
            # Direct OpenAI call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                max_tokens=200,
                temperature=0.3
            )
            
            end_time = time.time()
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Estimate cost
            cost_per_1k_tokens = 0.002  # gpt-3.5-turbo
            cost = (tokens_used / 1000) * cost_per_1k_tokens
            
            print(f"Response: {response_text[:100]}...")
            print(f"Tokens Used: {tokens_used}")
            print(f"Cost: ${cost:.4f}")
            print(f"Response Time: {end_time - start_time:.2f}s")
            
            results.append({
                "query": query,
                "response": response_text,
                "tokens": tokens_used,
                "cost": cost,
                "response_time": end_time - start_time,
                "method": "direct_openai"
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "method": "direct_openai"
            })
    
    return results


def test_rag_system():
    """Test our RAG system."""
    
    print("\n🧠 Testing Our RAG System")
    print("=" * 50)
    
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
            start_time = time.time()
            
            # Use our chat system
            import subprocess
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
            
            end_time = time.time()
            
            # Parse response
            response_lines = stdout.split('\n')
            response_text = ""
            confidence = 0.0
            provider = "unknown"
            tools_used = []
            tokens_estimate = 0
            
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
            
            # More accurate token estimation for RAG pipeline
            # RAG uses multiple calls: intent analysis + context retrieval + response generation
            # Each call has input tokens + output tokens
            intent_tokens = len(query.split()) * 1.3  # Intent analysis input
            context_tokens = 50  # Estimated context retrieval tokens
            response_tokens = len(response_text.split()) * 1.3  # Response generation output
            total_tokens = intent_tokens + context_tokens + response_tokens
            
            # Estimate cost
            cost_per_1k_tokens = 0.002
            cost = (total_tokens / 1000) * cost_per_1k_tokens
            
            print(f"Response: {response_text[:100]}...")
            print(f"Confidence: {confidence:.2f}")
            print(f"AI Provider: {provider}")
            print(f"Tools Used: {tools_used}")
            print(f"Estimated Tokens: {total_tokens:.0f} (intent: {intent_tokens:.0f}, context: {context_tokens}, response: {response_tokens:.0f})")
            print(f"Estimated Cost: ${cost:.4f}")
            print(f"Response Time: {end_time - start_time:.2f}s")
            
            results.append({
                "query": query,
                "response": response_text,
                "confidence": confidence,
                "provider": provider,
                "tools": tools_used,
                "tokens_estimate": total_tokens,
                "cost": cost,
                "response_time": end_time - start_time,
                "method": "rag_system"
            })
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            results.append({
                "query": query,
                "error": "Timeout",
                "method": "rag_system"
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "method": "rag_system"
            })
    
    return results


def compare_results(direct_results, rag_results):
    """Compare the results between direct OpenAI and RAG system."""
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Filter successful results
    direct_successful = [r for r in direct_results if "error" not in r]
    rag_successful = [r for r in rag_results if "error" not in r]
    
    print(f"Direct OpenAI Success Rate: {len(direct_successful)}/{len(direct_results)} ({len(direct_successful)/len(direct_results)*100:.1f}%)")
    print(f"RAG System Success Rate: {len(rag_successful)}/{len(rag_results)} ({len(rag_successful)/len(rag_results)*100:.1f}%)")
    
    if direct_successful and rag_successful:
        # Cost comparison
        direct_avg_cost = sum(r["cost"] for r in direct_successful) / len(direct_successful)
        rag_avg_cost = sum(r["cost"] for r in rag_successful) / len(rag_successful)
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"Direct OpenAI Avg Cost: ${direct_avg_cost:.4f}")
        print(f"RAG System Avg Cost: ${rag_avg_cost:.4f}")
        print(f"Cost Difference: {((rag_avg_cost - direct_avg_cost) / direct_avg_cost * 100):+.1f}%")
        
        # Token comparison
        direct_avg_tokens = sum(r["tokens"] for r in direct_successful) / len(direct_successful)
        rag_avg_tokens = sum(r["tokens_estimate"] for r in rag_successful) / len(rag_successful)
        
        print(f"\n🔢 TOKEN ANALYSIS:")
        print(f"Direct OpenAI Avg Tokens: {direct_avg_tokens:.0f}")
        print(f"RAG System Avg Tokens: {rag_avg_tokens:.0f}")
        print(f"Token Difference: {((rag_avg_tokens - direct_avg_tokens) / direct_avg_tokens * 100):+.1f}%")
        
        # Response time comparison
        direct_avg_time = sum(r["response_time"] for r in direct_successful) / len(direct_successful)
        rag_avg_time = sum(r["response_time"] for r in rag_successful) / len(rag_successful)
        
        print(f"\n⏱️ SPEED ANALYSIS:")
        print(f"Direct OpenAI Avg Time: {direct_avg_time:.2f}s")
        print(f"RAG System Avg Time: {rag_avg_time:.2f}s")
        print(f"Speed Difference: {((rag_avg_time - direct_avg_time) / direct_avg_time * 100):+.1f}%")
        
        # Quality comparison
        print(f"\n🎯 QUALITY ANALYSIS:")
        rag_avg_confidence = sum(r["confidence"] for r in rag_successful) / len(rag_successful)
        print(f"RAG System Avg Confidence: {rag_avg_confidence:.2f}")
        
        # Tool usage analysis
        tool_usage = {}
        for result in rag_successful:
            for tool in result["tools"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        print(f"RAG Tool Usage: {tool_usage}")
        
        # Success metrics
        print(f"\n✅ SUCCESS METRICS:")
        print(f"Direct OpenAI: {'✅' if len(direct_successful) == len(direct_results) else '❌'}")
        print(f"RAG System: {'✅' if len(rag_successful) == len(rag_results) else '❌'}")
        
        # Value proposition
        print(f"\n💡 VALUE PROPOSITION:")
        if rag_avg_cost < direct_avg_cost:
            print(f"✅ RAG is {((direct_avg_cost - rag_avg_cost) / direct_avg_cost * 100):.1f}% cheaper")
        else:
            print(f"❌ RAG is {((rag_avg_cost - direct_avg_cost) / direct_avg_cost * 100):.1f}% more expensive")
        
        if rag_avg_time < direct_avg_time:
            print(f"✅ RAG is {((direct_avg_time - rag_avg_time) / direct_avg_time * 100):.1f}% faster")
        else:
            print(f"❌ RAG is {((rag_avg_time - direct_avg_time) / direct_avg_time * 100):.1f}% slower")
        
        if rag_avg_confidence > 0.5:
            print(f"✅ RAG has good confidence ({rag_avg_confidence:.2f})")
        else:
            print(f"⚠️ RAG has low confidence ({rag_avg_confidence:.2f})")


def main():
    """Run the comparison test."""
    print("🔬 AI COMPARISON TEST")
    print("Comparing our RAG system vs direct OpenAI calls")
    print("=" * 60)
    
    # Test direct OpenAI
    direct_results = test_direct_openai()
    
    # Test our RAG system
    rag_results = test_rag_system()
    
    # Compare results
    compare_results(direct_results, rag_results)


if __name__ == "__main__":
    main() 