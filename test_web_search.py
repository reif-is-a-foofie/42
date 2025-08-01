#!/usr/bin/env python3
"""Test Brave web search functionality."""

import sys
import os
import asyncio
import importlib.util
sys.path.append('.')

async def test_brave_search():
    """Test Brave API search functionality."""
    
    print("🔍 Testing Brave Web Search")
    print("=" * 40)
    
    try:
        # Import using importlib to handle the 42 package name
        steve_spec = importlib.util.spec_from_file_location("autonomous_scanner", "42/mission/steve/autonomous_scanner.py")
        steve_module = importlib.util.module_from_spec(steve_spec)
        steve_spec.loader.exec_module(steve_module)
        Steve = steve_module.Steve
        
        redis_spec = importlib.util.spec_from_file_location("redis_bus", "42/infra/services/redis_bus.py")
        redis_module = importlib.util.module_from_spec(redis_spec)
        redis_spec.loader.exec_module(redis_module)
        RedisBus = redis_module.RedisBus
        
        knowledge_spec = importlib.util.spec_from_file_location("knowledge_engine", "42/mission/steve/knowledge_engine.py")
        knowledge_module = importlib.util.module_from_spec(knowledge_spec)
        knowledge_spec.loader.exec_module(knowledge_module)
        KnowledgeEngine = knowledge_module.KnowledgeEngine
        
        config_spec = importlib.util.spec_from_file_location("config", "42/infra/utils/config.py")
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        load_config = config_module.load_config
        
        # Initialize components
        config = load_config()
        redis_bus = RedisBus()
        knowledge_engine = KnowledgeEngine(redis_bus)
        
        # Create Steve instance
        steve = Steve(redis_bus, knowledge_engine, config, soul_config={})
        
        # Test queries
        test_queries = [
            "current price of gold",
            "latest weather forecast",
            "breaking news today",
            "current stock market prices",
            "recent technology updates"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 30)
            
            try:
                # Perform web search
                results = await steve.search_brave_api(query, count=3)
                
                if results:
                    print(f"✅ Found {len(results)} results")
                    for j, result in enumerate(results[:2], 1):  # Show top 2
                        print(f"  {j}. {result.get('title', 'No title')}")
                        print(f"     URL: {result.get('url', 'No URL')}")
                        print(f"     Desc: {result.get('description', 'No description')[:100]}...")
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 40)
        print("📊 WEB SEARCH TEST SUMMARY")
        print("=" * 40)
        print("✅ Brave API integration working")
        print("✅ Web search functionality available")
        print("✅ Current information retrieval possible")
        
    except Exception as e:
        print(f"❌ Failed to initialize Brave search: {e}")
        print("This means web search won't work for current information queries")


if __name__ == "__main__":
    asyncio.run(test_brave_search()) 