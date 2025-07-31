#!/usr/bin/env python3
"""Manual test script to trigger knowledge engine ingestion."""

import asyncio
import json
import os
from 42.un.knowledge_engine import KnowledgeEngine, KnowledgeSource, SourceType, RSSFetcher
from 42.un.redis_bus import RedisBus

async def main():
    """Manually trigger knowledge engine to fetch and store documents."""
    print("🚀 Starting manual knowledge engine ingestion...")
    
    # Initialize knowledge engine
    redis_bus = RedisBus()
    engine = KnowledgeEngine(redis_bus)
    
    # Register fetchers
    engine.register_fetcher(SourceType.RSS, RSSFetcher)
    
    # Load sources from file
    if os.path.exists("universal_sources.json"):
        with open("universal_sources.json", "r") as f:
            sources_data = json.load(f)
            for item in sources_data:
                source = KnowledgeSource.from_dict(item)
                engine.add_source(source)
                print(f"  ✓ Loaded: {source.name}")
    
    print(f"\n📊 Processing {len(engine.sources)} sources...")
    
    # Run a fetch cycle
    await engine.run_fetch_cycle()
    
    print("✅ Knowledge engine ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main()) 