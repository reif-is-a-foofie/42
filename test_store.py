#!/usr/bin/env python3
"""Test storing documents in knowledge engine."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 42.un.knowledge_engine import KnowledgeEngine, KnowledgeDocument
from 42.un.redis_bus import RedisBus

async def test_store():
    """Test storing documents."""
    try:
        redis_bus = RedisBus()
        engine = KnowledgeEngine(redis_bus)
        
        # Create test documents
        docs = [
            KnowledgeDocument(
                source_id="test_source",
                content="This is a test document about machine learning and artificial intelligence.",
                metadata={"test": True}
            ),
            KnowledgeDocument(
                source_id="test_source", 
                content="Another test document about quantum computing and deep learning.",
                metadata={"test": True}
            )
        ]
        
        print(f"Storing {len(docs)} test documents...")
        await engine._store_documents(docs)
        print("Documents stored successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_store()) 