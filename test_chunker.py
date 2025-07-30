#!/usr/bin/env python3
"""Test script to demonstrate 42.un chunking functionality."""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import sys
sys.path.append('.')
from 42.chunker import Chunker
from 42.embedding import EmbeddingEngine

def test_chunking():
    """Test the chunking functionality."""
    print("🧪 Testing 42.un Chunking Functionality")
    print("=" * 50)
    
    # Initialize components
    chunker = Chunker()
    embedding_engine = EmbeddingEngine()
    
    # Test with our own code
    test_file = "42/cli.py"
    
    print(f"📁 Chunking file: {test_file}")
    chunks = chunker.chunk_file(test_file)
    
    print(f"✅ Found {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n📄 Chunk {i+1}:")
        print(f"   File: {chunk.file_path}")
        print(f"   Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"   Type: {chunk.metadata.get('type', 'unknown')}")
        print(f"   Name: {chunk.metadata.get('name', 'N/A')}")
        print(f"   Text preview: {chunk.text[:100]}...")
        
        # Test embedding
        try:
            vector = embedding_engine.embed_text(chunk.text)
            print(f"   Vector dimension: {len(vector)}")
        except Exception as e:
            print(f"   Embedding error: {e}")
    
    # Test directory chunking
    print(f"\n📁 Chunking directory: 42/")
    dir_chunks = chunker.chunk_directory("42/")
    print(f"✅ Found {len(dir_chunks)} total chunks in directory")
    
    # Show file breakdown
    file_counts = {}
    for chunk in dir_chunks:
        file_path = chunk.file_path
        file_counts[file_path] = file_counts.get(file_path, 0) + 1
    
    print("\n📊 File breakdown:")
    for file_path, count in sorted(file_counts.items()):
        print(f"   {file_path}: {count} chunks")

if __name__ == "__main__":
    test_chunking() 