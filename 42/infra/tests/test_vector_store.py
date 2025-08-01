"""Tests for the vector store module."""

import pytest
import tempfile
import subprocess
import time
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from vector_store import VectorStore


class TestVectorStore:
    """Test vector store functionality."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store instance for testing."""
        # Start temporary Qdrant instance
        store = VectorStore()
        yield store
        # Cleanup handled by VectorStore
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4, 1.5]
        ]
    
    @pytest.fixture
    def sample_payloads(self):
        """Create sample payloads for testing."""
        return [
            {"text": "Hello world", "file_path": "test1.txt", "metadata": {"type": "text"}},
            {"text": "Python code", "file_path": "test2.py", "metadata": {"type": "code"}},
            {"text": "Documentation", "file_path": "test3.md", "metadata": {"type": "doc"}}
        ]
    
    def test_vector_store_initialization(self, vector_store):
        """Test that vector store can be initialized."""
        assert vector_store.client is not None
        assert vector_store.collection_name == "42_chunks"
    
    def test_create_collection(self, vector_store):
        """Test collection creation."""
        dimension = 384  # bge-small-en dimension
        result = vector_store.create_collection(dimension)
        assert result is True
    
    def test_upsert_vectors(self, vector_store, sample_vectors, sample_payloads):
        """Test upserting vectors."""
        # Create collection first
        vector_store.create_collection(len(sample_vectors[0]))
        
        # Create points
        from qdrant_client.models import PointStruct
        points = []
        for i, (vector, payload) in enumerate(zip(sample_vectors, sample_payloads)):
            point = PointStruct(
                id=i,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upsert
        result = vector_store.upsert(points)
        assert result is True
    
    def test_search_vectors(self, vector_store, sample_vectors, sample_payloads):
        """Test vector search functionality."""
        # Setup: create collection and upsert vectors
        vector_store.create_collection(len(sample_vectors[0]))
        
        from qdrant_client.models import PointStruct
        points = []
        for i, (vector, payload) in enumerate(zip(sample_vectors, sample_payloads)):
            point = PointStruct(id=i, vector=vector, payload=payload)
            points.append(point)
        
        vector_store.upsert(points)
        
        # Search
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = vector_store.search(query_vector, limit=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(r, dict) for r in results)
        assert all("id" in r and "score" in r and "payload" in r for r in results)
    
    def test_update_payload(self, vector_store, sample_vectors, sample_payloads):
        """Test updating vector payloads."""
        # Setup
        vector_store.create_collection(len(sample_vectors[0]))
        
        from qdrant_client.models import PointStruct
        points = []
        for i, (vector, payload) in enumerate(zip(sample_vectors, sample_payloads)):
            point = PointStruct(id=i, vector=vector, payload=payload)
            points.append(point)
        
        vector_store.upsert(points)
        
        # Update payload
        new_payload = {"text": "Updated text", "file_path": "updated.txt"}
        result = vector_store.update_payload(0, new_payload)
        assert result is True
    
    def test_get_all_vectors(self, vector_store, sample_vectors, sample_payloads):
        """Test retrieving all vectors."""
        # Setup
        vector_store.create_collection(len(sample_vectors[0]))
        
        from qdrant_client.models import PointStruct
        points = []
        for i, (vector, payload) in enumerate(zip(sample_vectors, sample_payloads)):
            point = PointStruct(id=i, vector=vector, payload=payload)
            points.append(point)
        
        vector_store.upsert(points)
        
        # Get all vectors
        all_vectors = vector_store.get_all_vectors()
        
        assert isinstance(all_vectors, list)
        assert len(all_vectors) == len(sample_vectors)
        assert all(isinstance(v, dict) for v in all_vectors)
        assert all("id" in v and "vector" in v and "payload" in v for v in all_vectors)
    
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Test with invalid host
        with pytest.raises(Exception):
            VectorStore(host="invalid-host")
    
    def test_collection_exists(self, vector_store):
        """Test checking if collection exists."""
        dimension = 384
        vector_store.create_collection(dimension)
        
        # Should return True for existing collection
        result = vector_store.create_collection(dimension)
        assert result is True  # Should not fail if collection exists
    
    def test_search_with_filters(self, vector_store, sample_vectors, sample_payloads):
        """Test search with payload filters."""
        # Setup
        vector_store.create_collection(len(sample_vectors[0]))
        
        from qdrant_client.models import PointStruct
        points = []
        for i, (vector, payload) in enumerate(zip(sample_vectors, sample_payloads)):
            point = PointStruct(id=i, vector=vector, payload=payload)
            points.append(point)
        
        vector_store.upsert(points)
        
        # Search with filter
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        filter_condition = {"must": [{"key": "metadata.type", "match": {"value": "text"}}]}
        
        results = vector_store.search(query_vector, limit=10, query_filter=filter_condition)
        
        assert isinstance(results, list)
        # Should only return text type results
        for result in results:
            assert result["payload"]["metadata"]["type"] == "text" 