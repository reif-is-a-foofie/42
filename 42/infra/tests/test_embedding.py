"""Tests for the embedding engine."""

import pytest
import sys
import os

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from embedding import EmbeddingEngine


def test_embedding_engine_initialization():
    """Test that the embedding engine can be initialized."""
    engine = EmbeddingEngine()
    assert engine.model is not None
    assert engine.model_name == "bge-small-en"


def test_embed_text():
    """Test that text can be embedded."""
    engine = EmbeddingEngine()
    text = "Hello, world!"
    vector = engine.embed_text(text)
    
    assert isinstance(vector, list)
    assert len(vector) == engine.get_dimension()
    assert all(isinstance(x, float) for x in vector)


def test_embed_text_batch():
    """Test that text batches can be embedded."""
    engine = EmbeddingEngine()
    texts = ["Hello", "world", "test"]
    vectors = engine.embed_text_batch(texts)
    
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(len(v) == engine.get_dimension() for v in vectors)


def test_get_dimension():
    """Test that the embedding dimension is correct."""
    engine = EmbeddingEngine()
    dimension = engine.get_dimension()
    assert isinstance(dimension, int)
    assert dimension > 0 