"""Tests for the prompt builder module."""

import pytest
from unittest.mock import Mock, patch
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from prompt import PromptBuilder


class TestPromptBuilder:
    """Test prompt builder functionality."""
    
    @pytest.fixture
    def prompt_builder(self):
        """Create a prompt builder instance for testing."""
        return PromptBuilder()
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        mock_store = Mock()
        
        # Mock search results
        mock_results = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "text": "This is a Python function that processes data.",
                    "file_path": "src/processor.py",
                    "start_line": 10,
                    "end_line": 25
                }
            },
            {
                "id": 2,
                "score": 0.87,
                "payload": {
                    "text": "Here's how to handle errors in Python.",
                    "file_path": "src/errors.py",
                    "start_line": 5,
                    "end_line": 15
                }
            },
            {
                "id": 3,
                "score": 0.82,
                "payload": {
                    "text": "Configuration settings for the application.",
                    "file_path": "config/settings.py",
                    "start_line": 1,
                    "end_line": 20
                }
            }
        ]
        
        mock_store.search.return_value = mock_results
        return mock_store
    
    def test_prompt_builder_initialization(self, prompt_builder):
        """Test that prompt builder can be initialized."""
        assert prompt_builder is not None
        assert hasattr(prompt_builder, 'max_tokens')
        assert hasattr(prompt_builder, 'template')
    
    def test_build_prompt_basic(self, prompt_builder, mock_vector_store):
        """Test basic prompt building functionality."""
        question = "How do I process data in Python?"
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert question in prompt
        assert "Python function that processes data" in prompt
    
    def test_build_prompt_with_limit(self, prompt_builder, mock_vector_store):
        """Test prompt building with custom limit."""
        question = "How do I handle errors?"
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question, top_k=2)
        
        assert isinstance(prompt, str)
        # Should only include top 2 results
        assert "Python function that processes data" in prompt
        assert "handle errors in Python" in prompt
        assert "Configuration settings" not in prompt
    
    def test_build_prompt_with_filters(self, prompt_builder, mock_vector_store):
        """Test prompt building with query filters."""
        question = "Show me Python code examples"
        
        # Mock filtered results
        filtered_results = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "text": "def process_data(data): return data.upper()",
                    "file_path": "src/processor.py",
                    "start_line": 10,
                    "end_line": 25,
                    "metadata": {"type": "code", "language": "python"}
                }
            }
        ]
        mock_vector_store.search.return_value = filtered_results
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(
                question, 
                query_filter={"must": [{"key": "metadata.type", "match": {"value": "code"}}]}
            )
        
        assert isinstance(prompt, str)
        assert "def process_data" in prompt
    
    def test_prompt_token_limit(self, prompt_builder, mock_vector_store):
        """Test that prompt respects token limits."""
        question = "What is Python?"
        
        # Create a very long mock result
        long_text = "This is a very long text. " * 1000
        mock_vector_store.search.return_value = [{
            "id": 1,
            "score": 0.95,
            "payload": {
                "text": long_text,
                "file_path": "long_file.txt"
            }
        }]
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question, max_tokens=1000)
        
        assert isinstance(prompt, str)
        assert len(prompt) <= 1000  # Approximate token limit
    
    def test_prompt_template_customization(self, prompt_builder, mock_vector_store):
        """Test custom prompt template."""
        question = "How do I use Python?"
        
        custom_template = """
        Question: {question}
        
        Relevant context:
        {context}
        
        Answer based on the context above:
        """
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(
                question, 
                template=custom_template
            )
        
        assert isinstance(prompt, str)
        assert "Question: How do I use Python?" in prompt
        assert "Relevant context:" in prompt
        assert "Answer based on the context above:" in prompt
    
    def test_empty_search_results(self, prompt_builder, mock_vector_store):
        """Test handling of empty search results."""
        question = "What is quantum computing?"
        
        # Mock empty results
        mock_vector_store.search.return_value = []
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question)
        
        assert isinstance(prompt, str)
        assert question in prompt
        assert "No relevant context found" in prompt or "I don't have enough information" in prompt
    
    def test_prompt_context_formatting(self, prompt_builder, mock_vector_store):
        """Test that context is properly formatted in prompt."""
        question = "Show me code examples"
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question)
        
        assert isinstance(prompt, str)
        # Check that file paths are included
        assert "src/processor.py" in prompt or "src/errors.py" in prompt
        # Check that line numbers are included
        assert "10:25" in prompt or "5:15" in prompt
    
    def test_prompt_score_threshold(self, prompt_builder, mock_vector_store):
        """Test filtering by score threshold."""
        question = "What is Python?"
        
        # Mock results with varying scores
        mixed_results = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {"text": "High relevance result"}
            },
            {
                "id": 2,
                "score": 0.3,
                "payload": {"text": "Low relevance result"}
            }
        ]
        mock_vector_store.search.return_value = mixed_results
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question, score_threshold=0.5)
        
        assert isinstance(prompt, str)
        assert "High relevance result" in prompt
        assert "Low relevance result" not in prompt
    
    def test_prompt_metadata_inclusion(self, prompt_builder, mock_vector_store):
        """Test that metadata is included in context."""
        question = "Show me configuration examples"
        
        # Mock results with metadata
        metadata_results = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "text": "DEBUG = True\nLOG_LEVEL = 'INFO'",
                    "file_path": "config/settings.py",
                    "metadata": {
                        "type": "config",
                        "language": "python",
                        "tags": ["debug", "logging"]
                    }
                }
            }
        ]
        mock_vector_store.search.return_value = metadata_results
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question, include_metadata=True)
        
        assert isinstance(prompt, str)
        assert "config/settings.py" in prompt
        assert "DEBUG = True" in prompt
    
    def test_prompt_error_handling(self, prompt_builder, mock_vector_store):
        """Test error handling in prompt building."""
        question = "What is Python?"
        
        # Mock search error
        mock_vector_store.search.side_effect = Exception("Search failed")
        
        with patch.object(prompt_builder, 'vector_store', mock_vector_store):
            prompt = prompt_builder.build_prompt(question)
        
        assert isinstance(prompt, str)
        assert question in prompt
        assert "Error" in prompt or "Unable to retrieve context" in prompt 