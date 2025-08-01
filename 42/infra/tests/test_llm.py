"""Tests for the LLM engine module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from llm import LLMEngine


class TestLLMEngine:
    """Test LLM engine functionality."""
    
    @pytest.fixture
    def llm_engine(self):
        """Create an LLM engine instance for testing."""
        return LLMEngine()
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock response for testing."""
        mock = Mock()
        mock.status_code = 200
        mock.json.return_value = {
            "response": "This is a test response from the LLM.",
            "done": True
        }
        return mock
    
    def test_llm_engine_initialization(self, llm_engine):
        """Test that LLM engine can be initialized."""
        assert llm_engine is not None
        assert hasattr(llm_engine, 'base_url')
        assert hasattr(llm_engine, 'model')
        assert hasattr(llm_engine, 'timeout')
    
    def test_respond_basic(self, llm_engine, mock_response):
        """Test basic response functionality."""
        prompt = "What is Python?"
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test response" in response.lower()
    
    def test_respond_with_streaming(self, llm_engine):
        """Test streaming response functionality."""
        prompt = "Explain Python in detail"
        
        # Mock streaming response
        mock_stream_response = Mock()
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = [
            b'{"response": "Python", "done": false}',
            b'{"response": " is", "done": false}',
            b'{"response": " a", "done": false}',
            b'{"response": " programming", "done": false}',
            b'{"response": " language.", "done": true}'
        ]
        
        with patch('requests.post', return_value=mock_stream_response):
            response = llm_engine.respond(prompt, stream=True)
        
        assert isinstance(response, str)
        assert "Python is a programming language" in response
    
    def test_respond_with_custom_model(self, llm_engine, mock_response):
        """Test response with custom model."""
        prompt = "What is machine learning?"
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(prompt, model="llama2")
        
        assert isinstance(response, str)
    
    def test_respond_with_parameters(self, llm_engine, mock_response):
        """Test response with custom parameters."""
        prompt = "Write a short story"
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(
                prompt,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9
            )
        
        assert isinstance(response, str)
    
    def test_respond_connection_error(self, llm_engine):
        """Test handling of connection errors."""
        prompt = "What is AI?"
        
        with patch('requests.post', side_effect=requests.ConnectionError("Connection failed")):
            with pytest.raises(Exception):
                llm_engine.respond(prompt)
    
    def test_respond_timeout_error(self, llm_engine):
        """Test handling of timeout errors."""
        prompt = "What is deep learning?"
        
        with patch('requests.post', side_effect=requests.Timeout("Request timeout")):
            with pytest.raises(Exception):
                llm_engine.respond(prompt)
    
    def test_respond_invalid_response(self, llm_engine):
        """Test handling of invalid response format."""
        prompt = "What is Python?"
        
        # Mock invalid response
        mock_invalid_response = Mock()
        mock_invalid_response.status_code = 200
        mock_invalid_response.json.return_value = {"invalid": "format"}
        
        with patch('requests.post', return_value=mock_invalid_response):
            with pytest.raises(Exception):
                llm_engine.respond(prompt)
    
    def test_respond_http_error(self, llm_engine):
        """Test handling of HTTP errors."""
        prompt = "What is programming?"
        
        # Mock HTTP error
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        
        with patch('requests.post', return_value=mock_error_response):
            with pytest.raises(Exception):
                llm_engine.respond(prompt)
    
    def test_respond_empty_prompt(self, llm_engine):
        """Test handling of empty prompt."""
        with pytest.raises(ValueError):
            llm_engine.respond("")
    
    def test_respond_none_prompt(self, llm_engine):
        """Test handling of None prompt."""
        with pytest.raises(ValueError):
            llm_engine.respond(None)
    
    def test_respond_large_prompt(self, llm_engine, mock_response):
        """Test handling of large prompts."""
        large_prompt = "This is a very large prompt. " * 1000
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(large_prompt)
        
        assert isinstance(response, str)
    
    def test_respond_with_context(self, llm_engine, mock_response):
        """Test response with context in prompt."""
        context = "Python is a programming language."
        question = "What are its features?"
        full_prompt = f"Context: {context}\nQuestion: {question}"
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(full_prompt)
        
        assert isinstance(response, str)
    
    def test_respond_with_system_prompt(self, llm_engine, mock_response):
        """Test response with system prompt."""
        system_prompt = "You are a helpful coding assistant."
        user_prompt = "How do I write a function in Python?"
        
        with patch('requests.post', return_value=mock_response):
            response = llm_engine.respond(
                user_prompt,
                system_prompt=system_prompt
            )
        
        assert isinstance(response, str)
    
    def test_respond_streaming_chunk_processing(self, llm_engine):
        """Test processing of streaming chunks."""
        prompt = "Count to 5"
        
        # Mock streaming with malformed chunks
        mock_stream_response = Mock()
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = [
            b'{"response": "1", "done": false}',
            b'invalid json',
            b'{"response": "2", "done": false}',
            b'{"response": "3", "done": true}'
        ]
        
        with patch('requests.post', return_value=mock_stream_response):
            response = llm_engine.respond(prompt, stream=True)
        
        assert isinstance(response, str)
        assert "1" in response and "2" in response and "3" in response
    
    def test_respond_with_retry(self, llm_engine, mock_response):
        """Test response with retry mechanism."""
        prompt = "What is testing?"
        
        # Mock first request fails, second succeeds
        with patch('requests.post', side_effect=[
            requests.ConnectionError("First attempt failed"),
            mock_response
        ]):
            response = llm_engine.respond(prompt, max_retries=2)
        
        assert isinstance(response, str)
    
    def test_respond_model_not_found(self, llm_engine):
        """Test handling of model not found error."""
        prompt = "What is AI?"
        
        # Mock 404 response for model not found
        mock_404_response = Mock()
        mock_404_response.status_code = 404
        mock_404_response.raise_for_status.side_effect = requests.HTTPError("Model not found")
        
        with patch('requests.post', return_value=mock_404_response):
            with pytest.raises(Exception):
                llm_engine.respond(prompt, model="nonexistent-model")
    
    def test_respond_rate_limit(self, llm_engine):
        """Test handling of rate limiting."""
        prompt = "What is programming?"
        
        # Mock 429 response for rate limiting
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        mock_429_response.raise_for_status.side_effect = requests.HTTPError("Rate limited")
        
        with patch('requests.post', return_value=mock_429_response):
            with pytest.raises(Exception):
                llm_engine.respond(prompt)
    
    def test_respond_with_callback(self, llm_engine):
        """Test response with progress callback."""
        prompt = "Write a poem"
        callback_called = False
        
        def progress_callback(chunk):
            nonlocal callback_called
            callback_called = True
            assert isinstance(chunk, str)
        
        # Mock streaming response
        mock_stream_response = Mock()
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = [
            b'{"response": "Roses", "done": false}',
            b'{"response": " are red", "done": true}'
        ]
        
        with patch('requests.post', return_value=mock_stream_response):
            response = llm_engine.respond(prompt, stream=True, progress_callback=progress_callback)
        
        assert isinstance(response, str)
        assert callback_called 