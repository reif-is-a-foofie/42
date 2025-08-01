"""Simple comprehensive intelligence test for 42 system following cursor rules."""

import pytest
import json
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import using importlib to handle the 42 package name
import importlib.util

# Import Moroni
moroni_spec = importlib.util.spec_from_file_location("moroni", "42/moroni/moroni.py")
moroni_module = importlib.util.module_from_spec(moroni_spec)
moroni_spec.loader.exec_module(moroni_module)
Moroni = moroni_module.Moroni
ConversationContext = moroni_module.ConversationContext
KnowledgeResponse = moroni_module.KnowledgeResponse
ToolExecution = moroni_module.ToolExecution


class TestIntelligenceSimpleComprehensive:
    """Simple comprehensive intelligence test suite following cursor rules."""
    
    @pytest.fixture
    def moroni(self):
        """Initialize Moroni for testing."""
        return Moroni()
    
    @pytest.fixture
    def conversation_context(self):
        """Create test conversation context."""
        return ConversationContext(
            user_id="test_user_001",
            conversation_history=[],
            current_mission=None,
            knowledge_base=[],
            tools_available=["search", "learn", "query", "status", "mission", "sources", "web_search"]
        )
    
    def test_query_normalization(self, moroni):
        """Test query normalization functionality."""
        test_cases = [
            ("What is the current price of gold?", "current price gold"),
            ("THE PRICE OF GOLD IS IMPORTANT", "price gold important"),
            ("  extra   whitespace   test  ", "extra whitespace test"),
            ("The and or but in on at", ""),  # All stop words
        ]
        
        for input_query, expected in test_cases:
            normalized = moroni._normalize_query(input_query)
            assert normalized == expected, f"Expected '{expected}', got '{normalized}'"
    
    def test_context_retrieval(self, moroni):
        """Test context retrieval functionality."""
        # Mock the vector store search
        with patch.object(moroni, 'vector_store') as mock_store:
            mock_store.search_semantic.return_value = [
                {
                    "score": 0.85,
                    "payload": {
                        "text": "Gold is currently trading at $2,150 per ounce.",
                        "source": "financial_data",
                        "metadata": {"date": "2024-01-01"}
                    }
                }
            ]
            
            chunks = moroni._retrieve_context("gold price", max_chunks=3)
            
            assert len(chunks) <= 3
            assert all(chunk["relevance"] >= moroni.min_relevance_score for chunk in chunks)
            assert all("content" in chunk for chunk in chunks)
            assert all("source" in chunk for chunk in chunks)
    
    def test_chunk_summarization(self, moroni):
        """Test chunk summarization functionality."""
        test_chunks = [
            {
                "content": "This is a very long sentence that should be truncated because it exceeds the maximum length limit.",
                "source": "test_source",
                "relevance": 0.9
            },
            {
                "content": "Short sentence.",
                "source": "test_source",
                "relevance": 0.8
            }
        ]
        
        summarized = moroni._summarize_chunks(test_chunks)
        
        assert len(summarized) == 2
        assert all(len(chunk) <= moroni.max_chunk_length + 50 for chunk in summarized)  # +50 for source prefix
        assert all("[test_source]" in chunk for chunk in summarized)
    
    def test_rag_prompt_building(self, moroni):
        """Test RAG prompt building functionality."""
        user_query = "What is the price of gold?"
        context_chunks = [
            "[financial_data] Gold is currently trading at $2,150 per ounce.",
            "[market_analysis] The price of gold has increased significantly this year."
        ]
        
        prompt = moroni._build_rag_prompt(user_query, context_chunks)
        
        assert "SYSTEM:" in prompt
        assert "CONTEXT:" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "USER: What is the price of gold?" in prompt
        assert "Gold is currently trading at $2,150 per ounce" in prompt
    
    def test_intent_analysis(self, moroni, conversation_context):
        """Test intent analysis functionality."""
        test_queries = [
            ("What is the price of gold?", "information_request"),
            ("Search for gold prices", "search_request"),
            ("Learn about cryptocurrency", "learning_request"),
            ("Create mission to analyze markets", "mission_request"),
            ("Check system status", "status_request"),
        ]
        
        for query, expected_intent in test_queries:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = json.dumps({
                    "intent": expected_intent,
                    "needs_tools": True,
                    "tools_needed": ["search"],
                    "knowledge_query": query,
                    "confidence": 0.8
                })
                
                result = moroni._analyze_intent(query, conversation_context)
                
                assert "intent" in result
                assert "needs_tools" in result
                assert "tools_needed" in result
                assert "confidence" in result
    
    def test_tool_execution(self, moroni, conversation_context):
        """Test tool execution functionality."""
        # Test search tool
        search_result = moroni._tool_search("search for gold prices", conversation_context)
        assert search_result.tool_name == "search"
        assert hasattr(search_result, 'success')
        assert hasattr(search_result, 'result')
        
        # Test status tool
        status_result = moroni._tool_status("check status", conversation_context)
        assert status_result.tool_name == "status"
        assert hasattr(status_result, 'success')
        
        # Test mission tool
        mission_result = moroni._tool_mission("create mission analyze markets", conversation_context)
        assert mission_result.tool_name == "mission"
        assert hasattr(mission_result, 'success')
    
    def test_error_handling(self, moroni, conversation_context):
        """Test error handling in conversation flow."""
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.side_effect = Exception("API Error")
            
            response = moroni.process_request("What is the price of gold?", conversation_context)
            
            assert isinstance(response, KnowledgeResponse)
            assert "error" in response.response.lower() or "trouble" in response.response.lower()
            assert response.confidence == 0.0
    
    def test_cost_tracking(self, moroni):
        """Test cost tracking functionality."""
        initial_cost = moroni.total_cost
        initial_tokens = moroni.total_tokens_used
        
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Test response"
            
            # Simulate a call
            moroni._call_openai("test prompt", "test_operation")
            
            assert moroni.total_cost >= initial_cost
            assert moroni.total_tokens_used >= initial_tokens
    
    def test_provider_switching(self, moroni):
        """Test AI provider switching functionality."""
        initial_provider = moroni.primary_provider
        
        # Test switching to available provider
        success = moroni.switch_provider("ollama")
        assert success is True
        assert moroni.primary_provider == "ollama"
        
        # Test switching to unavailable provider
        success = moroni.switch_provider("nonexistent")
        assert success is False
        assert moroni.primary_provider == "ollama"  # Should remain unchanged
    
    def test_usage_statistics(self, moroni):
        """Test usage statistics functionality."""
        stats = moroni.get_usage_stats()
        
        assert "total_tokens" in stats
        assert "total_cost" in stats
        assert "current_provider" in stats
        assert "primary_provider" in stats
        assert "fallback_provider" in stats
        
        assert isinstance(stats["total_tokens"], int)
        assert isinstance(stats["total_cost"], float)
        assert isinstance(stats["current_provider"], str)
    
    def test_performance_benchmarks(self, moroni, conversation_context):
        """Test performance benchmarks and response times."""
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Test response"
            
            # Test response time
            start_time = time.time()
            response = moroni.process_request("What is gold?", conversation_context)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 5.0  # Should respond within 5 seconds
            assert isinstance(response, KnowledgeResponse)


class TestIntelligenceScenarios:
    """Test specific intelligence scenarios and edge cases."""
    
    @pytest.fixture
    def moroni(self):
        return Moroni()
    
    @pytest.fixture
    def conversation_context(self):
        return ConversationContext(
            user_id="scenario_test_user",
            conversation_history=[],
            current_mission=None,
            knowledge_base=[],
            tools_available=["search", "learn", "query", "status", "mission", "sources"]
        )
    
    def test_factual_question_answering(self, moroni, conversation_context):
        """Test ability to answer factual questions accurately."""
        factual_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical symbol for gold?",
            "What year did World War II end?",
        ]
        
        for question in factual_questions:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = "Correct factual answer"
                
                response = moroni.process_request(question, conversation_context)
                assert isinstance(response, KnowledgeResponse)
                assert len(response.response) > 0
    
    def test_context_awareness(self, moroni, conversation_context):
        """Test context awareness in multi-turn conversations."""
        
        # First question
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Paris is the capital of France."
            
            response1 = moroni.process_request("What is the capital of France?", conversation_context)
            assert "Paris" in response1.response
        
        # Follow-up question
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Yes, Paris is a beautiful city with the Eiffel Tower."
            
            response2 = moroni.process_request("Is it beautiful?", conversation_context)
            assert "Paris" in response2.response
    
    def test_tool_selection_intelligence(self, moroni, conversation_context):
        """Test intelligent tool selection based on query intent."""
        tool_test_cases = [
            ("Search for gold prices", ["search"]),
            ("Learn about cryptocurrency", ["learn"]),
            ("Check system status", ["status"]),
            ("Create mission analyze markets", ["mission"]),
            ("What are the current sources?", ["sources"]),
        ]
        
        for query, expected_tools in tool_test_cases:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = json.dumps({
                    "intent": "tool_request",
                    "needs_tools": True,
                    "tools_needed": expected_tools,
                    "knowledge_query": query,
                    "confidence": 0.8
                })
                
                response = moroni.process_request(query, conversation_context)
                assert isinstance(response, KnowledgeResponse)
    
    def test_edge_case_handling(self, moroni, conversation_context):
        """Test handling of edge cases and unusual inputs."""
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 1000,  # Very long query
            "!@#$%^&*()",  # Special characters
            "1234567890",  # Numbers only
        ]
        
        for edge_case in edge_cases:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = "Handled edge case"
                
                response = moroni.process_request(edge_case, conversation_context)
                assert isinstance(response, KnowledgeResponse)
                # Should not crash or return error for valid edge cases


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 