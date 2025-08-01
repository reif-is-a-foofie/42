"""Test suite for 42 Chat API intelligence."""

import pytest
import asyncio
import json
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from loguru import logger

# Use absolute imports
import sys
import os
sys.path.append('.')

# Import using importlib to handle the 42 package name
import importlib.util

# Import Moroni
moroni_spec = importlib.util.spec_from_file_location("moroni", "42/moroni/moroni.py")
moroni_module = importlib.util.module_from_spec(moroni_spec)
moroni_spec.loader.exec_module(moroni_module)
Moroni = moroni_module.Moroni
ConversationContext = moroni_module.ConversationContext
KnowledgeResponse = moroni_module.KnowledgeResponse

# Import VectorStore
vector_store_spec = importlib.util.spec_from_file_location("vector_store", "42/infra/core/vector_store.py")
vector_store_module = importlib.util.module_from_spec(vector_store_spec)
vector_store_spec.loader.exec_module(vector_store_module)
VectorStore = vector_store_module.VectorStore

# Import EmbeddingEngine
embedding_spec = importlib.util.spec_from_file_location("embedding", "42/infra/core/embedding.py")
embedding_module = importlib.util.module_from_spec(embedding_spec)
embedding_spec.loader.exec_module(embedding_module)
EmbeddingEngine = embedding_module.EmbeddingEngine


class TestChatIntelligence:
    """Test suite for chat intelligence capabilities."""
    
    @pytest.fixture
    def moroni(self):
        """Initialize Moroni for testing."""
        return Moroni()
    
    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_history=[],
            current_mission=None,
            knowledge_base=[],
            tools_available=["search", "learn", "query", "status", "mission", "sources"]
        )
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        mock_store = Mock(spec=VectorStore)
        mock_store.search_semantic.return_value = [
            {
                "score": 0.85,
                "payload": {
                    "text": "Gold is currently trading at $3,300 per ounce.",
                    "source": "financial_data",
                    "metadata": {"date": "2024-01-01"}
                }
            },
            {
                "score": 0.75,
                "payload": {
                    "text": "The price of gold has increased significantly this year.",
                    "source": "market_analysis",
                    "metadata": {"date": "2024-01-01"}
                }
            }
        ]
        return mock_store
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Mock embedding engine for testing."""
        mock_engine = Mock(spec=EmbeddingEngine)
        mock_engine.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 76  # 384 dim vector
        return mock_engine
    
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
    
    def test_context_retrieval(self, moroni, mock_vector_store, mock_embedding_engine):
        """Test context retrieval functionality."""
        with patch.object(moroni, 'vector_store', mock_vector_store), \
             patch.object(moroni, 'embedding_engine', mock_embedding_engine):
            
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
            "[financial_data] Gold is currently trading at $3,300 per ounce.",
            "[market_analysis] The price of gold has increased significantly this year."
        ]
        
        prompt = moroni._build_rag_prompt(user_query, context_chunks)
        
        assert "SYSTEM:" in prompt
        assert "CONTEXT:" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "USER: What is the price of gold?" in prompt
        assert "Gold is currently trading at $3,300 per ounce" in prompt
    
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
        assert search_result.success is True
        
        # Test status tool
        status_result = moroni._tool_status("check status", conversation_context)
        assert status_result.tool_name == "status"
        assert status_result.success is True
        
        # Test mission tool
        mission_result = moroni._tool_mission("create mission analyze markets", conversation_context)
        assert mission_result.tool_name == "mission"
        assert mission_result.success is True
        assert conversation_context.current_mission == "analyze markets"
    
    def test_rag_context_building(self, moroni, mock_vector_store, mock_embedding_engine):
        """Test RAG context building functionality."""
        with patch.object(moroni, 'vector_store', mock_vector_store), \
             patch.object(moroni, 'embedding_engine', mock_embedding_engine):
            
            rag_context = moroni._build_rag_context("What is the price of gold?")
            
            assert rag_context.normalized_query == "price gold"
            assert len(rag_context.retrieved_chunks) > 0
            assert len(rag_context.summarized_context) > 0
            assert rag_context.total_tokens > 0
            assert 0 <= rag_context.retrieval_score <= 1
    
    def test_response_generation(self, moroni, conversation_context, mock_vector_store, mock_embedding_engine):
        """Test response generation with RAG context."""
        with patch.object(moroni, 'vector_store', mock_vector_store), \
             patch.object(moroni, 'embedding_engine', mock_embedding_engine), \
             patch.object(moroni, '_call_ai') as mock_call:
            
            mock_call.return_value = "The current price of gold is $3,300 per ounce."
            
            rag_context = moroni._build_rag_context("What is the price of gold?")
            response = moroni._generate_rag_response(
                "What is the price of gold?",
                {"intent": "information_request", "confidence": 0.8},
                rag_context,
                [],
                conversation_context
            )
            
            assert "text" in response
            assert "confidence" in response
            assert "reasoning" in response
    
    def test_full_conversation_flow(self, moroni, conversation_context, mock_vector_store, mock_embedding_engine):
        """Test full conversation flow."""
        with patch.object(moroni, 'vector_store', mock_vector_store), \
             patch.object(moroni, 'embedding_engine', mock_embedding_engine), \
             patch.object(moroni, '_call_ai') as mock_call:
            
            mock_call.side_effect = [
                # Intent analysis response
                json.dumps({
                    "intent": "information_request",
                    "needs_tools": True,
                    "tools_needed": ["search"],
                    "knowledge_query": "gold price",
                    "confidence": 0.8
                }),
                # RAG response
                "The current price of gold is $3,300 per ounce."
            ]
            
            response = moroni.process_request("What is the price of gold?", conversation_context)
            
            assert isinstance(response, KnowledgeResponse)
            assert response.response == "The current price of gold is $3,300 per ounce."
            assert response.ai_provider == moroni.current_provider
            assert len(response.sources) > 0
            assert "search" in response.tools_used
    
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
            
            assert moroni.total_cost > initial_cost
            assert moroni.total_tokens_used > initial_tokens
    
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


class TestIntelligenceScenarios:
    """Test specific intelligence scenarios."""
    
    @pytest.fixture
    def moroni(self):
        return Moroni()
    
    @pytest.fixture
    def conversation_context(self):
        return ConversationContext(
            user_id="test_user",
            conversation_history=[],
            current_mission=None,
            knowledge_base=[],
            tools_available=["search", "learn", "query", "status", "mission", "sources"]
        )
    
    def test_factual_question_answering(self, moroni, conversation_context):
        """Test ability to answer factual questions."""
        questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical symbol for gold?",
        ]
        
        for question in questions:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = "Correct answer"
                
                response = moroni.process_request(question, conversation_context)
                assert response.response == "Correct answer"
    
    def test_context_awareness(self, moroni, conversation_context):
        """Test context awareness in conversation."""
        # First question
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Paris is the capital of France."
            
            response1 = moroni.process_request("What is the capital of France?", conversation_context)
            assert "Paris" in response1.response
        
        # Follow-up question
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Yes, Paris is a beautiful city."
            
            response2 = moroni.process_request("Is it beautiful?", conversation_context)
            assert "Paris" in response2.response
    
    def test_tool_selection_intelligence(self, moroni, conversation_context):
        """Test intelligent tool selection."""
        test_cases = [
            ("Search for gold prices", ["search"]),
            ("Learn about cryptocurrency", ["learn"]),
            ("Check system status", ["status"]),
            ("Create mission analyze markets", ["mission"]),
        ]
        
        for query, expected_tools in test_cases:
            with patch.object(moroni, '_call_ai') as mock_call:
                mock_call.return_value = json.dumps({
                    "intent": "tool_request",
                    "needs_tools": True,
                    "tools_needed": expected_tools,
                    "knowledge_query": query,
                    "confidence": 0.8
                })
                
                response = moroni.process_request(query, conversation_context)
                assert all(tool in response.tools_used for tool in expected_tools)
    
    def test_knowledge_integration(self, moroni, conversation_context):
        """Test integration of knowledge base with responses."""
        with patch.object(moroni, '_call_ai') as mock_call:
            mock_call.return_value = "Based on the available information, gold is trading at $3,300."
            
            response = moroni.process_request("What is the price of gold?", conversation_context)
            
            assert "gold" in response.response.lower()
            assert "$3,300" in response.response
            assert len(response.sources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 