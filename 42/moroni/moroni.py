"""Moroni - The AI-Agnostic NLP Brain for 42

Moroni provides intelligent conversation handling, knowledge checking, and tool execution.
Acts as the brain layer between user requests and system actions.
Supports multiple AI providers with automatic fallback.
Implements RAG (Retrieval-Augmented Generation) for cost-effective, context-aware responses.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
from loguru import logger

from ..infra.utils.config import load_config
from ..infra.core.vector_store import VectorStore
from ..infra.core.embedding import EmbeddingEngine
from ..infra.core.llm import LLMEngine


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    user_id: str
    conversation_history: List[Dict[str, str]]
    current_mission: Optional[str] = None
    knowledge_base: List[Dict[str, Any]] = None
    tools_available: List[str] = None


@dataclass
class ToolExecution:
    """Result of tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class KnowledgeResponse:
    """Response with knowledge integration."""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    tools_used: List[str]
    reasoning: str
    ai_provider: str


@dataclass
class RAGContext:
    """RAG pipeline context for efficient retrieval and generation."""
    normalized_query: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    summarized_context: List[str]
    total_tokens: int
    retrieval_score: float


class Moroni:
    """The AI-Agnostic NLP Brain - Provides intelligent conversation and tool execution."""
    
    def __init__(self):
        """Initialize Moroni with AI-agnostic configuration."""
        config = load_config()
        
        # AI Provider configuration
        self.primary_provider = config.ai_provider
        self.primary_model = config.ai_model
        self.fallback_provider = config.ai_fallback_provider
        self.fallback_model = config.ai_fallback_model
        
        # Initialize AI clients
        self._init_ai_clients(config)
        
        # Initialize components
        self.vector_store = VectorStore()
        self.embedding_engine = EmbeddingEngine()
        self.llm_engine = LLMEngine()
        
        # RAG Pipeline settings
        self.max_context_chunks = 5
        self.max_chunk_length = 200
        self.min_relevance_score = 0.7
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.current_provider = self.primary_provider
        
        # Available tools
        self.tools = {
            "search": self._tool_search,
            "web_search": self._tool_web_search,
            "learn": self._tool_learn,
            "query": self._tool_query,
            "status": self._tool_status,
            "mission": self._tool_mission,
            "sources": self._tool_sources
        }
        
        logger.info(f"🧠 Moroni initialized - AI-Agnostic Brain ready (Primary: {self.primary_provider}, Fallback: {self.fallback_provider})")
    
    def _init_ai_clients(self, config):
        """Initialize AI provider clients."""
        self.ai_clients = {}
        
        # OpenAI client
        if config.openai_api_key:
            self.ai_clients['openai'] = {
                'client': openai.OpenAI(api_key=config.openai_api_key),
                'model': config.openai_model,
                'available': True
            }
        else:
            self.ai_clients['openai'] = {'available': False}
        
        # Ollama client (always available as local fallback)
        self.ai_clients['ollama'] = {
            'client': None,  # Uses LLMEngine
            'model': config.ai_model,
            'available': True
        }
        
        # Anthropic client (if implemented)
        if config.anthropic_api_key:
            self.ai_clients['anthropic'] = {
                'client': None,  # Would need anthropic client
                'model': config.anthropic_model,
                'available': False  # Not implemented yet
            }
        else:
            self.ai_clients['anthropic'] = {'available': False}
    
    def _normalize_query(self, text: str) -> str:
        """Normalize and clean user query for better retrieval."""
        # Basic normalization without external dependencies
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def _retrieve_context(self, query: str, max_chunks: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context using hybrid approach."""
        try:
            # Normalize query
            normalized_query = self._normalize_query(query)
            
            # Get query embedding
            query_embedding = self.embedding_engine.embed_text(normalized_query)
            
            # Search vector store
            search_results = self.vector_store.search_semantic(query_embedding, limit=max_chunks * 2)
            
            # Filter by relevance score
            relevant_chunks = []
            for result in search_results:
                if result.get("score", 0) >= self.min_relevance_score:
                    relevant_chunks.append({
                        "content": result.get("payload", {}).get("text", ""),
                        "source": result.get("payload", {}).get("source", ""),
                        "relevance": result.get("score", 0),
                        "metadata": result.get("payload", {})
                    })
            
            # Sort by relevance and take top chunks
            relevant_chunks.sort(key=lambda x: x["relevance"], reverse=True)
            return relevant_chunks[:max_chunks]
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def _summarize_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Summarize chunks to fit token budget."""
        summarized = []
        
        for chunk in chunks:
            content = chunk["content"]
            
            # Simple summarization: take first sentence or truncate
            if len(content) > self.max_chunk_length:
                # Try to find sentence boundary
                sentences = re.split(r'[.!?]', content)
                if sentences and len(sentences[0]) <= self.max_chunk_length:
                    content = sentences[0] + "."
                else:
                    content = content[:self.max_chunk_length] + "..."
            
            summarized.append(f"[{chunk['source']}] {content}")
        
        return summarized
    
    def _build_rag_prompt(self, user_query: str, context_chunks: List[str]) -> str:
        """Build RAG prompt with context."""
        if not context_chunks:
            return f"USER: {user_query}"
        
        prompt = f"""SYSTEM:
You are a knowledgeable assistant. Answer using the context below when available.
If the answer is not in the context, say "I don't have specific information about that."

CONTEXT:
"""
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt += f"[{i}] {chunk}\n"
        
        prompt += f"\nUSER: {user_query}"
        
        return prompt
    
    def _call_ai(self, prompt: str, operation: str) -> str:
        """Make AI call with automatic provider selection and fallback."""
        providers_to_try = [self.primary_provider, self.fallback_provider]
        
        for provider in providers_to_try:
            try:
                if provider == 'openai' and self.ai_clients['openai']['available']:
                    return self._call_openai(prompt, operation)
                elif provider == 'ollama':
                    return self._call_ollama(prompt, operation)
                elif provider == 'anthropic' and self.ai_clients['anthropic']['available']:
                    return self._call_anthropic(prompt, operation)
                else:
                    logger.warning(f"Provider {provider} not available, trying next")
                    continue
                    
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}, trying fallback")
                continue
        
        # All providers failed, return simple fallback response
        logger.error("All AI providers failed, using simple fallback")
        return self._simple_fallback_response(prompt, operation)
    
    def _simple_fallback_response(self, prompt: str, operation: str) -> str:
        """Provide simple fallback responses when AI is unavailable."""
        if "intent_analysis" in operation:
            return json.dumps({
                "intent": "general conversation",
                "needs_tools": False,
                "tools_needed": [],
                "knowledge_query": "general information",
                "confidence": 0.5
            })
        elif "response_generation" in operation:
            return json.dumps({
                "text": "I'm here to help! I can search for information, answer questions, and assist with various tasks. What would you like to know?",
                "confidence": 0.6,
                "reasoning": "Simple fallback response when AI is unavailable"
            })
        else:
            return "I'm available to help. What can I assist you with?"
    
    def _call_openai(self, prompt: str, operation: str) -> str:
        """Make OpenAI API call."""
        client = self.ai_clients['openai']['client']
        model = self.ai_clients['openai']['model']
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Track usage
        tokens_used = response.usage.total_tokens
        self.total_tokens_used += tokens_used
        
        # Estimate cost (rough calculation)
        cost_per_1k_tokens = 0.002  # gpt-3.5-turbo
        self.total_cost += (tokens_used / 1000) * cost_per_1k_tokens
        
        logger.info(f"OpenAI call ({operation}): {tokens_used} tokens, ${self.total_cost:.4f}")
        self.current_provider = 'openai'
        
        return response.choices[0].message.content
    
    def _call_ollama(self, prompt: str, operation: str) -> str:
        """Make Ollama API call."""
        # Use the query method instead of respond
        result = self.llm_engine.query(prompt, model=self.primary_model)
        logger.info(f"Ollama call ({operation}): using local LLM")
        self.current_provider = 'ollama'
        return result.get("response", "No response generated")
    
    def _call_anthropic(self, prompt: str, operation: str) -> str:
        """Make Anthropic API call (placeholder for future implementation)."""
        # TODO: Implement Anthropic client
        raise NotImplementedError("Anthropic provider not yet implemented")
    
    def process_request(self, user_input: str, context: ConversationContext) -> KnowledgeResponse:
        """Process user request with RAG pipeline for cost-effective responses."""
        try:
            # Step 1: Analyze user intent
            intent_analysis = self._analyze_intent(user_input, context)
            
            # Step 2: Retrieve relevant context using RAG
            rag_context = self._build_rag_context(user_input)
            
            # Step 3: Execute necessary tools
            tool_results = []
            if intent_analysis.get("needs_tools"):
                for tool_name in intent_analysis["tools_needed"]:
                    if tool_name in self.tools:
                        tool_result = self.tools[tool_name](user_input, context)
                        tool_results.append(tool_result)
            
            # Step 4: Generate comprehensive response with RAG context
            response = self._generate_rag_response(
                user_input, 
                intent_analysis, 
                rag_context,
                tool_results, 
                context
            )
            
            return KnowledgeResponse(
                response=response["text"],
                confidence=response["confidence"],
                sources=rag_context.retrieved_chunks,
                tools_used=[r.tool_name for r in tool_results if r.success],
                reasoning=response["reasoning"],
                ai_provider=self.current_provider
            )
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return KnowledgeResponse(
                response="I encountered an error processing your request. Please try again.",
                confidence=0.0,
                sources=[],
                tools_used=[],
                reasoning=f"Error: {str(e)}",
                ai_provider=self.current_provider
            )
    
    def _build_rag_context(self, user_input: str) -> RAGContext:
        """Build RAG context for the user input."""
        # Normalize query
        normalized_query = self._normalize_query(user_input)
        
        # Retrieve relevant chunks
        retrieved_chunks = self._retrieve_context(normalized_query, self.max_context_chunks)
        
        # Summarize chunks
        summarized_context = self._summarize_chunks(retrieved_chunks)
        
        # Calculate total tokens (rough estimate)
        total_tokens = len(normalized_query.split()) + sum(len(chunk.split()) for chunk in summarized_context)
        
        # Calculate average relevance score
        avg_relevance = sum(chunk["relevance"] for chunk in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        return RAGContext(
            normalized_query=normalized_query,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=retrieved_chunks,  # No reranking for now
            summarized_context=summarized_context,
            total_tokens=total_tokens,
            retrieval_score=avg_relevance
        )
    
    def _generate_rag_response(
        self, 
        user_input: str, 
        intent_analysis: Dict[str, Any],
        rag_context: RAGContext,
        tool_results: List[ToolExecution],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate response using RAG context."""
        try:
            # Build RAG prompt
            rag_prompt = self._build_rag_prompt(user_input, rag_context.summarized_context)
            
            # Add tool context if available
            tool_context = ""
            if tool_results:
                tool_context = "\nTool Results:\n"
                for result in tool_results:
                    if result.success:
                        tool_context += f"- {result.tool_name}: {str(result.result)[:100]}...\n"
                    else:
                        tool_context += f"- {result.tool_name}: Error - {result.error}\n"
                
                rag_prompt += f"\n{tool_context}"
            
            # Generate response
            response = self._call_ai(rag_prompt, "rag_response_generation")
            
            # Parse response (handle both JSON and plain text)
            try:
                response_data = json.loads(response)
                return response_data
            except json.JSONDecodeError:
                # If not JSON, treat as plain text response
                # Calculate confidence based on RAG context quality
                confidence = self._calculate_confidence(rag_context, tool_results)
                
                return {
                    "text": response,
                    "confidence": confidence,
                    "reasoning": f"RAG response with {len(rag_context.retrieved_chunks)} context chunks, relevance: {rag_context.retrieval_score:.2f}"
                }
            
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            return {
                "text": "I'm having trouble generating a response. Please try again.",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def _calculate_confidence(self, rag_context: RAGContext, tool_results: List[ToolExecution]) -> float:
        """Calculate confidence score based on RAG context and tool results."""
        confidence = 0.0
        
        # Base confidence from retrieval quality
        if rag_context.retrieval_score > 0.8:
            confidence += 0.4
        elif rag_context.retrieval_score > 0.6:
            confidence += 0.3
        elif rag_context.retrieval_score > 0.4:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Boost confidence if we have relevant chunks
        if len(rag_context.retrieved_chunks) > 0:
            confidence += 0.2
        
        # Boost confidence if tools were successful
        successful_tools = [r for r in tool_results if r.success]
        if successful_tools:
            confidence += 0.1 * len(successful_tools)
        
        # Cap confidence at 0.9 (never 100% certain)
        return min(confidence, 0.9)
    
    def _analyze_intent(self, user_input: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze user intent and determine required tools."""
        try:
            prompt = f"""
            Analyze this user request and determine:
            1. What the user wants to accomplish
            2. What tools are needed
            3. What knowledge should be checked
            
            User input: {user_input}
            Available tools: {list(self.tools.keys())}
            
            Respond in JSON format:
            {{
                "intent": "description of what user wants",
                "needs_tools": true/false,
                "tools_needed": ["tool1", "tool2"],
                "knowledge_query": "what to search in knowledge base",
                "confidence": 0.0-1.0
            }}
            """
            
            response = self._call_ai(prompt, "intent_analysis")
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to analyze intent: {e}")
            return {
                "intent": "unknown",
                "needs_tools": False,
                "tools_needed": [],
                "knowledge_query": user_input,
                "confidence": 0.5
            }
    
    # Tool implementations
    def _tool_search(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Search for information using vector store and web search."""
        try:
            # Extract search query from user input
            search_query = user_input.replace("search", "").replace("find", "").strip()
            
            # Check if this is a current information query that needs web search
            current_keywords = ["current", "latest", "today", "now", "price", "weather", "news", "recent", "update"]
            needs_web_search = any(keyword in search_query.lower() for keyword in current_keywords)
            
            if needs_web_search:
                # Use web search for current information
                return self._tool_web_search(search_query, context)
            else:
                # Use vector store search for knowledge base queries
                query_embedding = self.embedding_engine.embed_text(search_query)
                results = self.vector_store.search_semantic(query_embedding, limit=10)
                
                return ToolExecution(
                    tool_name="search",
                    success=True,
                    result=f"Found {len(results)} relevant results from knowledge base",
                    execution_time=0.1
                )
            
        except Exception as e:
            return ToolExecution(
                tool_name="search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_web_search(self, query: str, context: ConversationContext) -> ToolExecution:
        """Search the web using Brave API for current information."""
        try:
            import requests
            import os
            
            # Get API key from environment or use default
            api_key = os.getenv("BRAVE_API_KEY", "BSAyr39Gxgxm9R1YI_vvJ0CbOmqbEQ7")
            
            if not api_key:
                return ToolExecution(
                    tool_name="web_search",
                    success=False,
                    result=None,
                    error="No Brave API key configured"
                )
            
            # Make direct API call
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": api_key
            }
            
            params = {
                "q": query,
                "count": 5,
                "safesearch": "moderate"
            }
            
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("web", {}).get("results", [])
                
                if results:
                    # Format results for response
                    formatted_results = []
                    for result in results[:3]:  # Top 3 results
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", "")
                        })
                    
                    return ToolExecution(
                        tool_name="web_search",
                        success=True,
                        result=f"Found {len(results)} current results from web search",
                        execution_time=2.0
                    )
                else:
                    return ToolExecution(
                        tool_name="web_search",
                        success=False,
                        result="No current information found",
                        error="No web search results"
                    )
            else:
                return ToolExecution(
                    tool_name="web_search",
                    success=False,
                    result=None,
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolExecution(
                tool_name="web_search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_learn(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Learn new information."""
        try:
            # Extract learning target from user input
            if "learn about" in user_input:
                topic = user_input.split("learn about")[-1].strip()
            else:
                topic = user_input.replace("learn", "").strip()
            
            return ToolExecution(
                tool_name="learn",
                success=True,
                result=f"Learning about: {topic}",
                execution_time=0.1
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="learn",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_query(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Query the knowledge base."""
        try:
            # Use LLM to answer question
            result = self.llm_engine.query(user_input)
            
            return ToolExecution(
                tool_name="query",
                success=True,
                result=result.get("response", "No response"),
                execution_time=0.5
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="query",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_status(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Check system status."""
        try:
            # Check vector store status
            total_points = self.vector_store.get_total_points()
            
            return ToolExecution(
                tool_name="status",
                success=True,
                result=f"System has {total_points} knowledge chunks",
                execution_time=0.1
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="status",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_mission(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Handle mission-related requests."""
        try:
            if "create mission" in user_input or "start mission" in user_input:
                mission_objective = user_input.replace("create mission", "").replace("start mission", "").strip()
                context.current_mission = mission_objective
                
                return ToolExecution(
                    tool_name="mission",
                    success=True,
                    result=f"Mission created: {mission_objective}",
                    execution_time=0.1
                )
            
            return ToolExecution(
                tool_name="mission",
                success=True,
                result="Mission command processed",
                execution_time=0.1
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="mission",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _tool_sources(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Handle source-related requests."""
        try:
            return ToolExecution(
                tool_name="sources",
                success=True,
                result="Source information retrieved",
                execution_time=0.1
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="sources",
                success=False,
                result=None,
                error=str(e)
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "current_provider": self.current_provider,
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider
        }
    
    def switch_provider(self, provider: str) -> bool:
        """Switch to a different AI provider."""
        if provider in self.ai_clients and self.ai_clients[provider]['available']:
            self.primary_provider = provider
            logger.info(f"Switched to provider: {provider}")
            return True
        else:
            logger.warning(f"Provider {provider} not available")
            return False 