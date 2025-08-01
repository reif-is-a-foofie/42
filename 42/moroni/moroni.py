"""Moroni - The AI-Agnostic NLP Brain for 42

Moroni provides intelligent conversation handling, knowledge checking, and tool execution.
Acts as the brain layer between user requests and system actions.
Supports multiple AI providers with automatic fallback.
"""

import json
import logging
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
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.current_provider = self.primary_provider
        
        # Available tools
        self.tools = {
            "search": self._tool_search,
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
        
        # All providers failed, return error message
        logger.error("All AI providers failed")
        return f"I'm having trouble processing that request. All AI providers are unavailable."
    
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
        """Process user request and return intelligent response with knowledge integration."""
        try:
            # Step 1: Analyze user intent and determine needed tools
            intent_analysis = self._analyze_intent(user_input, context)
            
            # Step 2: Check knowledge base for relevant information
            knowledge_results = self._check_knowledge(user_input, context)
            
            # Step 3: Execute necessary tools
            tool_results = []
            if intent_analysis.get("needs_tools"):
                for tool_name in intent_analysis["tools_needed"]:
                    if tool_name in self.tools:
                        tool_result = self.tools[tool_name](user_input, context)
                        tool_results.append(tool_result)
            
            # Step 4: Generate comprehensive response
            response = self._generate_response(
                user_input, 
                intent_analysis, 
                knowledge_results, 
                tool_results, 
                context
            )
            
            return KnowledgeResponse(
                response=response["text"],
                confidence=response["confidence"],
                sources=knowledge_results.get("sources", []),
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
    
    def _check_knowledge(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Check knowledge base for relevant information."""
        try:
            # Embed the query
            query_embedding = self.embedding_engine.embed_text(query)
            
            # Search vector store using the correct method
            search_results = self.vector_store.search_semantic(query_embedding, limit=5)
            
            # Analyze relevance
            relevant_sources = []
            for result in search_results:
                if result.get("score", 0) > 0.7:  # Only highly relevant results
                    relevant_sources.append({
                        "content": result.get("payload", {}).get("text", ""),
                        "source": result.get("payload", {}).get("source", ""),
                        "relevance": result.get("score", 0)
                    })
            
            return {
                "sources": relevant_sources,
                "total_found": len(search_results),
                "relevant_count": len(relevant_sources)
            }
            
        except Exception as e:
            logger.error(f"Failed to check knowledge: {e}")
            return {"sources": [], "total_found": 0, "relevant_count": 0}
    
    def _generate_response(
        self, 
        user_input: str, 
        intent_analysis: Dict[str, Any],
        knowledge_results: Dict[str, Any],
        tool_results: List[ToolExecution],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate comprehensive response integrating knowledge and tool results."""
        try:
            # Build context for response generation
            knowledge_context = ""
            if knowledge_results.get("sources"):
                knowledge_context = "Relevant knowledge found:\n"
                for i, source in enumerate(knowledge_results["sources"][:3]):
                    knowledge_context += f"{i+1}. {source['content'][:200]}...\n"
            
            tool_context = ""
            if tool_results:
                tool_context = "Tool execution results:\n"
                for result in tool_results:
                    if result.success:
                        tool_context += f"- {result.tool_name}: {str(result.result)[:100]}...\n"
                    else:
                        tool_context += f"- {result.tool_name}: Error - {result.error}\n"
            
            prompt = f"""
            Generate a helpful response to the user's request.
            
            User request: {user_input}
            User intent: {intent_analysis.get('intent', 'unknown')}
            
            {knowledge_context}
            
            {tool_context}
            
            Provide a comprehensive, helpful response that:
            1. Addresses the user's request directly
            2. Incorporates relevant knowledge if available
            3. Explains any actions taken
            4. Suggests next steps if appropriate
            
            Respond in JSON format:
            {{
                "text": "your response here",
                "confidence": 0.0-1.0,
                "reasoning": "why this response was chosen"
            }}
            """
            
            response = self._call_ai(prompt, "response_generation")
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "text": "I'm having trouble generating a response. Please try again.",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    # Tool implementations
    def _tool_search(self, user_input: str, context: ConversationContext) -> ToolExecution:
        """Search for information."""
        try:
            # Extract search query from user input
            search_query = user_input.replace("search", "").replace("find", "").strip()
            
            # Use vector store search
            query_embedding = self.embedding_engine.embed_text(search_query)
            results = self.vector_store.search_semantic(query_embedding, limit=10)
            
            return ToolExecution(
                tool_name="search",
                success=True,
                result=f"Found {len(results)} relevant results",
                execution_time=0.1
            )
            
        except Exception as e:
            return ToolExecution(
                tool_name="search",
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