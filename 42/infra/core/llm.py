"""LLM engine for 42."""

import os
import httpx
import json
from typing import List, Dict, Optional
from loguru import logger

from ..utils.interfaces import SearchResult
from .vector_store import VectorStore
from .embedding import EmbeddingEngine
# Import PromptBuilder only when needed to avoid circular import


class LLMEngine:
    """LLM engine for generating responses using Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the LLM engine."""
        self.base_url = base_url
        self.vector_store = VectorStore()
        self.embedding_engine = EmbeddingEngine()
        # Initialize prompt builder when needed
        self.prompt_builder = None
        
    def query(self, question: str, model: str = "llama3.2", 
              top_k: int = 3, max_tokens: int = 2000) -> Dict:
        """Query the LLM with context from vector store."""
        try:
            # Build prompt with relevant context
            if self.prompt_builder is None:
                from ..services.prompt import PromptBuilder
                self.prompt_builder = PromptBuilder()
            prompt = self.prompt_builder.build_prompt(question, top_k, max_tokens)
            
            # Query Ollama
            response = self._query_ollama(prompt, model)
            
            return {
                "question": question,
                "response": response,
                "model": model,
                "prompt_length": len(prompt),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to query LLM: {e}")
            return {
                "question": question,
                "response": f"Error: {str(e)}",
                "model": model,
                "status": "error",
                "error": str(e)
            }
    
    def _query_ollama(self, prompt: str, model: str) -> str:
        """Query Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        try:
            logger.info(f"Sending request to Ollama with {len(prompt)} characters")
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Received response from Ollama")
                return result.get("response", "No response generated")
                
        except httpx.RequestError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                
                result = response.json()
                return [model["name"] for model in result.get("models", [])]
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test connection to Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                return response.status_code == 200
        except:
            return False 