"""Token counting utilities for 42 using tiktoken."""

import tiktoken
from typing import List, Dict, Any
from loguru import logger


class Tokenizer:
    """Token counting utility using tiktoken."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize tokenizer with specified model."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self.model = model
            logger.info(f"Initialized tokenizer for model: {model}")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer for {model}, using cl100k_base: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = "cl100k_base"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            return len(text.split())  # Fallback to word count
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [self.count_tokens(text) for text in texts]
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate and decode back to text
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        except Exception as e:
            logger.error(f"Failed to truncate text: {e}")
            # Fallback: simple character truncation
            return text[:max_tokens * 4]  # Rough estimate
    
    def estimate_cost(self, tokens: int, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Estimate cost for token usage (approximate)."""
        # Rough cost estimates per 1K tokens (as of 2024)
        cost_per_1k = {
            "gpt-3.5-turbo": 0.0015,  # $0.0015 per 1K input tokens
            "gpt-4": 0.03,            # $0.03 per 1K input tokens
            "llama3.2": 0.0001,       # Very rough estimate for local models
        }
        
        cost_per_1k_tokens = cost_per_1k.get(model, 0.001)
        estimated_cost = (tokens / 1000) * cost_per_1k_tokens
        
        return {
            "tokens": tokens,
            "model": model,
            "estimated_cost_usd": estimated_cost,
            "cost_per_1k_tokens": cost_per_1k_tokens
        }
    
    def analyze_prompt(self, prompt: str, max_tokens: int = None) -> Dict[str, Any]:
        """Analyze prompt token usage."""
        token_count = self.count_tokens(prompt)
        
        analysis = {
            "token_count": token_count,
            "character_count": len(prompt),
            "word_count": len(prompt.split()),
            "tokens_per_word": token_count / len(prompt.split()) if prompt.split() else 0,
            "is_within_limit": True
        }
        
        if max_tokens:
            analysis["is_within_limit"] = token_count <= max_tokens
            analysis["remaining_tokens"] = max_tokens - token_count
            analysis["truncated_prompt"] = self.truncate_to_tokens(prompt, max_tokens)
        
        return analysis


# Global tokenizer instance
_tokenizer = None


def get_tokenizer(model: str = "gpt-3.5-turbo") -> Tokenizer:
    """Get or create global tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer(model)
    return _tokenizer


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Quick token counting utility."""
    return get_tokenizer(model).count_tokens(text)


def analyze_prompt(prompt: str, max_tokens: int = None, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Quick prompt analysis utility."""
    return get_tokenizer(model).analyze_prompt(prompt, max_tokens) 