"""Prompt builder for 42."""

from typing import List, Optional
from loguru import logger

from .interfaces import SearchResult
from .vector_store import VectorStore
from .embedding import EmbeddingEngine


class PromptBuilder:
    """Builds context-aware prompts for LLM queries."""
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.vector_store = VectorStore()
        self.embedding_engine = EmbeddingEngine()
        
    def build_prompt(self, question: str, top_k: int = 5, max_tokens: int = 4000) -> str:
        """Build a prompt with relevant context from the vector store."""
        try:
            # Embed the question
            question_vector = self.embedding_engine.embed_text(question)
            
            # Search for relevant chunks
            search_results = self.vector_store.search(question_vector, limit=top_k)
            
            if not search_results:
                logger.warning("No relevant context found for question")
                return self._build_fallback_prompt(question)
            
            # Build context from search results
            context_parts = []
            total_tokens = 0
            
            for result in search_results:
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = len(result.text) // 4
                
                if total_tokens + estimated_tokens > max_tokens:
                    break
                
                context_parts.append(self._format_chunk(result))
                total_tokens += estimated_tokens
            
            # Build the final prompt
            prompt = self._build_final_prompt(question, context_parts)
            
            logger.info(f"Built prompt with {len(context_parts)} chunks ({total_tokens} estimated tokens)")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            return self._build_fallback_prompt(question)
    
    def _format_chunk(self, result: SearchResult) -> str:
        """Format a search result chunk for inclusion in the prompt."""
        file_info = f"File: {result.file_path}"
        if result.metadata and "start_line" in result.metadata:
            file_info += f" (lines {result.metadata['start_line']}-{result.metadata.get('end_line', '?')})"
        
        score_info = f"Relevance: {result.score:.3f}"
        
        return f"""
{file_info}
{score_info}
Code:
{result.text}
---"""
    
    def _build_final_prompt(self, question: str, context_parts: List[str]) -> str:
        """Build the final prompt with context and question."""
        context = "\n".join(context_parts)
        
        prompt = f"""You are an AI assistant that helps with code analysis and programming questions. 
You have access to relevant code snippets from a codebase. Use this context to provide accurate and helpful answers.

Context from codebase:
{context}

Question: {question}

Please provide a clear, helpful answer based on the code context above. If the context doesn't contain enough information to answer the question, say so and provide general guidance where possible.

Answer:"""
        
        return prompt
    
    def _build_fallback_prompt(self, question: str) -> str:
        """Build a fallback prompt when no context is available."""
        return f"""You are an AI assistant that helps with code analysis and programming questions.

Question: {question}

Please provide a helpful answer based on your general knowledge. If this is a specific question about code that would require access to a particular codebase, please mention that you would need more context to provide a complete answer.

Answer:"""
    
    def build_code_review_prompt(self, code: str, context_results: Optional[List[SearchResult]] = None) -> str:
        """Build a prompt for code review with optional context."""
        try:
            if context_results is None:
                # Search for similar code patterns
                code_vector = self.embedding_engine.embed_text(code)
                context_results = self.vector_store.search(code_vector, limit=3)
            
            context_parts = []
            for result in context_results:
                context_parts.append(self._format_chunk(result))
            
            context = "\n".join(context_parts) if context_parts else "No similar code patterns found in the codebase."
            
            prompt = f"""You are an expert code reviewer. Review the following code and provide feedback on:

1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Similar code patterns from the codebase (for reference):
{context}

Code to review:
```python
{code}
```

Please provide a comprehensive code review:"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build code review prompt: {e}")
            return f"""You are an expert code reviewer. Review the following code:

```python
{code}
```

Please provide a comprehensive code review:"""
    
    def build_pattern_search_prompt(self, query: str, results: List[SearchResult]) -> str:
        """Build a prompt for analyzing code pattern search results."""
        if not results:
            return f"""No code patterns found matching: "{query}"

This could mean:
1. The pattern doesn't exist in the codebase
2. The search terms need to be more specific
3. The codebase doesn't contain this type of functionality

Try searching with different terms or broader concepts."""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Pattern {i}:\n{self._format_chunk(result)}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are analyzing code patterns in a codebase. Here are the patterns found for the query: "{query}"

{context}

Please analyze these patterns and provide:
1. A summary of what these patterns do
2. Common themes or approaches used
3. Potential improvements or variations
4. How these patterns could be applied to similar problems

Analysis:"""
        
        return prompt 