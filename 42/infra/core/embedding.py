"""Embedding engine for 42."""

from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer

from ..utils.config import load_config


class EmbeddingEngine:
    """Wraps sentence-transformers for text embedding."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding engine (singleton)."""
        if not self._initialized:
            config = load_config()
            self.model_name = model_name or config.embedding_model
            self.model = None
            self._load_model()
            EmbeddingEngine._initialized = True
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text batch: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension() 