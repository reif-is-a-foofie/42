"""Common data structures and interfaces for 42 modules."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    file_path: str
    start_line: int
    end_line: int
    cluster_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    text: str
    file_path: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Cluster:
    """Represents a cluster of similar chunks."""
    cluster_id: int
    chunks: List[Chunk]
    centroid: Optional[List[float]] = None
    size: int = 0


@dataclass
class Config:
    """Configuration for 42 system."""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    redis_host: str = "localhost"
    redis_port: int = 6379
    embedding_model: str = "BAAI/bge-small-en"
    embedding_dimension: int = 384
    collection_name: str = "42_chunks"
    log_level: str = "INFO" 