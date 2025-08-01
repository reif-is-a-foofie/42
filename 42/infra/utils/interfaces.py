"""Common data structures and interfaces for 42 modules."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass(frozen=True)
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    file_path: str
    start_line: int
    end_line: int
    cluster_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SearchResult:
    """Represents a search result from the vector store."""
    text: str
    file_path: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Cluster:
    """Represents a cluster of similar chunks."""
    cluster_id: int
    chunks: List[Chunk]
    centroid: Optional[List[float]] = None
    size: int = 0


# Config class moved to config.py using Pydantic Settings 