"""42 - Code analysis and querying system."""

# Core features
from .soul import soul
from .mission import MissionOrchestrator
from .moroni import Moroni

# Core components
from .infra.core import EmbeddingEngine, VectorStore, Chunker, LLMEngine, ClusteringEngine

# Utilities
from .infra.utils import load_config, save_config, Config, Chunk, SearchResult, Cluster

# Services
from .infra.services import run_server, app, main

__version__ = "1.0.0"

__all__ = [
    # Core features
    'soul',
    'MissionOrchestrator', 
    'Moroni',
    
    # Core components
    'EmbeddingEngine',
    'VectorStore',
    'Chunker', 
    'LLMEngine',
    'ClusteringEngine',
    
    # Utilities
    'load_config',
    'save_config',
    'Config',
    'Chunk',
    'SearchResult',
    'Cluster',
    
    # Services
    'run_server',
    'app',
    'main'
] 