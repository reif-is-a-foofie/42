"""Configuration management for 42."""

import json
import os
from pathlib import Path
from typing import Optional
from loguru import logger

from .interfaces import Config


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment variables."""
    if config_path is None:
        config_path = "42.config.json"
    
    config = Config()
    
    # Load from file if it exists
    if Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                for key, value in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    # Override with environment variables
    config.qdrant_host = os.getenv("42_QDRANT_HOST", config.qdrant_host)
    config.qdrant_port = int(os.getenv("42_QDRANT_PORT", str(config.qdrant_port)))
    config.ollama_host = os.getenv("42_OLLAMA_HOST", config.ollama_host)
    config.ollama_port = int(os.getenv("42_OLLAMA_PORT", str(config.ollama_port)))
    config.redis_host = os.getenv("42_REDIS_HOST", config.redis_host)
    config.redis_port = int(os.getenv("42_REDIS_PORT", str(config.redis_port)))
    config.embedding_model = os.getenv("42_EMBEDDING_MODEL", config.embedding_model)
    config.embedding_dimension = int(os.getenv("42_EMBEDDING_DIMENSION", str(config.embedding_dimension)))
    config.collection_name = os.getenv("42_COLLECTION_NAME", config.collection_name)
    config.log_level = os.getenv("42_LOG_LEVEL", config.log_level)
    
    return config


def save_config(config: Config, config_path: str = "42.config.json") -> None:
    """Save configuration to file."""
    config_dict = {
        "qdrant_host": config.qdrant_host,
        "qdrant_port": config.qdrant_port,
        "ollama_host": config.ollama_host,
        "ollama_port": config.ollama_port,
        "redis_host": config.redis_host,
        "redis_port": config.redis_port,
        "embedding_model": config.embedding_model,
        "embedding_dimension": config.embedding_dimension,
        "collection_name": config.collection_name,
        "log_level": config.log_level,
    }
    
    try:
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise 