"""Configuration management for 42 using Pydantic Settings."""

import json
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Config(BaseSettings):
    """Configuration for 42 system using Pydantic Settings."""
    
    # Qdrant settings
    qdrant_host: str = Field(default="localhost", env="42_QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="42_QDRANT_PORT")
    
    # Ollama settings
    ollama_host: str = Field(default="localhost", env="42_OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="42_OLLAMA_PORT")
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="42_REDIS_HOST")
    redis_port: int = Field(default=6379, env="42_REDIS_PORT")
    
    # AI Provider settings
    ai_provider: str = Field(default="openai", env="42_AI_PROVIDER")  # "openai", "ollama", "anthropic"
    ai_model: str = Field(default="gpt-3.5-turbo", env="42_AI_MODEL")
    ai_fallback_provider: str = Field(default="ollama", env="42_AI_FALLBACK_PROVIDER")
    ai_fallback_model: str = Field(default="llama3.2", env="42_AI_FALLBACK_MODEL")
    
    # OpenAI settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="42_OPENAI_MODEL")
    
    # Anthropic settings
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-haiku", env="42_ANTHROPIC_MODEL")
    
    # Embedding settings
    embedding_model: str = Field(default="BAAI/bge-small-en", env="42_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="42_EMBEDDING_DIMENSION")
    
    # Collection settings
    collection_name: str = Field(default="42_chunks", env="42_COLLECTION_NAME")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="42_LOG_LEVEL")
    
    # Timeout settings
    default_timeout: int = Field(default=120, env="42_DEFAULT_TIMEOUT")
    extraction_timeout: int = Field(default=600, env="42_EXTRACTION_TIMEOUT")
    
    # Clustering settings
    min_cluster_size: int = Field(default=5, env="42_MIN_CLUSTER_SIZE")
    min_samples: int = Field(default=3, env="42_MIN_SAMPLES")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('ai_provider', 'ai_fallback_provider')
    def validate_ai_provider(cls, v):
        valid_providers = ['openai', 'ollama', 'anthropic']
        if v.lower() not in valid_providers:
            raise ValueError(f'ai_provider must be one of {valid_providers}')
        return v.lower()
    
    @validator('qdrant_port', 'ollama_port', 'redis_port')
    def validate_ports(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration using Pydantic Settings."""
    if config_path and Path(config_path).exists():
        # Load from custom config file
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
            
            # Create config with file values
            config = Config(**file_config)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
    
    # Load from environment variables and defaults
    config = Config()
    logger.info("Configuration loaded from environment variables and defaults")
    return config


def save_config(config: Config, config_path: str = "42.config.json") -> None:
    """Save configuration to file."""
    config_dict = config.dict()
    
    try:
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise 