"""Tests for the config module."""

import pytest
import tempfile
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from config import Config, load_config, save_config, get_default_config


class TestConfig:
    """Test config functionality."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "42_chunks"
                },
                "ollama": {
                    "host": "localhost",
                    "port": 11434,
                    "model": "mistral:latest"
                },
                "embedding": {
                    "model": "bge-small-en",
                    "dimension": 384
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000
                }
            }
            json.dump(config_data, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample config object."""
        return Config(
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection="42_chunks",
            ollama_host="localhost",
            ollama_port=11434,
            ollama_model="mistral:latest",
            embedding_model="bge-small-en",
            embedding_dimension=384,
            api_host="0.0.0.0",
            api_port=8000
        )
    
    def test_config_initialization(self, sample_config):
        """Test config object initialization."""
        assert sample_config.qdrant_host == "localhost"
        assert sample_config.qdrant_port == 6333
        assert sample_config.qdrant_collection == "42_chunks"
        assert sample_config.ollama_host == "localhost"
        assert sample_config.ollama_port == 11434
        assert sample_config.ollama_model == "mistral:latest"
        assert sample_config.embedding_model == "bge-small-en"
        assert sample_config.embedding_dimension == 384
        assert sample_config.api_host == "0.0.0.0"
        assert sample_config.api_port == 8000
    
    def test_load_config_from_file(self, temp_config_file):
        """Test loading config from file."""
        config = load_config(temp_config_file)
        
        assert isinstance(config, Config)
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.ollama_model == "mistral:latest"
        assert config.embedding_model == "bge-small-en"
    
    def test_load_config_default_path(self):
        """Test loading config from default path."""
        # Create a temporary config file in the expected location
        config_data = {
            "qdrant": {"host": "test-host", "port": 6333},
            "ollama": {"host": "test-host", "port": 11434, "model": "test-model"},
            "embedding": {"model": "test-embedding", "dimension": 384},
            "api": {"host": "0.0.0.0", "port": 8000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Mock the default config path
            with patch('42.config.CONFIG_FILE', temp_path):
                config = load_config()
                assert isinstance(config, Config)
                assert config.qdrant_host == "test-host"
                assert config.ollama_model == "test-model"
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_config(self, sample_config):
        """Test saving config to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(sample_config, temp_path)
            
            # Verify the file was created and contains correct data
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["qdrant"]["host"] == "localhost"
            assert saved_data["qdrant"]["port"] == 6333
            assert saved_data["ollama"]["model"] == "mistral:latest"
            assert saved_data["embedding"]["model"] == "bge-small-en"
        finally:
            os.unlink(temp_path)
    
    def test_save_config_default_path(self, sample_config):
        """Test saving config to default path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock the default config path
            with patch('42.config.CONFIG_FILE', temp_path):
                save_config(sample_config)
                
                assert os.path.exists(temp_path)
                
                with open(temp_path, 'r') as f:
                    saved_data = json.load(f)
                
                assert saved_data["qdrant"]["host"] == "localhost"
                assert saved_data["ollama"]["model"] == "mistral:latest"
        finally:
            os.unlink(temp_path)
    
    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.ollama_host == "localhost"
        assert config.ollama_port == 11434
        assert config.embedding_model == "bge-small-en"
        assert config.embedding_dimension == 384
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
    
    def test_config_environment_overrides(self):
        """Test config environment variable overrides."""
        # Set environment variables
        os.environ["QDRANT_HOST"] = "env-host"
        os.environ["OLLAMA_MODEL"] = "env-model"
        os.environ["EMBEDDING_MODEL"] = "env-embedding"
        os.environ["API_PORT"] = "9000"
        
        try:
            config = get_default_config()
            
            assert config.qdrant_host == "env-host"
            assert config.ollama_model == "env-model"
            assert config.embedding_model == "env-embedding"
            assert config.api_port == 9000
        finally:
            # Clean up environment variables
            for key in ["QDRANT_HOST", "OLLAMA_MODEL", "EMBEDDING_MODEL", "API_PORT"]:
                os.environ.pop(key, None)
    
    def test_config_validation(self):
        """Test config validation."""
        # Test invalid port numbers
        with pytest.raises(ValueError):
            Config(
                qdrant_host="localhost",
                qdrant_port=-1,  # Invalid port
                qdrant_collection="test",
                ollama_host="localhost",
                ollama_port=11434,
                ollama_model="test",
                embedding_model="test",
                embedding_dimension=384,
                api_host="0.0.0.0",
                api_port=8000
            )
        
        with pytest.raises(ValueError):
            Config(
                qdrant_host="localhost",
                qdrant_port=6333,
                qdrant_collection="test",
                ollama_host="localhost",
                ollama_port=70000,  # Invalid port
                ollama_model="test",
                embedding_model="test",
                embedding_dimension=384,
                api_host="0.0.0.0",
                api_port=8000
            )
    
    def test_config_to_dict(self, sample_config):
        """Test converting config to dictionary."""
        config_dict = sample_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["qdrant"]["host"] == "localhost"
        assert config_dict["qdrant"]["port"] == 6333
        assert config_dict["ollama"]["model"] == "mistral:latest"
        assert config_dict["embedding"]["model"] == "bge-small-en"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "qdrant": {
                "host": "test-host",
                "port": 6333,
                "collection_name": "test_collection"
            },
            "ollama": {
                "host": "test-host",
                "port": 11434,
                "model": "test-model"
            },
            "embedding": {
                "model": "test-embedding",
                "dimension": 384
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        
        config = Config.from_dict(config_dict)
        
        assert isinstance(config, Config)
        assert config.qdrant_host == "test-host"
        assert config.qdrant_collection == "test_collection"
        assert config.ollama_model == "test-model"
        assert config.embedding_model == "test-embedding"
    
    def test_config_merge(self, sample_config):
        """Test merging config with updates."""
        updates = {
            "qdrant_host": "updated-host",
            "ollama_model": "updated-model",
            "api_port": 9000
        }
        
        updated_config = sample_config.merge(updates)
        
        assert updated_config.qdrant_host == "updated-host"
        assert updated_config.ollama_model == "updated-model"
        assert updated_config.api_port == 9000
        
        # Original config should be unchanged
        assert sample_config.qdrant_host == "localhost"
        assert sample_config.ollama_model == "mistral:latest"
        assert sample_config.api_port == 8000
    
    def test_config_validation_missing_required_fields(self):
        """Test config validation with missing required fields."""
        with pytest.raises(TypeError):
            Config()  # Missing required arguments
    
    def test_config_validation_invalid_dimension(self):
        """Test config validation with invalid embedding dimension."""
        with pytest.raises(ValueError):
            Config(
                qdrant_host="localhost",
                qdrant_port=6333,
                qdrant_collection="test",
                ollama_host="localhost",
                ollama_port=11434,
                ollama_model="test",
                embedding_model="test",
                embedding_dimension=0,  # Invalid dimension
                api_host="0.0.0.0",
                api_port=8000
            )
    
    def test_config_serialization(self, sample_config):
        """Test config serialization and deserialization."""
        # Test JSON serialization
        config_json = sample_config.to_json()
        assert isinstance(config_json, str)
        
        # Test JSON deserialization
        deserialized_config = Config.from_json(config_json)
        assert isinstance(deserialized_config, Config)
        assert deserialized_config.qdrant_host == sample_config.qdrant_host
        assert deserialized_config.ollama_model == sample_config.ollama_model
        assert deserialized_config.embedding_model == sample_config.embedding_model
    
    def test_config_repr(self, sample_config):
        """Test config string representation."""
        config_repr = repr(sample_config)
        assert isinstance(config_repr, str)
        assert "Config" in config_repr
        assert "localhost" in config_repr
        assert "mistral:latest" in config_repr
    
    def test_config_equality(self, sample_config):
        """Test config equality comparison."""
        config_copy = Config(
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection="42_chunks",
            ollama_host="localhost",
            ollama_port=11434,
            ollama_model="mistral:latest",
            embedding_model="bge-small-en",
            embedding_dimension=384,
            api_host="0.0.0.0",
            api_port=8000
        )
        
        assert sample_config == config_copy
        
        # Test inequality
        different_config = Config(
            qdrant_host="different-host",
            qdrant_port=6333,
            qdrant_collection="42_chunks",
            ollama_host="localhost",
            ollama_port=11434,
            ollama_model="mistral:latest",
            embedding_model="bge-small-en",
            embedding_dimension=384,
            api_host="0.0.0.0",
            api_port=8000
        )
        
        assert sample_config != different_config 