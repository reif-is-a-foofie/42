"""
Tests for 42.un Source Scanner components.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add the 42 package to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from un.source_scanner import (
    SourceScanner,
    GitHubScanner,
    FileSystemScanner,
    RSSFeedScanner,
    APIEndpointScanner,
    SourceScannerOrchestrator
)
from un.redis_bus import RedisBus
from un.events import Event, EventType


class TestSourceScanner:
    """Test base SourceScanner functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,  # Short interval for testing
            "max_errors": 3,
            "github": {
                "repos": ["https://github.com/test/repo"],
                "webhook_secret": "test-secret",
                "api_token": "test-token"
            },
            "filesystem": {
                "watch_directories": ["/tmp/test"],
                "ignore_patterns": ["*.tmp"]
            },
            "rss": {
                "feeds": ["https://example.com/feed"]
            },
            "api": {
                "endpoints": [
                    {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "headers": {"Authorization": "Bearer test"}
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_source_scanner_initialization(self, mock_redis_bus, config):
        """Test SourceScanner initialization."""
        scanner = SourceScanner(mock_redis_bus, config)
        
        assert scanner.redis_bus == mock_redis_bus
        assert scanner.config == config
        assert scanner.scan_interval == 1
        assert scanner.max_errors == 3
        assert not scanner.running
        assert scanner.error_count == 0
    
    @pytest.mark.asyncio
    async def test_source_scanner_state_management(self, mock_redis_bus, config):
        """Test state storage and retrieval."""
        scanner = SourceScanner(mock_redis_bus, config)
        
        # Test state key generation
        key = scanner._get_state_key("test-source")
        assert key == "scanner:state:SourceScanner:test-source"
        
        # Test state storage
        test_state = {"last_commit": "abc123", "timestamp": "2024-01-01T00:00:00"}
        await scanner._store_state("test-source", test_state)
        
        mock_redis_bus.redis.set.assert_called_once()
        call_args = mock_redis_bus.redis.set.call_args
        assert call_args[0][0] == key
        assert json.loads(call_args[0][1]) == test_state
        assert call_args[1]["ex"] == 86400
        
        # Test state retrieval
        mock_redis_bus.redis.get.return_value = json.dumps(test_state)
        retrieved_state = await scanner._get_stored_state("test-source")
        assert retrieved_state == test_state
        
        # Test state retrieval with no stored state
        mock_redis_bus.redis.get.return_value = None
        retrieved_state = await scanner._get_stored_state("test-source")
        assert retrieved_state is None


class TestGitHubScanner:
    """Test GitHubScanner functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,
            "github": {
                "repos": ["https://github.com/test/repo"],
                "webhook_secret": "test-secret",
                "api_token": "test-token"
            }
        }
    
    @pytest.mark.asyncio
    async def test_github_scanner_initialization(self, mock_redis_bus, config):
        """Test GitHubScanner initialization."""
        scanner = GitHubScanner(mock_redis_bus, config)
        
        assert scanner.repos == ["https://github.com/test/repo"]
        assert scanner.webhook_secret == "test-secret"
        assert scanner.api_token == "test-token"
        assert scanner.api_base == "https://api.github.com"
    
    @pytest.mark.asyncio
    async def test_github_scanner_invalid_url(self, mock_redis_bus, config):
        """Test GitHubScanner with invalid URL."""
        scanner = GitHubScanner(mock_redis_bus, config)
        scanner.repos = ["invalid-url"]
        
        # Mock session
        mock_session = AsyncMock()
        scanner.session = mock_session
        
        await scanner._check_repo_changes("invalid-url", mock_session)
        # Should handle invalid URL gracefully
    
    @pytest.mark.asyncio
    async def test_github_scanner_api_error(self, mock_redis_bus, config):
        """Test GitHubScanner with API error."""
        scanner = GitHubScanner(mock_redis_bus, config)
        
        # Mock session with error response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        scanner.session = mock_session
        
        await scanner._check_repo_changes("https://github.com/test/repo", mock_session)
        # Should handle API error gracefully
    
    @pytest.mark.asyncio
    async def test_github_scanner_new_commit(self, mock_redis_bus, config):
        """Test GitHubScanner detecting new commit."""
        scanner = GitHubScanner(mock_redis_bus, config)
        
        # Mock session with successful response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            "sha": "abc123",
            "commit": {
                "author": {
                    "name": "Test Author",
                    "date": "2024-01-01T00:00:00Z"
                },
                "message": "Test commit"
            }
        }])
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock no stored state (new commit)
        mock_redis_bus.redis.get.return_value = None
        
        scanner.session = mock_session
        
        await scanner._check_repo_changes("https://github.com/test/repo", mock_session)
        
        # Verify state was stored
        mock_redis_bus.redis.set.assert_called_once()
        
        # Verify event was published
        mock_redis_bus.publish_event.assert_called_once()


class TestFileSystemScanner:
    """Test FileSystemScanner functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,
            "filesystem": {
                "watch_directories": ["/tmp/test"],
                "ignore_patterns": ["*.tmp", "*.log"]
            }
        }
    
    @pytest.mark.asyncio
    async def test_file_system_scanner_initialization(self, mock_redis_bus, config):
        """Test FileSystemScanner initialization."""
        scanner = FileSystemScanner(mock_redis_bus, config)
        
        assert scanner.watch_directories == ["/tmp/test"]
        assert scanner.ignore_patterns == ["*.tmp", "*.log"]
    
    def test_file_system_scanner_ignore_patterns(self, mock_redis_bus, config):
        """Test file ignore patterns."""
        scanner = FileSystemScanner(mock_redis_bus, config)
        
        # Test ignored files
        assert scanner._should_ignore_file("/tmp/test.tmp")
        assert scanner._should_ignore_file("/tmp/test.log")
        assert scanner._should_ignore_file("/tmp/.hidden")
        assert scanner._should_ignore_file("/tmp/binary.exe")
        
        # Test allowed files
        assert not scanner._should_ignore_file("/tmp/test.py")
        assert not scanner._should_ignore_file("/tmp/README.md")
        assert not scanner._should_ignore_file("/tmp/config.json")


class TestRSSFeedScanner:
    """Test RSSFeedScanner functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,
            "rss": {
                "feeds": ["https://example.com/feed"]
            }
        }
    
    @pytest.mark.asyncio
    async def test_rss_feed_scanner_initialization(self, mock_redis_bus, config):
        """Test RSSFeedScanner initialization."""
        scanner = RSSFeedScanner(mock_redis_bus, config)
        
        assert scanner.feeds == ["https://example.com/feed"]
    
    @pytest.mark.asyncio
    async def test_rss_feed_scanner_fetch_error(self, mock_redis_bus, config):
        """Test RSSFeedScanner with fetch error."""
        scanner = RSSFeedScanner(mock_redis_bus, config)
        
        # Mock session with error response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        scanner.session = mock_session
        
        await scanner._check_feed_updates("https://example.com/feed", mock_session)
        # Should handle fetch error gracefully


class TestAPIEndpointScanner:
    """Test APIEndpointScanner functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,
            "api": {
                "endpoints": [
                    {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "headers": {"Authorization": "Bearer test"}
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_api_endpoint_scanner_initialization(self, mock_redis_bus, config):
        """Test APIEndpointScanner initialization."""
        scanner = APIEndpointScanner(mock_redis_bus, config)
        
        assert len(scanner.endpoints) == 1
        assert scanner.endpoints[0]["url"] == "https://api.example.com/data"
        assert scanner.endpoints[0]["method"] == "GET"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_scanner_api_error(self, mock_redis_bus, config):
        """Test APIEndpointScanner with API error."""
        scanner = APIEndpointScanner(mock_redis_bus, config)
        
        # Mock session with error response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        scanner.session = mock_session
        
        await scanner._check_endpoint_changes(
            {"url": "https://api.example.com/data", "method": "GET"},
            mock_session
        )
        # Should handle API error gracefully


class TestSourceScannerOrchestrator:
    """Test SourceScannerOrchestrator functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.redis = AsyncMock()
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "scan_interval": 1,
            "github": {
                "repos": ["https://github.com/test/repo"]
            },
            "filesystem": {
                "watch_directories": ["/tmp/test"]
            },
            "rss": {
                "feeds": ["https://example.com/feed"]
            },
            "api": {
                "endpoints": [
                    {
                        "url": "https://api.example.com/data",
                        "method": "GET"
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_redis_bus, config):
        """Test SourceScannerOrchestrator initialization."""
        orchestrator = SourceScannerOrchestrator(mock_redis_bus, config)
        
        assert orchestrator.redis_bus == mock_redis_bus
        assert orchestrator.config == config
        assert not orchestrator.running
        assert len(orchestrator.scanners) == 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_start(self, mock_redis_bus, config):
        """Test SourceScannerOrchestrator start."""
        orchestrator = SourceScannerOrchestrator(mock_redis_bus, config)
        
        # Mock scanner start methods
        with patch('un.source_scanner.GitHubScanner') as mock_github_scanner_class, \
             patch('un.source_scanner.FileSystemScanner') as mock_fs_scanner_class, \
             patch('un.source_scanner.RSSFeedScanner') as mock_rss_scanner_class, \
             patch('un.source_scanner.APIEndpointScanner') as mock_api_scanner_class:
            
            # Mock scanner instances
            mock_github_scanner = AsyncMock()
            mock_fs_scanner = AsyncMock()
            mock_rss_scanner = AsyncMock()
            mock_api_scanner = AsyncMock()
            
            mock_github_scanner_class.return_value = mock_github_scanner
            mock_fs_scanner_class.return_value = mock_fs_scanner
            mock_rss_scanner_class.return_value = mock_rss_scanner
            mock_api_scanner_class.return_value = mock_api_scanner
            
            await orchestrator.start()
            
            assert orchestrator.running
            assert len(orchestrator.scanners) == 4
    
    @pytest.mark.asyncio
    async def test_orchestrator_stop(self, mock_redis_bus, config):
        """Test SourceScannerOrchestrator stop."""
        orchestrator = SourceScannerOrchestrator(mock_redis_bus, config)
        orchestrator.running = True
        
        # Add mock scanners
        mock_scanner = AsyncMock()
        orchestrator.scanners = [mock_scanner]
        
        await orchestrator.stop()
        
        assert not orchestrator.running
        assert len(orchestrator.scanners) == 0
        mock_scanner.stop.assert_called_once()
    
    def test_orchestrator_get_status(self, mock_redis_bus, config):
        """Test SourceScannerOrchestrator status."""
        orchestrator = SourceScannerOrchestrator(mock_redis_bus, config)
        orchestrator.running = True
        
        # Add mock scanner
        mock_scanner = Mock()
        mock_scanner.__class__.__name__ = "TestScanner"
        mock_scanner.running = True
        mock_scanner.error_count = 0
        orchestrator.scanners = [mock_scanner]
        
        status = orchestrator.get_status()
        
        assert status["running"] is True
        assert status["scanner_count"] == 1
        assert len(status["scanners"]) == 1
        assert status["scanners"][0]["type"] == "TestScanner"


if __name__ == "__main__":
    pytest.main([__file__]) 