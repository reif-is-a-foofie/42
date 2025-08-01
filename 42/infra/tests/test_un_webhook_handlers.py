"""
Tests for 42.un Webhook Handlers.
"""

import pytest
import json
import hashlib
import hmac
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import sys
import os

# Add the 42 package to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from un.webhook_handlers import (
    WebhookValidator,
    GitHubWebhookHandler,
    GenericWebhookHandler,
    WebhookManager
)
from un.redis_bus import RedisBus
from un.events import Event, EventType


class TestWebhookValidator:
    """Test WebhookValidator functionality."""
    
    def test_verify_github_signature_valid(self):
        """Test valid GitHub signature verification."""
        secret = "test-secret"
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        
        # Create valid signature
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        
        github_signature = f"sha256={signature}"
        
        assert WebhookValidator.verify_github_signature(payload_bytes, github_signature, secret)
    
    def test_verify_github_signature_invalid(self):
        """Test invalid GitHub signature verification."""
        secret = "test-secret"
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        
        # Invalid signature
        invalid_signature = "sha256=invalid"
        
        assert not WebhookValidator.verify_github_signature(payload_bytes, invalid_signature, secret)
    
    def test_verify_github_signature_no_secret(self):
        """Test GitHub signature verification with no secret."""
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        signature = "sha256=test"
        
        assert not WebhookValidator.verify_github_signature(payload_bytes, signature, "")
    
    def test_verify_generic_signature_sha256(self):
        """Test generic signature verification with SHA256."""
        secret = "test-secret"
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        
        # Create valid signature
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        
        assert WebhookValidator.verify_generic_signature(payload_bytes, signature, secret, "sha256")
    
    def test_verify_generic_signature_sha1(self):
        """Test generic signature verification with SHA1."""
        secret = "test-secret"
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        
        # Create valid signature
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha1
        ).hexdigest()
        
        assert WebhookValidator.verify_generic_signature(payload_bytes, signature, secret, "sha1")
    
    def test_verify_generic_signature_invalid_algorithm(self):
        """Test generic signature verification with invalid algorithm."""
        secret = "test-secret"
        payload = '{"test": "data"}'
        payload_bytes = payload.encode('utf-8')
        signature = "test"
        
        assert not WebhookValidator.verify_generic_signature(payload_bytes, signature, secret, "md5")


class TestGitHubWebhookHandler:
    """Test GitHubWebhookHandler functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def handler(self, mock_redis_bus):
        """Create a GitHub webhook handler."""
        return GitHubWebhookHandler(mock_redis_bus, "test-secret")
    
    @pytest.fixture
    def valid_signature(self):
        """Create a valid signature for testing."""
        # This will be updated for each specific test
        return "sha256=placeholder"
    
    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, handler):
        """Test webhook handling with invalid signature."""
        payload = {"test": "data"}
        invalid_signature = "sha256=invalid"
        
        result = await handler.handle_webhook(payload, invalid_signature)
        assert not result
    
    @pytest.mark.asyncio
    async def test_handle_push_event(self, handler, mock_redis_bus):
        """Test push event handling."""
        payload = {
            "ref_type": "push",
            "repository": {"html_url": "https://github.com/test/repo"},
            "after": "abc123",
            "ref": "refs/heads/main",
            "pusher": {"name": "test-user"},
            "head_commit": {"message": "Test commit"}
        }
        
        # Generate valid signature for this payload
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        github_signature = f"sha256={signature}"
        
        result = await handler.handle_webhook(payload, github_signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_pr_event(self, handler, mock_redis_bus):
        """Test pull request event handling."""
        payload = {
            "ref_type": "pull_request",
            "action": "opened",
            "repository": {"html_url": "https://github.com/test/repo"},
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "html_url": "https://github.com/test/repo/pull/123"
            }
        }
        
        # Generate valid signature for this payload
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        github_signature = f"sha256={signature}"
        
        result = await handler.handle_webhook(payload, github_signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_create_event(self, handler, mock_redis_bus):
        """Test create event handling."""
        payload = {
            "ref_type": "create",
            "repository": {"html_url": "https://github.com/test/repo"},
            "ref": "new-branch"
        }
        
        # Generate valid signature for this payload
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        github_signature = f"sha256={signature}"
        
        result = await handler.handle_webhook(payload, github_signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_delete_event(self, handler, mock_redis_bus):
        """Test delete event handling."""
        payload = {
            "ref_type": "delete",
            "repository": {"html_url": "https://github.com/test/repo"},
            "ref": "old-branch"
        }
        
        # Generate valid signature for this payload
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        github_signature = f"sha256={signature}"
        
        result = await handler.handle_webhook(payload, github_signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()


class TestGenericWebhookHandler:
    """Test GenericWebhookHandler functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def handler(self, mock_redis_bus):
        """Create a generic webhook handler."""
        return GenericWebhookHandler(mock_redis_bus, "test-secret")
    
    @pytest.fixture
    def valid_signature(self):
        """Create a valid signature for testing."""
        # This will be updated for each specific test
        return "placeholder"
    
    @pytest.mark.asyncio
    async def test_handle_webhook_valid(self, handler, mock_redis_bus):
        """Test valid webhook handling."""
        payload = {
            "event_type": "test_event",
            "source": "test_source",
            "data": "test_data"
        }
        
        # Generate valid signature for this payload
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        
        result = await handler.handle_webhook(payload, signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, handler):
        """Test webhook handling with invalid signature."""
        payload = {"test": "data"}
        invalid_signature = "invalid"
        
        result = await handler.handle_webhook(payload, invalid_signature)
        assert not result
    
    @pytest.mark.asyncio
    async def test_handle_webhook_sha1_algorithm(self, mock_redis_bus):
        """Test webhook handling with SHA1 algorithm."""
        handler = GenericWebhookHandler(mock_redis_bus, "test-secret", "sha1")
        
        payload = {"test": "data"}
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = hmac.new(
            "test-secret".encode('utf-8'),
            payload_bytes,
            hashlib.sha1
        ).hexdigest()
        
        result = await handler.handle_webhook(payload, signature)
        assert result
        mock_redis_bus.publish_event.assert_called_once()


class TestWebhookManager:
    """Test WebhookManager functionality."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create a mock Redis bus."""
        mock_bus = Mock(spec=RedisBus)
        mock_bus.publish_event = AsyncMock()
        return mock_bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "github": {
                "webhook_secret": "github-secret"
            },
            "webhooks": {
                "custom": {
                    "secret": "custom-secret",
                    "algorithm": "sha256"
                }
            }
        }
    
    @pytest.fixture
    def manager(self, mock_redis_bus, config):
        """Create a webhook manager."""
        return WebhookManager(mock_redis_bus, config)
    
    def test_webhook_manager_initialization(self, manager):
        """Test WebhookManager initialization."""
        assert "github" in manager.handlers
        assert "custom" in manager.handlers
        assert len(manager.handlers) == 2
    
    def test_get_supported_webhooks(self, manager):
        """Test getting supported webhook types."""
        supported = manager.get_supported_webhooks()
        assert "github" in supported
        assert "custom" in supported
        assert len(supported) == 2
    
    def test_is_webhook_supported(self, manager):
        """Test webhook support checking."""
        assert manager.is_webhook_supported("github")
        assert manager.is_webhook_supported("custom")
        assert not manager.is_webhook_supported("unknown")
    
    @pytest.mark.asyncio
    async def test_handle_webhook_github(self, manager, mock_redis_bus):
        """Test handling GitHub webhook."""
        payload = {"test": "data"}
        signature = "sha256=test"
        
        # Mock the handler to return True
        manager.handlers["github"].handle_webhook = AsyncMock(return_value=True)
        
        result = await manager.handle_webhook("github", payload, signature)
        assert result
    
    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_type(self, manager):
        """Test handling unknown webhook type."""
        payload = {"test": "data"}
        signature = "test"
        
        result = await manager.handle_webhook("unknown", payload, signature)
        assert not result


if __name__ == "__main__":
    pytest.main([__file__]) 