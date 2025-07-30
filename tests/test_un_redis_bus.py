"""Tests for the 42.un Redis bus module."""

import pytest
import json
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from un.redis_bus import RedisBus
from un.events import Event, EventType, create_github_repo_updated_event


class TestRedisBus:
    """Test Redis bus functionality."""
    
    @pytest.fixture
    def redis_bus(self):
        """Create a Redis bus instance for testing."""
        with patch('un.redis_bus.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            bus = RedisBus(host="localhost", port=6379, db=0)
            yield bus
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return create_github_repo_updated_event(
            repository="user/repo",
            commit_hash="abc123",
            branch="main",
            files_changed=5
        )
    
    def test_redis_bus_initialization(self, redis_bus):
        """Test Redis bus initialization."""
        assert redis_bus is not None
        assert hasattr(redis_bus, 'client')
        assert hasattr(redis_bus, 'connection_pool')
    
    def test_publish_event(self, redis_bus, sample_event):
        """Test publishing an event."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            mock_publish.return_value = 1  # Number of clients that received the message
            
            result = redis_bus.publish_event(sample_event)
            
            assert result == 1
            mock_publish.assert_called_once()
            
            # Check that the event was serialized to JSON
            call_args = mock_publish.call_args
            assert call_args[0][0] == "42_events"  # Channel name
            assert call_args[0][1] == sample_event.to_json()  # Event JSON
    
    def test_publish_event_to_specific_channel(self, redis_bus, sample_event):
        """Test publishing an event to a specific channel."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            mock_publish.return_value = 1
            
            result = redis_bus.publish_event(sample_event, channel="github_events")
            
            assert result == 1
            mock_publish.assert_called_once_with("github_events", sample_event.to_json())
    
    def test_publish_event_for_persistence(self, redis_bus, sample_event):
        """Test publishing an event for persistence."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            with patch.object(redis_bus, '_store_event') as mock_store:
                mock_publish.return_value = 1
                mock_store.return_value = True
                
                result = redis_bus.publish_event(sample_event, persist=True)
                
                assert result == 1
                mock_publish.assert_called_once()
                mock_store.assert_called_once_with(sample_event)
    
    def test_store_event(self, redis_bus, sample_event):
        """Test storing an event in Redis."""
        with patch.object(redis_bus.client, 'zadd') as mock_zadd:
            mock_zadd.return_value = 1
            
            result = redis_bus._store_event(sample_event)
            
            assert result is True
            mock_zadd.assert_called_once()
            
            # Check that the event was stored with timestamp as score
            call_args = mock_zadd.call_args
            assert call_args[0][0] == "42_events_history"  # Key name
            assert call_args[0][1] == {sample_event.to_json(): sample_event.timestamp.timestamp()}
    
    def test_store_event_with_history_trimming(self, redis_bus, sample_event):
        """Test storing event with history trimming."""
        with patch.object(redis_bus.client, 'zadd') as mock_zadd:
            with patch.object(redis_bus.client, 'zremrangebyrank') as mock_trim:
                mock_zadd.return_value = 1
                mock_trim.return_value = 0
                
                result = redis_bus._store_event(sample_event)
                
                assert result is True
                mock_zadd.assert_called_once()
                mock_trim.assert_called_once_with("42_events_history", 0, -1001)  # Keep last 1000 events
    
    def test_subscribe_to_events(self, redis_bus):
        """Test subscribing to events."""
        with patch.object(redis_bus.client, 'pubsub') as mock_pubsub:
            mock_pubsub_instance = Mock()
            mock_pubsub.return_value = mock_pubsub_instance
            
            pubsub = redis_bus.subscribe_to_events()
            
            assert pubsub == mock_pubsub_instance
            mock_pubsub_instance.subscribe.assert_called_once_with("42_events")
    
    def test_subscribe_to_specific_channel(self, redis_bus):
        """Test subscribing to a specific channel."""
        with patch.object(redis_bus.client, 'pubsub') as mock_pubsub:
            mock_pubsub_instance = Mock()
            mock_pubsub.return_value = mock_pubsub_instance
            
            pubsub = redis_bus.subscribe_to_events(channel="github_events")
            
            assert pubsub == mock_pubsub_instance
            mock_pubsub_instance.subscribe.assert_called_once_with("github_events")
    
    def test_start_subscription_thread(self, redis_bus):
        """Test starting subscription thread."""
        with patch.object(redis_bus, '_subscription_worker') as mock_worker:
            with patch('un.redis_bus.threading.Thread') as mock_thread:
                mock_thread_instance = Mock()
                mock_thread.return_value = mock_thread_instance
                
                redis_bus.start_subscription_thread()
                
                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()
                assert redis_bus.subscription_thread is not None
    
    def test_stop_subscription_thread(self, redis_bus):
        """Test stopping subscription thread."""
        # Mock the subscription thread
        mock_thread = Mock()
        redis_bus.subscription_thread = mock_thread
        
        redis_bus.stop_subscription_thread()
        
        mock_thread.join.assert_called_once()
        assert redis_bus.subscription_thread is None
    
    def test_get_event_history(self, redis_bus):
        """Test getting event history."""
        sample_events = [
            create_github_repo_updated_event("user/repo1", "abc123", "main", 5),
            create_github_repo_updated_event("user/repo2", "def456", "develop", 3)
        ]
        
        with patch.object(redis_bus.client, 'zrange') as mock_zrange:
            mock_zrange.return_value = [event.to_json() for event in sample_events]
            
            history = redis_bus.get_event_history(limit=10)
            
            assert len(history) == 2
            assert all(isinstance(event, Event) for event in history)
            assert history[0].data["repository"] == "user/repo1"
            assert history[1].data["repository"] == "user/repo2"
            mock_zrange.assert_called_once_with("42_events_history", 0, 9, desc=True)
    
    def test_get_event_history_with_time_range(self, redis_bus):
        """Test getting event history with time range."""
        with patch.object(redis_bus.client, 'zrangebyscore') as mock_zrangebyscore:
            mock_zrangebyscore.return_value = []
            
            start_time = datetime.now(timezone.utc)
            end_time = datetime.now(timezone.utc)
            
            history = redis_bus.get_event_history(
                start_time=start_time,
                end_time=end_time
            )
            
            assert isinstance(history, list)
            mock_zrangebyscore.assert_called_once()
    
    def test_get_event_stats(self, redis_bus):
        """Test getting event statistics."""
        with patch.object(redis_bus.client, 'zcard') as mock_zcard:
            with patch.object(redis_bus.client, 'zrange') as mock_zrange:
                mock_zcard.return_value = 100
                mock_zrange.return_value = []
                
                stats = redis_bus.get_event_stats()
                
                assert isinstance(stats, dict)
                assert "total_events" in stats
                assert "event_types" in stats
                assert stats["total_events"] == 100
                mock_zcard.assert_called_once_with("42_events_history")
    
    def test_clear_history(self, redis_bus):
        """Test clearing event history."""
        with patch.object(redis_bus.client, 'delete') as mock_delete:
            mock_delete.return_value = 1
            
            result = redis_bus.clear_history()
            
            assert result is True
            mock_delete.assert_called_once_with("42_events_history")
    
    def test_connection_error_handling(self):
        """Test handling of Redis connection errors."""
        with patch('un.redis_bus.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                RedisBus(host="invalid-host")
    
    def test_publish_event_connection_error(self, redis_bus, sample_event):
        """Test publishing event with connection error."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            mock_publish.side_effect = Exception("Redis error")
            
            with pytest.raises(Exception):
                redis_bus.publish_event(sample_event)
    
    def test_subscription_worker(self, redis_bus):
        """Test subscription worker functionality."""
        sample_event = create_github_repo_updated_event("user/repo", "abc123", "main", 5)
        
        with patch.object(redis_bus.client, 'pubsub') as mock_pubsub:
            mock_pubsub_instance = Mock()
            mock_pubsub.return_value = mock_pubsub_instance
            
            # Mock the message stream
            mock_message = Mock()
            mock_message.type = 'message'
            mock_message.channel = b'42_events'
            mock_message.data = sample_event.to_json().encode()
            
            mock_pubsub_instance.listen.return_value = [mock_message]
            
            # Mock the callback
            callback_called = False
            def test_callback(event):
                nonlocal callback_called
                callback_called = True
                assert isinstance(event, Event)
                assert event.data["repository"] == "user/repo"
            
            redis_bus.start_subscription_thread(callback=test_callback)
            
            # Simulate running the worker
            redis_bus._subscription_worker(test_callback)
            
            assert callback_called
    
    def test_event_serialization_in_publish(self, redis_bus, sample_event):
        """Test that events are properly serialized when published."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            mock_publish.return_value = 1
            
            redis_bus.publish_event(sample_event)
            
            call_args = mock_publish.call_args
            published_data = call_args[0][1]
            
            # Verify the published data is valid JSON
            parsed_event = Event.from_json(published_data)
            assert parsed_event.event_type == sample_event.event_type
            assert parsed_event.data["repository"] == sample_event.data["repository"]
    
    def test_multiple_channels(self, redis_bus, sample_event):
        """Test publishing to multiple channels."""
        with patch.object(redis_bus.client, 'publish') as mock_publish:
            mock_publish.return_value = 2  # 2 channels
            
            result = redis_bus.publish_event(sample_event, channels=["42_events", "github_events"])
            
            assert result == 2
            assert mock_publish.call_count == 2
    
    def test_event_filtering(self, redis_bus):
        """Test event filtering in history."""
        with patch.object(redis_bus.client, 'zrange') as mock_zrange:
            mock_zrange.return_value = []
            
            history = redis_bus.get_event_history(
                event_type=EventType.GITHUB_REPO_UPDATED
            )
            
            assert isinstance(history, list)
            mock_zrange.assert_called_once()
    
    def test_redis_bus_with_custom_config(self):
        """Test Redis bus with custom configuration."""
        with patch('un.redis_bus.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            bus = RedisBus(
                host="custom-host",
                port=6380,
                db=1,
                password="secret",
                decode_responses=True
            )
            
            assert bus is not None
            mock_redis.assert_called_once_with(
                host="custom-host",
                port=6380,
                db=1,
                password="secret",
                decode_responses=True
            )
    
    def test_event_history_pagination(self, redis_bus):
        """Test event history pagination."""
        with patch.object(redis_bus.client, 'zrange') as mock_zrange:
            mock_zrange.return_value = []
            
            # Test with offset and limit
            history = redis_bus.get_event_history(offset=10, limit=20)
            
            assert isinstance(history, list)
            mock_zrange.assert_called_once_with("42_events_history", 10, 29, desc=True)
    
    def test_event_history_time_based(self, redis_bus):
        """Test time-based event history retrieval."""
        with patch.object(redis_bus.client, 'zrangebyscore') as mock_zrangebyscore:
            mock_zrangebyscore.return_value = []
            
            start_time = datetime.now(timezone.utc)
            end_time = datetime.now(timezone.utc)
            
            history = redis_bus.get_event_history(
                start_time=start_time,
                end_time=end_time,
                limit=50
            )
            
            assert isinstance(history, list)
            mock_zrangebyscore.assert_called_once() 