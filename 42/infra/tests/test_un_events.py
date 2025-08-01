"""Tests for the 42.un events module."""

import pytest
import json
import sys
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from un.events import Event, EventType, create_github_repo_updated_event, create_file_ingested_event, create_embedding_completed_event, create_task_completed_event


class TestEvents:
    """Test events functionality."""
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            source="github",
            timestamp=datetime.now(timezone.utc),
            data={
                "repository": "user/repo",
                "commit_hash": "abc123",
                "branch": "main"
            },
            metadata={
                "priority": "high",
                "tags": ["code", "update"]
            }
        )
    
    def test_event_initialization(self, sample_event):
        """Test event object initialization."""
        assert sample_event.event_type == EventType.GITHUB_REPO_UPDATED
        assert sample_event.source == "github"
        assert isinstance(sample_event.timestamp, datetime)
        assert sample_event.data["repository"] == "user/repo"
        assert sample_event.data["commit_hash"] == "abc123"
        assert sample_event.metadata["priority"] == "high"
    
    def test_event_type_enum(self):
        """Test event type enum values."""
        assert EventType.GITHUB_REPO_UPDATED == "github_repo_updated"
        assert EventType.FILE_INGESTED == "file_ingested"
        assert EventType.EMBEDDING_COMPLETED == "embedding_completed"
        assert EventType.TASK_COMPLETED == "task_completed"
        assert EventType.SOURCE_SCANNED == "source_scanned"
        assert EventType.TASK_PRIORITIZED == "task_prioritized"
        assert EventType.WORKER_STARTED == "worker_started"
        assert EventType.WORKER_COMPLETED == "worker_completed"
        assert EventType.ERROR_OCCURRED == "error_occurred"
    
    def test_event_to_dict(self, sample_event):
        """Test converting event to dictionary."""
        event_dict = sample_event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == "github_repo_updated"
        assert event_dict["source"] == "github"
        assert "timestamp" in event_dict
        assert event_dict["data"]["repository"] == "user/repo"
        assert event_dict["metadata"]["priority"] == "high"
    
    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        event_dict = {
            "event_type": "file_ingested",
            "source": "filesystem",
            "timestamp": "2023-01-01T12:00:00Z",
            "data": {
                "file_path": "/path/to/file.txt",
                "file_size": 1024
            },
            "metadata": {
                "priority": "medium",
                "tags": ["file", "ingest"]
            }
        }
        
        event = Event.from_dict(event_dict)
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.FILE_INGESTED
        assert event.source == "filesystem"
        assert event.data["file_path"] == "/path/to/file.txt"
        assert event.data["file_size"] == 1024
        assert event.metadata["priority"] == "medium"
    
    def test_event_to_json(self, sample_event):
        """Test converting event to JSON."""
        event_json = sample_event.to_json()
        
        assert isinstance(event_json, str)
        event_dict = json.loads(event_json)
        assert event_dict["event_type"] == "github_repo_updated"
        assert event_dict["source"] == "github"
    
    def test_event_from_json(self):
        """Test creating event from JSON."""
        event_json = '''
        {
            "event_type": "embedding_completed",
            "source": "embedding_engine",
            "timestamp": "2023-01-01T12:00:00Z",
            "data": {
                "chunks_processed": 100,
                "model_used": "bge-small-en"
            },
            "metadata": {
                "priority": "low",
                "tags": ["embedding", "complete"]
            }
        }
        '''
        
        event = Event.from_json(event_json)
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.EMBEDDING_COMPLETED
        assert event.source == "embedding_engine"
        assert event.data["chunks_processed"] == 100
        assert event.data["model_used"] == "bge-small-en"
    
    def test_create_github_repo_updated_event(self):
        """Test creating GitHub repo updated event."""
        event = create_github_repo_updated_event(
            repository="user/repo",
            commit_hash="abc123",
            branch="main",
            files_changed=5
        )
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.GITHUB_REPO_UPDATED
        assert event.source == "github"
        assert event.data["repository"] == "user/repo"
        assert event.data["commit_hash"] == "abc123"
        assert event.data["branch"] == "main"
        assert event.data["files_changed"] == 5
    
    def test_create_file_ingested_event(self):
        """Test creating file ingested event."""
        event = create_file_ingested_event(
            file_path="/path/to/file.py",
            file_size=1024,
            chunks_created=3,
            file_type="python"
        )
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.FILE_INGESTED
        assert event.source == "filesystem"
        assert event.data["file_path"] == "/path/to/file.py"
        assert event.data["file_size"] == 1024
        assert event.data["chunks_created"] == 3
        assert event.data["file_type"] == "python"
    
    def test_create_embedding_completed_event(self):
        """Test creating embedding completed event."""
        event = create_embedding_completed_event(
            chunks_processed=100,
            model_used="bge-small-en",
            processing_time=30.5,
            vector_dimension=384
        )
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.EMBEDDING_COMPLETED
        assert event.source == "embedding_engine"
        assert event.data["chunks_processed"] == 100
        assert event.data["model_used"] == "bge-small-en"
        assert event.data["processing_time"] == 30.5
        assert event.data["vector_dimension"] == 384
    
    def test_create_task_completed_event(self):
        """Test creating task completed event."""
        event = create_task_completed_event(
            task_id="task_123",
            task_type="github_extraction",
            status="completed",
            duration=120.5,
            result_summary="Extracted 100 chunks"
        )
        
        assert isinstance(event, Event)
        assert event.event_type == EventType.TASK_COMPLETED
        assert event.source == "background_worker"
        assert event.data["task_id"] == "task_123"
        assert event.data["task_type"] == "github_extraction"
        assert event.data["status"] == "completed"
        assert event.data["duration"] == 120.5
        assert event.data["result_summary"] == "Extracted 100 chunks"
    
    def test_event_timestamp_handling(self):
        """Test event timestamp handling."""
        # Test with string timestamp
        event_dict = {
            "event_type": "test_event",
            "source": "test",
            "timestamp": "2023-01-01T12:00:00Z",
            "data": {},
            "metadata": {}
        }
        
        event = Event.from_dict(event_dict)
        assert isinstance(event.timestamp, datetime)
        
        # Test with datetime timestamp
        now = datetime.now(timezone.utc)
        event = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            source="test",
            timestamp=now,
            data={},
            metadata={}
        )
        
        assert event.timestamp == now
    
    def test_event_validation(self):
        """Test event validation."""
        # Test missing required fields
        with pytest.raises(TypeError):
            Event()
        
        # Test invalid event type
        with pytest.raises(ValueError):
            Event(
                event_type="invalid_type",
                source="test",
                timestamp=datetime.now(timezone.utc),
                data={},
                metadata={}
            )
    
    def test_event_equality(self):
        """Test event equality comparison."""
        timestamp = datetime.now(timezone.utc)
        
        event1 = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            source="github",
            timestamp=timestamp,
            data={"repository": "user/repo"},
            metadata={"priority": "high"}
        )
        
        event2 = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            source="github",
            timestamp=timestamp,
            data={"repository": "user/repo"},
            metadata={"priority": "high"}
        )
        
        assert event1 == event2
        
        # Test inequality
        event3 = Event(
            event_type=EventType.FILE_INGESTED,
            source="filesystem",
            timestamp=timestamp,
            data={"file_path": "/test.txt"},
            metadata={}
        )
        
        assert event1 != event3
    
    def test_event_repr(self, sample_event):
        """Test event string representation."""
        event_repr = repr(sample_event)
        assert isinstance(event_repr, str)
        assert "Event" in event_repr
        assert "github_repo_updated" in event_repr
        assert "github" in event_repr
    
    def test_event_serialization_roundtrip(self, sample_event):
        """Test event serialization and deserialization roundtrip."""
        # Convert to dict and back
        event_dict = sample_event.to_dict()
        reconstructed_event = Event.from_dict(event_dict)
        
        assert reconstructed_event == sample_event
        
        # Convert to JSON and back
        event_json = sample_event.to_json()
        reconstructed_event = Event.from_json(event_json)
        
        assert reconstructed_event == sample_event
    
    def test_event_with_complex_data(self):
        """Test event with complex nested data."""
        complex_data = {
            "repository": {
                "name": "user/repo",
                "url": "https://github.com/user/repo",
                "branches": ["main", "develop"]
            },
            "changes": [
                {"file": "src/main.py", "status": "modified"},
                {"file": "tests/test_main.py", "status": "added"}
            ],
            "statistics": {
                "files_changed": 5,
                "lines_added": 100,
                "lines_deleted": 20
            }
        }
        
        event = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            source="github",
            timestamp=datetime.now(timezone.utc),
            data=complex_data,
            metadata={"priority": "high", "tags": ["complex", "data"]}
        )
        
        # Test serialization
        event_dict = event.to_dict()
        event_json = event.to_json()
        
        assert isinstance(event_dict, dict)
        assert isinstance(event_json, str)
        
        # Test deserialization
        reconstructed_event = Event.from_dict(event_dict)
        assert reconstructed_event.data["repository"]["name"] == "user/repo"
        assert len(reconstructed_event.data["changes"]) == 2
        assert reconstructed_event.data["statistics"]["files_changed"] == 5 