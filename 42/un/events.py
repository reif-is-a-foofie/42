"""
Event definitions for 42.un

Defines the event types and schemas used throughout the 42.un system.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


class EventType(Enum):
    """Event types for the 42.un system."""
    
    # GitHub events
    GITHUB_REPO_UPDATED = "github.repo.updated"
    GITHUB_WEBHOOK_RECEIVED = "github.webhook.received"
    
    # File system events
    FILE_INGESTED = "file.ingested"
    FILE_CHANGED = "file.changed"
    FILE_DELETED = "file.deleted"
    
    # Processing events
    EMBEDDING_COMPLETED = "embedding.completed"
    EMBEDDING_FAILED = "embedding.failed"
    CLUSTER_UPDATED = "cluster.updated"
    CLUSTER_FAILED = "cluster.failed"
    
    # LLM events
    LLM_QUERY_COMPLETED = "llm.query.completed"
    LLM_QUERY_FAILED = "llm.query.failed"
    
    # Task events
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "health.check"
    
    # Knowledge engine events
    KNOWLEDGE_TRIGGER = "knowledge.trigger"
    KNOWLEDGE_DOCUMENT = "knowledge.document"


@dataclass
class Event:
    """Event object for 42.un system."""
    
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    event_id: Optional[str] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = f"{self.event_type.value}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            event_id=data.get("event_id")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Event factory functions for common events
def create_github_repo_updated_event(repo_url: str, commit_hash: str, **kwargs) -> Event:
    """Create a GitHub repository updated event."""
    return Event(
        event_type=EventType.GITHUB_REPO_UPDATED,
        data={
            "repo_url": repo_url,
            "commit_hash": commit_hash,
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="github"
    )


def create_file_ingested_event(file_path: str, chunk_count: int, **kwargs) -> Event:
    """Create a file ingested event."""
    return Event(
        event_type=EventType.FILE_INGESTED,
        data={
            "file_path": file_path,
            "chunk_count": chunk_count,
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="file_scanner"
    )


def create_embedding_completed_event(batch_size: int, duration: float, **kwargs) -> Event:
    """Create an embedding completed event."""
    return Event(
        event_type=EventType.EMBEDDING_COMPLETED,
        data={
            "batch_size": batch_size,
            "duration": duration,
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="embedding_engine"
    )


def create_task_completed_event(task_id: str, task_type: str, duration: float, **kwargs) -> Event:
    """Create a task completed event."""
    return Event(
        event_type=EventType.TASK_COMPLETED,
        data={
            "task_id": task_id,
            "task_type": task_type,
            "duration": duration,
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="background_worker"
    )


def create_source_discovered_event(url: str, source_type: str, domain: str, **kwargs) -> Event:
    """Create a source discovered event."""
    return Event(
        event_type=EventType.KNOWLEDGE_DOCUMENT,
        data={
            "url": url,
            "source_type": source_type,
            "domain": domain,
            "discovery_method": "autonomous_scanner",
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="autonomous_scanner"
    )


def create_crawl_completed_event(url: str, pages_crawled: int, sources_found: int, **kwargs) -> Event:
    """Create a crawl completed event."""
    return Event(
        event_type=EventType.KNOWLEDGE_DOCUMENT,
        data={
            "url": url,
            "pages_crawled": pages_crawled,
            "sources_found": sources_found,
            "crawl_type": "autonomous",
            **kwargs
        },
        timestamp=datetime.utcnow(),
        source="autonomous_scanner"
    ) 