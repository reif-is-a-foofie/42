"""
42.un - Reflex and ingestion phase

This module implements Phase un of the 42 masterplan:
- Redis event bus for fast reactions
- Source scanner for constant monitoring
- Task prioritizer with Bayesian scoring
- Background worker for async execution
"""

from .redis_bus import RedisBus
from .source_scanner import (
    SourceScanner,
    GitHubScanner,
    FileSystemScanner,
    RSSFeedScanner,
    APIEndpointScanner,
    SourceScannerOrchestrator
)
from .webhook_handlers import (
    WebhookValidator,
    GitHubWebhookHandler,
    GenericWebhookHandler,
    WebhookManager
)
from .knowledge_engine import (
    KnowledgeEngine,
    KnowledgeSource,
    KnowledgeEvent,
    KnowledgeDocument,
    SourceType,
    DomainType,
    TriggerType,
    RSSFetcher,
    APIFetcher
)
from .mission_config import MISSION_CONFIG
from .events import Event, EventType

__all__ = [
    'RedisBus',
    'SourceScanner',
    'GitHubScanner',
    'FileSystemScanner',
    'RSSFeedScanner',
    'APIEndpointScanner',
    'SourceScannerOrchestrator',
    'WebhookValidator',
    'GitHubWebhookHandler',
    'GenericWebhookHandler',
    'WebhookManager',
    'KnowledgeEngine',
    'KnowledgeSource',
    'KnowledgeEvent',
    'KnowledgeDocument',
    'SourceType',
    'DomainType',
    'TriggerType',
    'RSSFetcher',
    'APIFetcher',
    'MISSION_CONFIG',
    'Event',
    'EventType'
] 