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
    'Event',
    'EventType'
] 