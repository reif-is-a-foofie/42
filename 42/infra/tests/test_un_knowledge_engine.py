"""
Tests for 42.un Knowledge Engine

Tests the abstract knowledge engine, sources, events, and processing pipeline.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

# Add the 42 package to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from un.knowledge_engine import (
    KnowledgeEngine, KnowledgeSource, KnowledgeEvent, KnowledgeDocument,
    SourceType, DomainType, TriggerType, RSSFetcher, APIFetcher,
    KnowledgeNormalizer, KnowledgeTrigger
)
from un.events import Event, EventType
from un.redis_bus import RedisBus


class TestKnowledgeSource:
    """Test KnowledgeSource entity."""
    
    def test_knowledge_source_creation(self):
        """Test creating a knowledge source."""
        source = KnowledgeSource(
            id="test_source",
            name="Test Source",
            type=SourceType.RSS,
            domain=DomainType.WEATHER,
            url="https://example.com/feed",
            frequency="5min",
            parser="xml"
        )
        
        assert source.id == "test_source"
        assert source.name == "Test Source"
        assert source.type == SourceType.RSS
        assert source.domain == DomainType.WEATHER
        assert source.url == "https://example.com/feed"
        assert source.vectorize is True
        assert source.active is True
    
    def test_knowledge_source_to_dict(self):
        """Test converting source to dictionary."""
        source = KnowledgeSource(
            id="test_source",
            name="Test Source",
            type=SourceType.API,
            domain=DomainType.CRISIS,
            url="https://api.example.com",
            frequency="1hour",
            parser="json"
        )
        
        data = source.to_dict()
        assert data["id"] == "test_source"
        assert data["type"] == "api"
        assert data["domain"] == "crisis"
    
    def test_knowledge_source_from_dict(self):
        """Test creating source from dictionary."""
        data = {
            "id": "test_source",
            "name": "Test Source",
            "type": "rss",
            "domain": "weather",
            "url": "https://example.com/feed",
            "frequency": "5min",
            "parser": "xml",
            "vectorize": True,
            "active": True
        }
        
        source = KnowledgeSource.from_dict(data)
        assert source.type == SourceType.RSS
        assert source.domain == DomainType.WEATHER


class TestKnowledgeEvent:
    """Test KnowledgeEvent entity."""
    
    def test_knowledge_event_creation(self):
        """Test creating a knowledge event."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="earthquake",
            priority=1,
            actions=["dispatch_team", "alert_ops"]
        )
        
        assert event.id == "test_event"
        assert event.name == "Test Event"
        assert event.trigger_type == TriggerType.KEYWORD
        assert event.trigger_value == "earthquake"
        assert event.priority == 1
        assert "dispatch_team" in event.actions
    
    def test_knowledge_event_to_dict(self):
        """Test converting event to dictionary."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.THRESHOLD,
            trigger_value="5.0",
            priority=2
        )
        
        data = event.to_dict()
        assert data["id"] == "test_event"
        assert data["trigger_type"] == "threshold"
        assert data["trigger_value"] == "5.0"
    
    def test_knowledge_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "id": "test_event",
            "name": "Test Event",
            "trigger_type": "pattern",
            "trigger_value": r"magnitude \d+",
            "priority": 3,
            "actions": ["notify_team"]
        }
        
        event = KnowledgeEvent.from_dict(data)
        assert event.trigger_type == TriggerType.PATTERN
        assert event.priority == 3


class TestKnowledgeDocument:
    """Test KnowledgeDocument entity."""
    
    def test_knowledge_document_creation(self):
        """Test creating a knowledge document."""
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Test content",
            metadata={"url": "https://example.com", "title": "Test"}
        )
        
        assert doc.source_id == "test_source"
        assert doc.content == "Test content"
        assert doc.metadata["url"] == "https://example.com"
        assert doc.timestamp is not None
        assert doc.vector_id is None
    
    def test_knowledge_document_to_dict(self):
        """Test converting document to dictionary."""
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Test content",
            metadata={"domain": "weather"}
        )
        
        data = doc.to_dict()
        assert data["source_id"] == "test_source"
        assert data["content"] == "Test content"
        assert "timestamp" in data
    
    def test_knowledge_document_from_dict(self):
        """Test creating document from dictionary."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "source_id": "test_source",
            "content": "Test content",
            "metadata": {"domain": "crisis"},
            "timestamp": timestamp.isoformat(),
            "vector_id": "doc_123"
        }
        
        doc = KnowledgeDocument.from_dict(data)
        assert doc.source_id == "test_source"
        assert doc.vector_id == "doc_123"


class TestKnowledgeNormalizer:
    """Test KnowledgeNormalizer."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        normalizer = KnowledgeNormalizer()
        assert DomainType.WEATHER in normalizer.processors
        assert DomainType.GEOSPATIAL in normalizer.processors
    
    def test_normalize_weather_document(self):
        """Test normalizing weather documents."""
        normalizer = KnowledgeNormalizer()
        
        doc = KnowledgeDocument(
            source_id="weather_source",
            content="Hurricane warning issued for coastal areas",
            metadata={"domain": "weather"}
        )
        
        normalized = normalizer.normalize([doc])
        assert len(normalized) == 1
        assert normalized[0].metadata["alert_type"] == "severe_weather"
        assert normalized[0].metadata["priority"] == "high"
    
    def test_normalize_geospatial_document(self):
        """Test normalizing geospatial documents."""
        normalizer = KnowledgeNormalizer()
        
        doc = KnowledgeDocument(
            source_id="quake_source",
            content="Earthquake magnitude 6.2 detected",
            metadata={"domain": "geospatial"}
        )
        
        normalized = normalizer.normalize([doc])
        assert len(normalized) == 1
        assert normalized[0].metadata["alert_type"] == "earthquake"
        assert normalized[0].metadata["priority"] == "high"
    
    def test_normalize_medical_document(self):
        """Test normalizing medical documents."""
        normalizer = KnowledgeNormalizer()
        
        doc = KnowledgeDocument(
            source_id="health_source",
            content="Disease outbreak reported in region",
            metadata={"domain": "medical"}
        )
        
        normalized = normalizer.normalize([doc])
        assert len(normalized) == 1
        assert normalized[0].metadata["alert_type"] == "health_emergency"
        assert normalized[0].metadata["priority"] == "high"
    
    def test_normalize_generic_document(self):
        """Test normalizing generic documents."""
        normalizer = KnowledgeNormalizer()
        
        doc = KnowledgeDocument(
            source_id="generic_source",
            content="Regular news update",
            metadata={"domain": "research"}
        )
        
        normalized = normalizer.normalize([doc])
        assert len(normalized) == 1
        # Should not have alert_type or priority
        assert "alert_type" not in normalized[0].metadata


class TestKnowledgeTrigger:
    """Test KnowledgeTrigger."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create mock Redis bus."""
        mock = Mock(spec=RedisBus)
        mock.publish_event = AsyncMock()
        return mock
    
    @pytest.fixture
    def trigger(self, mock_redis_bus):
        """Create trigger instance."""
        return KnowledgeTrigger(mock_redis_bus)
    
    def test_trigger_initialization(self, trigger):
        """Test trigger initialization."""
        assert len(trigger.events) == 0
    
    def test_add_event(self, trigger):
        """Test adding events to trigger."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="earthquake"
        )
        
        trigger.add_event(event)
        assert len(trigger.events) == 1
        assert trigger.events[0].id == "test_event"
    
    def test_keyword_trigger_matching(self, trigger):
        """Test keyword trigger matching."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="earthquake"
        )
        trigger.add_event(event)
        
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Earthquake detected in California",
            metadata={"domain": "geospatial"}
        )
        
        matches = trigger.check_triggers(doc)
        assert len(matches) == 1
        assert matches[0].id == "test_event"
    
    def test_threshold_trigger_matching(self, trigger):
        """Test threshold trigger matching."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.THRESHOLD,
            trigger_value="5.0"
        )
        trigger.add_event(event)
        
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Earthquake magnitude 6.2 detected",
            metadata={"domain": "geospatial"}
        )
        
        matches = trigger.check_triggers(doc)
        assert len(matches) == 1
        assert matches[0].id == "test_event"
    
    def test_pattern_trigger_matching(self, trigger):
        """Test pattern trigger matching."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.PATTERN,
            trigger_value=r"magnitude \d+"
        )
        trigger.add_event(event)
        
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Earthquake magnitude 6.2 detected",
            metadata={"domain": "geospatial"}
        )
        
        matches = trigger.check_triggers(doc)
        assert len(matches) == 1
        assert matches[0].id == "test_event"
    
    def test_no_trigger_matching(self, trigger):
        """Test when no triggers match."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="hurricane"
        )
        trigger.add_event(event)
        
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Regular weather update",
            metadata={"domain": "weather"}
        )
        
        matches = trigger.check_triggers(doc)
        assert len(matches) == 0
    
    @pytest.mark.asyncio
    async def test_fire_events(self, trigger, mock_redis_bus):
        """Test firing events to Redis bus."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="earthquake",
            actions=["dispatch_team"]
        )
        trigger.add_event(event)
        
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Earthquake detected",
            metadata={"domain": "geospatial"}
        )
        
        await trigger.fire_events(doc, [event])
        
        # Verify Redis bus was called
        mock_redis_bus.publish_event.assert_called_once()
        call_args = mock_redis_bus.publish_event.call_args[0][0]
        assert call_args.event_type == EventType.KNOWLEDGE_TRIGGER
        assert call_args.data["trigger_id"] == "test_event"


class TestRSSFetcher:
    """Test RSSFetcher."""
    
    @pytest.mark.asyncio
    async def test_rss_fetcher_fetch(self):
        """Test RSS fetcher."""
        source = KnowledgeSource(
            id="test_rss",
            name="Test RSS",
            type=SourceType.RSS,
            domain=DomainType.WEATHER,
            url="https://example.com/feed",
            frequency="5min",
            parser="xml"
        )
        
        # Mock aiohttp session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Item</title>
                    <description>Test description</description>
                    <link>https://example.com/item</link>
                </item>
            </channel>
        </rss>
        """)
        
        # Create async context manager mock
        mock_context = Mock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context
        
        fetcher = RSSFetcher(mock_session)
        documents = await fetcher.fetch(source)
        
        assert len(documents) == 1
        assert documents[0].source_id == "test_rss"
        assert "Test description" in documents[0].content


class TestAPIFetcher:
    """Test APIFetcher."""
    
    @pytest.mark.asyncio
    async def test_api_fetcher_fetch_json(self):
        """Test API fetcher with JSON response."""
        source = KnowledgeSource(
            id="test_api",
            name="Test API",
            type=SourceType.API,
            domain=DomainType.WEATHER,
            url="https://api.example.com/weather",
            frequency="5min",
            parser="json"
        )
        
        # Mock aiohttp session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = AsyncMock(return_value='{"features": [{"properties": {"event": "Hurricane", "headline": "Warning"}}]}')
        
        # Create async context manager mock
        mock_context = Mock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context
        
        fetcher = APIFetcher(mock_session)
        documents = await fetcher.fetch(source)
        
        assert len(documents) == 1
        assert documents[0].source_id == "test_api"
        assert "Hurricane" in documents[0].content
    
    @pytest.mark.asyncio
    async def test_api_fetcher_fetch_text(self):
        """Test API fetcher with text response."""
        source = KnowledgeSource(
            id="test_api",
            name="Test API",
            type=SourceType.API,
            domain=DomainType.CRISIS,
            url="https://api.example.com/alerts",
            frequency="5min",
            parser="text"
        )
        
        # Mock aiohttp session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = AsyncMock(return_value="Emergency alert: Test crisis")
        
        # Create async context manager mock
        mock_context = Mock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context
        
        fetcher = APIFetcher(mock_session)
        documents = await fetcher.fetch(source)
        
        assert len(documents) == 1
        assert documents[0].source_id == "test_api"
        assert "Emergency alert" in documents[0].content


class TestKnowledgeEngine:
    """Test KnowledgeEngine."""
    
    @pytest.fixture
    def mock_redis_bus(self):
        """Create mock Redis bus."""
        mock = Mock(spec=RedisBus)
        mock.publish_event = AsyncMock()
        return mock
    
    @pytest.fixture
    def engine(self, mock_redis_bus):
        """Create knowledge engine instance."""
        return KnowledgeEngine(mock_redis_bus)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert len(engine.sources) == 0
        assert len(engine.events) == 0
        assert SourceType.RSS in engine.fetchers
        assert SourceType.API in engine.fetchers
    
    def test_add_source(self, engine):
        """Test adding sources to engine."""
        source = KnowledgeSource(
            id="test_source",
            name="Test Source",
            type=SourceType.RSS,
            domain=DomainType.WEATHER,
            url="https://example.com/feed",
            frequency="5min",
            parser="xml"
        )
        
        engine.add_source(source)
        assert len(engine.sources) == 1
        assert engine.sources[0].id == "test_source"
    
    def test_add_event(self, engine):
        """Test adding events to engine."""
        event = KnowledgeEvent(
            id="test_event",
            name="Test Event",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="earthquake"
        )
        
        engine.add_event(event)
        assert len(engine.events) == 1
        assert engine.events[0].id == "test_event"
        assert len(engine.trigger.events) == 1
    
    @pytest.mark.asyncio
    async def test_run_fetch_cycle_no_sources(self, engine):
        """Test running fetch cycle with no sources."""
        await engine.run_fetch_cycle()
        # Should complete without error
    
    @pytest.mark.asyncio
    async def test_run_fetch_cycle_with_sources(self, engine):
        """Test running fetch cycle with sources."""
        source = KnowledgeSource(
            id="test_source",
            name="Test Source",
            type=SourceType.RSS,
            domain=DomainType.WEATHER,
            url="https://example.com/feed",
            frequency="5min",
            parser="xml"
        )
        engine.add_source(source)
        
        # Mock the fetcher to return documents
        with patch.object(RSSFetcher, 'fetch', return_value=[]):
            await engine.run_fetch_cycle()
            # Should complete without error
    
    @pytest.mark.asyncio
    async def test_store_documents(self, engine, mock_redis_bus):
        """Test storing documents."""
        doc = KnowledgeDocument(
            source_id="test_source",
            content="Test content",
            metadata={"domain": "weather"}
        )
        
        await engine._store_documents([doc])
        
        # Verify Redis bus was called
        mock_redis_bus.publish_event.assert_called_once()
        call_args = mock_redis_bus.publish_event.call_args[0][0]
        assert call_args.event_type == EventType.KNOWLEDGE_DOCUMENT
        assert call_args.data["source_id"] == "test_source"


if __name__ == "__main__":
    pytest.main([__file__]) 