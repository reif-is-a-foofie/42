"""
Abstract Knowledge Engine for 42.un

Universal intelligence organ that can ingest, normalize, and trigger on any knowledge source.
Replaces mission-specific ingestion with a scalable, pluggable framework.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import feedparser
from loguru import logger

from .events import Event, EventType
from .redis_bus import RedisBus


class SourceType(Enum):
    """Types of knowledge sources."""
    RSS = "rss"
    API = "api"
    GITHUB = "github"
    FILESYSTEM = "filesystem"
    CRAWLER = "crawler"
    PDF = "pdf"
    DATABASE = "database"
    WEBHOOK = "webhook"


class DomainType(Enum):
    """Knowledge domains."""
    WEATHER = "weather"
    MEDICAL = "medical"
    FINANCE = "finance"
    RESEARCH = "research"
    LOGISTICS = "logistics"
    GEOSPATIAL = "geospatial"
    HUMANITARIAN = "humanitarian"
    INTELLIGENCE = "intelligence"
    TECHNOLOGY = "technology"
    CRISIS = "crisis"


class TriggerType(Enum):
    """Types of triggers."""
    KEYWORD = "keyword"
    THRESHOLD = "threshold"
    LLM = "llm"
    VECTOR_SIMILARITY = "vector_similarity"
    PATTERN = "pattern"
    ANOMALY = "anomaly"


@dataclass
class KnowledgeSource:
    """Universal knowledge source entity."""
    id: str
    name: str
    type: SourceType
    domain: DomainType
    url: str
    frequency: str  # "5min", "1hour", "daily"
    parser: str  # "xml", "json", "text", "github", "filesystem"
    auth_ref: Optional[str] = None
    vectorize: bool = True
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        data['domain'] = self.domain.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeSource':
        """Create from dictionary."""
        data['type'] = SourceType(data['type'])
        data['domain'] = DomainType(data['domain'])
        return cls(**data)


@dataclass
class KnowledgeEvent:
    """Abstract trigger for knowledge events."""
    id: str
    name: str
    trigger_type: TriggerType
    trigger_value: str
    priority: int = 1
    actions: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['trigger_type'] = self.trigger_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEvent':
        """Create from dictionary."""
        data['trigger_type'] = TriggerType(data['trigger_type'])
        return cls(**data)


@dataclass
class KnowledgeDocument:
    """Canonical knowledge document model."""
    source_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime = None
    vector_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeDocument':
        """Create from dictionary."""
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class KnowledgeFetcher:
    """Abstract fetcher for knowledge sources."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    
    async def fetch(self, source: KnowledgeSource) -> List[KnowledgeDocument]:
        """Fetch documents from a source."""
        raise NotImplementedError


class RSSFetcher(KnowledgeFetcher):
    """Fetch from RSS feeds."""
    
    async def fetch(self, source: KnowledgeSource) -> List[KnowledgeDocument]:
        """Fetch RSS feed content."""
        try:
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch RSS: {source.url} - {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                documents = []
                for entry in feed.entries:
                    doc = KnowledgeDocument(
                        source_id=source.id,
                        content=entry.get('summary', entry.get('title', '')),
                        metadata={
                            'url': entry.get('link', ''),
                            'title': entry.get('title', ''),
                            'published': entry.get('published', ''),
                            'author': entry.get('author', ''),
                            'domain': source.domain.value
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"Fetched {len(documents)} documents from RSS: {source.name}")
                return documents
                
        except Exception as e:
            logger.error(f"Error fetching RSS {source.name}: {e}")
            return []


class APIFetcher(KnowledgeFetcher):
    """Fetch from API endpoints."""
    
    async def fetch(self, source: KnowledgeSource) -> List[KnowledgeDocument]:
        """Fetch API content."""
        try:
            headers = source.metadata.get('headers', {}) if source.metadata else {}
            
            async with self.session.get(source.url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch API: {source.url} - {response.status}")
                    return []
                
                content = await response.text()
                
                # Parse based on content type
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = json.loads(content)
                    # Extract relevant fields based on domain
                    extracted_content = self._extract_api_content(data, source.domain)
                else:
                    extracted_content = content
                
                doc = KnowledgeDocument(
                    source_id=source.id,
                    content=extracted_content,
                    metadata={
                        'url': source.url,
                        'content_type': response.headers.get('content-type', ''),
                        'domain': source.domain.value,
                        'raw_data': data if 'data' in locals() else None
                    }
                )
                
                logger.info(f"Fetched 1 document from API: {source.name}")
                return [doc]
                
        except Exception as e:
            logger.error(f"Error fetching API {source.name}: {e}")
            return []
    
    def _extract_api_content(self, data: Dict[str, Any], domain: DomainType) -> str:
        """Extract relevant content based on domain."""
        if domain == DomainType.WEATHER:
            # Extract weather alerts
            if 'features' in data:
                alerts = []
                for feature in data['features']:
                    props = feature.get('properties', {})
                    alerts.append(f"{props.get('event', 'Alert')}: {props.get('headline', '')}")
                return '\n'.join(alerts)
            return str(data)
        
        elif domain == DomainType.GEOSPATIAL:
            # Extract earthquake data
            if 'features' in data:
                quakes = []
                for feature in data['features']:
                    props = feature.get('properties', {})
                    quakes.append(f"Magnitude {props.get('mag', 'N/A')} at {props.get('place', 'Unknown')}")
                return '\n'.join(quakes)
            return str(data)
        
        else:
            return str(data)


class KnowledgeNormalizer:
    """Normalize knowledge documents."""
    
    def __init__(self):
        self.processors = {
            DomainType.WEATHER: self._process_weather,
            DomainType.GEOSPATIAL: self._process_geospatial,
            DomainType.MEDICAL: self._process_medical,
            DomainType.CRISIS: self._process_crisis,
        }
    
    def normalize(self, documents: List[KnowledgeDocument]) -> List[KnowledgeDocument]:
        """Normalize documents based on their domain."""
        normalized = []
        
        for doc in documents:
            domain = DomainType(doc.metadata.get('domain', 'research'))
            processor = self.processors.get(domain, self._process_generic)
            
            normalized_doc = processor(doc)
            if normalized_doc:
                normalized.append(normalized_doc)
        
        return normalized
    
    def _process_weather(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Process weather-related documents."""
        # Extract key weather information
        content = doc.content.lower()
        if any(word in content for word in ['hurricane', 'tornado', 'storm', 'flood']):
            doc.metadata['alert_type'] = 'severe_weather'
            doc.metadata['priority'] = 'high'
        return doc
    
    def _process_geospatial(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Process geospatial documents."""
        # Extract location and magnitude information
        content = doc.content.lower()
        if 'earthquake' in content:
            doc.metadata['alert_type'] = 'earthquake'
            doc.metadata['priority'] = 'high'
        return doc
    
    def _process_medical(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Process medical documents."""
        # Extract disease outbreak information
        content = doc.content.lower()
        if any(word in content for word in ['outbreak', 'epidemic', 'pandemic', 'disease']):
            doc.metadata['alert_type'] = 'health_emergency'
            doc.metadata['priority'] = 'high'
        return doc
    
    def _process_crisis(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Process crisis documents."""
        # Extract crisis indicators
        content = doc.content.lower()
        if any(word in content for word in ['disaster', 'emergency', 'crisis', 'evacuation']):
            doc.metadata['alert_type'] = 'crisis'
            doc.metadata['priority'] = 'critical'
        return doc
    
    def _process_generic(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Process generic documents."""
        return doc


class KnowledgeTrigger:
    """Trigger system for knowledge events."""
    
    def __init__(self, redis_bus: RedisBus):
        self.redis_bus = redis_bus
        self.events: List[KnowledgeEvent] = []
    
    def add_event(self, event: KnowledgeEvent):
        """Add a trigger event."""
        self.events.append(event)
    
    def check_triggers(self, document: KnowledgeDocument) -> List[KnowledgeEvent]:
        """Check if document triggers any events."""
        triggered_events = []
        
        for event in self.events:
            if self._matches_trigger(document, event):
                triggered_events.append(event)
        
        return triggered_events
    
    def _matches_trigger(self, document: KnowledgeDocument, event: KnowledgeEvent) -> bool:
        """Check if document matches trigger."""
        content = document.content.lower()
        
        if event.trigger_type == TriggerType.KEYWORD:
            return event.trigger_value.lower() in content
        
        elif event.trigger_type == TriggerType.THRESHOLD:
            # For numeric thresholds (e.g., earthquake magnitude)
            try:
                # Extract numeric value from content
                import re
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    value = float(numbers[0])
                    threshold = float(event.trigger_value)
                    return value >= threshold
            except (ValueError, IndexError):
                pass
            return False
        
        elif event.trigger_type == TriggerType.PATTERN:
            # For regex patterns
            import re
            return bool(re.search(event.trigger_value, content, re.IGNORECASE))
        
        return False
    
    async def fire_events(self, document: KnowledgeDocument, events: List[KnowledgeEvent]):
        """Fire triggered events to Redis bus."""
        for event in events:
            # Create event for Redis bus
            event_data = {
                'document_id': document.source_id,
                'trigger_id': event.id,
                'content': document.content[:200],  # First 200 chars
                'metadata': document.metadata,
                'actions': event.actions
            }
            
            redis_event = Event(
                event_type=EventType.KNOWLEDGE_TRIGGER,
                data=event_data,
                timestamp=datetime.now(timezone.utc),
                source=f"knowledge_engine:{event.id}"
            )
            
            # Publish to Redis bus (non-async)
            self.redis_bus.publish_event(redis_event)
            logger.info(f"Fired trigger: {event.name} for document: {document.source_id}")


class KnowledgeEngine:
    """Main knowledge engine orchestrator."""
    
    def __init__(self, redis_bus: RedisBus):
        self.redis_bus = redis_bus
        self.sources: List[KnowledgeSource] = []
        self.events: List[KnowledgeEvent] = []
        self.normalizer = KnowledgeNormalizer()
        self.trigger = KnowledgeTrigger(redis_bus)
        self.fetchers = {
            SourceType.RSS: RSSFetcher,
            SourceType.API: APIFetcher,
        }
        
        # Note: Using JobManager for storage instead of direct embedding/vector store access
    
    def add_source(self, source: KnowledgeSource):
        """Add a knowledge source."""
        self.sources.append(source)
        logger.info(f"Added knowledge source: {source.name}")
    
    def add_event(self, event: KnowledgeEvent):
        """Add a trigger event."""
        self.events.append(event)
        self.trigger.add_event(event)
        logger.info(f"Added trigger event: {event.name}")
    
    def register_fetcher(self, source_type: SourceType, fetcher_class):
        """Register a new fetcher for a source type."""
        self.fetchers[source_type] = fetcher_class
        logger.info(f"Registered fetcher for {source_type.value}")
    
    async def run_fetch_cycle(self):
        """Run a complete fetch cycle."""
        async with aiohttp.ClientSession() as session:
            for source in self.sources:
                if not source.active:
                    continue
                
                try:
                    # Fetch documents
                    fetcher_class = self.fetchers.get(source.type)
                    if not fetcher_class:
                        logger.warning(f"No fetcher for source type: {source.type}")
                        continue
                    
                    fetcher = fetcher_class(session)
                    logger.info(f"🌐 FETCHING: {source.name} ({source.type}) from {source.url}")
                    documents = await fetcher.fetch(source)
                    
                    if not documents:
                        logger.warning(f"❌ No documents fetched from {source.name}")
                        continue
                    
                    logger.info(f"✅ FETCHED: {len(documents)} documents from {source.name}")
                    
                    # Normalize documents
                    logger.info(f"🔄 NORMALIZING: {len(documents)} documents")
                    normalized_docs = self.normalizer.normalize(documents)
                    logger.info(f"✅ NORMALIZED: {len(normalized_docs)} documents ready for storage")
                    
                    # Check triggers and fire events
                    triggered_count = 0
                    for doc in normalized_docs:
                        triggered_events = self.trigger.check_triggers(doc)
                        if triggered_events:
                            await self.trigger.fire_events(doc, triggered_events)
                            triggered_count += 1
                    
                    if triggered_count > 0:
                        logger.info(f"🚨 TRIGGERS: {triggered_count} documents triggered events")
                    
                    # Store in vector DB
                    logger.info(f"💾 STORING: {len(normalized_docs)} documents in vector database")
                    await self._store_documents(normalized_docs)
                    
                except Exception as e:
                    logger.error(f"Error processing source {source.name}: {e}")
    
    async def _store_documents(self, documents: List[KnowledgeDocument]):
        """Store documents using existing 42.zero import functionality."""
        if not documents:
            logger.warning("No documents to store")
            return
            
        logger.info(f"🔄 STORAGE PIPELINE: Starting storage of {len(documents)} documents")
        
        try:
            # Use existing 42.zero import functionality directly
            from ...infra.core.chunker import Chunker
            from ...infra.core.embedding import EmbeddingEngine
            from ...infra.core.vector_store import VectorStore
            from qdrant_client.models import PointStruct
            import tempfile
            import os
            import uuid
            
            logger.info("📦 PIPELINE STEP 1: Initializing 42.zero components")
            
            # Initialize 42.zero components
            chunker = Chunker()
            embedding_engine = EmbeddingEngine()
            vector_store = VectorStore()
            
            # Ensure collection exists
            vector_size = embedding_engine.get_dimension()
            vector_store.create_collection(vector_size)
            logger.info(f"✅ Vector store initialized (dimension: {vector_size})")
            
            # Create temporary files for each document and chunk them
            temp_files = []
            all_chunks = []
            
            logger.info("📝 PIPELINE STEP 2: Processing documents into chunks")
            
            for doc_idx, doc in enumerate(documents):
                logger.info(f"  📄 Processing document {doc_idx + 1}/{len(documents)}: {doc.source_id}")
                logger.info(f"     Content length: {len(doc.content)} chars")
                logger.info(f"     Title: {doc.metadata.get('title', 'No title')[:100]}...")
                
                # Create temporary file with knowledge metadata
                safe_name = doc.source_id.replace("/", "_").replace(":", "_")[:50]
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{safe_name}.txt', delete=False) as f:
                    # Add metadata as comments for context
                    f.write(f"# Source: {doc.source_id}\n")
                    f.write(f"# URL: {doc.metadata.get('url', 'unknown')}\n") 
                    f.write(f"# Title: {doc.metadata.get('title', 'unknown')}\n")
                    f.write(f"# Published: {doc.metadata.get('published', 'unknown')}\n")
                    f.write(f"# ---\n\n")
                    f.write(doc.content)
                    temp_files.append(f.name)
                    
                    # Chunk the file using 42.zero chunker
                    chunks = chunker.chunk_file(f.name)
                    all_chunks.extend(chunks)
                    logger.info(f"     Generated {len(chunks)} chunks from document")
            
            logger.info(f"✅ CHUNKING COMPLETE: {len(all_chunks)} total chunks generated")
            
            if not all_chunks:
                logger.warning("❌ No chunks generated - aborting storage")
                return
            
            logger.info("🧠 PIPELINE STEP 3: Embedding and storing chunks")
            
            # Embed and store chunks using 42.zero functionality
            stored_count = 0
            for i, chunk in enumerate(all_chunks):
                try:
                    logger.debug(f"  🔤 Embedding chunk {i + 1}/{len(all_chunks)}: {chunk.text[:100]}...")
                    
                    # Embed the chunk
                    vector = embedding_engine.embed_text(chunk.text)
                    logger.debug(f"     Embedding shape: {len(vector) if vector else 'None'}")
                    
                    if not vector:
                        logger.error(f"❌ Failed to generate embedding for chunk {i}")
                        continue
                    
                    # Create point with unique ID using UUID to avoid collisions
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk.text,
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "metadata": chunk.metadata or {},
                            "knowledge_source": True,  # Mark as knowledge content
                            "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    logger.debug(f"     Created point with ID: {point_id}")
                    
                    # Store in vector database
                    vector_store.upsert([point])
                    stored_count += 1
                    logger.debug(f"     ✅ Stored chunk {i + 1} in vector DB")
                    
                except Exception as chunk_error:
                    logger.error(f"❌ Failed to process chunk {i}: {chunk_error}")
                    continue
            
            logger.info(f"✅ EMBEDDING & STORAGE COMPLETE: {stored_count}/{len(all_chunks)} chunks stored")
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass  # Ignore cleanup errors
            
            logger.info("🧹 Temporary files cleaned up")
            
            # Also publish to Redis for event system
            logger.info("📡 PIPELINE STEP 4: Publishing to Redis event bus")
            for doc in documents:
                self.redis_bus.publish_event(Event(
                    event_type=EventType.KNOWLEDGE_DOCUMENT,
                    data=doc.to_dict(),
                    timestamp=datetime.now(timezone.utc),
                    source="knowledge_engine"
                ))
            
            logger.info(f"🎉 STORAGE PIPELINE COMPLETE: Successfully stored {len(documents)} documents ({stored_count} chunks) in vector database")
            
            # Verify storage by checking total count
            try:
                total_points = vector_store.get_total_points()
                logger.info(f"📊 Vector DB now contains {total_points} total points")
            except Exception as count_error:
                logger.warning(f"Could not verify point count: {count_error}")
            
        except Exception as e:
            logger.error(f"Failed to store documents using 42.zero import_data: {e}")
            logger.exception("Full traceback:")
            # Fallback to Redis only
            for doc in documents:
                self.redis_bus.publish_event(Event(
                    event_type=EventType.KNOWLEDGE_DOCUMENT,
                    data=doc.to_dict(),
                    timestamp=datetime.now(timezone.utc),
                    source="knowledge_engine"
                ))
            logger.info(f"Stored {len(documents)} documents in Redis only (fallback)")
    
    # Removed redundant search_knowledge method - use CLI search command with 42.zero functions instead
    
    async def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring."""
        logger.info(f"Starting knowledge engine monitoring (interval: {interval_seconds}s)")
        
        while True:
            await self.run_fetch_cycle()
            await asyncio.sleep(interval_seconds) 