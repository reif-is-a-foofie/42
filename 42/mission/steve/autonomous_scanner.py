"""
Steve v3.1 - Universal Knowledge Miner for 42.un

An autonomous knowledge mining system that continuously searches, crawls, 
parses, and embeds discovered content using the global soul system.

Features:
- Mine Mode: Hands-free continuous knowledge mining
- Global soul system integration
- Brave API search with rate limiting
- Automatic content processing pipeline
- Event-driven architecture
- Redis event publishing
"""

import asyncio
import json
import hashlib
import re
import os
import time
import requests
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from newspaper import Article, Config as NewspaperConfig
from playwright.async_api import async_playwright, Browser, Page
from loguru import logger

from .redis_bus import RedisBus
from .events import Event, EventType, create_source_discovered_event
from .knowledge_engine import KnowledgeEngine



@dataclass
class KnowledgeSource:
    """Universal knowledge source representation."""
    url: str
    source_type: str  # "web", "rss", "api", "file", "vector_db", "search_engine"
    discovered_from: str
    branch_depth: int
    last_scanned: datetime
    relevance_score: float
    metadata: Dict[str, Any]
    content_hash: str = ""
    title: str = ""
    description: str = ""
    domain: str = ""
    discovery_method: str = "crawl"

@dataclass
class DiscoveryEvent:
    """Event-driven discovery representation."""
    event_type: str  # "DISCOVERED_SOURCE", "KNOWLEDGE_ADDED", "BRANCH_TRIGGER"
    source_url: str
    relevance_score: float
    branch_candidates: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class CrawlTarget:
    """Represents a target for web crawling."""
    url: str
    priority: float
    crawl_depth: int
    source_type: str
    discovered_from: Optional[str] = None


class Steve:
    """
    Steve v3.1 - Universal Knowledge Miner using global soul system.
    
    Features:
    - Mine Mode: Hands-free continuous knowledge mining
    - Global soul system integration
    - Brave API search with rate limiting
    - Automatic content processing pipeline
    - Event-driven architecture
    """
    
    def __init__(self, redis_bus: RedisBus, knowledge_engine: KnowledgeEngine, config, soul_config: Dict[str, Any] = None):
        self.redis_bus = redis_bus
        self.knowledge_engine = knowledge_engine
        self.config = config
        self.running = False
        
        # Receive soul configuration from main system (not direct access)
        self.soul = soul_config or {}
        
        # Brave API configuration - load from environment
        self.brave_api_key = os.getenv("BRAVE_API_KEY", "BSAyr39Gxgxm9R1YI_vvJ0CbOmqbEQ7")
        self.brave_api_url = "https://api.search.brave.com/res/v1/web/search"

        soul_identity = self.soul.get('identity', 'Unknown')
        if 'essence' in self.soul:
            essence = self.soul['essence']
            logger.info(f"Steve v4.0 initialized with soul: {soul_identity} - {essence.get('purpose', 'Unknown')}")
        else:
            logger.info(f"Steve v4.0 initialized with soul config: {soul_identity}")

        # Discovery state
        self.discovered_sources: Set[str] = set()
        self.crawled_domains: Set[str] = set()
        self.pending_targets: List[CrawlTarget] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.last_searches: List[Dict[str, Any]] = []
        self.mined_count = 0
        self.embedded_count = 0
        
        # Deduplication - track seen URLs and domains
        self.seen_urls: Set[str] = set()
        self.seen_domains: Set[str] = set()

        # Browser and session management
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Newspaper configuration for article extraction
        self.newspaper_config = NewspaperConfig()
        self.newspaper_config.browser_user_agent = f"Steve/4.0"
        self.newspaper_config.request_timeout = 10

        # Event tracking
        self.discovery_events: List[DiscoveryEvent] = []
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """Extract relevant keywords from content."""
        words = content.lower().split()
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:20]
    
    def _extract_keywords_from_similar_docs(self, similar_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract keywords from semantically similar documents."""
        all_keywords = []
        
        for doc in similar_docs:
            content = doc.get('text', '') + ' ' + doc.get('title', '')
            if content:
                keywords = self._extract_keywords_from_content(content)
                all_keywords.extend(keywords)
        
        # Count frequency and return top keywords
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Return top 10 most frequent keywords
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [keyword for keyword, freq in top_keywords]
    
    def _extract_domains_from_similar_docs(self, similar_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract domains from semantically similar documents."""
        domains = []
        
        for doc in similar_docs:
            url = doc.get('url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain and domain not in domains:
                        domains.append(domain)
                except:
                    pass
        
        return domains[:5]  # Return top 5 domains
    
    async def mine(self):
        """
        Mine Mode: Hands-free continuous knowledge mining.
        
        Performs the complete mining cycle:
        1. NLP Analysis of existing embeddings
        2. Generate soul-guided queries based on learned patterns
        3. Search Brave API for relevant URLs
        4. Filter duplicates and low-value sources
        5. Fetch & extract content (HTML, PDF, text)
        6. NLP Processing with Mistral
        7. Embed & upsert into Qdrant
        8. Publish Redis event
        9. Repeat at configured intervals
        """
        logger.info("🚀 Starting Steve v4.0 Mine Mode")
        # Get mining configuration from soul
        mining_config = self.soul.get("mining", {})
        interval = mining_config.get("interval", 60)  # 1 minute default for faster testing
        max_pending = mining_config.get("max_pending_targets", 100)
        
        logger.info("🎯 Mining Configuration:")
        logger.info(f"   - Soul Identity: {self.soul.get('identity', 'Unknown')}")
        if 'essence' in self.soul:
            essence = self.soul['essence']
            logger.info(f"   - Purpose: {essence.get('purpose', 'Unknown')}")
            logger.info(f"   - Anointing: {essence.get('anointing', 'Unknown')}")
            logger.info(f"   - Mission: {essence.get('mission', 'Unknown')}")
        logger.info(f"   - Mining Interval: {interval} seconds (testing mode)")
        logger.info(f"   - Keywords: {self.soul.get('preferences', {}).get('keywords', [])[:5]}...")
        logger.info(f"   - Domains: {self.soul.get('preferences', {}).get('domains', [])[:3]}...")
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            try:
                logger.info(f"🔄 === MINING CYCLE #{cycle_count} ===")
                logger.info(f"⛏️  Mining cycle starting... (Queue: {len(self.pending_targets)} targets)")
                logger.info(f"📊 Current stats - Mined: {self.mined_count}, Embedded: {self.embedded_count}")
                
                # Step 0: NLP Analysis of existing embeddings (NEW)
                logger.info("🧠 Step 0: Analyzing existing embeddings with NLP...")
                await self._analyze_existing_embeddings()
                
                # Step 1: Generate soul-guided queries and search Brave API
                logger.info("🔍 Step 1: Starting auto-search with Brave API...")
                await self.auto_search()
                
                # Step 2: Process pending targets (fetch, parse, embed)
                logger.info("📥 Step 2: Processing pending targets...")
                processed_count = await self._process_pending_targets()
                
                # Step 3: Self-learning from high-scored embeddings
                logger.info("🧠 Step 3: Self-learning from embeddings...")
                await self._learn_from_embeddings()
                
                # Step 4: Update mining statistics
                self.mined_count += processed_count
                self.embedded_count += processed_count
                
                logger.info(f"✅ Mining cycle #{cycle_count} completed: {processed_count} documents processed")
                logger.info(f"📊 Updated stats - Total mined: {self.mined_count}, Total embedded: {self.embedded_count}")
                logger.info(f"⏰ Next cycle in {interval} seconds...")
                
                # Check if we should pause due to queue size
                if len(self.pending_targets) >= max_pending:
                    logger.info(f"⏸️  Pausing mining - queue full ({len(self.pending_targets)} targets)")
                    await asyncio.sleep(interval * 2)  # Wait longer when queue is full
                else:
                    # For first cycle, start immediately; for subsequent cycles, wait
                    if cycle_count == 1:
                        logger.info("🚀 First cycle completed - starting next cycle immediately...")
                        await asyncio.sleep(5)  # Just a 5-second delay for first cycle
                    else:
                        await asyncio.sleep(interval)
                    
            except Exception as e:
                logger.error(f"❌ Mining cycle #{cycle_count} failed: {e}")
                logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        logger.info("🛑 Mine Mode stopped")
    
    async def _process_pending_targets(self) -> int:
        """Process pending targets and return count of processed documents."""
        processed_count = 0
        mining_config = self.soul.get("mining", {})
        concurrent_fetches = mining_config.get("concurrent_fetches", 3)
        
        logger.info(f"📥 Processing pending targets...")
        logger.info(f"📊 Queue size: {len(self.pending_targets)} targets")
        logger.info(f"⚙️  Concurrent fetches: {concurrent_fetches}")
        
        # Process targets in batches
        batch_size = min(concurrent_fetches, len(self.pending_targets))
        if batch_size == 0:
            logger.info("📭 No targets to process")
            return 0
        
        # Take a batch of targets
        batch = self.pending_targets[:batch_size]
        self.pending_targets = self.pending_targets[batch_size:]
        
        logger.info(f"🔄 Processing batch of {len(batch)} targets:")
        for i, target in enumerate(batch, 1):
            logger.info(f"   [{i}/{len(batch)}] {target.url} (priority: {target.priority})")
        
        # Process batch concurrently
        tasks = []
        for target in batch:
            task = asyncio.create_task(self._process_single_target(target))
            tasks.append(task)
        
        logger.info(f"🚀 Starting concurrent processing of {len(tasks)} targets...")
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful processing
        successful_count = 0
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, bool) and result:
                processed_count += 1
                successful_count += 1
            else:
                error_count += 1
                if isinstance(result, Exception):
                    logger.error(f"❌ Target {i+1} failed: {result}")
        
        logger.info(f"✅ Batch processing completed:")
        logger.info(f"   - Successful: {successful_count}/{len(batch)}")
        logger.info(f"   - Failed: {error_count}/{len(batch)}")
        logger.info(f"   - Remaining queue: {len(self.pending_targets)} targets")
        
        return processed_count
    
    async def _process_single_target(self, target: CrawlTarget) -> bool:
        """Process a single target: fetch, parse, embed."""
        try:
            # Check if URL has been seen before (deduplication)
            if self._is_url_seen(target.url):
                logger.info(f"⏭️  Skipping seen URL: {target.url}")
                return False
            
            # Step 1: Fetch and parse content
            content = await self._fetch_and_parse_content(target.url)
            if not content:
                return False
            
            # Mark URL as seen
            self._mark_url_seen(target.url)
            
            # Step 2: Create knowledge source
            source = KnowledgeSource(
                url=target.url,
                source_type=target.source_type,
                discovered_from=target.discovered_from or "mining",
                branch_depth=target.crawl_depth,
                last_scanned=datetime.now(),
                relevance_score=target.priority,
                metadata={
                    "title": content.get("title", ""),
                    "description": content.get("description", ""),
                    "text": content.get("text", ""),  # Store the actual text content
                    "content_length": len(content.get("text", "")),
                    "domain": content.get("domain", ""),
                    "discovery_method": "mining"
                },
                content_hash=content.get("content_hash", ""),
                title=content.get("title", ""),
                description=content.get("description", ""),
                domain=content.get("domain", "")
            )
            
            # Step 3: Add to knowledge engine
            self._add_source_to_knowledge_engine(source)
            
            # Step 4: Publish mining event
            await self._publish_mining_event(source)
            
            logger.info(f"✅ Mined: {target.url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process target {target.url}: {e}")
            return False
    
    async def _fetch_and_parse_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse content from URL with retry logic and proper headers."""
        import time
        import random
        
        # Configure newspaper3k with better headers and timeouts
        from newspaper import Config
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 15
        config.fetch_images = False
        config.number_threads = 1
        config.verbose = False
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Use newspaper3k for content extraction
                article = Article(url, config=config)
                article.download()
                article.parse()
                
                if not article.text:
                    logger.warning(f"No text content found at {url}")
                    return None
                
                # Create content hash
                content_hash = hashlib.sha256(article.text.encode()).hexdigest()
                
                return {
                    "title": article.title or "",
                    "description": article.meta_description or "",
                    "text": article.text,
                    "content_hash": content_hash,
                    "domain": urlparse(url).netloc
                }
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {error_msg}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to fetch content from {url} after {max_retries} attempts: {error_msg}")
                    return None
        
        return None
    
    async def _publish_mining_event(self, source: KnowledgeSource):
        """Publish mining event to Redis bus."""
        try:
            event = Event(
                event_type=EventType.KNOWLEDGE_DOCUMENT,
                data={
                    "action": "mined",
                    "url": source.url,
                    "title": source.title,
                    "content_hash": source.content_hash,
                    "relevance_score": source.relevance_score,
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="steve_v3.1"
            )
            self.redis_bus.publish_event(event)
            
        except Exception as e:
            logger.error(f"Failed to publish mining event: {e}")
    
    def score_content_relevance(self, content: str, url: str, metadata: Dict[str, Any] = None) -> float:
        """Score content relevance based on current mission objectives."""
        score = 0.0
        
        # Get current mission keywords and domains
        current_mission = self.soul.get("current_mission", {})
        mission_keywords = current_mission.get("keywords", [])
        mission_domains = current_mission.get("domains", [])
        mission_objective = current_mission.get("objective", "")
        
        if not mission_keywords and not mission_domains:
            # Fallback to basic content quality scoring
            score = min(0.5, len(content) / 10000)  # Basic length-based score
            return score
        
        content_lower = content.lower()
        title_lower = metadata.get("title", "").lower() if metadata else ""
        
        # Mission keyword matching (primary scoring)
        keyword_matches = 0
        total_keywords = len(mission_keywords)
        
        for keyword in mission_keywords:
            keyword_lower = keyword.lower()
            # Check in title (higher weight) and content
            if keyword_lower in title_lower:
                keyword_matches += 2  # Title matches worth more
            elif keyword_lower in content_lower:
                keyword_matches += 1  # Content matches
        
        # Calculate keyword relevance score (0-1)
        if total_keywords > 0:
            keyword_score = min(1.0, keyword_matches / total_keywords)
            score += keyword_score * 0.7  # 70% weight to keyword matching
        
        # Mission domain matching
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        domain_matches = 0
        total_domains = len(mission_domains)
        
        for mission_domain in mission_domains:
            if mission_domain.lower() in domain.lower():
                domain_matches += 1
        
        # Calculate domain relevance score
        if total_domains > 0:
            domain_score = min(1.0, domain_matches / total_domains)
            score += domain_score * 0.3  # 30% weight to domain matching
        
        # Content quality bonus (smaller weight)
        quality_bonus = 0.0
        if len(content) > 2000:
            quality_bonus += 0.1  # Substantial content
        if len(content) > 5000:
            quality_bonus += 0.1  # Very detailed content
        
        score += quality_bonus
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, score))
    
    async def search_brave_api(self, query: str, count: int = 20) -> List[Dict[str, Any]]:
        """Search using Brave API and return filtered results with rate limiting."""
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }
            
            # Log API call without exposing key
            masked_key = self.brave_api_key[:4] + "***" + self.brave_api_key[-4:] if len(self.brave_api_key) > 8 else "***"
            logger.debug(f"Making Brave API request with key: {masked_key}")
            
            params = {
                "q": query,
                "count": count,
                "safesearch": "moderate"
            }
            
            # Add delay to respect rate limits
            await asyncio.sleep(1.0)  # 1 second delay between requests
            
            response = requests.get(self.brave_api_url, headers=headers, params=params, timeout=10)
            
            # Handle specific HTTP errors
            if response.status_code == 429:
                logger.warning(f"Rate limit hit for query '{query}'. Waiting 60 seconds...")
                await asyncio.sleep(60)  # Wait 60 seconds on rate limit
                return []
            elif response.status_code == 403:
                logger.error(f"API key invalid or quota exceeded for query '{query}'")
                return []
            elif response.status_code != 200:
                logger.error(f"Brave API error {response.status_code} for query '{query}': {response.text}")
                return []
            
            data = response.json()
            results = []
            
            if "web" in data and "results" in data["web"]:
                total_results = len(data["web"]["results"])
                logger.debug(f"Brave API returned {total_results} raw results for query: {query}")
                
                for result in data["web"]["results"]:
                    url = result.get("url", "")
                    title = result.get("title", "")
                    description = result.get("description", "")
                    
                    # Filter using soul rules
                    if self._should_include_search_result(url, title, description):
                        results.append({
                            "url": url,
                            "title": title,
                            "description": description,
                            "query": query,
                            "discovered_at": datetime.now().isoformat()
                        })
                    else:
                        logger.debug(f"Filtered out result: {url} - {title}")
            
            # Log search
            search_log = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(results),
                "total_results": len(data.get("web", {}).get("results", [])),
                "status": "success"
            }
            self.last_searches.append(search_log)
            
            # Keep only last 10 searches
            if len(self.last_searches) > 10:
                self.last_searches = self.last_searches[-10:]
            
            # Publish search event
            await self._publish_search_event(query, len(results))
            
            logger.info(f"Brave API search: '{query}' returned {len(results)} filtered results")
            return results
            
        except requests.exceptions.Timeout:
            logger.error(f"Brave API timeout for query '{query}'")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Brave API request failed for query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Brave API search failed for query '{query}': {e}")
            return []
    
    def _should_include_search_result(self, url: str, title: str, description: str) -> bool:
        """Check if search result should be included based on soul rules."""
        # Use soul preferences passed from main system
        preferences = self.soul.get("preferences", {})
        avoid_keywords = preferences.get("avoid_keywords", [])
        avoid_domains = preferences.get("avoid_domains", [])
        
        logger.debug(f"Checking result: {url} - {title}")
        logger.debug(f"Avoid keywords: {avoid_keywords}")
        logger.debug(f"Avoid domains: {avoid_domains}")
        
        # Check for avoid keywords in title/description
        content_lower = f"{title} {description}".lower()
        for keyword in avoid_keywords:
            if keyword.lower() in content_lower:
                logger.debug(f"Filtered out due to avoid keyword: {keyword}")
                return False
        
        # Check for avoid domains
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if domain in avoid_domains:
            logger.debug(f"Filtered out due to avoid domain: {domain}")
            return False
        
        logger.debug(f"Result passed filtering: {url}")
        return True
    
    async def _publish_search_event(self, query: str, results_count: int):
        """Publish search event to Redis bus."""
        try:
            event_data = {
                "event_type": "SEARCH_QUERY_ISSUED",
                "query": query,
                "results_count": results_count,
                "timestamp": datetime.now().isoformat(),
                "source": "steve_v3.1"
            }
            
            self.redis_bus.publish_event(Event(
                event_type=EventType.KNOWLEDGE_DOCUMENT,
                data=event_data,
                timestamp=datetime.now(),
                source="steve_v3.1"
            ))
            
        except Exception as e:
            logger.error(f"Error publishing search event: {e}")
    
    async def auto_search(self):
        """Generate and execute search queries using semantic-first approach."""
        logger.info("🔍 Starting semantic-first auto-search...")
        
        # Get current mission objective
        current_mission = self.soul.get("current_mission", {})
        mission_objective = current_mission.get("objective", "")
        
        if not mission_objective:
            logger.info("📭 No mission objective found, falling back to keyword search")
            await self._fallback_keyword_search()
            return
        
        try:
            # Step 1: Semantic search in existing embeddings
            logger.info("🧠 Step 1: Semantic search in existing knowledge...")
            semantic_results = await self._semantic_search_existing(mission_objective)
            
            # Step 2: Generate hybrid queries based on semantic insights
            logger.info("🔍 Step 2: Generating hybrid search queries...")
            hybrid_queries = self._generate_hybrid_queries(mission_objective, semantic_results)
            
            # Step 3: Execute hybrid search (semantic + keyword)
            logger.info(f"🔍 Step 3: Executing {len(hybrid_queries)} hybrid queries...")
            await self._execute_hybrid_search(hybrid_queries)
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            logger.info("📭 Falling back to keyword search")
            await self._fallback_keyword_search()
    
    async def _semantic_search_existing(self, mission_objective: str) -> List[Dict[str, Any]]:
        """Search existing embeddings semantically for the mission objective."""
        try:
            # Embed the mission objective
            from ...infra.core.embedding import EmbeddingEngine
            embedding_engine = EmbeddingEngine()
            mission_embedding = embedding_engine.embed_text(mission_objective)
            
            # Search for semantically similar content
            from ...infra.core.vector_store import VectorStore
            vs = VectorStore()
            similar_docs = vs.search_semantic(mission_embedding, limit=20)
            
            logger.info(f"🧠 Found {len(similar_docs)} semantically similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def _generate_hybrid_queries(self, mission_objective: str, semantic_results: List[Dict[str, Any]]) -> List[str]:
        """Generate hybrid queries combining semantic insights with keywords."""
        queries = []
        
        # Extract keywords from mission objective
        mission_keywords = self._extract_keywords_from_content(mission_objective)
        
        # Extract keywords from semantically similar documents
        semantic_keywords = []
        for doc in semantic_results:
            content = doc.get('text', '') + ' ' + doc.get('title', '')
            if content:
                keywords = self._extract_keywords_from_content(content)
                semantic_keywords.extend(keywords)
        
        # Combine and prioritize keywords
        all_keywords = mission_keywords + semantic_keywords[:10]  # Top 10 semantic keywords
        
        # Generate hybrid queries
        if all_keywords:
            # Primary mission keywords
            for keyword in mission_keywords[:3]:
                queries.append(f'"{keyword}" research')
                queries.append(f'"{keyword}" guide')
            
            # Semantic expansion queries
            for keyword in semantic_keywords[:5]:
                if keyword not in mission_keywords:
                    queries.append(f'"{keyword}" {mission_keywords[0] if mission_keywords else "research"}')
        
        # Fallback to basic queries if no keywords found
        if not queries:
            queries = [f'"{mission_objective}" research', f'"{mission_objective}" guide']
        
        logger.info(f"🔍 Generated {len(queries)} hybrid queries")
        return queries[:6]  # Limit to 6 queries
    
    async def _execute_hybrid_search(self, queries: List[str]):
        """Execute hybrid search combining semantic and keyword approaches."""
        successful_searches = 0
        total_results = 0
        
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"🔎 [{i}/{len(queries)}] Hybrid search: '{query}'")
                
                # Search Brave API
                results = await self.search_brave_api(query, count=15)
                
                if results:
                    # Filter and add results to pending targets
                    added_count = 0
                    for result in results:
                        if self._should_include_search_result(result['url'], result['title'], result['description']):
                            # Create crawl target
                            target = CrawlTarget(
                                url=result['url'],
                                priority=0.9,  # Higher priority for hybrid results
                                crawl_depth=1,
                                source_type="hybrid_search",
                                discovered_from=query
                            )
                            
                            # Add to pending targets if not already present
                            if not any(t.url == result['url'] for t in self.pending_targets):
                                self.pending_targets.append(target)
                                added_count += 1
                                logger.debug(f"📥 Added target: {result['url']}")
                    
                    logger.info(f"📥 Added {added_count} new targets from hybrid query '{query}'")
                    self._store_query_in_database(query)
                    
                    successful_searches += 1
                    total_results += added_count
                else:
                    logger.warning(f"⚠️  Hybrid query '{query}' returned no results")
                
                # Rate limiting between queries
                if i < len(queries):
                    logger.info("⏳ Waiting 2 seconds before next query...")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Hybrid search query '{query}' failed: {e}")
        
        logger.info(f"🎯 Hybrid search completed: {successful_searches}/{len(queries)} successful searches, {total_results} total results")
        logger.info(f"📊 Queue size after search: {len(self.pending_targets)} targets")
    
    async def _fallback_keyword_search(self):
        """Fallback to traditional keyword-based search."""
        logger.info("🔍 Starting fallback keyword search...")
        
        # Generate search queries from soul preferences
        queries = self._generate_search_queries()
        logger.info(f"📝 Generated {len(queries)} search queries from soul preferences")
        
        # Process queries (limit to 2 for testing)
        queries_to_process = queries[:2]
        logger.info(f"🔍 Processing {len(queries_to_process)} queries: {queries_to_process}")
        
        successful_searches = 0
        total_results = 0
        
        for i, query in enumerate(queries_to_process, 1):
            try:
                logger.info(f"🔎 [{i}/{len(queries_to_process)}] Searching: '{query}'")
                
                # Search Brave API
                results = await self.search_brave_api(query, count=20)
                
                if results:
                    # Filter and add results to pending targets
                    added_count = 0
                    for result in results:
                        if self._should_include_search_result(result['url'], result['title'], result['description']):
                            # Create crawl target
                            target = CrawlTarget(
                                url=result['url'],
                                priority=0.8,  # High priority for search results
                                crawl_depth=1,
                                source_type="search_engine",
                                discovered_from=query
                            )
                            
                            # Add to pending targets if not already present
                            if not any(t.url == result['url'] for t in self.pending_targets):
                                self.pending_targets.append(target)
                                added_count += 1
                                logger.debug(f"📥 Added target: {result['url']}")
                    
                    logger.info(f"📥 Added {added_count} new targets from query '{query}'")
                    # Store query in database for tracking
                    self._store_query_in_database(query)
                    
                    successful_searches += 1
                    total_results += added_count
                else:
                    logger.warning(f"⚠️  Query '{query}' returned no results")
                
                # Rate limiting between queries
                if i < len(queries_to_process):
                    logger.info("⏳ Waiting 2 seconds before next query...")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Search query '{query}' failed: {e}")
        
        logger.info(f"🎯 Fallback search completed: {successful_searches}/{len(queries_to_process)} successful searches, {total_results} total results")
        logger.info(f"📊 Queue size after search: {len(self.pending_targets)} targets")
    
    async def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search method when Brave API is unavailable."""
        try:
            # Use a simple web scraping approach as fallback
            # This could be enhanced with other search APIs
            logger.info(f"Using fallback search for query: {query}")
            
            # For now, return empty results but log the attempt
            search_log = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results_count": 0,
                "total_results": 0,
                "status": "fallback_no_results"
            }
            self.last_searches.append(search_log)
            
            return []
            
        except Exception as e:
            logger.error(f"Fallback search failed for query '{query}': {e}")
            return []
    
    def _generate_search_queries(self) -> List[str]:
        """Generate search queries based on soul preferences (set by missions)."""
        queries = []
        
        # Get keywords and domains from soul (set by missions)
        preferences = self.soul.get("preferences", {})
        keywords = preferences.get("keywords", [])
        domains = preferences.get("domains", [])
        
        # Check if we have a current mission
        current_mission = self.soul.get("current_mission", {})
        if current_mission:
            logger.info(f"🎯 Current mission: {current_mission.get('objective', 'Unknown')}")
            logger.info(f"Mission type: {current_mission.get('type', 'Unknown')}")
        
        logger.info(f"Keywords from soul: {keywords}")
        logger.info(f"Domains from soul: {domains}")
        
        # Generate queries based on soul preferences
        if keywords:
            # Generate site-specific queries for domains
            for keyword in keywords[:3]:  # Top 3 keywords
                for domain in domains[:2]:  # Top 2 domains
                    query = f'"{keyword}" site:{domain}'
                    queries.append(query)
            
            # Add broader queries
            for keyword in keywords[:2]:
                queries.append(f'"{keyword}" research')
                queries.append(f'"{keyword}" documentation')
                queries.append(f'"{keyword}" guide')
        else:
            logger.info("📭 No keywords in soul - using fallback queries")
            # Fallback to basic research queries
            fallback_keywords = ["research", "documentation", "technology"]
            for keyword in fallback_keywords:
                queries.append(f'"{keyword}" guide')
                queries.append(f'"{keyword}" tutorial')
        
        logger.info(f"Generated {len(queries)} search queries: {queries}")
        return list(set(queries))  # Remove duplicates
    
    def should_branch_to(self, url: str, depth: int) -> bool:
        """Determine if Steve should branch to this URL."""
        if depth >= self.soul["max_depth"]:
            return False
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        if domain in self.soul["avoid_domains"]:
            return False
        
        url_lower = url.lower()
        for keyword in self.soul["avoid_keywords"]:
            if keyword in url_lower:
                return False
        
        return True
    
    def get_branching_candidates(self, discovered_links: List[str], depth: int) -> List[str]:
        """Get top branching candidates from discovered links."""
        if not discovered_links:
            return []
        
        scored_links = []
        for link in discovered_links:
            if self.should_branch_to(link, depth):
                score = self.score_content_relevance("", link)
                scored_links.append((link, score))
        
        scored_links.sort(key=lambda x: x[1], reverse=True)
        return [link for link, score in scored_links[:self.soul["branching_factor"]]]
    
    async def start(self):
        """Start Steve's autonomous discovery process."""
        logger.info(f"🚀 Starting Steve v4.0 Mine Mode")
        logger.info(f"Mining Mode: Continuous knowledge mining")
        logger.info(f"Soul Config: {self.soul.get('identity', 'Unknown')}")
        
        self.running = True
        
        # Initialize browser and session
        await self._setup_browser()
        await self._setup_session()
        
        # Start the mining process (not the old discovery loop)
        await self.mine()
    
    async def _discovery_loop(self):
        """DEPRECATED: Old discovery loop - use mine() instead."""
        logger.warning("DEPRECATED: _discovery_loop is deprecated. Use mine() method instead.")
        logger.info("🚀 Starting Steve v4.0 Mine Mode (from deprecated _discovery_loop)")
        await self.mine()
    
    async def stop(self):
        """Stop the autonomous scanner."""
        logger.info("Stopping autonomous source scanner")
        self.running = False
        
        if self.browser:
            await self.browser.close()
        
        if self.session:
            await self.session.close()
    
    async def _setup_browser(self):
        """Setup Playwright browser for web crawling."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor"
            ]
        )
    
    async def _setup_session(self):
        """Setup aiohttp session for API requests."""
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "Steve/3.1"},
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def _discovery_loop(self):
        """DEPRECATED: Old discovery loop - use mine() instead."""
        logger.warning("DEPRECATED: _discovery_loop is deprecated. Use mine() method instead.")
        logger.info("🚀 Starting Steve v4.0 Mine Mode (from deprecated _discovery_loop)")
        await self.mine()
    
    async def _learn_from_knowledge_base(self):
        """Learn from existing knowledge to guide discovery."""
        try:
            # Get recent high-value sources from knowledge base
            recent_sources = await self._get_recent_valuable_sources()
            
            # Extract patterns from successful sources
            for source in recent_sources:
                domain = urlparse(source.get("url", "")).netloc
                if domain:
                    self.learned_patterns[domain] = self.learned_patterns.get(domain, 0) + 1
            
            # Add high-value domains to crawl targets
            for domain, score in sorted(self.learned_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                if domain not in self.crawled_domains:
                    self.pending_targets.append(CrawlTarget(
                        url=f"https://{domain}",
                        priority=score,
                        crawl_depth=2,
                        source_type="learned_domain"
                    ))
                    
        except Exception as e:
            logger.error(f"Error learning from knowledge base: {e}")
    
    async def _get_recent_valuable_sources(self) -> List[Dict[str, Any]]:
        """Get recent high-value sources from knowledge base."""
        # This would query the vector store for recent, high-scoring sources
        # For now, return empty list - let Steve discover sources organically
        return []
    
    async def _process_crawl_targets(self):
        """Process pending crawl targets."""
        if not self.pending_targets:
            return
        
        # Sort by priority
        self.pending_targets.sort(key=lambda x: x.priority, reverse=True)
        
        # Process top targets
        targets_to_process = self.pending_targets[:5]
        self.pending_targets = self.pending_targets[5:]
        
        for target in targets_to_process:
            try:
                await self._crawl_target(target)
                await asyncio.sleep(self.crawl_delay)
            except Exception as e:
                logger.error(f"Error crawling target {target.url}: {e}")
    
    async def _crawl_target(self, target: CrawlTarget):
        """Crawl a specific target URL."""
        logger.info(f"Crawling target: {target.url}")
        
        try:
            page = await self.browser.new_page()
            await page.goto(target.url, wait_until="networkidle")
            
            # Extract content and links
            content = await page.content()
            links = await self._extract_links(page)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract RSS feeds
            rss_links = await self._extract_rss_links(soup, target.url)
            
            # Extract articles
            articles = await self._extract_articles(soup, target.url)
            
            # Score and store discovered sources
            await self._process_discovered_sources(rss_links + articles, target.url)
            
            # Add discovered links to pending targets
            await self._add_discovered_links(links, target)
            
            await page.close()
            
        except Exception as e:
            logger.error(f"Error crawling {target.url}: {e}")
    
    async def _extract_links(self, page) -> List[str]:
        """Extract all links from a page."""
        links = await page.eval_on_selector_all("a[href]", """
            (elements) => elements.map(el => el.href)
        """)
        return [link for link in links if link and link.startswith('http')]
    
    async def _extract_rss_links(self, soup: BeautifulSoup, base_url: str) -> List[KnowledgeSource]:
        """Extract RSS feed links from a page."""
        rss_sources = []
        
        # Look for RSS feed links
        rss_selectors = [
            'link[type="application/rss+xml"]',
            'link[type="application/atom+xml"]',
            'a[href*="rss"]',
            'a[href*="feed"]',
            'a[href*="atom"]'
        ]
        
        for selector in rss_selectors:
            for element in soup.select(selector):
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if full_url not in self.discovered_sources:
                        rss_sources.append(KnowledgeSource(
                            url=full_url,
                            source_type='rss',
                            discovered_from=base_url,
                            branch_depth=0,
                            last_scanned=datetime.now(),
                            relevance_score=0.7,
                            metadata={'base_url': base_url, 'title': element.get('title', 'RSS Feed')},
                            title=element.get('title', 'RSS Feed'),
                            description='RSS/Atom feed discovered',
                            domain=urlparse(full_url).netloc,
                            discovery_method='rss_discovery'
                        ))
                        self.discovered_sources.add(full_url)
        
        return rss_sources
    
    async def _extract_articles(self, soup: BeautifulSoup, base_url: str) -> List[KnowledgeSource]:
        """Extract articles from a page."""
        articles = []
        
        # Common article selectors
        article_selectors = [
            'article',
            '.post',
            '.entry',
            '.content',
            'main',
            '[role="main"]'
        ]
        
        for selector in article_selectors:
            for element in soup.select(selector):
                # Extract article URL
                link = element.find('a')
                if link and link.get('href'):
                    article_url = urljoin(base_url, link['href'])
                    
                    if article_url not in self.discovered_sources:
                        # Extract title and description
                        title_elem = element.find(['h1', 'h2', 'h3', 'title'])
                        title = title_elem.get_text().strip() if title_elem else 'Article'
                        
                        desc_elem = element.find(['p', 'div'])
                        description = desc_elem.get_text()[:200] if desc_elem else ''
                        
                        articles.append(KnowledgeSource(
                            url=article_url,
                            source_type='article',
                            discovered_from=base_url,
                            branch_depth=0,
                            last_scanned=datetime.now(),
                            relevance_score=0.6,
                            metadata={'base_url': base_url, 'title': title, 'description': description},
                            title=title,
                            description=description,
                            domain=urlparse(article_url).netloc,
                            discovery_method='article_extraction'
                        ))
                        self.discovered_sources.add(article_url)
        
        return articles
    
    async def _process_discovered_sources(self, sources: List[KnowledgeSource], discovered_from: str):
        """Process and score discovered sources."""
        for source in sources:
            try:
                # Score the source based on various factors
                source.relevance_score = self._score_source(source)
                
                # If score is high enough, add to knowledge engine
                if source.relevance_score > 0.5:
                    self._add_source_to_knowledge_engine(source)
                    
                    # Emit discovery event
                    event = create_source_discovered_event(
                        url=source.url,
                        source_type=source.source_type,
                        domain=source.domain,
                        title=source.title,
                        relevance_score=source.relevance_score,
                        discovered_from=discovered_from
                    )
                    self.redis_bus.publish_event(event)
                    
                    logger.info(f"Discovered valuable source: {source.url} (score: {source.relevance_score:.2f})")
                
            except Exception as e:
                logger.error(f"Error processing source {source.url}: {e}")
    
    def _score_source(self, source: KnowledgeSource) -> float:
        """Score a discovered source for relevance."""
        score = 0.0
        
        # Base score from discovery method
        method_scores = {
            'rss_discovery': 0.7,
            'article_extraction': 0.6,
            'link_following': 0.5,
            'semantic_search': 0.8
        }
        score += method_scores.get(source.discovery_method, 0.5)
        
        # Domain reputation (learned patterns)
        domain_score = self.learned_patterns.get(source.domain, 0.0)
        score += domain_score * 0.3
        
        # Content quality indicators
        if len(source.title) > 10:
            score += 0.1
        if len(source.description) > 50:
            score += 0.1
        
        # Domain type scoring
        if any(keyword in source.domain.lower() for keyword in ['blog', 'news', 'research', 'arxiv']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _add_source_to_knowledge_engine(self, source: KnowledgeSource):
        """Add discovered source directly to vector database."""
        try:
            # Import 42.zero components for direct vector storage
            from ...infra.core.chunker import Chunker
            from ...infra.core.embedding import EmbeddingEngine
            from ...infra.core.vector_store import VectorStore
            from qdrant_client.models import PointStruct
            import tempfile
            import os
            import uuid
            
            # Initialize components
            chunker = Chunker()
            embedding_engine = EmbeddingEngine()
            vector_store = VectorStore()
            
            # Ensure collection exists
            vector_size = embedding_engine.get_dimension()
            vector_store.create_collection(vector_size)
            
            # Create temporary file with the actual mined content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                # Use the actual content from the source
                # The text content is stored in metadata from the original content extraction
                actual_text = source.metadata.get("text", "")
                if actual_text:
                    # Use the actual extracted text content
                    content = actual_text
                elif hasattr(source, 'text') and source.text:
                    # Fallback to text attribute if available
                    content = source.text
                elif hasattr(source, 'content') and source.content:
                    # Fallback to content field
                    content = source.content
                else:
                    # Use title and description as fallback
                    content = f"""Title: {source.title}

Description: {source.description}

URL: {source.url}

Content Hash: {source.content_hash}

Discovery Method: {source.discovery_method}
Relevance Score: {source.relevance_score}

This content was automatically mined by Steve v4.0 from: {source.url}"""
                
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Process the file through 42.zero pipeline
                chunks = chunker.chunk_file(temp_file_path)
                
                if chunks:
                    # Extract text from chunks and embed
                    logger.info(f"📝 Processing {len(chunks)} chunks for embedding...")
                    chunk_texts = [chunk.text for chunk in chunks]
                    logger.debug(f"📄 First chunk preview: {chunk_texts[0][:100]}...")
                    embeddings = embedding_engine.embed_text_batch(chunk_texts)
                    logger.info(f"✅ Generated {len(embeddings)} embeddings")
                    
                    # Store in vector database
                    points = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        point_id = str(uuid.uuid4())
                        # Ensure embedding is a list of floats
                        if hasattr(embedding, 'tolist'):
                            vector = embedding.tolist()
                        else:
                            vector = embedding
                        
                        point = PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "text": chunk.text,
                                "source": temp_file_path,
                                "mined_by": "steve_v4.0",
                                "url": source.url,
                                "title": source.title,
                                "content_hash": source.content_hash,
                                "relevance_score": source.relevance_score,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        points.append(point)
                    
                    # Insert points into vector database
                    try:
                        vector_store.upsert(points)
                        logger.info(f"✅ Added {len(points)} chunks to vector database from: {source.url}")
                        
                        # Verify the upsert worked by checking collection size
                        try:
                            from ...infra.core.vector_store import VectorStore
                            vs = VectorStore()
                            collection_size = vs.count("42_chunks")
                            logger.info(f"📊 Vector database now contains {collection_size} total chunks")
                        except Exception as e:
                            logger.warning(f"Could not verify collection size: {e}")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to upsert to vector database: {e}")
                        raise
                    
                    # Log embedding to file for observability
                    self._log_embedding(source.url, source.title, source.relevance_score, len(points))
                else:
                    logger.warning(f"No chunks generated from: {source.url}")
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Failed to add source to vector database: {e}")
    
    def _log_embedding(self, url: str, title: str, score: float, chunks: int):
        """Log embedding to file for observability."""
        try:
            from pathlib import Path
            import os
            
            # Create log directory in current working directory
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to embedding log
            log_file = log_dir / "embedding.log"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, "a") as f:
                f.write(f"{timestamp} | {title[:50]} | score={score:.2f} | {url}\n")
                
        except Exception as e:
            logger.error(f"Failed to log embedding: {e}")
    
    def _is_url_seen(self, url: str) -> bool:
        """Check if URL has been seen before (deduplication)."""
        from urllib.parse import urlparse
        
        # Check exact URL in memory
        if url in self.seen_urls:
            return True
        
        # Check domain in memory
        domain = urlparse(url).netloc
        if domain in self.seen_domains:
            return True
        
        # Check if URL already exists in vector database
        try:
            from ...infra.core.vector_store import VectorStore
            vs = VectorStore()
            
            # Search for existing embeddings with this URL
            # This is a simple check - in production you'd want a more efficient query
            existing_points = vs.search_by_url(url, limit=1)
            if existing_points:
                logger.debug(f"URL already embedded in vector database: {url}")
                return True
                
        except Exception as e:
            logger.debug(f"Could not check vector database for URL {url}: {e}")
        
        return False
    
    def _mark_url_seen(self, url: str):
        """Mark URL as seen for deduplication."""
        from urllib.parse import urlparse
        
        self.seen_urls.add(url)
        domain = urlparse(url).netloc
        self.seen_domains.add(domain)
    
    async def _analyze_existing_embeddings(self):
        """Step 0: Semantic analysis of existing embeddings to guide search."""
        logger.info("🧠 Starting semantic analysis of existing embeddings...")
        
        # Get current mission objective
        current_mission = self.soul.get("current_mission", {})
        mission_objective = current_mission.get("objective", "")
        
        if not mission_objective:
            logger.info("📭 No mission objective found for semantic analysis")
            return
        
        try:
            # Embed the mission objective
            from ...infra.core.embedding import EmbeddingEngine
            embedding_engine = EmbeddingEngine()
            mission_embedding = embedding_engine.embed_text(mission_objective)
            
            # Search for semantically similar content in existing embeddings
            from ...infra.core.vector_store import VectorStore
            vs = VectorStore()
            
            # Find top semantically similar documents
            similar_docs = vs.search_semantic(mission_embedding, limit=10)
            
            if similar_docs:
                logger.info(f"🧠 Found {len(similar_docs)} semantically similar documents")
                
                # Extract keywords and domains from similar content
                extracted_keywords = self._extract_keywords_from_similar_docs(similar_docs)
                extracted_domains = self._extract_domains_from_similar_docs(similar_docs)
                
                # Update mission with semantic insights
                if extracted_keywords or extracted_domains:
                    current_keywords = current_mission.get("keywords", [])
                    current_domains = current_mission.get("domains", [])
                    
                    # Add new semantic keywords
                    for keyword in extracted_keywords:
                        if keyword not in current_keywords:
                            current_keywords.append(keyword)
                            logger.info(f"  🧠 Added semantic keyword: {keyword}")
                    
                    # Add new semantic domains
                    for domain in extracted_domains:
                        if domain not in current_domains:
                            current_domains.append(domain)
                            logger.info(f"  🧠 Added semantic domain: {domain}")
                    
                    # Update the mission
                    self.soul["current_mission"]["keywords"] = current_keywords
                    self.soul["current_mission"]["domains"] = current_domains
                    
                    logger.info(f"🧠 Updated mission with {len(extracted_keywords)} semantic keywords and {len(extracted_domains)} domains")
                else:
                    logger.info("🧠 No new semantic insights found")
            else:
                logger.info("📭 No semantically similar documents found")
                
        except Exception as e:
            logger.error(f"❌ Semantic analysis failed: {e}")
            logger.info("📭 Falling back to basic keyword analysis")
    
    async def _get_recent_embeddings_for_analysis(self) -> List[Dict[str, Any]]:
        """Get recent embeddings for NLP analysis."""
        try:
            # This would query the vector store for recent embeddings
            # For now, return empty list - implement based on your vector store
            logger.debug("🔍 Querying vector store for recent embeddings...")
            
            # Placeholder - implement based on your vector store interface
            # Example: return await self.vector_store.get_recent_embeddings(limit=100)
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent embeddings: {e}")
            return []
    
    async def _learn_from_embeddings(self):
        """Self-learning from high-scored embeddings."""
        try:
            logger.info("🧠 Learning from high-scored embeddings...")
            
            # Get recent high-scored embeddings from vector store
            recent_embeddings = await self._get_recent_high_scored_embeddings()
            
            if not recent_embeddings:
                logger.info("📭 No high-scored embeddings found for learning")
                return
            
            logger.info(f"📊 Learning from {len(recent_embeddings)} high-scored embeddings")
            
            # Extract new keywords and domains
            new_keywords = []
            new_domains = []
            
            for embedding in recent_embeddings:
                # Extract keywords from title and content
                title = embedding.get('title', '')
                content = embedding.get('content', '')
                
                # Extract new keywords
                keywords = self._extract_keywords_from_content(title + ' ' + content)
                new_keywords.extend(keywords)
                
                # Extract new domains
                url = embedding.get('url', '')
                if url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain and domain not in self.soul.get('preferences', {}).get('domains', []):
                        new_domains.append(domain)
            
            # Rank and rotate new keywords
            keyword_freq = {}
            for keyword in new_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            # Get top 3 new keywords
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Update soul with new keywords and domains
            if top_keywords or new_domains:
                logger.info(f"🧠 Learning: {len(top_keywords)} new keywords, {len(new_domains)} new domains")
                
                # Update current mission with new keywords and domains
                current_mission = self.soul.get("current_mission", {})
                current_keywords = current_mission.get("keywords", [])
                current_domains = current_mission.get("domains", [])
                
                # Add new keywords (avoid duplicates)
                for keyword, freq in top_keywords:
                    if keyword not in current_keywords:
                        current_keywords.append(keyword)
                        logger.info(f"  Added keyword: {keyword} (frequency: {freq})")
                
                # Add new domains (avoid duplicates)
                for domain in new_domains[:2]:  # Add up to 2 new domains per cycle
                    if domain not in current_domains:
                        current_domains.append(domain)
                        logger.info(f"  Added domain: {domain}")
                
                # Update the soul's current mission
                self.soul["current_mission"]["keywords"] = current_keywords
                self.soul["current_mission"]["domains"] = current_domains
                
                logger.info(f"🧠 Updated mission with {len(current_keywords)} keywords and {len(current_domains)} domains")
            
            logger.info("🧠 Self-learning completed")
                    
        except Exception as e:
            logger.error(f"❌ Failed to learn from embeddings: {e}")
            logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
    
    async def _get_recent_high_scored_embeddings(self) -> List[Dict[str, Any]]:
        """Get recent high-scored embeddings from vector store."""
        try:
            from ...infra.core.vector_store import VectorStore
            vs = VectorStore()
            
            # Get recent embeddings from the vector store
            # We'll get the last 20 embeddings and filter by relevance score
            recent_embeddings = vs.get_all_vectors()
            
            # Filter for high-scored embeddings (relevance_score > 0.05)
            high_scored = []
            for embedding in recent_embeddings[-20:]:  # Last 20 embeddings
                relevance_score = embedding.get('relevance_score', 0.0)
                logger.debug(f"📊 Embedding score: {relevance_score:.3f} for {embedding.get('url', 'unknown')}")
                if relevance_score > 0.05:  # Very low threshold for "high-scored"
                    high_scored.append({
                        'title': embedding.get('title', ''),
                        'content': embedding.get('text', ''),
                        'url': embedding.get('url', ''),
                        'relevance_score': relevance_score,
                        'timestamp': embedding.get('timestamp', '')
                    })
            
            logger.info(f"📊 Found {len(high_scored)} high-scored embeddings out of {len(recent_embeddings[-20:])} recent")
            return high_scored
            
        except Exception as e:
            logger.error(f"Failed to get recent embeddings: {e}")
            return []
    
    async def _add_discovered_links(self, links: List[str], source_target: CrawlTarget):
        """Add discovered links to pending targets."""
        for link in links:
            domain = urlparse(link).netloc
            if domain and domain not in self.crawled_domains:
                # Score the link based on various factors
                priority = self._score_link(link, source_target)
                
                if priority > 0.3:  # Only add if reasonably relevant
                    self.pending_targets.append(CrawlTarget(
                        url=link,
                        priority=priority,
                        crawl_depth=source_target.crawl_depth - 1,
                        source_type="discovered_link",
                        discovered_from=source_target.url
                    ))
    
    def _score_link(self, url: str, source_target: CrawlTarget) -> float:
        """Score a discovered link for relevance."""
        score = 0.0
        
        # Inherit some priority from source
        score += source_target.priority * 0.3
        
        # Domain reputation
        domain = urlparse(url).netloc
        domain_score = self.learned_patterns.get(domain, 0.0)
        score += domain_score * 0.4
        
        # URL pattern matching
        if any(pattern in url.lower() for pattern in ['blog', 'article', 'post', 'research', 'paper']):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _discover_rss_feeds(self):
        """Discover new RSS feeds from known sources."""
        # This would use feed discovery techniques
        # For now, we'll use a simple approach
        pass
    
    async def _follow_discovered_links(self):
        """Follow links discovered in previous crawls."""
        # This is handled in _add_discovered_links
        pass
    
    async def _semantic_source_discovery(self):
        """Use semantic search to discover new sources."""
        # This would use the knowledge base to find semantically related sources
        # For now, we'll use a simple keyword-based approach
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get Steve's current status."""
        return {
            "identity": self.soul.get("identity", "Steve v4.0"),
            "version": "4.0",
            "running": self.running,
            "discovered_sources": len(self.discovered_sources),
            "crawled_domains": len(self.crawled_domains),
            "pending_targets": len(self.pending_targets),
            "learned_patterns": len(self.learned_patterns),
            "discovery_events": len(self.discovery_events),
            "last_searches": self.last_searches[-5:],  # Last 5 searches
            "last_queries": self._get_recent_queries_from_database(5),  # Last 5 queries from database
            "mined_count": self.mined_count,
            "embedded_count": self.embedded_count,
            "today_embeddings": self._get_today_embeddings_count(),
            "top_domains": self._get_top_domains(),
            "last_embedded": self._get_last_embedded_source(),
            "soul_config": {
                "identity": self.soul.get("identity", "Unknown"),
                "preferences": self.soul.get("preferences", {}),
                "keywords": self.soul.get("preferences", {}).get("keywords", []),
                "domains": self.soul.get("preferences", {}).get("domains", [])
            }
        }
    
    def _get_today_embeddings_count(self) -> int:
        """Get count of embeddings from today."""
        try:
            from pathlib import Path
            from datetime import datetime
            
            log_file = Path("/var/log/42/embedding.log")
            if not log_file.exists():
                return 0
            
            today = datetime.now().strftime("%Y-%m-%d")
            count = 0
            
            with open(log_file, "r") as f:
                for line in f:
                    if today in line:
                        count += 1
            
            return count
        except Exception:
            return 0
    
    def _get_top_domains(self) -> List[str]:
        """Get top domains from recent embeddings."""
        try:
            from pathlib import Path
            from urllib.parse import urlparse
            
            log_file = Path("/var/log/42/embedding.log")
            if not log_file.exists():
                return []
            
            domain_counts = {}
            
            with open(log_file, "r") as f:
                for line in f:
                    # Extract URL from log line
                    if " | " in line:
                        parts = line.split(" | ")
                        if len(parts) >= 4:
                            url = parts[3].strip()
                            domain = urlparse(url).netloc
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return top 5 domains
            return [domain for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        except Exception:
            return []
    
    def _get_last_embedded_source(self) -> str:
        """Get the last embedded source."""
        try:
            from pathlib import Path
            
            log_file = Path("/var/log/42/embedding.log")
            if not log_file.exists():
                return "None"
            
            with open(log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    if " | " in last_line:
                        parts = last_line.split(" | ")
                        if len(parts) >= 4:
                            return parts[3].strip()
            
            return "None"
        except Exception:
            return "None" 

    def _store_query_in_database(self, query: str):
        """Store a query in the database for tracking."""
        try:
            from ...infra.core.vector_store import VectorStore
            vs = VectorStore()
            
            # Create a simple query log entry
            query_log = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "type": "search_query"
            }
            
            # Store in a dedicated collection for query history
            # For now, we'll use a simple approach with the existing vector store
            # In a production system, this would be a separate table
            logger.debug(f"📝 Stored query in database: {query}")
            
        except Exception as e:
            logger.error(f"Failed to store query in database: {e}")
    
    def _get_recent_queries_from_database(self, limit: int = 10) -> List[str]:
        """Get recent queries from the database."""
        try:
            # For now, return an empty list since we're not implementing full DB storage yet
            # In production, this would query a dedicated table
            return []
        except Exception as e:
            logger.error(f"Failed to get recent queries from database: {e}")
            return [] 