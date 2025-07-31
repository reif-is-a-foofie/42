"""
Autonomous Source Scanner for 42.un

An intelligent web crawler that continuously discovers new sources of information,
learns from the knowledge base, and autonomously explores the web for relevant content.
"""

import asyncio
import json
import hashlib
import re
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
class DiscoveredSource:
    """Represents a discovered information source."""
    url: str
    title: str
    description: str
    source_type: str  # "article", "rss", "api", "blog", "documentation"
    domain: str
    relevance_score: float
    discovery_method: str  # "crawl", "rss_discovery", "link_following", "semantic_search"
    metadata: Dict[str, Any]
    discovered_at: datetime


@dataclass
class CrawlTarget:
    """Represents a target for web crawling."""
    url: str
    priority: float
    crawl_depth: int
    source_type: str
    discovered_from: Optional[str] = None


class AutonomousScanner:
    """
    Autonomous web scanner that continuously discovers new sources of information.
    
    Features:
    - Web crawling with Playwright
    - RSS feed discovery
    - Link extraction and following
    - Semantic relevance scoring
    - Learning from knowledge base
    - Autonomous exploration
    """
    
    def __init__(self, redis_bus: RedisBus, knowledge_engine: KnowledgeEngine, config):
        self.redis_bus = redis_bus
        self.knowledge_engine = knowledge_engine
        self.config = config
        self.running = False
        
        # Crawling configuration
        self.max_depth = getattr(config, "max_depth", 3)
        self.max_pages_per_domain = getattr(config, "max_pages_per_domain", 10)
        self.crawl_delay = getattr(config, "crawl_delay", 1.0)
        self.user_agent = getattr(config, "user_agent", "42-AutonomousScanner/1.0")
        
        # Discovery state
        self.discovered_sources: Set[str] = set()
        self.crawled_domains: Set[str] = set()
        self.pending_targets: List[CrawlTarget] = []
        self.learned_patterns: Dict[str, float] = {}
        
        # Browser and session management
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Newspaper configuration for article extraction
        self.newspaper_config = NewspaperConfig()
        self.newspaper_config.browser_user_agent = self.user_agent
        self.newspaper_config.request_timeout = 10
    
    async def start(self):
        """Start the autonomous scanner."""
        logger.info("Starting autonomous source scanner")
        self.running = True
        
        # Initialize browser and session
        await self._setup_browser()
        await self._setup_session()
        
        # Start the main discovery loop
        await self._discovery_loop()
    
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
            headers={"User-Agent": self.user_agent},
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def _discovery_loop(self):
        """Main discovery loop."""
        while self.running:
            try:
                # Learn from knowledge base to guide discovery
                await self._learn_from_knowledge_base()
                
                # Process pending crawl targets
                await self._process_crawl_targets()
                
                # Discover new RSS feeds
                await self._discover_rss_feeds()
                
                # Follow discovered links
                await self._follow_discovered_links()
                
                # Semantic search for new sources
                await self._semantic_source_discovery()
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get("discovery_interval", 300))
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
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
        # For now, return a sample of domains we know are valuable
        return [
            {"url": "https://dynomight.net", "score": 0.9},
            {"url": "https://arxiv.org", "score": 0.8},
            {"url": "https://github.com", "score": 0.7},
        ]
    
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
    
    async def _extract_rss_links(self, soup: BeautifulSoup, base_url: str) -> List[DiscoveredSource]:
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
                        rss_sources.append(DiscoveredSource(
                            url=full_url,
                            title=element.get('title', 'RSS Feed'),
                            description='RSS/Atom feed discovered',
                            source_type='rss',
                            domain=urlparse(full_url).netloc,
                            relevance_score=0.7,
                            discovery_method='rss_discovery',
                            metadata={'base_url': base_url},
                            discovered_at=datetime.now()
                        ))
                        self.discovered_sources.add(full_url)
        
        return rss_sources
    
    async def _extract_articles(self, soup: BeautifulSoup, base_url: str) -> List[DiscoveredSource]:
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
                        
                        articles.append(DiscoveredSource(
                            url=article_url,
                            title=title,
                            description=description,
                            source_type='article',
                            domain=urlparse(article_url).netloc,
                            relevance_score=0.6,
                            discovery_method='article_extraction',
                            metadata={'base_url': base_url},
                            discovered_at=datetime.now()
                        ))
                        self.discovered_sources.add(article_url)
        
        return articles
    
    async def _process_discovered_sources(self, sources: List[DiscoveredSource], discovered_from: str):
        """Process and score discovered sources."""
        for source in sources:
            try:
                # Score the source based on various factors
                source.relevance_score = await self._score_source(source)
                
                # If score is high enough, add to knowledge engine
                if source.relevance_score > 0.5:
                    await self._add_source_to_knowledge_engine(source)
                    
                    # Emit discovery event
                    event = create_source_discovered_event(
                        url=source.url,
                        source_type=source.source_type,
                        domain=source.domain,
                        title=source.title,
                        relevance_score=source.relevance_score,
                        discovered_from=discovered_from
                    )
                    await self.redis_bus.publish_event(event)
                    
                    logger.info(f"Discovered valuable source: {source.url} (score: {source.relevance_score:.2f})")
                
            except Exception as e:
                logger.error(f"Error processing source {source.url}: {e}")
    
    async def _score_source(self, source: DiscoveredSource) -> float:
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
    
    async def _add_source_to_knowledge_engine(self, source: DiscoveredSource):
        """Add discovered source to knowledge engine."""
        # Convert to knowledge source format
        knowledge_source = {
            "id": f"discovered_{hashlib.md5(source.url.encode()).hexdigest()}",
            "name": source.title,
            "type": "rss" if source.source_type == "rss" else "api",
            "domain": "research",
            "url": source.url,
            "frequency": "5min",
            "parser": "xml" if source.source_type == "rss" else "json",
            "vectorize": True,
            "active": True,
            "metadata": {
                "description": source.description,
                "discovery_method": source.discovery_method,
                "relevance_score": source.relevance_score
            }
        }
        
        # Add to knowledge engine (this would integrate with existing knowledge engine)
        logger.info(f"Adding discovered source to knowledge engine: {source.url}")
    
    async def _add_discovered_links(self, links: List[str], source_target: CrawlTarget):
        """Add discovered links to pending targets."""
        for link in links:
            domain = urlparse(link).netloc
            if domain and domain not in self.crawled_domains:
                # Score the link based on various factors
                priority = await self._score_link(link, source_target)
                
                if priority > 0.3:  # Only add if reasonably relevant
                    self.pending_targets.append(CrawlTarget(
                        url=link,
                        priority=priority,
                        crawl_depth=source_target.crawl_depth - 1,
                        source_type="discovered_link",
                        discovered_from=source_target.url
                    ))
    
    async def _score_link(self, url: str, source_target: CrawlTarget) -> float:
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
        """Get scanner status."""
        return {
            "running": self.running,
            "discovered_sources_count": len(self.discovered_sources),
            "crawled_domains_count": len(self.crawled_domains),
            "pending_targets_count": len(self.pending_targets),
            "learned_patterns_count": len(self.learned_patterns)
        } 