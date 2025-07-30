"""
Source Scanner for 42.un

This module implements the source scanning functionality for 42.un,
enabling constant monitoring of data sources and automatic event emission.
"""

import asyncio
import os
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
import feedparser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger

from .redis_bus import RedisBus
from .events import Event, EventType, create_github_repo_updated_event, create_file_ingested_event


class SourceScanner:
    """Base scanner for monitoring data sources."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.running = False
        self.scan_interval = config.get("scan_interval", 300)
        self.last_scan = datetime.now()
        
        # State tracking
        self.last_states = {}
        self.error_count = 0
        self.max_errors = config.get("max_errors", 5)
    
    async def start(self):
        """Start the scanner."""
        logger.info(f"Starting {self.__class__.__name__}")
        self.running = True
        await self._scan_loop()
    
    async def stop(self):
        """Stop the scanner."""
        logger.info(f"Stopping {self.__class__.__name__}")
        self.running = False
    
    async def _scan_loop(self):
        """Main scanning loop."""
        while self.running:
            try:
                await self._scan_sources()
                self.error_count = 0  # Reset error count on successful scan
            except Exception as e:
                self.error_count += 1
                logger.error(f"Scanner error ({self.error_count}/{self.max_errors}): {e}")
                
                if self.error_count >= self.max_errors:
                    logger.error(f"Too many errors, stopping {self.__class__.__name__}")
                    break
            
            await asyncio.sleep(self.scan_interval)
    
    async def _scan_sources(self):
        """Scan all configured sources."""
        raise NotImplementedError
    
    def _get_state_key(self, source_id: str) -> str:
        """Get Redis key for storing source state."""
        return f"scanner:state:{self.__class__.__name__}:{source_id}"
    
    async def _store_state(self, source_id: str, state: Dict[str, Any]):
        """Store current state for a source."""
        key = self._get_state_key(source_id)
        await self.redis_bus.redis.set(key, json.dumps(state), ex=86400)  # 24h expiry
    
    async def _get_stored_state(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get stored state for a source."""
        key = self._get_state_key(source_id)
        state_json = await self.redis_bus.redis.get(key)
        if state_json:
            return json.loads(state_json)
        return None


class GitHubScanner(SourceScanner):
    """Monitor GitHub repositories for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.repos = config.get("github", {}).get("repos", [])
        self.webhook_secret = config.get("github", {}).get("webhook_secret")
        self.api_token = config.get("github", {}).get("api_token")
        self.api_base = "https://api.github.com"
        
        # Session for API requests
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"token {self.api_token}"
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _scan_sources(self):
        """Scan GitHub repositories for changes."""
        session = await self._get_session()
        
        for repo_url in self.repos:
            try:
                await self._check_repo_changes(repo_url, session)
            except Exception as e:
                logger.error(f"Error checking repo {repo_url}: {e}")
    
    async def _check_repo_changes(self, repo_url: str, session: aiohttp.ClientSession):
        """Check for changes in a specific repository."""
        # Extract owner and repo from URL
        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 2:
            logger.error(f"Invalid GitHub URL: {repo_url}")
            return
        
        owner, repo = parts[-2], parts[-1]
        
        # Get latest commit
        api_url = f"{self.api_base}/repos/{owner}/{repo}/commits"
        params = {"per_page": 1}
        
        async with session.get(api_url, params=params) as response:
            if response.status != 200:
                logger.error(f"GitHub API error: {response.status}")
                return
            
            commits = await response.json()
            if not commits:
                return
            
            latest_commit = commits[0]
            commit_hash = latest_commit["sha"]
            commit_date = datetime.fromisoformat(latest_commit["commit"]["author"]["date"].replace('Z', '+00:00'))
            
            # Check if this is a new commit
            stored_state = await self._get_stored_state(repo_url)
            if stored_state and stored_state.get("last_commit") == commit_hash:
                return  # No change
            
            # Store new state
            new_state = {
                "last_commit": commit_hash,
                "last_commit_date": commit_date.isoformat(),
                "last_scan": datetime.now().isoformat()
            }
            await self._store_state(repo_url, new_state)
            
            # Emit event
            event = create_github_repo_updated_event(
                repo_url=repo_url,
                commit_hash=commit_hash,
                branch="main",  # Default branch
                author=latest_commit["commit"]["author"]["name"],
                message=latest_commit["commit"]["message"]
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"GitHub change detected: {repo_url} -> {commit_hash[:8]}")
    
    async def stop(self):
        """Stop the scanner and close session."""
        await super().stop()
        if self.session and not self.session.closed:
            await self.session.close()


class FileSystemScanner(SourceScanner):
    """Monitor local file system for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.watch_directories = config.get("filesystem", {}).get("watch_directories", [])
        self.ignore_patterns = config.get("filesystem", {}).get("ignore_patterns", ["*.tmp", "*.log"])
        self.observer = None
        self.handler = None
    
    async def start(self):
        """Start file system monitoring."""
        await super().start()
        await self._setup_watchers()
    
    async def _setup_watchers(self):
        """Setup file system watchers."""
        if not self.watch_directories:
            logger.warning("No directories configured for file system monitoring")
            return
        
        self.handler = FileSystemEventHandler()
        self.handler.on_created = self._on_file_created
        self.handler.on_modified = self._on_file_modified
        self.handler.on_deleted = self._on_file_deleted
        
        self.observer = Observer()
        
        for directory in self.watch_directories:
            if os.path.exists(directory):
                self.observer.schedule(self.handler, directory, recursive=True)
                logger.info(f"Watching directory: {directory}")
            else:
                logger.warning(f"Directory does not exist: {directory}")
        
        self.observer.start()
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        path = Path(file_path)
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if path.match(pattern):
                return True
        
        # Ignore hidden files
        if path.name.startswith('.'):
            return True
        
        # Ignore common non-text files
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db'}
        if path.suffix.lower() in binary_extensions:
            return True
        
        return False
    
    def _on_file_created(self, event):
        """Handle file creation event."""
        if event.is_directory or self._should_ignore_file(event.src_path):
            return
        
        asyncio.create_task(self._emit_file_event(event.src_path, "created"))
    
    def _on_file_modified(self, event):
        """Handle file modification event."""
        if event.is_directory or self._should_ignore_file(event.src_path):
            return
        
        asyncio.create_task(self._emit_file_event(event.src_path, "modified"))
    
    def _on_file_deleted(self, event):
        """Handle file deletion event."""
        if event.is_directory or self._should_ignore_file(event.src_path):
            return
        
        asyncio.create_task(self._emit_file_event(event.src_path, "deleted"))
    
    async def _emit_file_event(self, file_path: str, event_type: str):
        """Emit file system event."""
        try:
            # Get file stats
            stat = os.stat(file_path)
            file_size = stat.st_size
            
            # Only process text files under reasonable size
            if file_size > 1024 * 1024:  # 1MB limit
                return
            
            event = Event(
                event_type=EventType.FILE_INGESTED if event_type in ["created", "modified"] else EventType.FILE_DELETED,
                data={
                    "file_path": file_path,
                    "file_size": file_size,
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="filesystem_scanner"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"File system event: {event_type} -> {file_path}")
            
        except Exception as e:
            logger.error(f"Error emitting file event: {e}")
    
    async def _scan_sources(self):
        """Periodic scan of file system (backup to real-time events)."""
        # This is mainly for initial scan and verification
        # Real-time events are handled by the file system watcher
        pass
    
    async def stop(self):
        """Stop the scanner and observer."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        await super().stop()


class RSSFeedScanner(SourceScanner):
    """Monitor RSS/Atom feeds for updates."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.feeds = config.get("rss", {}).get("feeds", [])
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _scan_sources(self):
        """Scan RSS feeds for new content."""
        session = await self._get_session()
        
        for feed_url in self.feeds:
            try:
                await self._check_feed_updates(feed_url, session)
            except Exception as e:
                logger.error(f"Error checking feed {feed_url}: {e}")
    
    async def _check_feed_updates(self, feed_url: str, session: aiohttp.ClientSession):
        """Check for updates in an RSS feed."""
        async with session.get(feed_url) as response:
            if response.status != 200:
                logger.error(f"Feed fetch error: {response.status}")
                return
            
            content = await response.text()
            feed = feedparser.parse(content)
            
            if not feed.entries:
                return
            
            # Get latest entry
            latest_entry = feed.entries[0]
            entry_id = latest_entry.get("id", latest_entry.get("link", ""))
            
            # Check if this is a new entry
            stored_state = await self._get_stored_state(feed_url)
            if stored_state and stored_state.get("last_entry_id") == entry_id:
                return  # No change
            
            # Store new state
            new_state = {
                "last_entry_id": entry_id,
                "last_entry_title": latest_entry.get("title", ""),
                "last_entry_date": latest_entry.get("published", ""),
                "last_scan": datetime.now().isoformat()
            }
            await self._store_state(feed_url, new_state)
            
            # Emit event
            event = Event(
                event_type=EventType.FILE_INGESTED,  # Reuse for feed updates
                data={
                    "feed_url": feed_url,
                    "entry_title": latest_entry.get("title", ""),
                    "entry_link": latest_entry.get("link", ""),
                    "entry_id": entry_id,
                    "feed_title": feed.feed.get("title", ""),
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="rss_scanner"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"RSS update detected: {feed_url} -> {latest_entry.get('title', '')[:50]}")
    
    async def stop(self):
        """Stop the scanner and close session."""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().stop()


class APIEndpointScanner(SourceScanner):
    """Monitor API endpoints for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.endpoints = config.get("api", {}).get("endpoints", [])
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _scan_sources(self):
        """Scan API endpoints for changes."""
        session = await self._get_session()
        
        for endpoint_config in self.endpoints:
            try:
                await self._check_endpoint_changes(endpoint_config, session)
            except Exception as e:
                logger.error(f"Error checking endpoint {endpoint_config.get('url', '')}: {e}")
    
    async def _check_endpoint_changes(self, endpoint_config: Dict[str, Any], session: aiohttp.ClientSession):
        """Check for changes in an API endpoint."""
        url = endpoint_config.get("url")
        method = endpoint_config.get("method", "GET")
        headers = endpoint_config.get("headers", {})
        
        if not url:
            return
        
        # Make request
        async with session.request(method, url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"API endpoint error: {response.status}")
                return
            
            content = await response.text()
            
            # Create hash of content for change detection
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if content has changed
            stored_state = await self._get_stored_state(url)
            if stored_state and stored_state.get("content_hash") == content_hash:
                return  # No change
            
            # Store new state
            new_state = {
                "content_hash": content_hash,
                "last_scan": datetime.now().isoformat(),
                "response_size": len(content)
            }
            await self._store_state(url, new_state)
            
            # Emit event
            event = Event(
                event_type=EventType.FILE_INGESTED,  # Reuse for API updates
                data={
                    "endpoint_url": url,
                    "method": method,
                    "content_hash": content_hash,
                    "response_size": len(content),
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="api_scanner"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"API endpoint change detected: {url}")
    
    async def stop(self):
        """Stop the scanner and close session."""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().stop()


class SourceScannerOrchestrator:
    """Orchestrate all source scanners."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.scanners = []
        self.running = False
    
    async def start(self):
        """Start all scanners."""
        logger.info("Starting Source Scanner Orchestrator")
        self.running = True
        
        # Create scanners based on configuration
        scanners = []
        
        if self.config.get("github", {}).get("repos"):
            scanners.append(GitHubScanner(self.redis_bus, self.config))
        
        if self.config.get("filesystem", {}).get("watch_directories"):
            scanners.append(FileSystemScanner(self.redis_bus, self.config))
        
        if self.config.get("rss", {}).get("feeds"):
            scanners.append(RSSFeedScanner(self.redis_bus, self.config))
        
        if self.config.get("api", {}).get("endpoints"):
            scanners.append(APIEndpointScanner(self.redis_bus, self.config))
        
        # Start all scanners
        for scanner in scanners:
            self.scanners.append(scanner)
            asyncio.create_task(scanner.start())
        
        logger.info(f"Started {len(scanners)} scanners")
    
    async def stop(self):
        """Stop all scanners."""
        logger.info("Stopping Source Scanner Orchestrator")
        self.running = False
        
        for scanner in self.scanners:
            await scanner.stop()
        
        self.scanners.clear()
        logger.info("All scanners stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all scanners."""
        return {
            "running": self.running,
            "scanner_count": len(self.scanners),
            "scanners": [
                {
                    "type": scanner.__class__.__name__,
                    "running": scanner.running,
                    "error_count": scanner.error_count
                }
                for scanner in self.scanners
            ]
        } 