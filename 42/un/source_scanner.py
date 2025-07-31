"""
Source Scanner for 42.un

This module implements the source scanning functionality for 42.un,
enabling constant monitoring of data sources and automatic event emission.
"""

import asyncio
import os
import hashlib
import json
import aiohttp
import feedparser
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
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
        if "github.com" in repo_url:
            parts = repo_url.split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
            else:
                logger.error(f"Invalid GitHub URL format: {repo_url}")
                return
        else:
            logger.error(f"Not a GitHub URL: {repo_url}")
            return
        
        # Get latest commit
        api_url = f"{self.api_base}/repos/{owner}/{repo}/commits"
        async with session.get(api_url, params={"per_page": 1}) as response:
            if response.status != 200:
                logger.error(f"GitHub API error: {response.status}")
                return
            
            commits = await response.json()
            if not commits:
                return
            
            latest_commit = commits[0]
            commit_sha = latest_commit["sha"]
            commit_date = latest_commit["commit"]["author"]["date"]
            
            # Check if this is a new commit
            stored_state = await self._get_stored_state(repo_url)
            if stored_state and stored_state.get("last_commit_sha") == commit_sha:
                return  # No change
            
            # Store new state
            new_state = {
                "last_commit_sha": commit_sha,
                "last_commit_date": commit_date,
                "last_scan": datetime.now().isoformat()
            }
            await self._store_state(repo_url, new_state)
            
            # Emit event
            event = create_github_repo_updated_event(
                repo_url=repo_url,
                commit_sha=commit_sha,
                commit_message=latest_commit["commit"]["message"],
                author=latest_commit["commit"]["author"]["name"],
                timestamp=commit_date
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"GitHub update detected: {repo_url} -> {commit_sha[:8]}")
    
    async def stop(self):
        """Stop the scanner and close session."""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().stop()


class FileSystemScanner(SourceScanner):
    """Monitor local file system for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.watch_directories = config.get("filesystem", {}).get("watch_directories", [])
        self.observer = None
        self.event_handler = None
    
    async def start(self):
        """Start file system monitoring."""
        await super().start()
        await self._setup_watchers()
    
    async def _setup_watchers(self):
        """Setup file system watchers."""
        if not self.watch_directories:
            return
        
        self.observer = Observer()
        self.event_handler = FileSystemEventHandler()
        
        # Set up event handlers
        self.event_handler.on_created = self._on_file_created
        self.event_handler.on_modified = self._on_file_modified
        self.event_handler.on_deleted = self._on_file_deleted
        
        # Schedule watchers
        for directory in self.watch_directories:
            if os.path.exists(directory):
                self.observer.schedule(self.event_handler, directory, recursive=True)
                logger.info(f"Watching directory: {directory}")
            else:
                logger.warning(f"Directory does not exist: {directory}")
        
        self.observer.start()
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = [
            ".git", ".svn", ".hg",
            "__pycache__", ".pyc", ".pyo",
            ".DS_Store", "Thumbs.db",
            "*.tmp", "*.temp", "*.log"
        ]
        
        for pattern in ignore_patterns:
            if pattern in file_path:
                return True
        return False
    
    def _on_file_created(self, event):
        """Handle file creation event."""
        if not event.is_directory and not self._should_ignore_file(event.src_path):
            asyncio.create_task(self._emit_file_event(event.src_path, "created"))
    
    def _on_file_modified(self, event):
        """Handle file modification event."""
        if not event.is_directory and not self._should_ignore_file(event.src_path):
            asyncio.create_task(self._emit_file_event(event.src_path, "modified"))
    
    def _on_file_deleted(self, event):
        """Handle file deletion event."""
        if not event.is_directory and not self._should_ignore_file(event.src_path):
            asyncio.create_task(self._emit_file_event(event.src_path, "deleted"))
    
    async def _emit_file_event(self, file_path: str, event_type: str):
        """Emit file system event."""
        try:
            # Get file metadata
            stat = os.stat(file_path)
            file_size = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            event = create_file_ingested_event(
                file_path=file_path,
                event_type=event_type,
                file_size=file_size,
                modified_time=modified_time.isoformat()
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"File system event: {event_type} -> {file_path}")
            
        except Exception as e:
            logger.error(f"Error emitting file event: {e}")
    
    async def _scan_sources(self):
        """Scan file system sources (called periodically)."""
        # This is mainly for initial scan, ongoing monitoring is via watchers
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
                logger.error(f"Error checking endpoint {endpoint_config.get('url', 'unknown')}: {e}")
    
    async def _check_endpoint_changes(self, endpoint_config: Dict[str, Any], session: aiohttp.ClientSession):
        """Check for changes in an API endpoint."""
        url = endpoint_config.get("url")
        if not url:
            return
        
        headers = endpoint_config.get("headers", {})
        method = endpoint_config.get("method", "GET")
        
        async with session.request(method, url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"API endpoint error: {response.status}")
                return
            
            content = await response.text()
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
                    "content_hash": content_hash,
                    "response_size": len(content),
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="api_scanner"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"API endpoint update detected: {url}")
    
    async def stop(self):
        """Stop the scanner and close session."""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().stop()


class SourceScannerOrchestrator:
    """Orchestrates multiple source scanners."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.scanners = []
        self.running = False
    
    def add_scanner(self, scanner: SourceScanner):
        """Add a scanner to the orchestrator."""
        self.scanners.append(scanner)
    
    async def start(self):
        """Start all scanners."""
        logger.info("Starting source scanner orchestrator")
        self.running = True
        
        # Start all scanners concurrently
        tasks = [scanner.start() for scanner in self.scanners]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all scanners."""
        logger.info("Stopping source scanner orchestrator")
        self.running = False
        
        # Stop all scanners
        for scanner in self.scanners:
            await scanner.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all scanners."""
        return {
            "running": self.running,
            "scanner_count": len(self.scanners),
            "scanners": [scanner.__class__.__name__ for scanner in self.scanners]
        } 