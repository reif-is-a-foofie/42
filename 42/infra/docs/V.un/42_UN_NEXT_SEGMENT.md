# 42.un Next Segment Implementation Plan

## Current Status ✅

### **Completed Components:**
- ✅ **Event System** (`42/un/events.py`)
  - Event types and schemas defined
  - Event serialization/deserialization
  - Factory functions for common events
  - Comprehensive event coverage

- ✅ **Redis Event Bus** (`42/un/redis_bus.py`)
  - Redis connection with pooling
  - Event publishing and subscription
  - Event persistence and history
  - Channel management and filtering

- ✅ **Basic Infrastructure**
  - Docker Redis service configured
  - Event system foundation
  - Basic pub/sub functionality

## Next Segment: Source Scanner Implementation 🚧

### **Week 2 Priority: Source Scanner (`42/un/source_scanner.py`)**

The source scanner is the **core component** that will enable 42.un to constantly monitor for new data sources and trigger the event system.

## Implementation Plan

### **Phase 1: Core Scanner Framework (Days 1-2)**

#### **1.1 Base Scanner Class**
```python
class SourceScanner:
    """Base scanner for monitoring data sources."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.running = False
        self.scan_interval = config.get("scan_interval", 300)
    
    async def start(self):
        """Start the scanner."""
        self.running = True
        await self._scan_loop()
    
    async def stop(self):
        """Stop the scanner."""
        self.running = False
    
    async def _scan_loop(self):
        """Main scanning loop."""
        while self.running:
            await self._scan_sources()
            await asyncio.sleep(self.scan_interval)
    
    async def _scan_sources(self):
        """Scan all configured sources."""
        raise NotImplementedError
```

#### **1.2 GitHub Scanner**
```python
class GitHubScanner(SourceScanner):
    """Monitor GitHub repositories for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.repos = config.get("github", {}).get("repos", [])
        self.webhook_secret = config.get("github", {}).get("webhook_secret")
        self.api_token = config.get("github", {}).get("api_token")
    
    async def _scan_sources(self):
        """Scan GitHub repositories for changes."""
        for repo in self.repos:
            await self._check_repo_changes(repo)
    
    async def _check_repo_changes(self, repo_url: str):
        """Check for changes in a specific repository."""
        # Implementation: Check last commit, compare with stored state
        # Emit events for detected changes
```

#### **1.3 File System Scanner**
```python
class FileSystemScanner(SourceScanner):
    """Monitor local file system for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.watch_directories = config.get("filesystem", {}).get("watch_directories", [])
        self.watcher = None
    
    async def start(self):
        """Start file system monitoring."""
        await super().start()
        await self._setup_watchers()
    
    async def _setup_watchers(self):
        """Setup file system watchers."""
        # Implementation: Use watchdog for file system events
        # Emit events for file changes
```

### **Phase 2: Webhook Integration (Days 3-4)**

#### **2.1 GitHub Webhook Handler**
```python
class GitHubWebhookHandler:
    """Handle GitHub webhook events."""
    
    def __init__(self, redis_bus: RedisBus, secret: str):
        self.redis_bus = redis_bus
        self.secret = secret
    
    async def handle_webhook(self, payload: Dict[str, Any], signature: str):
        """Handle incoming GitHub webhook."""
        if not self._verify_signature(payload, signature):
            raise ValueError("Invalid webhook signature")
        
        event_type = payload.get("ref_type")
        if event_type == "push":
            await self._handle_push_event(payload)
        elif event_type == "pull_request":
            await self._handle_pr_event(payload)
    
    async def _handle_push_event(self, payload: Dict[str, Any]):
        """Handle push events."""
        repo_url = payload["repository"]["html_url"]
        commit_hash = payload["after"]
        
        event = create_github_repo_updated_event(
            repo_url=repo_url,
            commit_hash=commit_hash,
            branch=payload["ref"],
            author=payload["pusher"]["name"]
        )
        
        await self.redis_bus.publish_event(event)
```

#### **2.2 FastAPI Webhook Endpoint**
```python
@app.post("/un/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature: str = Header(None)
):
    """GitHub webhook endpoint."""
    payload = await request.json()
    
    handler = GitHubWebhookHandler(redis_bus, config["github"]["webhook_secret"])
    await handler.handle_webhook(payload, x_hub_signature)
    
    return {"status": "ok"}
```

### **Phase 3: RSS/API Monitoring (Days 5-6)**

#### **3.1 RSS Feed Scanner**
```python
class RSSFeedScanner(SourceScanner):
    """Monitor RSS/Atom feeds for updates."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.feeds = config.get("rss", {}).get("feeds", [])
        self.last_checks = {}
    
    async def _scan_sources(self):
        """Scan RSS feeds for new content."""
        for feed_url in self.feeds:
            await self._check_feed_updates(feed_url)
    
    async def _check_feed_updates(self, feed_url: str):
        """Check for updates in an RSS feed."""
        # Implementation: Parse RSS feed, compare with last check
        # Emit events for new articles
```

#### **3.2 API Endpoint Monitor**
```python
class APIEndpointScanner(SourceScanner):
    """Monitor API endpoints for changes."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        super().__init__(redis_bus, config)
        self.endpoints = config.get("api", {}).get("endpoints", [])
    
    async def _scan_sources(self):
        """Scan API endpoints for changes."""
        for endpoint in self.endpoints:
            await self._check_endpoint_changes(endpoint)
```

### **Phase 4: Integration & Testing (Days 7)**

#### **4.1 Main Scanner Orchestrator**
```python
class SourceScannerOrchestrator:
    """Orchestrate all source scanners."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.scanners = []
    
    async def start(self):
        """Start all scanners."""
        scanners = [
            GitHubScanner(self.redis_bus, self.config),
            FileSystemScanner(self.redis_bus, self.config),
            RSSFeedScanner(self.redis_bus, self.config),
            APIEndpointScanner(self.redis_bus, self.config)
        ]
        
        for scanner in scanners:
            self.scanners.append(scanner)
            await scanner.start()
    
    async def stop(self):
        """Stop all scanners."""
        for scanner in self.scanners:
            await scanner.stop()
```

## File Structure for Next Segment

```
42/
├── un/
│   ├── __init__.py              # Updated imports
│   ├── events.py                # ✅ Complete
│   ├── redis_bus.py             # ✅ Complete
│   ├── source_scanner.py        # 🚧 Next implementation
│   │   ├── SourceScanner        # Base scanner class
│   │   ├── GitHubScanner        # GitHub monitoring
│   │   ├── FileSystemScanner    # File system monitoring
│   │   ├── RSSFeedScanner       # RSS feed monitoring
│   │   ├── APIEndpointScanner   # API endpoint monitoring
│   │   └── SourceScannerOrchestrator # Main orchestrator
│   ├── webhook_handlers.py      # 🚧 Webhook handlers
│   │   ├── GitHubWebhookHandler # GitHub webhook processing
│   │   └── WebhookValidator     # Signature validation
│   └── config.py                # 🚧 Scanner configuration
```

## Configuration Updates

### **Updated 42.config.json:**
```json
{
  "un": {
    "scan_interval": 300,
    "github": {
      "repos": [
        "https://github.com/user/repo1",
        "https://github.com/user/repo2"
      ],
      "webhook_secret": "your-webhook-secret",
      "api_token": "your-github-token"
    },
    "filesystem": {
      "watch_directories": [
        "/path/to/watch1",
        "/path/to/watch2"
      ],
      "ignore_patterns": ["*.tmp", "*.log"]
    },
    "rss": {
      "feeds": [
        "https://blog.example.com/feed",
        "https://docs.example.com/feed"
      ]
    },
    "api": {
      "endpoints": [
        {
          "url": "https://api.example.com/data",
          "method": "GET",
          "headers": {"Authorization": "Bearer token"}
        }
      ]
    }
  }
}
```

## CLI Commands for Testing

```bash
# Start source scanner
python3 -m 42 un scanner start

# Stop source scanner
python3 -m 42 un scanner stop

# View scanner status
python3 -m 42 un scanner status

# Test GitHub webhook
curl -X POST http://localhost:8000/un/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature: sha1=..." \
  -d '{"ref_type": "push", "repository": {...}}'

# View scanner events
python3 -m 42 un events --source scanner
```

## Testing Strategy

### **Unit Tests:**
```python
# tests/test_un_source_scanner.py
def test_github_scanner_initialization():
    """Test GitHub scanner setup."""
    
def test_file_system_scanner_watchers():
    """Test file system watcher setup."""
    
def test_rss_feed_scanner_parsing():
    """Test RSS feed parsing."""
    
def test_webhook_signature_verification():
    """Test webhook signature validation."""
```

### **Integration Tests:**
```python
def test_scanner_event_emission():
    """Test that scanners emit events."""
    
def test_webhook_to_event_flow():
    """Test webhook → event → Redis flow."""
    
def test_multiple_scanner_orchestration():
    """Test orchestrator with multiple scanners."""
```

## Success Metrics

1. **Response Time**: <5 seconds to detect source changes
2. **Event Emission**: All source changes emit events
3. **Webhook Processing**: 100% webhook signature validation
4. **File Monitoring**: Real-time file system change detection
5. **Feed Monitoring**: RSS feed updates detected within scan interval

## Dependencies to Add

```txt
# .config/requirements.txt additions
watchdog==3.0.0          # File system monitoring
feedparser==6.0.10       # RSS feed parsing
aiohttp==3.9.1           # Async HTTP requests
cryptography==41.0.0     # Webhook signature verification
```

## Implementation Timeline

### **Day 1-2: Core Framework**
- [ ] Base SourceScanner class
- [ ] GitHubScanner implementation
- [ ] FileSystemScanner implementation
- [ ] Basic event emission

### **Day 3-4: Webhook Integration**
- [ ] GitHubWebhookHandler
- [ ] Webhook signature verification
- [ ] FastAPI webhook endpoint
- [ ] Webhook to event conversion

### **Day 5-6: Additional Sources**
- [ ] RSSFeedScanner implementation
- [ ] APIEndpointScanner implementation
- [ ] Configuration management
- [ ] Error handling

### **Day 7: Integration & Testing**
- [ ] SourceScannerOrchestrator
- [ ] CLI commands
- [ ] Comprehensive testing
- [ ] Documentation updates

## Next Steps After Source Scanner

1. **Week 3**: Task Prioritizer implementation
2. **Week 4**: Background Worker implementation
3. **Week 5**: Integration testing and optimization

This plan delivers a **comprehensive source scanning system** that will enable 42.un to automatically detect and respond to changes across multiple data sources. 🚀 