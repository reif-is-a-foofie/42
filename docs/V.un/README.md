# V.un Documentation

## Overview
V.un represents **Phase un** in the 42 masterplan - the "Reflex and ingestion" phase. This version adds autonomous monitoring, event-driven processing, and background task execution.

## Current Status 🚧

### **Completed Components ✅**
- **Event System** (`42/un/events.py`)
  - Event types and schemas defined
  - Event serialization/deserialization
  - Factory functions for common events
  - Comprehensive event coverage

- **Redis Event Bus** (`42/un/redis_bus.py`)
  - Redis connection with pooling
  - Event publishing and subscription
  - Event persistence and history
  - Channel management and filtering

- **Basic Infrastructure**
  - Docker Redis service configured
  - Event system foundation
  - Basic pub/sub functionality

### **In Development 🚧**
- **Source Scanner** - Constant monitoring of data sources
- **Task Prioritizer** - Bayesian scoring and prioritization
- **Background Worker** - Async task execution

## Implementation Plan

### **Week 2: Source Scanner Implementation**
**Priority**: Source Scanner (`42/un/source_scanner.py`)

#### **Phase 1: Core Scanner Framework (Days 1-2)**
- [ ] Base SourceScanner class
- [ ] GitHubScanner for repository monitoring
- [ ] FileSystemScanner for local file changes
- [ ] Basic event emission

#### **Phase 2: Webhook Integration (Days 3-4)**
- [ ] GitHubWebhookHandler
- [ ] Webhook signature verification
- [ ] FastAPI webhook endpoints
- [ ] Webhook to event conversion

#### **Phase 3: Additional Sources (Days 5-6)**
- [ ] RSSFeedScanner for blog/documentation feeds
- [ ] APIEndpointScanner for external APIs
- [ ] Configuration management
- [ ] Error handling

#### **Phase 4: Integration & Testing (Day 7)**
- [ ] SourceScannerOrchestrator
- [ ] CLI commands
- [ ] Comprehensive testing
- [ ] Documentation updates

### **Week 3: Task Prioritizer**
- [ ] Bayesian task scoring
- [ ] Priority queue with dynamic reordering
- [ ] Task dependency resolution
- [ ] Resource availability checking

### **Week 4: Background Worker**
- [ ] Worker pool with configurable concurrency
- [ ] Task execution with timeout handling
- [ ] Progress tracking and status updates
- [ ] Error handling and retry logic

### **Week 5: Integration & Optimization**
- [ ] End-to-end workflows
- [ ] Performance testing
- [ ] Error recovery scenarios
- [ ] Production deployment

## Documentation Files

### **42_UN_PLAN.md**
Comprehensive implementation plan for V.un phase including architecture, components, and timeline.

### **42_UN_NEXT_SEGMENT.md**
Detailed plan for the next segment (Source Scanner) with implementation phases, code examples, and testing strategy.

## Core Components

### **1. Redis Event Bus**
```python
# Event publishing
await redis_bus.publish_event(event)

# Event subscription
await redis_bus.subscribe_to_events([EventType.GITHUB_REPO_UPDATED], callback)
```

### **2. Source Scanner (In Development)**
```python
# Monitor GitHub repositories
github_scanner = GitHubScanner(redis_bus, config)
await github_scanner.start()

# Monitor file system changes
fs_scanner = FileSystemScanner(redis_bus, config)
await fs_scanner.start()
```

### **3. Task Prioritizer (Planned)**
```python
# Score and prioritize tasks
prioritizer = TaskPrioritizer()
priority = prioritizer.score_task(task)
```

### **4. Background Worker (Planned)**
```python
# Execute tasks in background
worker = BackgroundWorker(redis_bus, config)
await worker.start()
```

## Configuration

### **42.config.json additions:**
```json
{
  "un": {
    "scan_interval": 300,
    "github": {
      "repos": ["https://github.com/user/repo"],
      "webhook_secret": "your-secret",
      "api_token": "your-token"
    },
    "filesystem": {
      "watch_directories": ["/path/to/watch"]
    },
    "rss": {
      "feeds": ["https://blog.example.com/feed"]
    }
  }
}
```

## CLI Commands (Planned)

```bash
# Start 42.un background service
python3 -m 42 un start

# Stop background service
python3 -m 42 un stop

# View current tasks
python3 -m 42 un tasks

# View event history
python3 -m 42 un events

# Manually trigger source scan
python3 -m 42 un scan

# View worker status
python3 -m 42 un workers
```

## FastAPI Endpoints (Planned)

```python
# Event management
POST /un/events/publish
GET /un/events/history
POST /un/events/subscribe

# Task management
GET /un/tasks
POST /un/tasks/create
DELETE /un/tasks/{task_id}

# Source management
GET /un/sources
POST /un/sources/add
DELETE /un/sources/{source_id}
```

## Dependencies

```txt
redis==5.0.1              # Event bus
watchdog==3.0.0           # File system monitoring
feedparser==6.0.10        # RSS feed parsing
aiohttp==3.9.1            # Async HTTP requests
celery==5.3.4             # Background task execution
cryptography==41.0.0      # Webhook signature verification
```

## Success Metrics

1. **Event Processing**: 1000+ events/second
2. **Task Throughput**: 100+ tasks/minute
3. **Source Scanning**: <5 second response to changes
4. **Error Recovery**: 99.9% uptime
5. **Resource Usage**: <1GB memory, <10% CPU

## Architecture

```
42/
├── un/                    # V.un components
│   ├── events.py          # Event system ✅
│   ├── redis_bus.py       # Event bus ✅
│   ├── source_scanner.py  # Source monitoring 🚧
│   ├── task_prioritizer.py # Task scoring 🚧
│   ├── background_worker.py # Task execution 🚧
│   └── webhook_handlers.py # Webhook processing 🚧
├── 42/                    # V.zero components ✅
└── tests/                 # Test suite ✅
```

## Migration from V.zero

V.un builds upon V.zero by adding:
- ✅ **Event System** - Real-time event processing
- ✅ **Redis Integration** - Persistent event bus
- 🚧 **Source Monitoring** - Autonomous data source scanning
- 🚧 **Task Prioritization** - Intelligent task scoring
- 🚧 **Background Execution** - Async task processing

## Next Steps

1. **Complete Source Scanner** - Week 2 implementation
2. **Implement Task Prioritizer** - Week 3
3. **Build Background Worker** - Week 4
4. **Integration Testing** - Week 5

V.un will enable **autonomous, event-driven code analysis**! 🚀 