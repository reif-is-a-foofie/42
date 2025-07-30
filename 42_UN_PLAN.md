# 42.un Implementation Plan

## Overview
42.un represents **Phase un** in the masterplan - the "Reflex and ingestion" phase. This module will handle sensors, constant scanning for new sources, and Redis-based event relay.

## Masterplan Context
From MASTERPLAN.md:
> "Redis is introduced to relay events, and `42.un` constantly scans for new sources. Tasks are prioritized before execution, and a test suite ensures module stability."

## Core Components

### 1. Redis Event Bus (`42/redis_bus.py`)
**Purpose**: Central event relay system for fast reactions

**Implementation**:
- Redis connection wrapper with connection pooling
- Event publishing: `publish_event(event_type, data)`
- Event subscription: `subscribe_to_events(event_types, callback)`
- Event persistence: Store events for replay/recovery
- Event filtering: Route events to appropriate handlers

**Events**:
- `github.repo.updated` - Repository changes detected
- `file.ingested` - New file processed
- `embedding.completed` - Embedding batch finished
- `cluster.updated` - Clustering results available
- `llm.query.completed` - LLM response ready

### 2. Source Scanner (`42/source_scanner.py`)
**Purpose**: Constantly scan for new data sources

**Implementation**:
- GitHub webhook listener for repository changes
- File system watcher for local directory changes
- RSS/Atom feed monitor for blog updates
- API endpoint monitor for external data sources
- Configurable scan intervals and patterns

**Sources**:
- GitHub repositories (webhooks + polling)
- Local directories (file system events)
- RSS feeds (blog posts, documentation)
- API endpoints (external data)
- Database changes (monitoring)

### 3. Task Prioritizer (`42/task_prioritizer.py`)
**Purpose**: Score and prioritize tasks before execution

**Implementation**:
- Bayesian filter for task scoring
- Priority queue with dynamic reordering
- Task dependency resolution
- Resource availability checking
- Deadline and urgency handling

**Task Types**:
- `ingest_repository` - Process new GitHub repo
- `update_embeddings` - Re-embed changed files
- `recluster_vectors` - Re-run clustering
- `process_webhook` - Handle GitHub webhook
- `cleanup_old_data` - Remove stale embeddings

### 4. Background Worker (`42/background_worker.py`)
**Purpose**: Execute prioritized tasks in background

**Implementation**:
- Worker pool with configurable concurrency
- Task execution with timeout handling
- Progress tracking and status updates
- Error handling and retry logic
- Resource monitoring and throttling

**Features**:
- Async task execution
- Progress callbacks
- Error recovery
- Resource limits
- Task cancellation

## Implementation Phases

### Phase 1: Redis Foundation (Week 1)
1. **Setup Redis infrastructure**
   - Add Redis to docker-compose.yml
   - Create Redis connection wrapper
   - Implement basic pub/sub

2. **Event system**
   - Define event schemas
   - Implement event publishing
   - Add event subscription

3. **Basic event handlers**
   - GitHub webhook handler
   - File ingestion events
   - Embedding completion events

### Phase 2: Source Scanning (Week 2)
1. **GitHub integration**
   - Webhook endpoint setup
   - Repository change detection
   - Polling for missed events

2. **File system monitoring**
   - Directory watcher implementation
   - File change detection
   - Recursive directory scanning

3. **External source monitoring**
   - RSS feed parser
   - API endpoint monitoring
   - Database change detection

### Phase 3: Task Prioritization (Week 3)
1. **Bayesian task scoring**
   - Implement scoring algorithm
   - Add task metadata
   - Create scoring rules

2. **Priority queue**
   - Dynamic priority adjustment
   - Dependency resolution
   - Resource allocation

3. **Task management**
   - Task creation and scheduling
   - Status tracking
   - Result handling

### Phase 4: Background Execution (Week 4)
1. **Worker pool**
   - Configurable worker count
   - Task distribution
   - Load balancing

2. **Execution engine**
   - Task execution with timeouts
   - Progress tracking
   - Error handling

3. **Integration testing**
   - End-to-end workflows
   - Performance testing
   - Error scenario testing

## File Structure

```
42/
├── un/
│   ├── __init__.py
│   ├── redis_bus.py          # Redis event bus
│   ├── source_scanner.py     # Source monitoring
│   ├── task_prioritizer.py   # Task scoring & prioritization
│   ├── background_worker.py  # Task execution
│   └── events.py            # Event definitions
├── redis_bus.py             # Main Redis interface
└── config.py               # Redis configuration
```

## Configuration

### 42.config.json additions:
```json
{
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": null
  },
  "un": {
    "scan_interval": 300,
    "max_workers": 4,
    "github_webhook_secret": "your-secret",
    "sources": {
      "github": ["repo1", "repo2"],
      "directories": ["/path/to/watch"],
      "feeds": ["https://blog.example.com/feed"]
    }
  }
}
```

## CLI Commands

```bash
# Start 42.un background service
42 un start

# Stop background service
42 un stop

# View current tasks
42 un tasks

# View event history
42 un events

# Manually trigger source scan
42 un scan

# View worker status
42 un workers
```

## FastAPI Endpoints

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

# Worker management
GET /un/workers
POST /un/workers/scale
```

## Testing Strategy

### Unit Tests
- Redis connection and pub/sub
- Event serialization/deserialization
- Task prioritization algorithms
- Source scanning logic
- Worker pool management

### Integration Tests
- End-to-end event flow
- GitHub webhook processing
- File system monitoring
- Background task execution
- Error recovery scenarios

### Performance Tests
- Event throughput
- Task processing speed
- Memory usage under load
- Redis connection pooling
- Worker scaling

## Success Metrics

1. **Event Processing**: 1000+ events/second
2. **Task Throughput**: 100+ tasks/minute
3. **Source Scanning**: <5 second response to changes
4. **Error Recovery**: 99.9% uptime
5. **Resource Usage**: <1GB memory, <10% CPU

## Dependencies

```txt
redis==5.0.1
watchdog==3.0.0
feedparser==6.0.10
aiohttp==3.9.1
celery==5.3.4
```

## Migration Path

1. **Phase 0 → Phase un**: Add Redis and event system
2. **Gradual migration**: Move existing sync operations to background
3. **Feature flags**: Enable/disable 42.un features
4. **Rollback plan**: Fallback to current sync operations

## Next Steps

1. **Week 1**: Implement Redis foundation and basic event system
2. **Week 2**: Add source scanning capabilities
3. **Week 3**: Implement task prioritization
4. **Week 4**: Complete background execution engine
5. **Week 5**: Integration testing and optimization

This plan delivers the core 42.un functionality while maintaining compatibility with existing Phase zéro components. 