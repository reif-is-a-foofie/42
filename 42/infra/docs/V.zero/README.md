# V.zero Documentation

## Overview
V.zero represents **Phase zéro** in the 42 masterplan - the foundation phase. This version established the core components for intelligent code analysis and querying.

## Completed Components ✅

### **Core Architecture**
- **Embedding Engine** - Text-to-vector conversion with `bge-small-en`
- **Vector Store** - Qdrant integration for similarity search
- **Clustering Engine** - HDBSCAN for semantic grouping
- **Prompt Builder** - Context-aware prompt generation
- **LLM Engine** - Ollama integration for responses
- **Chunker** - Intelligent file splitting
- **CLI Interface** - Typer-based command line
- **FastAPI Backend** - RESTful API endpoints
- **Job Manager** - Background task processing

### **Performance Optimizations**
- **Parallel Processing** - ThreadPoolExecutor for file processing
- **Batch Embeddings** - 64 chunks per embedding batch
- **Vector Batching** - 100 chunks per vector store batch
- **File Filtering** - Skip non-text and large files
- **Dynamic Workers** - Auto-detect CPU cores (4-12 workers)

### **Testing Framework**
- **15 Test Modules** - Comprehensive test coverage
- **Mock Dependencies** - Isolated testing
- **Error Handling** - Edge case coverage
- **Performance Tests** - Benchmarking capabilities

## Documentation Files

### **TASKS.md**
Complete implementation tasks and development guidelines for V.zero phase.

### **SETUP.md**
Detailed setup instructions for V.zero components and dependencies.

## Key Features

### **GitHub Extraction**
```bash
# Extract repository with verbose logging
python3 -m 42 extract-github https://github.com/user/repo --verbose

# Extract with embedding dump
python3 -m 42 extract-github https://github.com/user/repo --dump-embeddings debug.jsonl
```

### **Code Querying**
```bash
# Ask questions about code
python3 -m 42 ask --question "How does authentication work?"

# Search for specific content
python3 -m 42 search --query "database connection"
```

### **API Endpoints**
```bash
# Start API server
python3 -m 42 api

# Query via HTTP
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this code work?"}'
```

## Performance Metrics

- **GitHub Extraction**: 4x speedup with parallel processing
- **Embedding Generation**: 64 chunks per batch
- **Vector Search**: Sub-second response times
- **LLM Queries**: 120-second timeout with streaming
- **File Processing**: Skip images, binaries, >2MB files

## Architecture

```
42/
├── 42/                    # Core modules
│   ├── embedding.py       # Text embedding
│   ├── vector_store.py    # Qdrant integration
│   ├── cluster.py         # HDBSCAN clustering
│   ├── prompt.py          # Context building
│   ├── llm.py            # Ollama integration
│   ├── chunker.py        # File splitting
│   ├── cli.py            # Command interface
│   ├── api.py            # FastAPI backend
│   ├── job_manager.py    # Background jobs
│   └── github.py         # Repository extraction
├── tests/                 # Test suite
├── docs/                  # Documentation
└── .config/              # Configuration
```

## Migration to V.un

V.zero provides the **foundation** for V.un (Phase un) by establishing:
- ✅ **Event System** - Event types and schemas
- ✅ **Redis Integration** - Event bus infrastructure
- ✅ **Modular Architecture** - Extensible component design
- ✅ **Testing Framework** - Quality assurance foundation

## Success Criteria

- [x] **Core Components** - All foundation modules implemented
- [x] **Performance** - Sub-2-minute GitHub extraction
- [x] **Testing** - 100% test coverage for core modules
- [x] **Documentation** - Complete setup and usage guides
- [x] **CLI Interface** - User-friendly command line
- [x] **API Backend** - RESTful endpoints for integration

V.zero is **complete and production-ready**! 🚀 