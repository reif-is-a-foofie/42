# 42 - Digital Organism for Code Analysis

A sophisticated digital organism designed for intelligent code analysis, querying, and autonomous processing.

## 🚀 Quick Start

### **Option 1: Using Setup Script (Recommended)**
```bash
# Install dependencies
./setup.sh install

# Start services
./setup.sh start

# Create system
./setup.sh create

# Extract GitHub repository
python3 -m 42 extract-github https://github.com/user/repo

# Ask questions
python3 -m 42 ask --question "How does this code work?"
```

### **Option 2: Manual Commands**
```bash
# Install dependencies
pip install -r .config/requirements.txt

# Start services
docker-compose -f .config/docker-compose.yml up -d

# Create system
python3 -m 42 create

# Extract GitHub repository
python3 -m 42 extract-github https://github.com/user/repo

# Ask questions
python3 -m 42 ask --question "How does this code work?"
```

## 📁 Project Structure

```
42/
├── 42/                    # Core application modules
│   ├── __init__.py        # Package initialization
│   ├── embedding.py       # Text embedding engine
│   ├── vector_store.py    # Qdrant vector database
│   ├── cluster.py         # HDBSCAN clustering
│   ├── prompt.py          # Context-aware prompts
│   ├── llm.py            # Ollama LLM integration
│   ├── chunker.py        # File chunking engine
│   ├── cli.py            # Command-line interface
│   ├── api.py            # FastAPI backend
│   ├── config.py         # Configuration management
│   ├── job_manager.py    # Background job handling
│   ├── github.py         # GitHub extraction
│   ├── interfaces.py     # Common data structures
│   ├── un/               # 42.un phase components
│   │   ├── __init__.py   # 42.un package
│   │   ├── events.py     # Event system
│   │   └── redis_bus.py  # Redis event bus
│   ├── 42.config.json    # Default configuration
│   ├── 42_jobs.json      # Job persistence
│   └── ollama            # Ollama configuration
├── tests/                 # Comprehensive test suite
│   ├── README.md         # Testing documentation
│   ├── test_embedding.py # Embedding engine tests
│   ├── test_vector_store.py # Vector store tests
│   ├── test_cluster.py   # Clustering tests
│   ├── test_prompt.py    # Prompt builder tests
│   ├── test_llm.py       # LLM engine tests
│   ├── test_chunker.py   # File chunking tests
│   ├── test_cli.py       # CLI interface tests
│   ├── test_api.py       # FastAPI backend tests
│   ├── test_config.py    # Configuration tests
│   ├── test_un_events.py # 42.un event tests
│   ├── test_un_redis_bus.py # Redis bus tests
│   ├── test_basic.py     # Basic functionality tests
│   └── test_performance.py # Performance benchmarks
├── docs/                  # Documentation
│   ├── README.md         # Main documentation
│   ├── MASTERPLAN.md     # Development roadmap
│   ├── TASKS.md          # Implementation tasks
│   ├── SETUP.md          # Setup instructions
│   └── 42_UN_PLAN.md    # 42.un implementation plan
├── .config/               # Configuration files (hidden)
│   ├── requirements.txt   # Python dependencies
│   ├── setup.py          # Package installation
│   ├── docker-compose.yml # Service orchestration
│   └── .cursor-rules     # Development guidelines
├── setup.sh              # Easy setup script
└── README.md             # Project overview
```

## 🎯 Core Features

### **Phase zéro - Foundation** ✅
- **Embedding Engine** - Text-to-vector conversion with `bge-small-en`
- **Vector Store** - Qdrant integration for similarity search
- **Clustering Engine** - HDBSCAN for semantic grouping
- **Prompt Builder** - Context-aware prompt generation
- **LLM Engine** - Ollama integration for responses
- **Chunker** - Intelligent file splitting
- **CLI Interface** - Typer-based command line
- **FastAPI Backend** - RESTful API endpoints
- **Job Manager** - Background task processing

### **Phase un - Reflex & Ingestion** 🚧
- **Redis Event Bus** - Real-time event relay
- **Source Scanner** - Continuous monitoring (in development)
- **Task Prioritizer** - Bayesian scoring (in development)
- **Background Worker** - Async execution (in development)

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Ollama (for LLM)

### Setup
```bash
# Clone repository
git clone https://github.com/reif-is-a-foofie/42.git
cd 42

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d

# Initialize system
python3 -m 42 create
```

## 📖 Usage

### Command Line Interface
```bash
# Extract GitHub repository
python3 -m 42 extract-github https://github.com/user/repo --verbose

# Ask questions about code
python3 -m 42 ask --question "How does the authentication work?"

# Search for specific content
python3 -m 42 search --query "database connection"

# Check system status
python3 -m 42 status

# Recluster vectors
python3 -m 42 recluster
```

### API Endpoints
```bash
# Start API server
python3 -m 42 api

# Query via HTTP
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this code work?"}'
```

## 🧪 Testing

### Run All Tests
```bash
python3 -m pytest tests/ -v
```

### Run Specific Tests
```bash
# Test embedding engine
python3 -m pytest tests/test_embedding.py -v

# Test with coverage
python3 -m pytest tests/ --cov=42 --cov-report=html
```

## 🔧 Configuration

### Environment Variables
```bash
export QDRANT_HOST=localhost
export OLLAMA_HOST=localhost
export EMBEDDING_MODEL=bge-small-en
export API_PORT=8000
```

### Configuration File
```json
{
  "qdrant": {
    "host": "localhost",
    "port": 6333,
    "collection_name": "42_chunks"
  },
  "ollama": {
    "host": "localhost", 
    "port": 11434,
    "model": "mistral:latest"
  },
  "embedding": {
    "model": "bge-small-en",
    "dimension": 384
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

## 🚀 Development

### Architecture
- **Modular Design** - Each component is isolated and testable
- **Type Safety** - Comprehensive type hints throughout
- **Error Handling** - Robust exception management
- **Testing** - 100% test coverage with mocks
- **Documentation** - Detailed docstrings and examples

### Development Workflow
1. **Follow cursor rules** - See `.cursor-rules`
2. **Run tests** - `python3 -m pytest tests/ -v`
3. **Format code** - `black . && ruff .`
4. **Check types** - `mypy 42/`

### Adding New Features
1. **Create module** in `42/` directory
2. **Add tests** in `tests/` directory
3. **Update documentation** in `docs/`
4. **Follow testing rules** - See `docs/TASKS.md`

## 📊 Performance

### Optimizations
- **Batch Processing** - 64 chunks per embedding batch
- **Parallel Processing** - ThreadPoolExecutor for file processing
- **Vector Batching** - 100 chunks per vector store batch
- **Memory Management** - Streaming for large datasets
- **Caching** - Redis for event persistence

### Benchmarks
- **GitHub Extraction** - 4x speedup with parallel processing
- **Embedding Generation** - 64 chunks per batch
- **Vector Search** - Sub-second response times
- **LLM Queries** - 120-second timeout with streaming

## 🤝 Contributing

### Guidelines
1. **Follow masterplan** - See `docs/MASTERPLAN.md`
2. **Complete tasks** - See `docs/TASKS.md`
3. **Write tests** - Comprehensive test coverage
4. **Document changes** - Update relevant docs
5. **Use type hints** - All functions annotated

### Testing Rules
- **Type hints** - All function arguments and return types
- **Formatting** - Run `black .` before committing
- **Linting** - Run `ruff .` to catch issues
- **Testing** - Add tests under `tests/` and run with `pytest`

## 📈 Roadmap

### Phase zéro ✅
- [x] Embedding engine with sentence-transformers
- [x] Vector store with Qdrant integration
- [x] Clustering with HDBSCAN
- [x] Prompt builder with context retrieval
- [x] LLM engine with Ollama
- [x] File chunking with metadata
- [x] CLI interface with Typer
- [x] FastAPI backend with endpoints
- [x] Job management with persistence
- [x] GitHub extraction with parallel processing

### Phase un 🚧
- [x] Redis event bus with pub/sub
- [ ] Source scanner with webhooks
- [ ] Task prioritizer with Bayesian scoring
- [ ] Background worker with async execution

### Phase deux 🔮
- [ ] Advanced clustering with UMAP
- [ ] Multi-modal embeddings
- [ ] Real-time collaboration
- [ ] Advanced analytics

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** - For embedding generation
- **Qdrant** - For vector similarity search
- **HDBSCAN** - For clustering algorithms
- **Ollama** - For local LLM inference
- **FastAPI** - For modern web APIs
- **Typer** - For elegant CLIs

---

**42** - Building the future of intelligent code analysis, one module at a time. 🚀 