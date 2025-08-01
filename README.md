# 42 - Autonomous Intelligence Platform

*Following MASTERPLAN.md vision of "digital organism capable of reasoning and acting without constant human supervision"*

## 🏗️ **Architecture**

```
42/
├── README.md              # This file
├── startup/               # Startup scripts and configuration
├── 42/                    # Core platform
│   ├── moroni/           # NLP Brain (intelligent analysis)
│   ├── mission/          # Mission Management
│   │   └── steve/       # Autonomous Mining Agent
│   ├── soul/             # Conscience & Values
│   ├── embedding.py      # Text embedding engine
│   ├── vector_store.py   # Vector database wrapper
│   ├── cluster.py        # Clustering engine
│   ├── prompt.py         # Prompt builder
│   ├── llm.py           # LLM engine (Ollama)
│   ├── chunker.py       # Content chunking
│   ├── cli.py           # Command line interface
│   ├── api.py           # FastAPI backend
│   └── config.py        # Configuration management
├── docs/                 # Documentation
├── tests/                # Test suite
└── TODO.md              # Development tasks
```

## 🎯 **Core Modules**

### **Moroni** - NLP Brain
- Intelligent mission analysis
- Query optimization
- Content strategy planning
- Learning orchestration

### **Mission** - Mission Management
- Mission creation and assignment
- Progress tracking
- Mission chaining
- Template management

### **Steve** - Autonomous Mining Agent
- Web crawling and content extraction
- Semantic search and discovery
- Self-learning from high-quality content
- Continuous knowledge acquisition

### **Soul** - Conscience & Values
- Behavioral preferences
- Decision filtering
- Value alignment
- Security and safety

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d

# Create a mission
42 mission "learn everything about AI healthcare research"

# Check status
42 status

# Ask questions
42 ask "What are the latest trends in healthcare AI?"
```

## 📊 **Current Status**

- ✅ **Phase un** - Reflex and ingestion (COMPLETED)
- 🚀 **Phase deux** - Advanced optimization (IN PROGRESS)
- 📋 **Phase trois** - Autonomous learning (PLANNED)
- 🎯 **Phase quatre** - Multi-agent orchestration (PLANNED)

## 🔧 **Development**

Following `.cursor-rules` discipline:
- Type hints required
- Black formatting
- Ruff linting
- Comprehensive testing
- Timeout estimates for all operations

## 📚 **Documentation**

- `docs/masterplan.md` - Overall vision and phases
- `docs/V.zero/TASKS.md` - Implementation tasks
- `docs/42.un.tasks.md` - Current development tasks
- `TODO.md` - Active task tracking

## 🎯 **Success Metrics**

- Embedding quality: semantic diversity > 0.3
- Mission success: completion rate > 80%
- Code quality: > 80% test coverage
- Performance: < 2 seconds per document processing 