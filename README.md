# 42 - Autonomous Intelligence Platform

*Following the vision of a "digital organism capable of reasoning and acting without constant human supervision"*

## 🏗️ **Architecture**

```
42/
├── soul/             # Conscience & Values
├── moroni/           # NLP Brain (intelligent analysis)  
├── mission/          # Mission Management
│   └── steve/       # Autonomous Mining Agent
└── infra/           # Infrastructure
    ├── core/        # Core services (embedding, vector_store, etc.)
    ├── services/    # API and CLI services
    ├── utils/       # Utilities and configuration
    ├── docs/        # Documentation
    ├── logs/        # Log files
    ├── startup/     # Configuration
    └── tests/       # Test suite
```

## 🎯 **Core Modules**

### **Soul** - Conscience & Values
- Behavioral preferences and decision filtering
- Value alignment and security
- Moral compass for all system decisions

### **Moroni** - NLP Brain
- Intelligent mission analysis and planning
- Query optimization and content strategy
- Learning orchestration and reasoning

### **Mission** - Mission Management
- Mission creation, assignment, and tracking
- Progress monitoring and chaining
- Template management and execution

### **Steve** - Autonomous Mining Agent
- Web crawling and content extraction
- Semantic search and discovery
- Self-learning from high-quality content

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r .config/requirements.txt

# Start services
docker-compose up -d

# Create a mission
42 mission "learn everything about AI healthcare research"

# Check status
42 status

# Ask questions
42 ask "What are the latest trends in healthcare AI?"
```

## 📊 **Development Phases**

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

- `42/infra/docs/MASTERPLAN.md` - Overall vision and phases
- `42/infra/docs/42.un.tasks.md` - Current development tasks
- `42/infra/docs/42.un.next.md` - Next phase planning

## 🎯 **Success Metrics**

- Embedding quality: semantic diversity > 0.3
- Mission success: completion rate > 80%
- Code quality: > 80% test coverage
- Performance: < 2 seconds per document processing 