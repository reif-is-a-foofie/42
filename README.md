# 42

42 is an autonomous, self-learning AI system.
It ingests, compresses, and exploits knowledge to reason and act.
The goal is a local loop of perception, memory, and morally aligned execution.

```
Sensory -> Reflex -> Brain -> Memory -> Muscles -> Soul
    |        |        |       |         |        |
42.un    Queue    42.deux  DB    Actions   Conscience
```

## Current Status: 42.un (Reflex Layer) ✅

**42.un is ready for GitHub repository extraction!**

### ✅ Completed Features

- **Embedding Engine** - Converts text to vectors using BAAI/bge-small-en
- **Vector Store** - Stores and searches vectors using Qdrant
- **Chunker** - Splits files into meaningful chunks (Python AST, Markdown headers)
- **CLI Interface** - Command-line interface with Typer
- **Configuration** - Centralized config management
- **Import System** - Import files and directories

### 🚀 Ready for GitHub Repos

The system can now:
1. **Extract code patterns** from repositories
2. **Chunk by function/class** for Python files
3. **Chunk by headers** for Markdown files
4. **Embed code chunks** into searchable vectors
5. **Store patterns** for similarity search

## Quick Start

```bash
# Install
python3 -m pip install -e .

# Initialize
python3 -m 42 create

# Start services (optional for now)
docker compose up -d  # Qdrant
ollama serve          # Ollama

# Test functionality
python3 -m 42 embed "def hello(): return 'world'"
python3 -m 42 status
```

## Commands

- `42 create` - Initialize the system
- `42 embed <text>` - Embed text and show vector
- `42 import-data <path>` - Import files/directories
- `42 status` - Check system status
- `42 purge` - Clear all data

## Architecture

### 42.un (Current)
```
Embedding Engine (sentence-transformers)
├── Vector Store (Qdrant)
├── Chunker (AST-based)
├── CLI Interface (Typer)
└── Configuration (JSON + ENV)
```

### Future Phases
- **42.deux** - Bayesian optimization and knowledge graphs
- **42.trois** - Autonomous reinforcement learning
- **42.quatre** - Multi-agent orchestration

## Self-Tuning Embeddings Subsystem

Classical (Mistral/LoRA) → Quantum (PennyLane + CUDA-Q) → Alexandrian Memory → RL Feedback → Self-tuning loop.
Uses PEFT, Triplet Learning, and RLHF for continuous fine-tuning. Embedding refinement drives smarter quantum reasoning and symbolic alignment.

## Setup Guide

See `SETUP.md` for detailed installation instructions.

## Development

```bash
# Run tests
python3 -m pytest tests/

# Format code
python3 -m black 42/

# Lint code
python3 -m ruff 42/
```
