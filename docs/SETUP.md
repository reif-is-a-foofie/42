# 42.un Setup Guide

This guide will help you set up 42.un (the reflex layer) for extracting and analyzing GitHub repositories.

## Prerequisites

1. **Python 3.9+** (tested with 3.9.6)
2. **Docker** (for Qdrant vector database)
3. **Ollama** (for local LLM inference)

## Installation

### 1. Install Python Dependencies

```bash
# Install the 42 package
python3 -m pip install -e .

# Verify installation
python3 -m 42 --help
```

### 2. Install Docker

Download and install Docker from: https://docs.docker.com/get-docker/

### 3. Install Ollama

Download and install Ollama from: https://ollama.ai/

## Quick Start

### 1. Initialize 42

```bash
python3 -m 42 create
```

This will:
- Download the embedding model (BAAI/bge-small-en)
- Create configuration files
- Set up the basic system

### 2. Start Services

```bash
# Start Qdrant vector database
docker compose up -d

# Start Ollama (in a separate terminal)
ollama serve
```

### 3. Verify Setup

```bash
python3 -m 42 status
```

You should see:
- ✓ Embedding engine: OK
- ✓ Vector store: OK

### 4. Test Basic Functionality

```bash
# Test embedding
python3 -m 42 embed "Hello, world!"

# Import a directory
python3 -m 42 import ./src

# Check status
python3 -m 42 status
```

## Current Features (42.un)

✅ **Embedding Engine** - Converts text to vectors using BAAI/bge-small-en
✅ **Vector Store** - Stores and searches vectors using Qdrant
✅ **Chunker** - Splits files into meaningful chunks
✅ **CLI Interface** - Command-line interface with Typer
✅ **Configuration** - Centralized config management
✅ **Import System** - Import files and directories

## Next Steps

Once 42.un is running, you can:

1. **Point it at GitHub repos** - Use the import command to analyze repositories
2. **Add clustering** - Group similar code patterns
3. **Add LLM integration** - Connect to Ollama for code analysis
4. **Add Redis** - For real-time event processing

## Troubleshooting

### Docker Issues
- Make sure Docker is running: `docker --version`
- Check if ports are available: `lsof -i :6333`

### Ollama Issues
- Install Ollama: https://ollama.ai/
- Start the service: `ollama serve`
- Pull a model: `ollama pull mistral`

### Python Issues
- Use Python 3.9+: `python3 --version`
- Install dependencies: `python3 -m pip install -e .`

## Architecture

```
42.un (Reflex Layer)
├── Embedding Engine (sentence-transformers)
├── Vector Store (Qdrant)
├── Chunker (AST-based)
├── CLI Interface (Typer)
└── Configuration (JSON + ENV)
```

## Development

```bash
# Run tests
python3 -m pytest tests/

# Format code
python3 -m black 42/

# Lint code
python3 -m ruff 42/
``` 