# Alexandrian

Alexandrian is a zero-cloud, open-source Retrieval Augmented Generation (RAG) framework for local codebases and documents. It embeds, clusters and searches your files entirely offline, powering local Q&A via an Ollama-hosted LLM. Alexandrian exposes both a CLI and FastAPI backend so you can ingest repositories, query them, and build automations without any external services.

## Installation

Alexandrian installs everything it needs, including Docker images and Ollama models.

```bash
# bootstrap a project
npx alexandrian create
```

This command installs Docker and Ollama if missing, pulls the default `mistral` and `bge-small-en` models, and generates `alexandrian.config.json` with sensible defaults.

### Requirements

- Docker
- Node.js (for `npx`)
- Python 3.11+

After running `create`, activate your environment and install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# index your source folder
alexandrian import ./src

# ask a question
alexandrian ask "How do we start the server?"
```

Other commands include `recluster`, `embed --text "..."`, `purge`, and `status`.

## Development Workflow

To keep the code base consistent run these commands before committing:

```bash
# format and lint
ruff .
black .

# run the tests
pytest -q
```

Type hints are required for all new functions. Include docstrings that explain
arguments and return values so team members and code generation tools can easily
understand the purpose of each module.

## Configuration

`alexandrian.config.json` stores model paths, ports, embedding dimensions and clustering parameters. Edit this file to point to custom models or tweak HDBSCAN settings.

## Extending

Alexandrian is modular. Swap out the embedding model, plug in a different vector database, or build additional FastAPI routes. The CLI uses the same API endpoints so extensions automatically benefit from backend updates.

## Screenshot

![Placeholder screenshot](docs/screenshot.png)

## License

MIT
