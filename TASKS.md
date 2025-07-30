# Implementation Tasks

This document expands the PRD into concrete development steps for each component. Follow these tasks sequentially when building 42.

## System Overview
42 consists of several cooperating modules, each built on a stable
open‑source library:

1. **Embedding Engine** – turns text into numeric vectors using a local model.
2. **Vector Store** – saves those vectors for similarity search.
3. **Clustering Engine** – groups vectors so related files can be explored together.
4. **Prompt Builder** – prepares context to send to the LLM.
5. **LLM Engine** – uses Ollama to generate answers.
6. **Chunker** – splits files into smaller pieces for embedding.
7. **CLI and FastAPI backend** – provide command line and HTTP interfaces.
8. **Configuration layer** – central place for model paths and ports.

Keep this picture in mind while you work; each task below maps back to one of
these pieces. **Every subsystem should wrap the third‑party dependency in a
thin module** so we control the surface area and can pin library versions.

## Open‑Source Module Approach

Leverage these proven projects and pin their versions in `requirements.txt`:

| Component          | Library                                | Version |
| ------------------ | -------------------------------------- | ------- |
| Embedding Engine   | `sentence-transformers` (`bge-small-en`) | `2.2.2` |
| Vector Store       | `qdrant-client` + Docker `qdrant`        | `1.7.0` |
| Chunking           | `tree-sitter` & `unstructured`           | `0.20.3` / `0.11.1` |
| Clustering         | `hdbscan` & `umap-learn`                 | `0.8.33` / `0.5.6` |
| LLM Engine         | `ollama`                                | pinned via Docker |
| CLI                | `typer`                                 | `0.12.3` |
| API                | `fastapi`                               | `0.110.1` |

Wrap each library behind an interface in `42` so the rest of the
codebase never calls these dependencies directly.

Keep this picture in mind while you work; each task below maps back to one of these pieces.

## Development Basics
New engineers should set up a Python 3.11 virtual environment and install the dependencies from `requirements.txt`. Before committing code:

1. **Type hints** – annotate all function arguments and return types. Use `mypy` if you want extra checking.
2. **Formatting** – run `black .` to automatically format the code.
3. **Linting** – run `ruff .` to catch common issues.
4. **Testing** – add tests under `tests/` and execute with `pytest -q`.

Even small modules should have at least one test so we know they work as expected.

## Repository Setup
1. Scaffold a Python package named `42` with submodules:
   `embedding.py`, `vector_store.py`, `cluster.py`, `prompt.py`, `llm.py`,
   `chunker.py`, `cli.py`, `api.py`, and `config.py`.
2. Create `requirements.txt` enumerating all dependencies listed in the PRD with
   **pinned versions** so updates are predictable.
3. Provide a starter `42.config.json` with default model names, ports
   and embedding dimensions.
4. Define common data structures in `interfaces.py` to keep module APIs
   consistent.
5. Configure linting and formatters (`black` + `ruff`).

## Embedding Engine
- Load `bge-small-en` via `sentence-transformers` when the application starts.
- Wrap the library in `embedding.py` so callers use a stable API.
   `embedding.py`, `vector_store.py`, `cluster.py`, `prompt.py`, `llm.py`, `chunker.py`, `cli.py`, `api.py`, and `config.py`.
2. Create `requirements.txt` enumerating all dependencies listed in the PRD (FastAPI, sentence-transformers, qdrant-client, hdbscan, etc.).
3. Provide a starter `42.config.json` with default model names, ports and embedding dimensions.
4. Configure linting and formatters (`black` + `ruff`).

## Embedding Engine
- Load `bge-small-en` via `sentence-transformers` when the application starts.
- Implement two helper functions with type hints:
  - `embed_text(text: str) -> list[float]`
  - `embed_text_batch(texts: list[str]) -> list[list[float]]`
- Add docstrings describing the expected input and output.
- Expose a CLI command `42 embed --text TEXT` that prints the vector as JSON.
- Create tests in `tests/test_embedding.py` asserting that the returned vector has the correct dimension and datatype.

## Vector Store (Qdrant)
- Supply a Docker Compose file that launches Qdrant.
- During `42 create`, start the container and wait for readiness.
- Wrap `qdrant-client` inside `vector_store.py` exposing
  `upsert`, `search`, `update_payload`, and `get_all_vectors` so callers never
  import the library directly.
- Implement wrapper methods `upsert`, `search`, `update_payload`, and `get_all_vectors`.
- Read connection info from `42.config.json`.
- Add tests using a temporary Qdrant instance to verify inserts and searches.
- Document each method with type hints and explain what parameters mean.

## Clustering Engine
- Build `recluster_vectors()` that loads all vectors and runs `hdbscan` to
  assign a `cluster_id` to each payload. Keep the library wrapped inside
  `cluster.py` so future upgrades are isolated.
- Build `recluster_vectors()` that loads all vectors, runs HDBSCAN and updates each payload with `cluster_id`.
- Optionally generate a UMAP plot saved under `docs/cluster.png`.
- Provide CLI command `42 recluster` and FastAPI endpoint `/recluster`.
- Include unit tests that feed a small set of vectors and assert clusters are returned.

## Prompt Builder
- Fetch the top‑N vectors for a query from Qdrant.
- Insert those chunks into a template trimmed to the model token limit.
- Keep this logic in `prompt.py` so the API and CLI share the same implementation.
- Export `build_prompt(question: str) -> str` used by the LLM engine.
- Write tests that mock the vector store to ensure the top-N logic returns predictable prompts.

## LLM Engine
- Connect to the local Ollama server (default `mistral`).
- Implement `respond(prompt: str) -> str` using the Ollama HTTP API. Place this
  logic in `llm.py` so the rest of the project calls a single helper function.
- Implement `respond(prompt: str) -> str` using the Ollama HTTP API.
- Stream tokens back to the CLI for progress feedback.
- Add a test that mocks the Ollama API and checks that streaming works without errors.

## Chunker
- Parse Python files using `tree-sitter` to split by function or class.
- Split Markdown files by heading level with `unstructured` as a fallback.
- Parse Python files with `ast` to split by function or class.
- Split Markdown files by heading level.
- Emit metadata: file path, start/end lines and cluster ID.
- Implement `42 import PATH` to ingest a folder recursively.
- Provide tests covering Python and Markdown splitting so future changes do not break chunk generation.

## CLI Interface
- Build the CLI with `typer`.
- Commands: `create`, `import`, `ask`, `recluster`, `embed`, `purge`, `status`.
- Each command should call the corresponding FastAPI route via HTTP. Keep the CLI
  logic separate in `cli.py` so other tools can reuse the functions.
- Each command should call the corresponding FastAPI route via HTTP.
- Add tests that invoke the CLI commands using `typer`'s testing utilities.

## FastAPI Backend
- Expose routes `/ask`, `/import/file`, `/import/folder`, `/recluster`, `/status`, `/search`.
- Use Pydantic models for request and response schemas.
- Run the server with `uvicorn`.
- Add tests using FastAPI's `TestClient` to ensure each route returns the expected response code.
- Keep route handlers thin and delegate to the wrapper modules so API logic stays minimal.

## Config Layer
- Generate `42.config.json` during `create` if missing.
- Provide a `load_config()` helper returning a typed dataclass.
- Allow overriding values via environment variables.
- Add tests that load a temporary config file and ensure values map to the dataclass correctly.
- Record dependency versions in the config so upgrades can be audited.

## Installer
- `npx 42 create` installs Docker and Ollama if absent, pulls the default models and Python dependencies, then launches Qdrant and the API server.
- Print next steps after setup completes.
- Provide installation logs so users can troubleshoot if something fails.
- Verify that pinned versions from `requirements.txt` are installed.

## Postinstall Verification
- Implement `42 status` to check Qdrant and Ollama health, report model names, cluster count and total chunks.
- Run a small embedding and LLM request to confirm the full pipeline works.
- Document common errors and where to look in the logs.
- Ensure the command also reports the versions of all pinned dependencies.

## Optional Features
- UMAP-based cluster explorer UI served from the FastAPI backend.
- GitHub listener that triggers re‑indexing on push events.
- VS Code extension for inline querying.
- Web dashboard with usage analytics.
- Memory cache to speed up common questions.
