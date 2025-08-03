# Development Guidelines

These instructions apply to all files in this repository.

## Style
- Use Python type hints for all functions and methods.
- Format code with `black` and lint with `ruff`.
- Prefer small, focused modules with clear interfaces.

## Testing
- Provide matching tests for every new module.
- Run the full test suite with `pytest` before committing.

## Architecture
- Follow the personas defined in `PLAN.md`:
  - Alma orchestrates tools and conversations.
  - Librarian manages long-term memory via LlamaIndex + Qdrant.
  - Hephaestus builds and validates tools.
  - Sons of Helaman execute queued jobs.
  - Moroni plans missions and tracks progress.
  - Forty-Two evaluates modules and proposes self-upgrades.
- Keep scope tight; reuse open-source components for each role.

## Repository Practices
- Do not create new branches.
- Keep commits focused and descriptive.
- Avoid adding dependencies without discussion.
