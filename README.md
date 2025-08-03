# Alma - Autonomous Intelligence Platform

Alma is a modular, open-source personal intelligence system. It features multiple personas that map to specific responsibilities and open-source components. The goal is a tightly scoped, extensible framework for autonomous reasoning and learning.

## Personas

### Alma – Companion / Overseer
- Central CLI interface.
- Routes tasks to other personas.
- Powered by a LangChain or LlamaIndex agent.

### Librarian – Memory & Knowledge Store
- Stores conversations and documents.
- Uses LlamaIndex with Qdrant and HDBSCAN for clustering.

### Hephaestus – Tool & Skill Builder
- Creates or integrates new tools.
- Validates tools before Alma uses them.

### Sons of Helaman – Worker Agents
- Execute queued jobs and ingestion tasks.
- Built on Redis Queue or Celery workers.

### Moroni – Mission Receiver & Planner
- Accepts missions from Alma and breaks them into subtasks.
- Tracks progress and orchestrates workers.

### Forty-Two – Self-Upgrader / Meta-Optimizer
- Evaluates all modules and runs adversarial tests.
- Proposes upgrades and triggers self-improvement loops.

## Self-Learning Loop
1. Observe interactions and store embeddings.
2. Cluster and summarise with HDBSCAN.
3. Reflect daily to consolidate lessons.
4. Upgrade tools when gaps are detected.
5. Feed mission outcomes back into planning.

## Stack
- LangChain or LlamaIndex agents.
- Qdrant vector database.
- HDBSCAN for clustering.
- Redis Queue / Celery for workers.
- FastAPI for module communication.

See `PLAN.md` for the detailed roadmap and `AGENTS.md` for development guidelines.
