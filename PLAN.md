# Alma Architectural Plan

This document defines the high-level architecture for the Alma autonomous intelligence platform. The system is composed of several personas that map to open-source components.

## 1. Alma – Companion / Overseer
- **Responsibilities:** CLI interface, task routing, decision making.
- **Tech:** LangChain or LlamaIndex agent for tool calling and reasoning.

## 2. Librarian – Memory & Knowledge Store
- **Responsibilities:** Long-term storage of conversations and documents, summarisation and context retrieval.
- **Tech:** LlamaIndex with Qdrant for persistent vector storage; HDBSCAN for clustering and compression.

## 3. Hephaestus – Tool & Skill Builder
- **Responsibilities:** Build or integrate new tools, sandbox testing, validation before use.
- **Tech:** LangChain tool builder, function calling, FastAPI interfaces.

## 4. Sons of Helaman – Worker Agents / Executors
- **Responsibilities:** Execute queued jobs, scale out heavy tasks, report results back to the Librarian.
- **Tech:** Redis Queue or Celery, FastAPI workers, LangChain or LlamaIndex tasks for batch work.

## 5. Moroni – Mission Receiver & Planner
- **Responsibilities:** Accept missions from Alma, break into subtasks, track progress.
- **Tech:** Prefect or Dagster orchestration, or lightweight Redis mission ledger.

## 6. Forty-Two – Self-Upgrader / Meta-Optimizer
- **Responsibilities:** Periodically evaluates all modules, runs adversarial tests, and proposes upgrades.
- **Tech:** LangChain agent or custom evaluator paired with `pytest` and GitOps-style self-testing.

## 7. Core Self-Learning Loop
1. Observe: log interactions in Qdrant.
2. Cluster & Summarise: HDBSCAN groups embeddings, summaries stored.
3. Reflect: nightly review for "What did I learn today?".
4. Upgrade: Hephaestus fetches/builds tools for unmet needs.
5. Mission feedback improves planning.

## 8. Minimum Open-Source Stack
- LangChain or LlamaIndex agents for orchestration.
- Qdrant for vector memory.
- HDBSCAN for clustering.
- Redis Queue or Celery for workers.
- FastAPI for inter-module communication.

The plan emphasises tight scope, modular design and open-source counterparts for each persona.
