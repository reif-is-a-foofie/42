# 42.un Development TODO

## ✅ **Codebase Cleanup COMPLETED**
- ✅ Removed unused garbage files
- ✅ Organized architecture: `42/moroni/`, `42/mission/steve/`, `42/soul/`
- ✅ Updated imports and module structure
- ✅ Created clean README reflecting new architecture

## 🚀 **Current Phase: Phase deux - Advanced optimization**

### **Task 1: Clustering Engine** ✅ **COMPLETED**
*Estimated: 4 hours | Timeout: 8 hours*

- ✅ Create `42/cluster.py` with HDBSCAN wrapper
- ✅ Implement `recluster_vectors() -> Dict[str, Any]`
- ✅ Add CLI command `42 recluster`
- ✅ Add FastAPI endpoint `/recluster`
- ✅ Write tests in `tests/test_cluster.py`

### **Task 2: Prompt Builder**
*Estimated: 3 hours | Timeout: 6 hours*

- [ ] Create `42/prompt.py` with prompt building logic
- [ ] Implement `build_prompt(question: str, top_n: int = 5) -> str`
- [ ] Add CLI command `42 ask --question TEXT`
- [ ] Add FastAPI endpoint `/ask`
- [ ] Write tests in `tests/test_prompt.py`

### **Task 3: LLM Engine**
*Estimated: 3 hours | Timeout: 6 hours*

- [ ] Create `42/llm.py` with Ollama wrapper
- [ ] Implement `respond(prompt: str, stream: bool = False) -> str`
- [ ] Add streaming support for CLI
- [ ] Write tests in `tests/test_llm.py`

### **Task 4: Enhanced Semantic Search**
*Estimated: 6 hours | Timeout: 12 hours*

- [ ] Implement query embedding in `auto_search`
- [ ] Add hybrid retrieval combining semantic + keyword
- [ ] Add semantic similarity search to Brave API results
- [ ] Write tests in `tests/test_semantic_search.py`

### **Task 5: Mission Intelligence Enhancement**
*Estimated: 8 hours | Timeout: 16 hours*

- [ ] Enhance Moroni's mission analysis in `42/un/moroni.py`
- [ ] Add mission templates for common scenarios
- [ ] Implement mission chaining
- [ ] Add progress tracking
- [ ] Write tests in `tests/test_mission_intelligence.py`

### **Task 6: Monitoring & Analytics**
*Estimated: 6 hours | Timeout: 12 hours*

- [ ] Add embedding quality monitoring
- [ ] Create mission progress tracking
- [ ] Implement learning effectiveness metrics
- [ ] Build dashboard for system health
- [ ] Write tests in `tests/test_monitoring.py`

## 📊 **Success Metrics**

- [ ] Clustering Quality: silhouette score > 0.3
- [ ] Prompt Effectiveness: relevance score > 0.7
- [ ] Semantic Search: precision@5 > 0.6
- [ ] Mission Success: completion rate > 80%
- [ ] Code Coverage: > 80% test coverage
- [ ] Zero ruff violations
- [ ] All functions properly typed 