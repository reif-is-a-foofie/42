# Testing Framework for 42

This directory contains comprehensive tests for the 42 system following the testing rules outlined in `docs/TASKS.md`.

## Test Structure

### Core Module Tests
- **`test_embedding.py`** - Tests for the embedding engine
- **`test_vector_store.py`** - Tests for Qdrant vector store integration
- **`test_cluster.py`** - Tests for HDBSCAN clustering engine
- **`test_prompt.py`** - Tests for prompt builder with vector search
- **`test_llm.py`** - Tests for LLM engine (Ollama integration)
- **`test_chunker.py`** - Tests for file chunking functionality
- **`test_cli.py`** - Tests for CLI interface using Typer
- **`test_api.py`** - Tests for FastAPI backend using TestClient
- **`test_config.py`** - Tests for configuration management

### 42.un Module Tests
- **`test_un_events.py`** - Tests for event system and types
- **`test_un_redis_bus.py`** - Tests for Redis event bus

### Basic Tests
- **`test_basic.py`** - Basic functionality verification

## Testing Rules Compliance

### ✅ Implemented Standards
1. **Type hints** - All function arguments and return types annotated
2. **Formatting** - Code formatted with `black`
3. **Linting** - Code linted with `ruff`
4. **Testing** - Comprehensive test coverage under `tests/`

### 🎯 Test Coverage by Module

#### Embedding Engine ✅
- ✅ Tests in `tests/test_embedding.py`
- ✅ Assert vector dimension and datatype
- ✅ Test both single and batch embedding
- ✅ Model loading and error handling

#### Vector Store (Qdrant) ✅
- ✅ Tests in `tests/test_vector_store.py`
- ✅ Tests using temporary Qdrant instance
- ✅ Verify inserts and searches
- ✅ Test connection handling and error scenarios
- ✅ Test collection creation and management
- ✅ Test payload updates and filtering

#### Clustering Engine ✅
- ✅ Tests in `tests/test_cluster.py`
- ✅ Unit tests with small vector sets
- ✅ Assert clusters are returned correctly
- ✅ Test HDBSCAN integration
- ✅ Test cluster quality metrics
- ✅ Test noise handling and visualization

#### Prompt Builder ✅
- ✅ Tests in `tests/test_prompt.py`
- ✅ Mock vector store tests
- ✅ Test top-N logic returns predictable prompts
- ✅ Test token limits and filtering
- ✅ Test custom templates and metadata

#### LLM Engine ✅
- ✅ Tests in `tests/test_llm.py`
- ✅ Mock Ollama API tests
- ✅ Test streaming functionality
- ✅ Test error handling and timeouts
- ✅ Test parameter customization

#### Chunker ✅
- ✅ Tests in `tests/test_chunker.py`
- ✅ Tests covering Python and Markdown splitting
- ✅ Test chunk generation doesn't break
- ✅ Test file size limits and encoding
- ✅ Test metadata preservation

#### CLI Interface ✅
- ✅ Tests in `tests/test_cli.py`
- ✅ Tests using `typer` testing utilities
- ✅ Test all CLI commands
- ✅ Test error handling and validation
- ✅ Test verbose mode and options

#### FastAPI Backend ✅
- ✅ Tests in `tests/test_api.py`
- ✅ Tests using FastAPI's `TestClient`
- ✅ Test all routes return expected response codes
- ✅ Test request validation and error handling
- ✅ Test CORS and documentation endpoints

#### Config Layer ✅
- ✅ Tests in `tests/test_config.py`
- ✅ Tests loading temporary config files
- ✅ Test environment variable overrides
- ✅ Test validation and serialization

#### 42.un Components ✅
- ✅ Tests in `tests/test_un_events.py`
- ✅ Event type definitions and serialization
- ✅ Event factory functions
- ✅ Complex data handling

- ✅ Tests in `tests/test_un_redis_bus.py`
- ✅ Redis connection and pub/sub
- ✅ Event persistence and history
- ✅ Error handling and recovery

## Running Tests

### Prerequisites
```bash
pip install pytest
```

### Run All Tests
```bash
python3 -m pytest tests/ -v
```

### Run Specific Test Module
```bash
python3 -m pytest tests/test_embedding.py -v
```

### Run Tests with Coverage
```bash
pip install pytest-cov
python3 -m pytest tests/ --cov=42 --cov-report=html
```

## Test Categories

### Unit Tests
- Individual module functionality
- Mock external dependencies
- Fast execution (< 1 second per test)

### Integration Tests
- Module interaction testing
- Real service connections (when available)
- End-to-end workflow validation

### Error Handling Tests
- Exception scenarios
- Invalid input handling
- Service unavailability

### Performance Tests
- Large dataset handling
- Memory usage validation
- Timeout scenarios

## Test Data

### Sample Vectors
- 3D test vectors for clustering
- Various dimensions for embedding tests
- Realistic file content for chunking

### Mock Responses
- HTTP responses for API testing
- Redis pub/sub messages
- LLM response streams

## Testing Best Practices

### 1. Mock External Dependencies
```python
with patch('module.external_service') as mock_service:
    mock_service.return_value = expected_result
    # Test code here
```

### 2. Use Fixtures for Common Setup
```python
@pytest.fixture
def sample_data():
    return {"test": "data"}
```

### 3. Test Error Conditions
```python
def test_error_handling():
    with pytest.raises(ExpectedException):
        function_that_should_fail()
```

### 4. Validate Output Formats
```python
def test_output_format():
    result = function_under_test()
    assert isinstance(result, expected_type)
    assert required_fields in result
```

## Future Test Additions

### Missing Test Areas
- **Performance benchmarks** - Load testing and optimization
- **End-to-end workflows** - Complete user scenarios
- **Security testing** - Input validation and sanitization
- **Concurrency testing** - Thread safety and race conditions

### 42.un Phase Tests
- **Source Scanner** - GitHub webhooks, file monitoring
- **Task Prioritizer** - Bayesian scoring algorithms
- **Background Worker** - Task execution and management

## Test Maintenance

### Adding New Tests
1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Test both success and failure cases
4. Mock external dependencies appropriately

### Updating Tests
1. Update when module interfaces change
2. Maintain backward compatibility where possible
3. Add regression tests for bug fixes

### Test Documentation
1. Keep this README updated
2. Document test data requirements
3. Explain complex test scenarios 