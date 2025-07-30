"""Tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from api import app


class TestAPI:
    """Test FastAPI backend functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)
    
    def test_api_root(self, client):
        """Test API root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "42" in response.json()["message"]
    
    def test_api_health(self, client):
        """Test API health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_ask_endpoint(self, client):
        """Test ask endpoint."""
        with patch('42.api.ask_question') as mock_ask:
            mock_ask.return_value = "Python is a programming language."
            
            response = client.post("/ask", json={"question": "What is Python?"})
            
            assert response.status_code == 200
            assert "Python is a programming language" in response.json()["answer"]
            mock_ask.assert_called_once_with("What is Python?")
    
    def test_ask_endpoint_with_options(self, client):
        """Test ask endpoint with options."""
        with patch('42.api.ask_question') as mock_ask:
            mock_ask.return_value = "Detailed answer about Python."
            
            response = client.post("/ask", json={
                "question": "What is Python?",
                "model": "llama2",
                "temperature": 0.7,
                "max_tokens": 100
            })
            
            assert response.status_code == 200
            mock_ask.assert_called_once()
    
    def test_ask_endpoint_missing_question(self, client):
        """Test ask endpoint with missing question."""
        response = client.post("/ask", json={})
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_empty_question(self, client):
        """Test ask endpoint with empty question."""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_error_handling(self, client):
        """Test ask endpoint error handling."""
        with patch('42.api.ask_question') as mock_ask:
            mock_ask.side_effect = Exception("LLM failed")
            
            response = client.post("/ask", json={"question": "What is Python?"})
            
            assert response.status_code == 500
            assert "error" in response.json()
    
    def test_import_file_endpoint(self, client):
        """Test import file endpoint."""
        with patch('42.api.import_file') as mock_import:
            mock_import.return_value = {"status": "success", "chunks": 10}
            
            response = client.post("/import/file", json={"file_path": "test.py"})
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            assert response.json()["chunks"] == 10
            mock_import.assert_called_once_with("test.py")
    
    def test_import_file_endpoint_missing_path(self, client):
        """Test import file endpoint with missing path."""
        response = client.post("/import/file", json={})
        assert response.status_code == 422  # Validation error
    
    def test_import_file_endpoint_file_not_found(self, client):
        """Test import file endpoint with file not found."""
        with patch('42.api.import_file') as mock_import:
            mock_import.side_effect = FileNotFoundError("File not found")
            
            response = client.post("/import/file", json={"file_path": "nonexistent.py"})
            
            assert response.status_code == 404
            assert "error" in response.json()
    
    def test_import_folder_endpoint(self, client):
        """Test import folder endpoint."""
        with patch('42.api.import_folder') as mock_import:
            mock_import.return_value = {"status": "success", "files": 5, "chunks": 50}
            
            response = client.post("/import/folder", json={"folder_path": "src/"})
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            assert response.json()["files"] == 5
            assert response.json()["chunks"] == 50
            mock_import.assert_called_once_with("src/")
    
    def test_import_folder_endpoint_missing_path(self, client):
        """Test import folder endpoint with missing path."""
        response = client.post("/import/folder", json={})
        assert response.status_code == 422  # Validation error
    
    def test_extract_github_endpoint(self, client):
        """Test extract GitHub endpoint."""
        with patch('42.api.extract_github_repository') as mock_extract:
            mock_extract.return_value = {"status": "success", "chunks": 100}
            
            response = client.post("/extract-github", json={
                "repository_url": "https://github.com/user/repo"
            })
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            assert response.json()["chunks"] == 100
            mock_extract.assert_called_once()
    
    def test_extract_github_endpoint_with_options(self, client):
        """Test extract GitHub endpoint with options."""
        with patch('42.api.extract_github_repository') as mock_extract:
            mock_extract.return_value = {"status": "success", "chunks": 100}
            
            response = client.post("/extract-github", json={
                "repository_url": "https://github.com/user/repo",
                "max_workers": 8,
                "verbose": True,
                "dump_embeddings": "test.jsonl"
            })
            
            assert response.status_code == 200
            mock_extract.assert_called_once()
    
    def test_extract_github_endpoint_invalid_url(self, client):
        """Test extract GitHub endpoint with invalid URL."""
        with patch('42.api.extract_github_repository') as mock_extract:
            mock_extract.side_effect = ValueError("Invalid GitHub URL")
            
            response = client.post("/extract-github", json={
                "repository_url": "invalid-url"
            })
            
            assert response.status_code == 400
            assert "error" in response.json()
    
    def test_recluster_endpoint(self, client):
        """Test recluster endpoint."""
        with patch('42.api.recluster_vectors') as mock_recluster:
            mock_recluster.return_value = {"status": "success", "clusters": 5}
            
            response = client.post("/recluster")
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            assert response.json()["clusters"] == 5
            mock_recluster.assert_called_once()
    
    def test_recluster_endpoint_with_options(self, client):
        """Test recluster endpoint with options."""
        with patch('42.api.recluster_vectors') as mock_recluster:
            mock_recluster.return_value = {"status": "success", "clusters": 3}
            
            response = client.post("/recluster", json={
                "min_cluster_size": 3,
                "generate_plot": True
            })
            
            assert response.status_code == 200
            mock_recluster.assert_called_once()
    
    def test_status_endpoint(self, client):
        """Test status endpoint."""
        with patch('42.api.get_system_status') as mock_status:
            mock_status.return_value = {
                "qdrant": "healthy",
                "ollama": "healthy",
                "chunks": 1000,
                "clusters": 5
            }
            
            response = client.get("/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["qdrant"] == "healthy"
            assert data["ollama"] == "healthy"
            assert data["chunks"] == 1000
            assert data["clusters"] == 5
            mock_status.assert_called_once()
    
    def test_status_endpoint_service_unavailable(self, client):
        """Test status endpoint when services are unavailable."""
        with patch('42.api.get_system_status') as mock_status:
            mock_status.return_value = {
                "qdrant": "unhealthy",
                "ollama": "unhealthy",
                "chunks": 0,
                "clusters": 0
            }
            
            response = client.get("/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["qdrant"] == "unhealthy"
            assert data["ollama"] == "unhealthy"
    
    def test_search_endpoint(self, client):
        """Test search endpoint."""
        with patch('42.api.search_vectors') as mock_search:
            mock_search.return_value = [
                {"id": 1, "score": 0.95, "payload": {"text": "Result 1"}},
                {"id": 2, "score": 0.87, "payload": {"text": "Result 2"}}
            ]
            
            response = client.post("/search", json={"query": "Python"})
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["results"][0]["score"] == 0.95
            assert data["results"][1]["score"] == 0.87
            mock_search.assert_called_once_with("Python")
    
    def test_search_endpoint_with_options(self, client):
        """Test search endpoint with options."""
        with patch('42.api.search_vectors') as mock_search:
            mock_search.return_value = [{"id": 1, "score": 0.95, "payload": {"text": "Result"}}]
            
            response = client.post("/search", json={
                "query": "Python",
                "limit": 5,
                "score_threshold": 0.8
            })
            
            assert response.status_code == 200
            mock_search.assert_called_once()
    
    def test_search_endpoint_missing_query(self, client):
        """Test search endpoint with missing query."""
        response = client.post("/search", json={})
        assert response.status_code == 422  # Validation error
    
    def test_embed_endpoint(self, client):
        """Test embed endpoint."""
        with patch('42.api.embed_text') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            response = client.post("/embed", json={"text": "Hello world"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embed.assert_called_once_with("Hello world")
    
    def test_embed_endpoint_missing_text(self, client):
        """Test embed endpoint with missing text."""
        response = client.post("/embed", json={})
        assert response.status_code == 422  # Validation error
    
    def test_embed_endpoint_error_handling(self, client):
        """Test embed endpoint error handling."""
        with patch('42.api.embed_text') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            response = client.post("/embed", json={"text": "Hello"})
            
            assert response.status_code == 500
            assert "error" in response.json()
    
    def test_purge_endpoint(self, client):
        """Test purge endpoint."""
        with patch('42.api.purge_system') as mock_purge:
            mock_purge.return_value = {"status": "success"}
            
            response = client.post("/purge")
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            mock_purge.assert_called_once()
    
    def test_job_status_endpoint(self, client):
        """Test job status endpoint."""
        with patch('42.api.get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": "123",
                "status": "completed",
                "progress": 100
            }
            
            response = client.get("/job-status?job_id=123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "123"
            assert data["status"] == "completed"
            assert data["progress"] == 100
            mock_status.assert_called_once_with("123")
    
    def test_job_status_endpoint_missing_id(self, client):
        """Test job status endpoint with missing job ID."""
        response = client.get("/job-status")
        assert response.status_code == 422  # Validation error
    
    def test_job_status_endpoint_not_found(self, client):
        """Test job status endpoint with job not found."""
        with patch('42.api.get_job_status') as mock_status:
            mock_status.side_effect = ValueError("Job not found")
            
            response = client.get("/job-status?job_id=nonexistent")
            
            assert response.status_code == 404
            assert "error" in response.json()
    
    def test_api_docs(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_api_openapi(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "components" in data
    
    def test_api_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/ask")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_api_rate_limiting(self, client):
        """Test API rate limiting."""
        # Make multiple requests quickly
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_api_error_responses(self, client):
        """Test API error response format."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "detail" in data
    
    def test_api_request_validation(self, client):
        """Test API request validation."""
        # Test invalid JSON
        response = client.post("/ask", data="invalid json")
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/ask", json={"invalid_field": "value"})
        assert response.status_code == 422
    
    def test_api_response_format(self, client):
        """Test API response format consistency."""
        with patch('42.api.ask_question') as mock_ask:
            mock_ask.return_value = "Test answer"
            
            response = client.post("/ask", json={"question": "Test question?"})
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "timestamp" in data
            assert "question" in data 