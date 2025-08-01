"""Tests for the CLI module."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from cli import app


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands:" in result.stdout
    
    def test_create_command(self, runner):
        """Test create command."""
        with patch('42.cli.create_system') as mock_create:
            mock_create.return_value = {"status": "success"}
            result = runner.invoke(app, ["create"])
            
            assert result.exit_code == 0
            mock_create.assert_called_once()
    
    def test_create_command_with_options(self, runner):
        """Test create command with options."""
        with patch('42.cli.create_system') as mock_create:
            mock_create.return_value = {"status": "success"}
            result = runner.invoke(app, ["create", "--force"])
            
            assert result.exit_code == 0
            mock_create.assert_called_once()
    
    def test_embed_command(self, runner):
        """Test embed command."""
        with patch('42.cli.embed_text') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            result = runner.invoke(app, ["embed", "--text", "Hello world"])
            
            assert result.exit_code == 0
            mock_embed.assert_called_once_with("Hello world")
            assert "[0.1, 0.2, 0.3, 0.4, 0.5]" in result.stdout
    
    def test_embed_command_missing_text(self, runner):
        """Test embed command with missing text."""
        result = runner.invoke(app, ["embed"])
        assert result.exit_code != 0
        assert "Error" in result.stdout
    
    def test_ask_command(self, runner):
        """Test ask command."""
        with patch('42.cli.ask_question') as mock_ask:
            mock_ask.return_value = "Python is a programming language."
            result = runner.invoke(app, ["ask", "--question", "What is Python?"])
            
            assert result.exit_code == 0
            mock_ask.assert_called_once_with("What is Python?")
            assert "Python is a programming language" in result.stdout
    
    def test_ask_command_with_options(self, runner):
        """Test ask command with options."""
        with patch('42.cli.ask_question') as mock_ask:
            mock_ask.return_value = "Detailed answer about Python."
            result = runner.invoke(app, [
                "ask", 
                "--question", "What is Python?",
                "--model", "llama2",
                "--temperature", "0.7"
            ])
            
            assert result.exit_code == 0
            mock_ask.assert_called_once()
    
    def test_import_file_command(self, runner):
        """Test import file command."""
        with patch('42.cli.import_file') as mock_import:
            mock_import.return_value = {"status": "success", "chunks": 10}
            result = runner.invoke(app, ["import", "file", "test.py"])
            
            assert result.exit_code == 0
            mock_import.assert_called_once_with("test.py")
    
    def test_import_folder_command(self, runner):
        """Test import folder command."""
        with patch('42.cli.import_folder') as mock_import:
            mock_import.return_value = {"status": "success", "files": 5, "chunks": 50}
            result = runner.invoke(app, ["import", "folder", "src/"])
            
            assert result.exit_code == 0
            mock_import.assert_called_once_with("src/")
    
    def test_extract_github_command(self, runner):
        """Test extract-github command."""
        with patch('42.cli.extract_github_repository') as mock_extract:
            mock_extract.return_value = {"status": "success", "chunks": 100}
            result = runner.invoke(app, [
                "extract-github", 
                "https://github.com/user/repo"
            ])
            
            assert result.exit_code == 0
            mock_extract.assert_called_once()
    
    def test_extract_github_command_with_options(self, runner):
        """Test extract-github command with options."""
        with patch('42.cli.extract_github_repository') as mock_extract:
            mock_extract.return_value = {"status": "success", "chunks": 100}
            result = runner.invoke(app, [
                "extract-github",
                "https://github.com/user/repo",
                "--max-workers", "8",
                "--verbose",
                "--dump-embeddings", "test.jsonl"
            ])
            
            assert result.exit_code == 0
            mock_extract.assert_called_once()
    
    def test_recluster_command(self, runner):
        """Test recluster command."""
        with patch('42.cli.recluster_vectors') as mock_recluster:
            mock_recluster.return_value = {"status": "success", "clusters": 5}
            result = runner.invoke(app, ["recluster"])
            
            assert result.exit_code == 0
            mock_recluster.assert_called_once()
    
    def test_recluster_command_with_options(self, runner):
        """Test recluster command with options."""
        with patch('42.cli.recluster_vectors') as mock_recluster:
            mock_recluster.return_value = {"status": "success", "clusters": 3}
            result = runner.invoke(app, [
                "recluster",
                "--min-cluster-size", "3",
                "--generate-plot"
            ])
            
            assert result.exit_code == 0
            mock_recluster.assert_called_once()
    
    def test_status_command(self, runner):
        """Test status command."""
        with patch('42.cli.get_system_status') as mock_status:
            mock_status.return_value = {
                "qdrant": "healthy",
                "ollama": "healthy",
                "chunks": 1000,
                "clusters": 5
            }
            result = runner.invoke(app, ["status"])
            
            assert result.exit_code == 0
            mock_status.assert_called_once()
            assert "healthy" in result.stdout
            assert "1000" in result.stdout
    
    def test_purge_command(self, runner):
        """Test purge command."""
        with patch('42.cli.purge_system') as mock_purge:
            mock_purge.return_value = {"status": "success"}
            result = runner.invoke(app, ["purge"])
            
            assert result.exit_code == 0
            mock_purge.assert_called_once()
    
    def test_purge_command_with_confirmation(self, runner):
        """Test purge command with confirmation."""
        with patch('42.cli.purge_system') as mock_purge:
            mock_purge.return_value = {"status": "success"}
            result = runner.invoke(app, ["purge", "--force"])
            
            assert result.exit_code == 0
            mock_purge.assert_called_once()
    
    def test_search_command(self, runner):
        """Test search command."""
        with patch('42.cli.search_vectors') as mock_search:
            mock_search.return_value = [
                {"id": 1, "score": 0.95, "payload": {"text": "Result 1"}},
                {"id": 2, "score": 0.87, "payload": {"text": "Result 2"}}
            ]
            result = runner.invoke(app, ["search", "--query", "Python"])
            
            assert result.exit_code == 0
            mock_search.assert_called_once_with("Python")
            assert "Result 1" in result.stdout
            assert "Result 2" in result.stdout
    
    def test_search_command_with_options(self, runner):
        """Test search command with options."""
        with patch('42.cli.search_vectors') as mock_search:
            mock_search.return_value = [{"id": 1, "score": 0.95, "payload": {"text": "Result"}}]
            result = runner.invoke(app, [
                "search",
                "--query", "Python",
                "--limit", "5",
                "--score-threshold", "0.8"
            ])
            
            assert result.exit_code == 0
            mock_search.assert_called_once()
    
    def test_job_status_command(self, runner):
        """Test job-status command."""
        with patch('42.cli.get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": "123",
                "status": "completed",
                "progress": 100
            }
            result = runner.invoke(app, ["job-status", "--job-id", "123"])
            
            assert result.exit_code == 0
            mock_status.assert_called_once_with("123")
            assert "completed" in result.stdout
    
    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        assert "Error" in result.stdout
    
    def test_command_with_invalid_options(self, runner):
        """Test command with invalid options."""
        result = runner.invoke(app, ["ask", "--invalid-option"])
        assert result.exit_code != 0
    
    def test_embed_command_error_handling(self, runner):
        """Test embed command error handling."""
        with patch('42.cli.embed_text') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            result = runner.invoke(app, ["embed", "--text", "Hello"])
            
            assert result.exit_code != 0
            assert "Error" in result.stdout
    
    def test_ask_command_error_handling(self, runner):
        """Test ask command error handling."""
        with patch('42.cli.ask_question') as mock_ask:
            mock_ask.side_effect = Exception("LLM failed")
            result = runner.invoke(app, ["ask", "--question", "What is Python?"])
            
            assert result.exit_code != 0
            assert "Error" in result.stdout
    
    def test_import_command_file_not_found(self, runner):
        """Test import command with file not found."""
        with patch('42.cli.import_file') as mock_import:
            mock_import.side_effect = FileNotFoundError("File not found")
            result = runner.invoke(app, ["import", "file", "nonexistent.py"])
            
            assert result.exit_code != 0
            assert "Error" in result.stdout
    
    def test_extract_github_command_invalid_url(self, runner):
        """Test extract-github command with invalid URL."""
        with patch('42.cli.extract_github_repository') as mock_extract:
            mock_extract.side_effect = ValueError("Invalid GitHub URL")
            result = runner.invoke(app, [
                "extract-github", 
                "invalid-url"
            ])
            
            assert result.exit_code != 0
            assert "Error" in result.stdout
    
    def test_status_command_service_unavailable(self, runner):
        """Test status command when services are unavailable."""
        with patch('42.cli.get_system_status') as mock_status:
            mock_status.return_value = {
                "qdrant": "unhealthy",
                "ollama": "unhealthy",
                "chunks": 0,
                "clusters": 0
            }
            result = runner.invoke(app, ["status"])
            
            assert result.exit_code == 0
            assert "unhealthy" in result.stdout
    
    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "42" in result.stdout
    
    def test_cli_verbose_mode(self, runner):
        """Test CLI verbose mode."""
        with patch('42.cli.ask_question') as mock_ask:
            mock_ask.return_value = "Answer"
            result = runner.invoke(app, [
                "ask", 
                "--question", "Test question",
                "--verbose"
            ])
            
            assert result.exit_code == 0
            mock_ask.assert_called_once()
    
    def test_cli_dry_run_mode(self, runner):
        """Test CLI dry run mode."""
        with patch('42.cli.import_folder') as mock_import:
            mock_import.return_value = {"status": "dry_run", "files": 0}
            result = runner.invoke(app, [
                "import", 
                "folder", 
                "src/",
                "--dry-run"
            ])
            
            assert result.exit_code == 0
            mock_import.assert_called_once() 