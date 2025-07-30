"""CLI interface for 42."""

import typer
import time
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from .config import load_config, save_config
from .embedding import EmbeddingEngine
from .vector_store import VectorStore
from .chunker import Chunker
from .github import GitHubExtractor
from .llm import LLMEngine
from .api import run_server

app = typer.Typer()
console = Console()


@app.command()
def create():
    """Create and initialize 42 system."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing 42...", total=None)
        
        try:
            # Load config
            config = load_config()
            progress.update(task, description="Loading configuration...")
            
            # Initialize embedding engine
            progress.update(task, description="Loading embedding model...")
            embedding_engine = EmbeddingEngine()
            
            # Save config
            progress.update(task, description="Saving configuration...")
            save_config(config)
            
            progress.update(task, description="42 system initialized successfully!")
            console.print("[green]✓[/green] 42 system created and ready!")
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Start Qdrant: docker compose up -d")
            console.print("2. Start Ollama: ollama serve")
            console.print("3. Run: python3 -m 42 status")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create 42 system: {e}")
            console.print("\n[bold]Setup required:[/bold]")
            console.print("1. Install Docker: https://docs.docker.com/get-docker/")
            console.print("2. Install Ollama: https://ollama.ai/")
            console.print("3. Start services: docker compose up -d && ollama serve")
            raise typer.Exit(1)


@app.command()
def embed(text: str = typer.Argument(..., help="Text to embed")):
    """Embed text and print the vector."""
    try:
        embedding_engine = EmbeddingEngine()
        vector = embedding_engine.embed_text(text)
        console.print(f"Embedding dimension: {len(vector)}")
        console.print(f"Vector: {vector[:5]}...")  # Show first 5 values
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to embed text: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Check system status."""
    try:
        config = load_config()
        console.print("[bold]42 System Status[/bold]")
        console.print(f"Qdrant: {config.qdrant_host}:{config.qdrant_port}")
        console.print(f"Ollama: {config.ollama_host}:{config.ollama_port}")
        console.print(f"Redis: {config.redis_host}:{config.redis_port}")
        console.print(f"Embedding model: {config.embedding_model}")
        console.print(f"Collection: {config.collection_name}")
        
        # Test connections
        try:
            embedding_engine = EmbeddingEngine()
            console.print("[green]✓[/green] Embedding engine: OK")
        except Exception as e:
            console.print(f"[red]✗[/red] Embedding engine: {e}")
        
        try:
            vector_store = VectorStore()
            console.print("[green]✓[/green] Vector store: OK")
        except Exception as e:
            console.print(f"[red]✗[/red] Vector store: {e}")
            console.print("[yellow]Note:[/yellow] Start Qdrant with: docker compose up -d")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to check status: {e}")
        raise typer.Exit(1)


@app.command()
def import_data(path: str = typer.Argument(..., help="Path to file or directory to import")):
    """Import files or directories into 42."""
    try:
        from pathlib import Path
        from qdrant_client.models import PointStruct
        
        chunker = Chunker()
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStore()
        
        # Ensure collection exists
        vector_store.create_collection(embedding_engine.get_dimension())
        
        # Chunk the files
        if Path(path).is_file():
            chunks = chunker.chunk_file(path)
        else:
            chunks = chunker.chunk_directory(path)
        
        if not chunks:
            console.print(f"[yellow]No chunks found in {path}[/yellow]")
            return
        
        # Embed and store chunks
        with Progress() as progress:
            task = progress.add_task("Importing chunks...", total=len(chunks))
            
            for i, chunk in enumerate(chunks):
                # Embed the chunk
                vector = embedding_engine.embed_text(chunk.text)
                
                # Create point
                point = PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "metadata": chunk.metadata or {}
                    }
                )
                
                # Store in vector database
                vector_store.upsert([point])
                progress.update(task, advance=1)
        
        console.print(f"[green]✓[/green] Imported {len(chunks)} chunks from {path}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to import {path}: {e}")
        raise typer.Exit(1)


@app.command()
def extract_github(
    repo_url: str = typer.Argument(..., help="GitHub repository URL to extract"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    max_workers: int = typer.Option(None, "--max-workers", "-w", help="Number of parallel workers (default: CPU count)"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Enable verbose logging"),
    dump_embeddings: Optional[str] = typer.Option(None, "--dump-embeddings", help="Save embeddings to JSONL file")
):
    """Extract and analyze a GitHub repository."""
    try:
        extractor = GitHubExtractor(max_workers=max_workers, verbose=verbose, dump_embeddings_path=dump_embeddings)
        
        if background:
            # Start background job
            result = extractor.analyze_repository(repo_url, background=True)
            console.print(f"[yellow]→[/yellow] Started background analysis for {repo_url}")
            console.print(f"Job ID: {result['job_id']}")
            console.print("Use '42 job-status <job_id>' to check progress")
            return
        
        # Progress callback for real-time updates
        def progress_callback(message: str, progress_val: float):
            progress.update(task, description=f"{message} ({progress_val:.1%})")
        
        with Progress() as progress:
            task = progress.add_task("Analyzing repository...", total=None)
            
            result = extractor.analyze_repository(repo_url, progress_callback=progress_callback)
            
            if result["status"] == "success":
                console.print(f"[green]✓[/green] Extracted {result['chunks']} chunks from {repo_url}")
                console.print(f"Latest commit: {result['repo_info']['latest_commit']}")
                if "elapsed_time" in result:
                    console.print(f"Completed in {result['elapsed_time']:.2f}s")
            else:
                console.print(f"[red]✗[/red] Failed to extract {repo_url}: {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to extract repository: {e}")
        raise typer.Exit(1)


@app.command()
def job_status(job_id: str = typer.Argument(..., help="Background job ID")):
    """Check status of a background job."""
    try:
        extractor = GitHubExtractor()
        status = extractor.get_job_status(job_id)
        
        if status["status"] == "not_found":
            console.print(f"[red]✗[/red] Job {job_id} not found")
            raise typer.Exit(1)
        elif status["status"] == "running":
            console.print(f"[yellow]→[/yellow] Job {job_id} is running")
            console.print(f"Repository: {status['repo_url']}")
            console.print(f"Elapsed time: {status['elapsed_time']:.1f}s")
        elif status["status"] == "completed":
            result = status["result"]
            console.print(f"[green]✓[/green] Job {job_id} completed")
            console.print(f"Repository: {result['repo_url']}")
            console.print(f"Chunks extracted: {result['chunks']}")
            if "elapsed_time" in result:
                console.print(f"Total time: {result['elapsed_time']:.2f}s")
        elif status["status"] == "failed":
            console.print(f"[red]✗[/red] Job {job_id} failed")
            console.print(f"Error: {status['error']}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get job status: {e}")
        raise typer.Exit(1)


@app.command()
def list_jobs():
    """List all background jobs."""
    try:
        extractor = GitHubExtractor()
        jobs = extractor.list_background_jobs()
        
        if not jobs:
            console.print("[yellow]No background jobs found[/yellow]")
            return
        
        console.print("[bold]Background Jobs:[/bold]")
        for job_id, job in jobs.items():
            status_icon = {
                "running": "[yellow]→[/yellow]",
                "completed": "[green]✓[/green]",
                "failed": "[red]✗[/red]"
            }.get(job["status"], "?")
            
            console.print(f"{status_icon} {job_id}")
            console.print(f"  Repository: {job.get('repo_url', 'unknown')}")
            console.print(f"  Status: {job['status']}")
            
            if job["status"] == "running":
                elapsed = time.time() - job["started_at"]
                console.print(f"  Elapsed: {elapsed:.1f}s")
            elif job["status"] == "completed":
                result = job["result"]
                console.print(f"  Chunks: {result['chunks']}")
                if "elapsed_time" in result:
                    console.print(f"  Time: {result['elapsed_time']:.2f}s")
            elif job["status"] == "failed":
                console.print(f"  Error: {job['error']}")
            console.print()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list jobs: {e}")
        raise typer.Exit(1)


@app.command()
def search_patterns(query: str = typer.Argument(..., help="Search query for code patterns")):
    """Search for similar code patterns."""
    try:
        extractor = GitHubExtractor()
        results = extractor.search_patterns(query, limit=5)
        
        if not results:
            console.print("[yellow]No patterns found matching your query.[/yellow]")
            return
        
        console.print(f"[bold]Found {len(results)} similar patterns:[/bold]")
        for i, result in enumerate(results, 1):
            console.print(f"\n{i}. [bold]{result['file_path']}[/bold] (score: {result['score']:.3f})")
            console.print(f"   {result['text'][:200]}...")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to search patterns: {e}")
        raise typer.Exit(1)


@app.command()
def repo_stats():
    """Show statistics about extracted repositories."""
    try:
        extractor = GitHubExtractor()
        stats = extractor.get_repository_stats()
        
        if not stats:
            console.print("[yellow]No repositories have been extracted yet.[/yellow]")
            return
        
        console.print("[bold]Repository Statistics:[/bold]")
        for repo_url, repo_data in stats.items():
            console.print(f"\n[bold]{repo_url}[/bold]")
            console.print(f"  Chunks: {repo_data['chunks']}")
            console.print(f"  Files: {repo_data['files']}")
            console.print(f"  Latest commit: {repo_data['latest_commit']}")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get repository stats: {e}")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about the codebase"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model to use"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of relevant chunks to include"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Query the codebase using LLM with context from vector store."""
    try:
        llm_engine = LLMEngine()
        
        # Test Ollama connection
        if not llm_engine.test_connection():
            console.print("[red]✗[/red] Cannot connect to Ollama")
            console.print("[yellow]Note:[/yellow] Start Ollama with: ollama serve")
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"[bold]Question:[/bold] {question}")
            console.print(f"[bold]Model:[/bold] {model}")
            console.print(f"[bold]Top-k:[/bold] {top_k}")
            console.print()
        
        with Progress() as progress:
            task = progress.add_task("Querying LLM...", total=None)
            
            result = llm_engine.query(question, model, top_k)
            
            if result["status"] == "success":
                console.print(f"\n[bold]Response:[/bold]")
                console.print(result["response"])
                
                if verbose:
                    console.print(f"\n[dim]Prompt length: {result['prompt_length']} characters[/dim]")
            else:
                console.print(f"[red]✗[/red] Query failed: {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to query: {e}")
        raise typer.Exit(1)


@app.command()
def list_models():
    """List available Ollama models."""
    try:
        llm_engine = LLMEngine()
        models = llm_engine.list_models()
        
        if models:
            console.print("[bold]Available models:[/bold]")
            for model in models:
                console.print(f"  • {model}")
        else:
            console.print("[yellow]No models found[/yellow]")
            console.print("[yellow]Note:[/yellow] Install models with: ollama pull <model_name>")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list models: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to")
):
    """Start the FastAPI server."""
    try:
        console.print(f"[bold]Starting 42 API server on {host}:{port}[/bold]")
        console.print(f"[dim]API docs: http://{host}:{port}/docs[/dim]")
        console.print(f"[dim]Health check: http://{host}:{port}/status[/dim]")
        console.print()
        
        run_server(host=host, port=port)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start server: {e}")
        raise typer.Exit(1)


@app.command()
def purge():
    """Purge all data from the system."""
    if typer.confirm("Are you sure you want to purge all data?"):
        try:
            vector_store = VectorStore()
            vector_store.delete_collection()
            console.print("[green]✓[/green] All data purged")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to purge data: {e}")
            raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main() 