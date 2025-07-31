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
# Import heavy components only when needed
# from .un.knowledge_engine import KnowledgeEngine, KnowledgeSource, SourceType, DomainType
# from .un.redis_bus import RedisBus

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


@app.command()
def watch():
    """Start the universal knowledge engine."""
    try:
        import asyncio
        from .un.knowledge_engine import KnowledgeEngine, KnowledgeSource
        from .un.redis_bus import RedisBus
        
        console.print("[bold]Starting Universal Knowledge Engine...[/bold]")
        
        async def run_knowledge_engine():
            redis_bus = RedisBus()
            engine = KnowledgeEngine(redis_bus)
            
            # Load sources from file if exists
            import json
            import os
            
            if os.path.exists("universal_sources.json"):
                with open("universal_sources.json", "r") as f:
                    sources_data = json.load(f)
                    for item in sources_data:
                        source = KnowledgeSource.from_dict(item)
                        engine.add_source(source)
                        console.print(f"  [green]✓[/green] Loaded: {source.name}")
            
            console.print(f"\n[bold]Monitoring {len(engine.sources)} sources...[/bold]")
            console.print("Press Ctrl+C to stop")
            
            await engine.start_monitoring(interval_seconds=300)
        
        asyncio.run(run_knowledge_engine())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Knowledge engine stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start knowledge engine: {e}")
        raise typer.Exit(1)


@app.command()
def learn(
    name: str = typer.Argument(..., help="Source name"),
    url: str = typer.Argument(..., help="Source URL"),
    source_type: str = typer.Argument(..., help="Source type (rss, api, github)"),
    domain: str = typer.Argument(..., help="Domain (weather, medical, finance, research, etc.)"),
    description: str = typer.Option("", "--description", "-d", help="Description"),
    validate: bool = typer.Option(True, "--validate", "-v", help="Validate source before adding")
):
    """Add a knowledge source to the universal engine."""
    try:
        import json
        import os
        import asyncio
        import aiohttp
        
        # Validate source if requested - without initializing heavy components
        if validate:
            console.print(f"[bold]Validating source: {name}[/bold]")
            
            async def validate_source():
                if source_type.lower() == "rss":
                    # Use lightweight validation
                    import subprocess
                    import sys
                    
                    try:
                        # Get the directory where the CLI module is located
                        import os
                        cli_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(cli_dir)
                        validate_script = os.path.join(project_root, "validate_rss.py")
                        
                        result = subprocess.run([
                            sys.executable, validate_script, url
                        ], capture_output=True, text=True, timeout=15)
                        
                        if result.returncode == 0:
                            console.print(result.stdout.strip())
                            return True
                        else:
                            console.print(result.stderr.strip())
                            return False
                    except subprocess.TimeoutExpired:
                        console.print(f"[red]✗[/red] Validation timed out after 15 seconds")
                        return False
                    except Exception as e:
                        console.print(f"[red]✗[/red] Validation failed: {e}")
                        return False
                else:
                    console.print(f"[yellow]⚠[/yellow] Validation not implemented for {source_type}")
                    return False
            
            validation_result = asyncio.run(validate_source())
            if not validation_result:
                console.print("[yellow]Source not added - validation failed[/yellow]")
                return
        
        # Only initialize heavy components after validation passes
        from .un.knowledge_engine import KnowledgeSource, SourceType, DomainType
        
        # Load existing sources
        sources = []
        if os.path.exists("universal_sources.json"):
            with open("universal_sources.json", "r") as f:
                sources_data = json.load(f)
                for item in sources_data:
                    sources.append(KnowledgeSource.from_dict(item))
        
        # Create new source
        source_id = f"source_{name.lower().replace(' ', '_')}"
        new_source = KnowledgeSource(
            id=source_id,
            name=name,
            type=SourceType(source_type.lower()),
            domain=DomainType(domain.lower()),
            url=url,
            frequency="5min",
            parser=source_type.lower(),
            active=True,
            metadata={"description": description}
        )
        
        # Add to list
        sources.append(new_source)
        
        # Save back to file
        data = [source.to_dict() for source in sources]
        with open("universal_sources.json", "w") as f:
            json.dump(data, f, indent=2)
        
        console.print(f"[green]✓[/green] Added source: {name}")
        console.print(f"   ID: {source_id}")
        console.print(f"   Type: {source_type}")
        console.print(f"   Domain: {domain}")
        console.print(f"   URL: {url}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to add source: {e}")
        raise typer.Exit(1)


@app.command()
def sources():
    """List all knowledge sources."""
    try:
        import json
        import os
        
        if not os.path.exists("universal_sources.json"):
            console.print("[yellow]No sources found. Add some with: 42 learn[/yellow]")
            return
        
        with open("universal_sources.json", "r") as f:
            sources_data = json.load(f)
        
        console.print("[bold]Knowledge Sources:[/bold]")
        console.print("=" * 60)
        
        for item in sources_data:
            status = "[green]Active[/green]" if item.get("active", True) else "[red]Inactive[/red]"
            console.print(f"{item['name']} ({item['id']})")
            console.print(f"  Type: {item['type']}")
            console.print(f"  Domain: {item['domain']}")
            console.print(f"  URL: {item['url']}")
            console.print(f"  Status: {status}")
            console.print()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list sources: {e}")
        raise typer.Exit(1)


@app.command()
def test(
    source_id: str = typer.Argument(..., help="Source ID to test")
):
    """Test a knowledge source."""
    try:
        import json
        import os
        import asyncio
        
        if not os.path.exists("universal_sources.json"):
            console.print("[red]No sources found[/red]")
            return
        
        with open("universal_sources.json", "r") as f:
            sources_data = json.load(f)
        
        # Find the source
        source_data = None
        for item in sources_data:
            if item["id"] == source_id:
                source_data = item
                break
        
        if not source_data:
            console.print(f"[red]Source not found: {source_id}[/red]")
            return
        
        from .un.knowledge_engine import KnowledgeSource
        source = KnowledgeSource.from_dict(source_data)
        console.print(f"[bold]Testing source: {source.name}[/bold]")
        
        async def test_source():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                from .un.knowledge_engine import SourceType, RSSFetcher, APIFetcher
                if source.type == SourceType.RSS:
                    fetcher = RSSFetcher(session)
                elif source.type == SourceType.API:
                    fetcher = APIFetcher(session)
                else:
                    console.print(f"[red]No fetcher for type: {source.type}[/red]")
                    return
                
                documents = await fetcher.fetch(source)
                console.print(f"[green]✓[/green] Fetched {len(documents)} documents")
                
                for i, doc in enumerate(documents[:3]):  # Show first 3
                    console.print(f"  {i+1}. {doc.content[:100]}...")
        
        asyncio.run(test_source())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to test source: {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to return")
):
    """Search the knowledge base."""
    try:
        import asyncio
        from .un.knowledge_engine import KnowledgeEngine
        from .un.redis_bus import RedisBus
        
        console.print(f"[bold]Searching knowledge base for: {query}[/bold]")
        
        async def search_knowledge():
            # Use existing 42.zero search functionality directly
            embedding_engine = EmbeddingEngine()
            vector_store = VectorStore()  # Uses default 42_chunks collection
            
            # Load sources from file if exists
            import json
            import os
            
            # Generate embedding for query using existing 42.zero tools
            query_embedding = embedding_engine.embed_text(query)
            
            # Search using existing 42.zero VectorStore
            search_results = vector_store.search(query_embedding, limit=limit)
            
            # Convert to display format
            results = []
            for result in search_results:
                results.append({
                    "content": result.text,
                    "source_id": result.metadata.get("source_id", result.file_path),
                    "score": result.score,
                    "timestamp": result.metadata.get("timestamp", ""),
                    "metadata": result.metadata
                })
            
            if not results:
                console.print("[yellow]No results found[/yellow]")
                return
            
            console.print(f"\n[bold]Found {len(results)} results:[/bold]")
            console.print("=" * 80)
            
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                source = result.get("source_id", "unknown")
                content = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
                
                console.print(f"{i}. [bold]{source}[/bold] (score: {score:.3f})")
                console.print(f"   {content}")
                console.print()
        
        asyncio.run(search_knowledge())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to search knowledge base: {e}")
        raise typer.Exit(1)

@app.command()
def status():
    """Check knowledge engine status and queue information."""
    try:
        import redis
        import json
        import os
        from datetime import datetime
        
        console.print("[bold]Knowledge Engine Status[/bold]")
        console.print("=" * 50)
        
        # Check Redis connection and queues
        try:
            r = redis.Redis()
            r.ping()
            console.print("[green]✓[/green] Redis: Connected")
            
            # Check event queues
            doc_queue_len = r.llen("42:events:knowledge.document")
            console.print(f"  Documents in queue: {doc_queue_len}")
            
            # Check if there are any recent events
            recent_events = r.lrange("42:events:knowledge.document", -5, -1)
            if recent_events:
                console.print(f"  Recent events: {len(recent_events)}")
                for event in recent_events[-3:]:  # Show last 3
                    event_data = json.loads(event)
                    timestamp = event_data.get('timestamp', 'Unknown')
                    console.print(f"    - {timestamp[:19]} | {event_data.get('data', {}).get('source_id', 'Unknown')}")
            else:
                console.print("  No recent events found")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Redis: {e}")
        
        # Check knowledge sources
        if os.path.exists("universal_sources.json"):
            with open("universal_sources.json", "r") as f:
                sources_data = json.load(f)
            
            active_sources = [s for s in sources_data if s.get("active", True)]
            console.print(f"[blue]ℹ[/blue] Sources: {len(active_sources)} active, {len(sources_data)} total")
            
            for source in active_sources[:3]:  # Show first 3 active sources
                console.print(f"  - {source['name']} ({source['type']}) | {source['url'][:50]}...")
        else:
            console.print("[yellow]⚠[/yellow] No sources configured")
        
        # Check vector database
        try:
            from .vector_store import VectorStore
            vs = VectorStore()
            total_points = vs.get_total_points()
            console.print(f"[blue]ℹ[/blue] Vector DB: {total_points} documents stored")
        except Exception as e:
            console.print(f"[red]✗[/red] Vector DB: {e}")
        
        # Check if knowledge engine process is running
        import subprocess
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            if "42 watch" in result.stdout:
                console.print("[green]✓[/green] Knowledge Engine: Running")
            else:
                console.print("[yellow]⚠[/yellow] Knowledge Engine: Not running")
        except:
            console.print("[yellow]⚠[/yellow] Knowledge Engine: Status unknown")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to check status: {e}")
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main() 