"""CLI interface for 42."""

import typer
import time
import asyncio
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from loguru import logger

from ..utils.config import load_config, save_config
from ..core.embedding import EmbeddingEngine
from ..core.vector_store import VectorStore
from ..core.chunker import Chunker
from .github import GitHubExtractor
from ..core.llm import LLMEngine
from ..core.cluster import ClusteringEngine
from .jobs import get_task_status, list_tasks, extract_github_repository, recluster_vectors, import_data
from .api import run_server
from ...moroni.moroni import Moroni, ConversationContext, KnowledgeResponse
# Import heavy components only when needed
# from .un.knowledge_engine import KnowledgeEngine, KnowledgeSource, SourceType, DomainType
# from .un.redis_bus import RedisBus

app = typer.Typer()
console = Console()


@app.command()
def chat():
    """Start interactive chat with 42 using Moroni brain.
    
    Provides intelligent conversation with knowledge integration and tool execution.
    """
    console.print(Panel.fit(
        "[bold blue]42 Chat Agent[/bold blue]\n"
        "[dim]Powered by Moroni - The AI-Agnostic NLP Brain[/dim]\n\n"
        "Type your questions or requests. I can:\n"
        "• Answer questions using my knowledge base\n"
        "• Search for information\n"
        "• Help you learn new topics\n"
        "• Create and manage missions\n"
        "• Check system status\n\n"
        "Commands:\n"
        "• 'switch to openai' - Switch to OpenAI provider\n"
        "• 'switch to ollama' - Switch to Ollama provider\n"
        "• 'status' - Show current AI provider and usage\n"
        "• 'quit' or 'exit' - End conversation",
        title="🤖 42 Chat"
    ))
    
    # Initialize Moroni
    try:
        moroni = Moroni()
        console.print(f"[green]✓[/green] Moroni brain initialized (Provider: {moroni.primary_provider})")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize Moroni: {e}")
        raise typer.Exit(1)
    
    # Initialize conversation context
    context = ConversationContext(
        user_id="cli_user",
        conversation_history=[],
        current_mission=None,
        knowledge_base=[],
        tools_available=list(moroni.tools.keys())
    )
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[dim]Goodbye! Thanks for chatting with 42.[/dim]")
                break
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.lower().startswith('switch to '):
                provider = user_input.lower().replace('switch to ', '').strip()
                if moroni.switch_provider(provider):
                    console.print(f"[green]✓[/green] Switched to {provider} provider")
                else:
                    console.print(f"[red]✗[/red] Provider {provider} not available")
                continue
            
            if user_input.lower() == 'status':
                stats = moroni.get_usage_stats()
                console.print(f"\n[bold]AI Provider Status:[/bold]")
                console.print(f"• Current: {stats['current_provider']}")
                console.print(f"• Primary: {stats['primary_provider']}")
                console.print(f"• Fallback: {stats['fallback_provider']}")
                console.print(f"• Tokens used: {stats['total_tokens']}")
                console.print(f"• Cost: ${stats['total_cost']:.4f}")
                continue
            
            # Process request with Moroni
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("🧠 Moroni is thinking...", total=None)
                
                response = moroni.process_request(user_input, context)
                
                # Update conversation history
                context.conversation_history.append({
                    "user": user_input,
                    "assistant": response.response
                })
            
            # Display response
            console.print(f"\n[bold blue]42[/bold blue] {response.response}")
            
            # Show provider and confidence
            console.print(f"[dim]AI Provider: {response.ai_provider}[/dim]")
            
            if response.confidence < 0.7:
                console.print(f"[dim]Confidence: {response.confidence:.2f}[/dim]")
            
            if response.tools_used:
                console.print(f"[dim]Tools used: {', '.join(response.tools_used)}[/dim]")
            
            if response.sources:
                console.print(f"[dim]Sources: {len(response.sources)} relevant items found[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Chat interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logger.error(f"Chat error: {e}")


@app.command()
def create():
    """Create and initialize 42 system.
    
    Initializes the embedding engine, loads configuration, and sets up
    the basic 42 system infrastructure.
    
    Raises:
        typer.Exit: If system initialization fails.
    """
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
    """Check system status.
    
    Tests connections to Qdrant, Ollama, and Redis services,
    and displays current system configuration.
    
    Raises:
        typer.Exit: If status check fails.
    """
    try:
        config = load_config()
        
        # Create status table
        table = Table(title="42 System Status", show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Host:Port", style="green")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="yellow")
        
        # Add configuration rows
        table.add_row("Qdrant", f"{config.qdrant_host}:{config.qdrant_port}", "Config", "")
        table.add_row("Ollama", f"{config.ollama_host}:{config.ollama_port}", "Config", "")
        table.add_row("Redis", f"{config.redis_host}:{config.redis_port}", "Config", "")
        table.add_row("Embedding", config.embedding_model, "Config", f"Dim: {config.embedding_dimension}")
        table.add_row("Collection", config.collection_name, "Config", "")
        
        console.print(table)
        console.print()
        
        # Test connections with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Test embedding engine
            task1 = progress.add_task("Testing embedding engine...", total=1)
            try:
                embedding_engine = EmbeddingEngine()
                progress.update(task1, completed=1, description="✓ Embedding engine: OK")
            except Exception as e:
                progress.update(task1, completed=1, description=f"✗ Embedding engine: {str(e)[:50]}")
            
            # Test vector store
            task2 = progress.add_task("Testing vector store...", total=1)
            try:
                vector_store = VectorStore()
                if vector_store.test_connection():
                    progress.update(task2, completed=1, description="✓ Vector store: OK")
                else:
                    progress.update(task2, completed=1, description="✗ Vector store: Connection failed")
            except Exception as e:
                progress.update(task2, completed=1, description=f"✗ Vector store: {str(e)[:50]}")
            
            # Test LLM engine
            task3 = progress.add_task("Testing LLM engine...", total=1)
            try:
                llm_engine = LLMEngine()
                if llm_engine.test_connection():
                    models = llm_engine.list_models()
                    progress.update(task3, completed=1, description=f"✓ LLM engine: OK ({len(models)} models)")
                else:
                    progress.update(task3, completed=1, description="✗ LLM engine: Connection failed")
            except Exception as e:
                progress.update(task3, completed=1, description=f"✗ LLM engine: {str(e)[:50]}")
        
        # Show setup instructions if needed
        console.print()
        setup_panel = Panel(
            Text("Setup Instructions", style="bold"),
            Text("1. Start Qdrant: docker compose up -d\n2. Start Ollama: ollama serve\n3. Run: python3 -m 42 status"),
            title="[bold blue]Next Steps[/bold blue]",
            border_style="blue"
        )
        console.print(setup_panel)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to check status: {e}")
        raise typer.Exit(1)


@app.command()
def recluster(
    min_cluster_size: int = typer.Option(5, "--min-cluster-size", "-m", help="Minimum cluster size"),
    min_samples: int = typer.Option(3, "--min-samples", "-s", help="Minimum samples for core points"),
    generate_plot: bool = typer.Option(False, "--generate-plot", "-p", help="Generate cluster visualization plot")
):
    """Recluster all vectors using HDBSCAN."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Reclustering vectors...", total=None)
            
            # Initialize clustering engine
            progress.update(task, description="Initializing clustering engine...")
            clustering_engine = ClusteringEngine()
            
            # Perform reclustering
            progress.update(task, description="Performing HDBSCAN clustering...")
            clusters = clustering_engine.recluster_vectors(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
            
            # Get cluster statistics
            progress.update(task, description="Calculating cluster statistics...")
            stats = clustering_engine.get_cluster_stats()
            
            progress.update(task, description="Clustering complete!")
            
            # Display results
            console.print(f"[green]✓[/green] Reclustering complete!")
            console.print(f"[bold]Results:[/bold]")
            console.print(f"  • Total vectors: {stats.get('total_vectors', 0)}")
            console.print(f"  • Total clusters: {stats.get('total_clusters', 0)}")
            console.print(f"  • Noise points: {stats.get('noise_points', 0)}")
            
            if clusters:
                console.print(f"  • Clusters found: {len(clusters)}")
                for cluster_id, cluster in clusters.items():
                    console.print(f"    - Cluster {cluster_id}: {cluster.size} chunks")
            
            if generate_plot:
                console.print(f"[green]✓[/green] Cluster visualization saved to docs/cluster.png")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to recluster vectors: {e}")
        console.print("[yellow]Note:[/yellow] Install clustering dependencies with: pip install hdbscan umap-learn matplotlib")
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
    dump_embeddings: Optional[str] = typer.Option(None, "--dump-embeddings", help="Save embeddings to JSONL file"),
    timeout: int = typer.Option(600, "--timeout", "-t", help="Timeout in seconds (default: 600)")
):
    """Extract and analyze a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL to extract.
        background: Run in background mode.
        max_workers: Number of parallel workers.
        verbose: Enable verbose logging.
        dump_embeddings: Save embeddings to JSONL file.
        timeout: Timeout in seconds for the operation.
        
    Raises:
        typer.Exit: If extraction fails or times out.
    """
    try:
        extractor = GitHubExtractor(max_workers=max_workers, verbose=verbose, dump_embeddings_path=dump_embeddings)
        
        if background:
            # Start background job using Celery
            task = extract_github_repository.delay(repo_url, max_workers)
            console.print(f"[yellow]→[/yellow] Started background analysis for {repo_url}")
            console.print(f"Job ID: {task.id}")
            console.print("Use '42 job-status <job_id>' to check progress")
            return
        
        # Progress callback for real-time updates
        def progress_callback(message: str, progress_val: float):
            progress.update(task, description=f"{message} ({progress_val:.1%})")
        
        with Progress() as progress:
            task = progress.add_task("Analyzing repository...", total=None)
            
            # Add timeout handling
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = extractor.analyze_repository(repo_url, progress_callback=progress_callback)
            finally:
                signal.alarm(0)  # Cancel the alarm
            
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
        status = get_task_status(job_id)
        
        # Create status table
        table = Table(title=f"Job Status: {job_id}", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Task ID", job_id)
        table.add_row("State", status.get('state', 'unknown'))
        table.add_row("Status", status.get('status', 'Unknown'))
        
        if 'current' in status and 'total' in status:
            progress = f"{status['current']}/{status['total']} ({status['current']/status['total']*100:.1f}%)"
            table.add_row("Progress", progress)
        
        if 'result' in status:
            result = status['result']
            if isinstance(result, dict):
                for key, value in result.items():
                    table.add_row(f"Result.{key}", str(value))
        
        if 'error' in status:
            table.add_row("Error", status['error'])
        
        console.print(table)
        
        # Show color-coded status
        state = status.get('state', 'unknown')
        if state == 'success':
            console.print("[green]✓[/green] Job completed successfully")
        elif state == 'progress':
            console.print("[yellow]→[/yellow] Job is running")
        elif state == 'failure':
            console.print("[red]✗[/red] Job failed")
            raise typer.Exit(1)
        elif state == 'pending':
            console.print("[blue]⏳[/blue] Job is pending")
        else:
            console.print(f"[gray]?[/gray] Job state: {state}")
            
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="Timeout in seconds (default: 120)")
):
    """Query the codebase using LLM with context from vector store.
    
    Args:
        question: The question to ask about the codebase.
        model: Ollama model to use for response generation.
        top_k: Number of relevant chunks to include in context.
        verbose: Show detailed information about the query process.
        timeout: Timeout in seconds for the operation.
        
    Raises:
        typer.Exit: If query fails or times out.
    """
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
            from ..core.vector_store import VectorStore
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


steve_app = typer.Typer(help="Manage Steve - the autonomous source scanner.")

steve2_app = typer.Typer(help="Manage Steve 2.0 - the universal discovery agent.")


@steve_app.command()
def start():
    """Start Steve - the autonomous source scanner."""
    try:
        async def start_scanner():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve
            
            # Load configuration
            config = load_config()
            
            # Initialize components
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            
            # Create and start Steve
            from .un.soul import soul
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            await steve.start()
        
        asyncio.run(start_scanner())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start Steve: {e}")
        raise typer.Exit(1)


@steve_app.command()
def status():
    """Get Steve's status."""
    try:
        async def get_scanner_status():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            from .un.soul import soul
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            status = steve.get_status()
            
            console.print("[bold]Steve's Status[/bold]")
            console.print("=" * 40)
            console.print(f"Identity: {status['identity']}")
            console.print(f"Version: {status['version']}")
            console.print(f"Running: {'✓' if status['running'] else '✗'}")
            console.print(f"Discovered Sources: {status['discovered_sources']}")
            console.print(f"Crawled Domains: {status['crawled_domains']}")
            console.print(f"Pending Targets: {status['pending_targets']}")
            console.print(f"Learned Patterns: {status['learned_patterns']}")
            console.print(f"Discovery Events: {status['discovery_events']}")
            console.print(f"Total Mined: {status['mined_count']}")
            console.print(f"Total Embedded: {status['embedded_count']}")
            console.print(f"Today's Embeddings: {status['today_embeddings']}")
            
            if status['top_domains']:
                console.print(f"Top Domains: {', '.join(status['top_domains'])}")
            
            if status['last_embedded'] != "None":
                console.print(f"Last Embedded: {status['last_embedded']}")
            
            # Show soul configuration
            if 'soul_config' in status:
                console.print("\n[bold]Soul Configuration:[/bold]")
                soul = status['soul_config']
                if 'identity' in soul:
                    console.print(f"Identity: {soul['identity']}")
                if 'preferences' in soul:
                    prefs = soul['preferences']
                    if 'keywords' in prefs:
                        console.print(f"Keywords: {', '.join(prefs['keywords'][:5])}...")
                    if 'domains' in prefs:
                        console.print(f"Domains: {', '.join(prefs['domains'][:3])}...")
            
            # Show soul status
            if 'soul_status' in status:
                soul_status = status['soul_status']
                console.print(f"\n[bold]Soul Status:[/bold]")
                console.print(f"Locked: {'Yes' if soul_status['is_locked'] else 'No'}")
                console.print(f"Password Attempts: {soul_status['password_attempts']}/{soul_status['max_attempts']}")
                if soul_status['is_locked']:
                    console.print(f"Locked Until: {soul_status['locked_until']}")
            
            # Show recent searches
            if 'last_searches' in status and status['last_searches']:
                console.print(f"\n[bold]Recent Searches:[/bold]")
                for search in status['last_searches'][-3:]:  # Last 3
                    console.print(f"  • {search['query']} ({search['results_count']} results)")
            
            # Show recent queries
            if 'last_queries' in status and status['last_queries']:
                console.print(f"\n[bold]Recent Queries:[/bold]")
                for query in status['last_queries'][-3:]:  # Last 3
                    console.print(f"  • {query}")
        
        asyncio.run(get_scanner_status())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get Steve's status: {e}")
        raise typer.Exit(1)


@steve_app.command()
def discover(url: str = typer.Argument(..., help="URL to discover sources from")):
    """Manually trigger Steve to discover sources from a URL."""
    try:
        async def discover_sources():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve, CrawlTarget
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            from .un.soul import soul
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            # Setup browser
            await steve._setup_browser()
            
            # Create crawl target
            target = CrawlTarget(
                url=url,
                priority=1.0,
                crawl_depth=2,
                source_type="manual_discovery"
            )
            
            # Crawl the target
            await steve._crawl_target(target)
            
            console.print(f"[green]✓[/green] Discovered sources from {url}")
        
        asyncio.run(discover_sources())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to discover sources with Steve: {e}")
        raise typer.Exit(1)


@steve_app.command()
def learn():
    """Trigger Steve to learn from knowledge base."""
    try:
        async def learn_from_knowledge():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            from .un.soul import soul
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            # Learn from knowledge base
            await steve._learn_from_knowledge_base()
            
            console.print("[green]✓[/green] Steve learned from knowledge base")
            console.print(f"Learned patterns: {len(steve.learned_patterns)}")
        
        asyncio.run(learn_from_knowledge())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to learn from knowledge base with Steve: {e}")
        raise typer.Exit(1)

@steve_app.command()
def search(query: str = typer.Argument(..., help="Search query for Brave API")):
    """Perform manual search using Brave API."""
    try:
        async def perform_search():
            from .redis_bus import RedisBus
            import sys
            import os
            sys.path.append('.')
            
            # Import using importlib for the mission modules
            import importlib.util
            knowledge_spec = importlib.util.spec_from_file_location("knowledge_engine", "42/mission/steve/knowledge_engine.py")
            knowledge_module = importlib.util.module_from_spec(knowledge_spec)
            knowledge_spec.loader.exec_module(knowledge_module)
            KnowledgeEngine = knowledge_module.KnowledgeEngine
            
            steve_spec = importlib.util.spec_from_file_location("autonomous_scanner", "42/mission/steve/autonomous_scanner.py")
            steve_module = importlib.util.module_from_spec(steve_spec)
            steve_spec.loader.exec_module(steve_module)
            Steve = steve_module.Steve
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            from .un.soul import soul
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            # Perform search
            results = await steve.search_brave_api(query)
            
            console.print(f"[green]✓[/green] Search completed: '{query}'")
            console.print(f"Found {len(results)} filtered results")
            
            for i, result in enumerate(results[:5], 1):  # Show top 5
                console.print(f"  {i}. {result['title']}")
                console.print(f"     {result['url']}")
        
        asyncio.run(perform_search())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to perform search: {e}")
        raise typer.Exit(1)

@steve_app.command()
def mine():
    """Start Steve v3.1 Mine Mode - continuous knowledge mining."""
    try:
        async def perform_mine():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve
            from .un.soul import soul
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            
            # Get soul configuration from main system
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            # Start mine mode
            await steve.start()
            
            console.print("[green]✓[/green] Mine Mode completed")
        
        asyncio.run(perform_mine())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start Mine Mode: {e}")
        raise typer.Exit(1)

@steve_app.command()
def auto_search():
    """Trigger Steve's autonomous search."""
    try:
        async def perform_auto_search():
            from .un.redis_bus import RedisBus
            from .un.knowledge_engine import KnowledgeEngine
            from .un.autonomous_scanner import Steve
            from .un.soul import soul
            
            config = load_config()
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            
            # Get soul configuration from main system
            soul_config = soul.get_soul("steve")
            steve = Steve(redis_bus, knowledge_engine, config, soul_config)
            
            # Perform auto-search
            await steve.auto_search()
            
            console.print("[green]✓[/green] Auto-search completed")
            console.print(f"Added {len(steve.pending_targets)} new targets to queue")
        
        asyncio.run(perform_auto_search())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to perform auto-search: {e}")
        raise typer.Exit(1)

@steve_app.command()
def soul():
    """Show soul status (read-only)."""
    try:
        from .un.soul import soul
        
        soul_status = soul.get_status()
        
        console.print("[bold]Soul Status[/bold]")
        console.print("=" * 40)
        console.print(f"Identity: {soul_status['identity']}")
        console.print(f"Version: {soul_status['version']}")
        console.print(f"Locked: {'Yes' if soul_status['locked'] else 'No'}")
        console.print(f"Password Attempts: {soul_status['password_attempts']}/{soul_status['max_attempts']}")
        console.print(f"Last Updated: {soul_status['last_updated']}")
        console.print(f"Total Mined: {soul_status['total_mined']}")
        console.print(f"Total Embedded: {soul_status['total_embedded']}")
        
        if soul_status['last_queries']:
            console.print(f"Recent Queries: {', '.join(soul_status['last_queries'])}")
        
        if soul_status['last_discoveries']:
            console.print(f"Recent Discoveries: {len(soul_status['last_discoveries'])} items")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to show soul: {e}")
        raise typer.Exit(1)

@steve_app.command()
def soul_edit():
    """Edit soul (password required)."""
    try:
        from .un.soul import soul
        
        # Check if soul is locked
        soul_status = soul.get_status()
        if soul_status['locked']:
            console.print(f"[red]❌[/red] Soul is locked")
            raise typer.Exit(1)
        
        # Prompt for password
        password = typer.prompt("Enter soul password", hide_input=True)
        
        # Verify password
        if not soul.verify_password(password):
            console.print(f"[red]❌[/red] Invalid password. {soul_status['max_attempts'] - soul_status['password_attempts']} attempts remaining")
            raise typer.Exit(1)
        
        # Show current soul configuration
        console.print("[bold]Current Soul Configuration[/bold]")
        console.print("=" * 40)
        
        soul_config = soul.get_soul()
        console.print(f"Identity: {soul_config['identity']}")
        console.print(f"Version: {soul_config['version']}")
        console.print(f"Keywords: {', '.join(soul_config['preferences']['keywords'][:5])}...")
        console.print(f"Domains: {', '.join(soul_config['preferences']['domains'][:3])}...")
        console.print(f"Mining Interval: {soul_config['mining']['interval']} seconds")
        
        console.print("\n[yellow]Note: Full soul editing interface not implemented yet[/yellow]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to edit soul: {e}")
        raise typer.Exit(1)

@app.command()
def mission(objective: str = typer.Argument(..., help="Mission objective for Steve")):
    """Give Steve a mission to learn about something."""
    try:
        from .un.redis_bus import RedisBus
        from .un.mission_config import MissionOrchestrator
        from .un.autonomous_scanner import Steve
        from .un.knowledge_engine import KnowledgeEngine
        
        async def execute_mission():
            console.print(f"[blue]🚀[/blue] Creating mission: {objective}")
            
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine(redis_bus)
            
            # Create Steve instance
            config = {"mining_interval": 60}
            steve = Steve(redis_bus, knowledge_engine, config)
            
            # Create mission orchestrator
            orchestrator = MissionOrchestrator(redis_bus, steve)
            
            # Use Moroni to analyze the mission objective
            console.print(f"[blue]🧠[/blue] Analyzing mission with Moroni...")
            
            try:
                from .un.moroni import Moroni
                moroni = Moroni()
                
                # Basic keyword extraction for initial analysis
                words = objective.lower().split()
                basic_keywords = [word for word in words if len(word) > 3 and word not in 
                                ["the", "and", "for", "with", "about", "that", "this", "they", "have", "will", "from"]]
                basic_domains = []
                
                # Get intelligent search strategy from Moroni
                strategy = moroni.analyze_mission(objective, basic_keywords, basic_domains)
                
                keywords = strategy.primary_queries + strategy.secondary_queries
                domains = strategy.focus_domains
                
                console.print(f"[green]✓[/green] Moroni analysis completed")
                console.print(f"[blue]🧠 Moroni Strategy:[/blue]")
                console.print(f"Primary Queries: {', '.join(strategy.primary_queries[:3])}")
                console.print(f"Secondary Queries: {', '.join(strategy.secondary_queries[:3])}")
                console.print(f"Focus Domains: {', '.join(strategy.focus_domains[:5])}")
                console.print(f"Content Types: {', '.join(strategy.content_types)}")
                console.print(f"Search Depth: {strategy.search_depth}")
                console.print(f"Reasoning: {strategy.reasoning}")
                
                # Get usage stats
                stats = moroni.get_usage_stats()
                console.print(f"[dim]Cost: ${stats['total_cost']:.4f} ({stats['total_tokens']} tokens)[/dim]")
                    
            except Exception as e:
                console.print(f"[yellow]⚠️[/yellow] Moroni analysis failed: {e}")
                console.print(f"[dim]Using basic keyword extraction[/dim]")
                # Fallback to simple extraction
                words = objective.lower().split()
                keywords = [word for word in words if len(word) > 3 and word not in 
                           ["the", "and", "for", "with", "about", "that", "this", "they", "have", "will", "from"]]
                domains = []
            
            # Check existing knowledge before creating mission
            console.print(f"[blue]🔍[/blue] Checking existing knowledge...")
            
            try:
                # Query existing embeddings for relevant content
                from ..core.vector_store import VectorStore
                vector_store = VectorStore()
                
                # Search for existing knowledge about the mission
                search_query = " ".join(keywords[:3])  # Use top 3 keywords
                
                # For now, just check if we have any embeddings at all
                total_points = vector_store.count()
                if total_points > 0:
                    console.print(f"[green]✓[/green] Found {total_points} existing knowledge chunks")
                    console.print(f"[dim]Topics already covered:[/dim]")
                    console.print(f"  - {total_points} documents embedded in vector database")
                else:
                    console.print(f"[yellow]⚠️[/yellow] No existing knowledge found - starting fresh")
                
                # existing_results logic removed - using count instead
                    
            except Exception as e:
                console.print(f"[yellow]⚠️[/yellow] Could not check existing knowledge: {e}")
            
            # Create mission with enhanced data
            mission_id = orchestrator.create_mission(
                mission_type="learning",
                objective=objective,
                keywords=keywords,
                domains=domains,
                priority=8
            )
            
            console.print(f"[green]✓[/green] Mission assigned to Steve")
            console.print(f"Objective: {objective}")
            console.print(f"Keywords: {', '.join(keywords[:5])}")
            console.print(f"Mission ID: {mission_id}")
            
            # Automatically start mining
            console.print(f"\n[blue]🚀[/blue] Starting automatic mining for: {objective}")
            
            try:
                # Start Steve's mining process
                await steve.start()
                console.print(f"[green]✓[/green] Mining process started successfully")
                console.print(f"[dim]Steve is now actively learning about: {objective}[/dim]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️[/yellow] Could not start mining automatically: {e}")
                console.print(f"[dim]Run '42 steve mine' manually to start mining[/dim]")
        
        asyncio.run(execute_mission())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create mission: {e}")
        raise typer.Exit(1)


@steve_app.command()
def mission_list():
    """List all active missions."""
    try:
        from .un.redis_bus import RedisBus
        from .un.mission_config import MissionOrchestrator
        from .un.autonomous_scanner import Steve
        from .un.knowledge_engine import KnowledgeEngine
        
        async def list_missions():
            redis_bus = RedisBus()
            knowledge_engine = KnowledgeEngine()
            
            # Create Steve instance
            config = {"mining_interval": 60}
            steve = Steve(redis_bus, knowledge_engine, config)
            
            # Create mission orchestrator
            orchestrator = MissionOrchestrator(redis_bus, steve)
            
            missions = orchestrator.get_active_missions()
            
            console.print("[bold]Active Missions[/bold]")
            console.print(f"Total missions: {len(missions)}")
            
            for mission in missions:
                console.print(f"\n[bold]{mission['id']}[/bold]")
                console.print(f"  Type: {mission['type']}")
                console.print(f"  Objective: {mission['objective']}")
                console.print(f"  Priority: {mission['priority']}")
                console.print(f"  Keywords: {', '.join(mission['keywords'])}")
                console.print(f"  Domains: {', '.join(mission['domains'])}")
                console.print(f"  Expires: {mission['expires_at']}")
        
        asyncio.run(list_missions())
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list missions: {e}")
        raise typer.Exit(1)


@steve_app.command()
def tail(
    filter_domain: str = typer.Option(None, "--filter", "-f", help="Filter by domain"),
    filter_keyword: str = typer.Option(None, "--keyword", "-k", help="Filter by keyword")
):
    """Tail embedding log with optional filters."""
    try:
        from pathlib import Path
        import subprocess
        
        log_file = Path("/var/log/42/embedding.log")
        
        if not log_file.exists():
            console.print("[yellow]No embedding log found. Run '42 steve mine' first.[/yellow]")
            return
        
        # Build tail command
        cmd = ["tail", "-f", str(log_file)]
        
        # Add filters if specified
        if filter_domain:
            cmd.extend(["|", "grep", filter_domain])
        elif filter_keyword:
            cmd.extend(["|", "grep", filter_keyword])
        
        console.print(f"[bold]Tailing embedding log...[/bold]")
        console.print(f"Log file: {log_file}")
        if filter_domain:
            console.print(f"Filter: domain={filter_domain}")
        elif filter_keyword:
            console.print(f"Filter: keyword={filter_keyword}")
        console.print("=" * 50)
        
        # Run tail command
        subprocess.run(cmd)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to tail embedding log: {e}")
        raise typer.Exit(1)

@steve_app.command()
def unlock_status():
    """Show soul lock status."""
    try:
        from .un.soul import soul
        
        soul_status = soul.get_status()
        
        console.print("[bold]Soul Lock Status[/bold]")
        console.print("=" * 40)
        
        if soul_status['locked']:
            console.print(f"[red]❌ LOCKED[/red]")
            console.print("Soul is locked for 1000 years due to failed password attempts")
            console.print("The system will continue to function with default settings")
        else:
            console.print(f"[green]✓ UNLOCKED[/green]")
            console.print(f"Password Attempts: {soul_status['password_attempts']}/{soul_status['max_attempts']}")
            console.print("Soul is ready for editing")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get unlock status: {e}")
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app.add_typer(steve_app, name="steve")
    app()


if __name__ == "__main__":
    main() 