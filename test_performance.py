#!/usr/bin/env python3
"""Performance test for 42 GitHub extraction."""

import time
import sys
from pathlib import Path

# Add the 42 module to the path
sys.path.insert(0, str(Path(__file__).parent / "42"))

from github import GitHubExtractor
from cli import console


def test_small_repo():
    """Test with a small repository."""
    console.print("[bold]Testing Small Repository[/bold]")
    
    extractor = GitHubExtractor()
    
    # Test with a small, focused repo
    repo_url = "https://github.com/tiangolo/typer"
    
    start_time = time.time()
    
    def progress_callback(message: str, progress: float):
        console.print(f"  {message} ({progress:.1%})")
    
    result = extractor.analyze_repository(repo_url, progress_callback=progress_callback)
    
    elapsed_time = time.time() - start_time
    
    if result["status"] == "success":
        console.print(f"[green]✓[/green] Success! {result['chunks']} chunks in {elapsed_time:.2f}s")
        console.print(f"  Rate: {result['chunks']/elapsed_time:.1f} chunks/second")
    else:
        console.print(f"[red]✗[/red] Failed: {result.get('error', 'Unknown error')}")


def test_background_processing():
    """Test background processing."""
    console.print("\n[bold]Testing Background Processing[/bold]")
    
    extractor = GitHubExtractor()
    
    # Start background job
    result = extractor.analyze_repository("https://github.com/tiangolo/fastapi", background=True)
    job_id = result["job_id"]
    
    console.print(f"  Started background job: {job_id}")
    
    # Monitor progress
    for i in range(10):
        time.sleep(5)
        status = extractor.get_job_status(job_id)
        
        if status["status"] == "running":
            elapsed = status["elapsed_time"]
            console.print(f"  Still running... ({elapsed:.1f}s elapsed)")
        elif status["status"] == "completed":
            result = status["result"]
            console.print(f"[green]✓[/green] Completed! {result['chunks']} chunks")
            break
        elif status["status"] == "failed":
            console.print(f"[red]✗[/red] Failed: {status.get('error', 'Unknown error')}")
            break
    else:
        console.print("[yellow]→[/yellow] Still running after 50s...")


def test_batch_performance():
    """Test batch embedding performance."""
    console.print("\n[bold]Testing Batch Performance[/bold]")
    
    from embedding import EmbeddingEngine
    
    engine = EmbeddingEngine()
    
    # Test texts
    texts = [
        "def hello(): return 'world'",
        "class User: pass",
        "import os",
        "print('hello')",
        "def main(): pass"
    ] * 10  # 50 texts total
    
    # Test individual embedding
    start_time = time.time()
    for text in texts:
        engine.embed_text(text)
    individual_time = time.time() - start_time
    
    # Test batch embedding
    start_time = time.time()
    engine.embed_text_batch(texts)
    batch_time = time.time() - start_time
    
    console.print(f"  Individual embedding: {individual_time:.2f}s")
    console.print(f"  Batch embedding: {batch_time:.2f}s")
    console.print(f"  Speedup: {individual_time/batch_time:.1f}x")


if __name__ == "__main__":
    console.print("[bold blue]42 Performance Test[/bold blue]\n")
    
    try:
        test_batch_performance()
        test_small_repo()
        test_background_processing()
        
        console.print("\n[bold green]Performance test completed![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Test failed: {e}[/red]") 