"""GitHub integration for 42."""

import os
import tempfile
import subprocess
import asyncio
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .chunker import Chunker
from .embedding import EmbeddingEngine
from .vector_store import VectorStore
from .interfaces import Chunk
from .job_manager import JobManager


class GitHubExtractor:
    """Extracts and analyzes GitHub repositories with background processing."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the GitHub extractor."""
        self.chunker = Chunker()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.max_workers = max_workers
        self.job_manager = JobManager()
        
    def clone_repository(self, repo_url: str, temp_dir: Optional[str] = None) -> str:
        """Clone a GitHub repository to a temporary directory."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="42_github_")
        
        try:
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, temp_dir],
                check=True,
                capture_output=True
            )
            logger.info(f"Repository cloned to: {temp_dir}")
            return temp_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def extract_repository_info(self, repo_path: str) -> Dict:
        """Extract basic information about the repository."""
        try:
            # Get git info
            result = subprocess.run(
                ["git", "-C", repo_path, "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True
            )
            repo_url = result.stdout.strip()
            
            # Get commit info
            result = subprocess.run(
                ["git", "-C", repo_path, "log", "-1", "--format=%H %s"],
                capture_output=True,
                text=True,
                check=True
            )
            commit_info = result.stdout.strip()
            
            return {
                "repo_url": repo_url,
                "local_path": repo_path,
                "latest_commit": commit_info,
                "extracted_at": str(Path(repo_path).stat().st_mtime)
            }
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not extract git info: {e}")
            return {
                "repo_url": "unknown",
                "local_path": repo_path,
                "latest_commit": "unknown",
                "extracted_at": str(Path(repo_path).stat().st_mtime)
            }
    
    def _batch_embed_chunks(self, chunks: List[Chunk], batch_size: int = 32, 
                           progress_callback: Optional[Callable] = None) -> List[List[float]]:
        """Embed chunks in batches for better performance."""
        vectors = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]
            
            # Use batch embedding if available
            try:
                batch_vectors = self.embedding_engine.embed_text_batch(batch_texts)
                vectors.extend(batch_vectors)
            except AttributeError:
                # Fallback to individual embedding
                for chunk in batch:
                    vector = self.embedding_engine.embed_text(chunk.text)
                    vectors.append(vector)
            
            batch_num = i // batch_size + 1
            logger.info(f"Embedded batch {batch_num}/{total_batches}")
            
            # Report progress if callback provided
            if progress_callback:
                progress = 0.5 + (0.2 * batch_num / total_batches)  # 50% to 70% range
                progress_callback(f"Embedding chunks... (batch {batch_num}/{total_batches})", progress)
        
        return vectors
    
    def _create_points_batch(self, chunks: List[Chunk], vectors: List[List[float]], 
                           repo_url: str, repo_info: Dict) -> List:
        """Create Qdrant points in batches."""
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # Add repository metadata
            chunk_metadata = dict(chunk.metadata) if chunk.metadata else {}
            chunk_metadata.update({
                "repo_url": repo_url,
                "repo_info": repo_info,
                "chunk_id": i
            })
            
            # Create point with integer ID
            point = PointStruct(
                id=hash(f"{repo_url}_{i}") % (2**63),  # Use hash for integer ID
                vector=vector,
                payload={
                    "text": chunk.text,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "metadata": chunk_metadata
                }
            )
            points.append(point)
        
        return points
    
    def analyze_repository(self, repo_url: str, background: bool = False, 
                          progress_callback: Optional[Callable] = None) -> Dict:
        """Analyze a GitHub repository and store its patterns."""
        if background:
            return self._analyze_repository_background(repo_url, progress_callback)
        else:
            return self._analyze_repository_sync(repo_url, progress_callback)
    
    def _analyze_repository_sync(self, repo_url: str, 
                                progress_callback: Optional[Callable] = None) -> Dict:
        """Synchronous repository analysis with progress tracking."""
        temp_dir = None
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback("Cloning repository...", 0.1)
            
            # Clone the repository
            temp_dir = self.clone_repository(repo_url)
            
            if progress_callback:
                progress_callback("Extracting repository info...", 0.2)
            
            # Extract repository info
            repo_info = self.extract_repository_info(temp_dir)
            
            if progress_callback:
                progress_callback("Chunking files...", 0.3)
            
            # Chunk the repository
            chunks = self.chunker.chunk_directory(temp_dir)
            
            if not chunks:
                logger.warning(f"No chunks found in repository: {repo_url}")
                return {
                    "repo_url": repo_url,
                    "chunks": 0,
                    "status": "no_chunks_found"
                }
            
            if progress_callback:
                progress_callback("Creating vector collection...", 0.4)
            
            # Ensure collection exists
            self.vector_store.create_collection(self.embedding_engine.get_dimension())
            
            # Embed chunks in batches with progress tracking
            vectors = self._batch_embed_chunks(chunks, progress_callback=progress_callback)
            
            if progress_callback:
                progress_callback("Creating database points...", 0.7)
            
            # Create points in batches
            points = self._create_points_batch(chunks, vectors, repo_url, repo_info)
            
            if progress_callback:
                progress_callback("Storing in vector database...", 0.8)
            
            # Store in vector database in batches
            batch_size = 100
            total_storage_batches = (len(points) + batch_size - 1) // batch_size
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.vector_store.upsert(batch)
                
                if progress_callback:
                    batch_num = i // batch_size + 1
                    progress = 0.7 + (0.25 * batch_num / total_storage_batches)  # 70% to 95% range
                    progress_callback(f"Storing batch {batch_num}/{total_storage_batches}...", min(progress, 0.95))
            
            elapsed_time = time.time() - start_time
            logger.info(f"Analyzed repository {repo_url}: {len(chunks)} chunks stored in {elapsed_time:.2f}s")
            
            if progress_callback:
                progress_callback("Analysis complete!", 1.0)
            
            return {
                "repo_url": repo_url,
                "chunks": len(chunks),
                "status": "success",
                "repo_info": repo_info,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to analyze repository {repo_url}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "repo_url": repo_url,
                "chunks": 0,
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    subprocess.run(["rm", "-rf", temp_dir], check=True)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {e}")
    
    def _analyze_repository_background(self, repo_url: str, 
                                      progress_callback: Optional[Callable] = None) -> Dict:
        """Start repository analysis in background thread."""
        job_id = self.job_manager.create_job("github_extract", repo_url=repo_url)
        
        def background_worker():
            try:
                self.job_manager.update_job(job_id, status="running", started_at=time.time())
                result = self._analyze_repository_sync(repo_url, progress_callback)
                self.job_manager.update_job(job_id, 
                                          status="completed", 
                                          result=result, 
                                          completed_at=time.time())
            except Exception as e:
                self.job_manager.update_job(job_id, 
                                          status="failed", 
                                          error=str(e), 
                                          completed_at=time.time())
        
        # Start background thread
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
        
        logger.info(f"Started background analysis for {repo_url} (job_id: {job_id})")
        
        return {
            "job_id": job_id,
            "status": "started",
            "repo_url": repo_url
        }
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a background job."""
        job = self.job_manager.get_job(job_id)
        if not job:
            return {"status": "not_found"}
        
        if job["status"] == "running":
            elapsed = time.time() - job.get("started_at", time.time())
            return {
                "status": "running",
                "elapsed_time": elapsed,
                "repo_url": job.get("kwargs", {}).get("repo_url", "unknown")
            }
        else:
            return job
    
    def list_background_jobs(self) -> Dict:
        """List all background jobs."""
        return self.job_manager.list_jobs("github_extract")
    
    def search_patterns(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for code patterns similar to the query."""
        try:
            # Embed the query
            query_vector = self.embedding_engine.embed_text(query)
            
            # Search for similar patterns
            results = self.vector_store.search(query_vector, limit=limit)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.text,
                    "file_path": result.file_path,
                    "score": result.score,
                    "metadata": result.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search patterns: {e}")
            return []
    
    def get_repository_stats(self) -> Dict:
        """Get statistics about stored repositories."""
        try:
            all_vectors = self.vector_store.get_all_vectors()
            
            # Group by repository
            repo_stats = {}
            for vector_data in all_vectors:
                repo_url = vector_data.get("metadata", {}).get("repo_url", "unknown")
                if repo_url not in repo_stats:
                    repo_stats[repo_url] = {
                        "chunks": 0,
                        "files": set(),
                        "latest_commit": vector_data.get("metadata", {}).get("repo_info", {}).get("latest_commit", "unknown")
                    }
                
                repo_stats[repo_url]["chunks"] += 1
                file_path = vector_data.get("file_path", "unknown")
                repo_stats[repo_url]["files"].add(file_path)
            
            # Convert sets to counts
            for repo in repo_stats:
                repo_stats[repo]["files"] = len(repo_stats[repo]["files"])
            
            return repo_stats
            
        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            return {} 