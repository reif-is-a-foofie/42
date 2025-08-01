"""GitHub integration for 42."""

import os
import tempfile
import subprocess
import asyncio
import threading
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import tqdm

from .chunker import Chunker
from .embedding import EmbeddingEngine
from .vector_store import VectorStore
from .interfaces import Chunk
# JobManager removed - using Celery for job management


class GitHubExtractor:
    """Extracts and analyzes GitHub repositories with parallel processing."""
    
    def __init__(self, max_workers: int = None, verbose: bool = True, dump_embeddings_path: Optional[str] = None):
        """Initialize the GitHub extractor."""
        self.chunker = Chunker()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.max_workers = max_workers or min(max(4, os.cpu_count()), 12)  # Auto-detect with 4-12 range
        # Job management now handled by Celery
        self.verbose = verbose
        self.dump_embeddings_path = dump_embeddings_path
        # Embedding dump is now handled directly in _stream_to_vector_db
        if self.dump_embeddings_path:
            logger.info(f"Embedding dump enabled: {self.dump_embeddings_path}")
        
        # File size and type filters
        self.max_file_size = 2 * 1024 * 1024  # 2MB
        self.text_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.html', '.css', '.scss',
            '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.ini',
            '.cfg', '.conf', '.sh', '.bash', '.zsh', '.fish',
            '.sql', '.r', '.m', '.scala', '.kt', '.swift', '.dart'
        }
    
    def __del__(self):
        """Cleanup method."""
        pass
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        try:
            # Check file size
            if os.path.getsize(file_path) > self.max_file_size:
                return False
            
            # Check file extension
            ext = Path(file_path).suffix.lower()
            if ext not in self.text_extensions:
                return False
            
            # Skip common non-code directories
            path_parts = Path(file_path).parts
            skip_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 
                        'build', 'dist', 'target', 'bin', 'obj', '.venv', 'venv'}
            
            for part in path_parts:
                if part in skip_dirs:
                    return False
            
            return True
            
        except Exception:
            return False
    

    
    def _process_single_file(self, file_path: str) -> Tuple[str, List[Chunk]]:
        """Process a single file and return chunks."""
        try:
            if not self._should_process_file(file_path):
                if self.verbose:
                    logger.debug(f"Skipping {file_path} (filtered out)")
                return file_path, []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            chunks = self.chunker.chunk_file(file_path)
            
            # Note: Verbose logging moved to main thread to avoid thread-safety issues
            
            return file_path, chunks
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return file_path, []
    
    def _process_files_parallel(self, repo_path: str, progress_callback: Optional[Callable] = None) -> List[Chunk]:
        """Process files in parallel using ThreadPoolExecutor."""
        # Find all files to process
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self._should_process_file(file_path):
                    all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process files in parallel
        chunks = []
        file_chunks_map = {}  # Track chunks per file for verbose logging
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path 
                for file_path in all_files
            }
            
            # Collect results with progress
            with tqdm.tqdm(total=len(all_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path, file_chunks = future.result()
                    chunks.extend(file_chunks)
                    file_chunks_map[file_path] = file_chunks
                    pbar.update(1)
                    
                    if progress_callback:
                        progress = len(chunks) / max(len(all_files), 1)
                        progress_callback(f"Processing files... ({len(chunks)} chunks)", progress * 0.3)
        
        # Verbose logging in main thread
        if self.verbose:
            for file_path, file_chunks in file_chunks_map.items():
                if file_chunks:
                    logger.info(f"Processed {file_path}: {len(file_chunks)} chunks")
                    for i, chunk in enumerate(file_chunks):
                        preview = chunk.text[:80].replace('\n', '\\n')
                        logger.info(f"  Chunk {i+1}: {preview}...")
        
        return chunks
    
    def _batch_embed_chunks_optimized(self, chunks: List[Chunk], 
                                     progress_callback: Optional[Callable] = None) -> List[List[float]]:
        """Embed chunks in optimized batches."""
        vectors = []
        batch_size = 64  # Increased batch size for better performance
        
        with tqdm.tqdm(total=len(chunks), desc="Embedding chunks") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk.text for chunk in batch]
                
                # Use batch embedding
                try:
                    batch_vectors = self.embedding_engine.embed_text_batch(batch_texts)
                    vectors.extend(batch_vectors)
                    
                    # Verbose logging for batch
                    if self.verbose:
                        logger.info(f"Embedded batch {i//batch_size + 1}: {len(batch)} chunks")
                        for j, (chunk, vector) in enumerate(zip(batch, batch_vectors)):
                            preview = chunk.text[:80].replace('\n', '\\n')
                            vector_preview = vector[:5] if len(vector) >= 5 else vector
                            logger.info(f"  Chunk {i+j+1}: {preview}... -> vector[{len(vector)}] {vector_preview}...")
                    
                except AttributeError:
                    # Fallback to individual embedding
                    for chunk in batch:
                        vector = self.embedding_engine.embed_text(chunk.text)
                        vectors.append(vector)
                        
                        if self.verbose:
                            preview = chunk.text[:80].replace('\n', '\\n')
                            vector_preview = vector[:5] if len(vector) >= 5 else vector
                            logger.info(f"  Chunk: {preview}... -> vector[{len(vector)}] {vector_preview}...")
                
                pbar.update(len(batch))
                
                if progress_callback:
                    progress = 0.3 + (0.4 * (i + batch_size) / len(chunks))
                    progress_callback(f"Embedding chunks... (batch {i//batch_size + 1})", min(progress, 0.7))
        
        return vectors
    
    def _stream_to_vector_db(self, chunks: List[Chunk], vectors: List[List[float]], 
                            repo_url: str, repo_info: Dict,
                            progress_callback: Optional[Callable] = None) -> None:
        """Stream results to vector database in batches and dump embeddings."""
        from qdrant_client.models import PointStruct
        
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        total_stored = 0
        
        # Open dump file if enabled
        dump_file = None
        if self.dump_embeddings_path:
            dump_file = open(self.dump_embeddings_path, 'w')
            logger.info(f"Writing embeddings to: {self.dump_embeddings_path}")
        
        try:
            with tqdm.tqdm(total=len(chunks), desc="Storing in vector DB") as pbar:
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    
                    # Create points for this batch
                    points = []
                    for j, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
                        chunk_metadata = dict(chunk.metadata) if chunk.metadata else {}
                        chunk_metadata.update({
                            "repo_url": repo_url,
                            "repo_info": repo_info,
                            "chunk_id": i + j
                        })
                        
                        point = PointStruct(
                            id=hash(f"{repo_url}_{i + j}") % (2**63),
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
                    
                    # Store batch in vector DB
                    self.vector_store.upsert(points)
                    total_stored += len(batch_chunks)
                    
                    # Dump embeddings to file
                    if dump_file:
                        for chunk, vector in zip(batch_chunks, batch_vectors):
                            data = {
                                "file": chunk.file_path,
                                "chunk": chunk.text,
                                "embedding": vector
                            }
                            dump_file.write(json.dumps(data) + '\n')
                        dump_file.flush()  # Ensure data is written
                    
                    if self.verbose:
                        logger.info(f"Stored batch {i//batch_size + 1}/{total_batches}: {len(batch_chunks)} chunks (total: {total_stored})")
                    
                    pbar.update(len(batch_chunks))
                    
                    if progress_callback:
                        progress = 0.7 + (0.3 * (i + batch_size) / len(chunks))
                        progress_callback(f"Storing batch {i//batch_size + 1}/{total_batches}...", min(progress, 0.95))
        finally:
            if dump_file:
                dump_file.close()
                logger.info(f"✓ Dumped {len(chunks)} embeddings → {self.dump_embeddings_path}")
    
    def clone_repository(self, repo_url: str, temp_dir: Optional[str] = None) -> str:
        """Clone a GitHub repository."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="42_github_")
        
        try:
            # Use shallow clone for speed
            subprocess.run([
                "git", "clone", "--depth", "1", "--single-branch", 
                repo_url, temp_dir
            ], check=True, capture_output=True)
            
            logger.info(f"Cloned repository to {temp_dir}")
            return temp_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def extract_repository_info(self, repo_path: str) -> Dict:
        """Extract repository information."""
        try:
            # Get latest commit
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H %s"],
                cwd=repo_path, capture_output=True, text=True, check=True
            )
            latest_commit = result.stdout.strip()
            
            return {
                "repo_url": "unknown",
                "local_path": repo_path,
                "latest_commit": latest_commit,
                "extracted_at": str(Path(repo_path).stat().st_mtime)
            }
        except Exception:
            return {
                "repo_url": "unknown",
                "local_path": repo_path,
                "latest_commit": "unknown",
                "extracted_at": str(Path(repo_path).stat().st_mtime)
            }
    
    def analyze_repository(self, repo_url: str, background: bool = False, 
                          progress_callback: Optional[Callable] = None) -> Dict:
        """Analyze a GitHub repository with parallel processing."""
        if background:
            return self._analyze_repository_background(repo_url, progress_callback)
        else:
            return self._analyze_repository_sync_optimized(repo_url, progress_callback)
    
    def _analyze_repository_sync_optimized(self, repo_url: str, 
                                          progress_callback: Optional[Callable] = None) -> Dict:
        """Optimized synchronous repository analysis."""
        temp_dir = None
        start_time = time.time()
        
        # Check if repository already processed (skip for now to test performance)
        # existing_stats = self.get_repository_stats()
        # if repo_url in existing_stats:
        #     logger.info(f"Repository {repo_url} already processed with {existing_stats[repo_url]['chunks']} chunks")
        #     return {
        #         "repo_url": repo_url,
        #         "chunks": existing_stats[repo_url]["chunks"],
        #         "status": "already_processed",
        #         "elapsed_time": 0
        #     }
        
        try:
            if progress_callback:
                progress_callback("Cloning repository...", 0.05)
            
            # Clone the repository
            temp_dir = self.clone_repository(repo_url)
            
            if progress_callback:
                progress_callback("Extracting repository info...", 0.1)
            
            # Extract repository info
            repo_info = self.extract_repository_info(temp_dir)
            
            if progress_callback:
                progress_callback("Processing files in parallel...", 0.15)
            
            # Process files in parallel
            chunks = self._process_files_parallel(temp_dir, progress_callback)
            
            if not chunks:
                logger.warning(f"No chunks found in repository: {repo_url}")
                return {
                    "repo_url": repo_url,
                    "chunks": 0,
                    "status": "no_chunks_found"
                }
            
            if progress_callback:
                progress_callback("Creating vector collection...", 0.25)
            
            # Ensure collection exists
            self.vector_store.create_collection(self.embedding_engine.get_dimension())
            
            # Embed chunks in optimized batches
            vectors = self._batch_embed_chunks_optimized(chunks, progress_callback)
            
            # Stream to vector database and dump embeddings
            self._stream_to_vector_db(chunks, vectors, repo_url, repo_info, progress_callback)
            
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
            
            # Embedding dump is now handled in _stream_to_vector_db
    
    def _analyze_repository_background(self, repo_url: str, 
                                      progress_callback: Optional[Callable] = None) -> Dict:
        """Start repository analysis using Celery."""
        from .jobs import extract_github_repository
        
        # Start Celery task
        task = extract_github_repository.delay(repo_url, self.max_workers)
        
        logger.info(f"Started background analysis for {repo_url} (task_id: {task.id})")
        
        return {
            "job_id": task.id,
            "status": "started",
            "repo_url": repo_url
        }
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a background job using Celery."""
        from .jobs import get_task_status
        return get_task_status(job_id)
    
    def list_background_jobs(self) -> Dict:
        """List all background jobs using Celery."""
        from .jobs import list_tasks
        return list_tasks()
    
    def search_patterns(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for similar code patterns."""
        try:
            # Embed the query
            query_vector = self.embedding_engine.embed_text(query)
            
            # Search in vector store
            results = self.vector_store.search(query_vector, limit)
            
            return [
                {
                    "text": result.text,
                    "file_path": result.file_path,
                    "score": result.score,
                    "metadata": result.metadata
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_repository_stats(self) -> Dict:
        """Get statistics about extracted repositories."""
        try:
            # Get all vectors and group by repository
            all_vectors = self.vector_store.get_all_vectors()
            
            repo_stats = {}
            for vector in all_vectors:
                metadata = vector.get("metadata", {})
                repo_url = metadata.get("repo_url", "unknown")
                
                if repo_url not in repo_stats:
                    repo_stats[repo_url] = {
                        "chunks": 0,
                        "files": set(),
                        "latest_commit": metadata.get("repo_info", {}).get("latest_commit", "unknown")
                    }
                
                repo_stats[repo_url]["chunks"] += 1
                repo_stats[repo_url]["files"].add(vector.get("file_path", ""))
            
            # Convert sets to counts
            for repo in repo_stats:
                repo_stats[repo]["files"] = len(repo_stats[repo]["files"])
            
            return repo_stats
        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            return {} 