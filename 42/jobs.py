"""Job management system for 42 using Celery."""

import os
import time
from typing import Dict, Any, Optional
from celery import Celery
from loguru import logger

from .config import load_config


# Initialize Celery
config = load_config()

celery_app = Celery(
    '42',
    broker=f'redis://{config.redis_host}:{config.redis_port}/0',
    backend=f'redis://{config.redis_host}:{config.redis_port}/0',
    include=['42.jobs']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


@celery_app.task(bind=True)
def extract_github_repository(self, repo_url: str, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Extract and analyze a GitHub repository."""
    try:
        from .github import GitHubExtractor
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing extraction...'}
        )
        
        # Initialize extractor
        extractor = GitHubExtractor(max_workers=max_workers)
        
        # Progress callback
        def progress_callback(message: str, progress_val: float):
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': int(progress_val * 100),
                    'total': 100,
                    'status': message
                }
            )
        
        # Perform extraction
        result = extractor.analyze_repository(repo_url, progress_callback=progress_callback)
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Extraction completed',
                'result': result
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"GitHub extraction failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


@celery_app.task(bind=True)
def recluster_vectors(self, min_cluster_size: int = 5, min_samples: int = 3) -> Dict[str, Any]:
    """Recluster all vectors using HDBSCAN."""
    try:
        from .cluster import ClusteringEngine
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing clustering...'}
        )
        
        # Initialize clustering engine
        clustering_engine = ClusteringEngine()
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 25, 'total': 100, 'status': 'Performing clustering...'}
        )
        
        # Perform reclustering
        clusters = clustering_engine.recluster_vectors(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        # Get statistics
        stats = clustering_engine.get_cluster_stats()
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Clustering completed',
                'result': {
                    'clusters': len(clusters),
                    'stats': stats
                }
            }
        )
        
        return {
            'clusters': len(clusters),
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


@celery_app.task(bind=True)
def import_data(self, path: str) -> Dict[str, Any]:
    """Import data from file or directory."""
    try:
        from .chunker import Chunker
        from .embedding import EmbeddingEngine
        from .vector_store import VectorStore
        from qdrant_client.models import PointStruct
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing import...'}
        )
        
        # Initialize components
        chunker = Chunker()
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStore()
        
        # Ensure collection exists
        vector_store.create_collection(embedding_engine.get_dimension())
        
        # Chunk the files
        from pathlib import Path
        if Path(path).is_file():
            chunks = chunker.chunk_file(path)
        else:
            chunks = chunker.chunk_directory(path)
        
        if not chunks:
            return {
                'status': 'error',
                'error': f'No chunks found in {path}',
                'chunks': 0
            }
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 25, 'total': 100, 'status': f'Processing {len(chunks)} chunks...'}
        )
        
        # Embed and store chunks
        total_chunks = len(chunks)
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
            
            # Update progress
            progress = int(25 + (i / total_chunks) * 75)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress,
                    'total': 100,
                    'status': f'Processed {i+1}/{total_chunks} chunks...'
                }
            )
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Import completed',
                'result': {
                    'status': 'success',
                    'chunks': total_chunks,
                    'path': path
                }
            }
        )
        
        return {
            'status': 'success',
            'chunks': total_chunks,
            'path': path
        }
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task."""
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            return {
                'task_id': task_id,
                'state': 'pending',
                'status': 'Task is waiting for execution'
            }
        elif result.state == 'PROGRESS':
            return {
                'task_id': task_id,
                'state': 'progress',
                'status': result.info.get('status', 'Processing...'),
                'current': result.info.get('current', 0),
                'total': result.info.get('total', 100)
            }
        elif result.state == 'SUCCESS':
            return {
                'task_id': task_id,
                'state': 'success',
                'status': 'Task completed successfully',
                'result': result.info.get('result', result.result)
            }
        elif result.state == 'FAILURE':
            return {
                'task_id': task_id,
                'state': 'failure',
                'status': 'Task failed',
                'error': result.info.get('error', str(result.result))
            }
        else:
            return {
                'task_id': task_id,
                'state': result.state,
                'status': 'Unknown state'
            }
            
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        return {
            'task_id': task_id,
            'state': 'error',
            'status': f'Failed to get status: {str(e)}'
        }


def list_tasks() -> Dict[str, Any]:
    """List recent tasks."""
    try:
        # This is a simplified implementation
        # In production, you'd want to store task metadata in Redis
        return {
            'active_tasks': 0,  # Would need to implement task tracking
            'recent_tasks': [],
            'status': 'Task listing not fully implemented'
        }
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        return {
            'error': str(e)
        } 