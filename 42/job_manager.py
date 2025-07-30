"""Job manager for background processing in 42."""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from loguru import logger


class JobManager:
    """Manages background jobs with persistence."""
    
    def __init__(self, jobs_file: str = "42_jobs.json"):
        """Initialize the job manager."""
        self.jobs_file = Path(jobs_file)
        self._jobs = {}
        self._lock = threading.Lock()
        self._load_jobs()
    
    def _load_jobs(self) -> None:
        """Load jobs from file."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r') as f:
                    self._jobs = json.load(f)
                logger.info(f"Loaded {len(self._jobs)} jobs from {self.jobs_file}")
            except Exception as e:
                logger.warning(f"Failed to load jobs: {e}")
                self._jobs = {}
    
    def _save_jobs(self) -> None:
        """Save jobs to file."""
        try:
            with open(self.jobs_file, 'w') as f:
                json.dump(self._jobs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")
    
    def create_job(self, job_type: str, **kwargs) -> str:
        """Create a new job."""
        job_id = f"{job_type}_{int(time.time())}_{hash(str(kwargs)) % 10000}"
        
        with self._lock:
            self._jobs[job_id] = {
                "type": job_type,
                "status": "created",
                "created_at": time.time(),
                "kwargs": kwargs
            }
            self._save_jobs()
        
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id
    
    def update_job(self, job_id: str, **updates) -> None:
        """Update job status and data."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)
                self._jobs[job_id]["updated_at"] = time.time()
                self._save_jobs()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self, job_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List all jobs, optionally filtered by type."""
        if job_type:
            return {jid: job for jid, job in self._jobs.items() 
                   if job.get("type") == job_type}
        return self._jobs.copy()
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Remove completed jobs older than specified hours."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0
        
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if (job.get("status") in ["completed", "failed"] and 
                    job.get("completed_at", 0) < cutoff_time):
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
                removed_count += 1
            
            if removed_count > 0:
                self._save_jobs()
                logger.info(f"Cleaned up {removed_count} old jobs")
        
        return removed_count 