import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class BackgroundJobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, video_source: str) -> str:
        """Initializes a new background search job in queued state."""
        job_id = f"JOB_{uuid.uuid4().hex[:8].upper()}"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "video_source": video_source,
                "matches_found": 0,
                "error": None,
                "created_at": datetime.utcnow().isoformat()
            }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Thread-safe retrieval of job status."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return job.copy()
            return None

    def _update_job(self, job_id: str, **kwargs):
        """Internal helper to update job attributes in a thread-safe way."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def start_job(self, job_id: str, pipeline, video_path: Path, camera_id: str,
                  camera_location: str, target_id: Optional[str] = None,
                  threshold: float = 0.40, output_video_path: Optional[Path] = None):
        """Spawns a background thread to process the video search."""
        thread = threading.Thread(
            target=self._run_job_sync,
            args=(job_id, pipeline, video_path, camera_id, camera_location, target_id, threshold, output_video_path),
            daemon=True
        )
        self._update_job(job_id, status="running")
        thread.start()

    def _run_job_sync(self, job_id: str, pipeline, video_path: Path, camera_id: str,
                      camera_location: str, target_id: Optional[str],
                      threshold: float, output_video_path: Optional[Path]):
        """Runs the pipeline search synchronously inside a background thread."""
        def progress_callback(frame_count: int, total_frames: int, matches_found: int):
            progress = (frame_count / total_frames) if total_frames > 0 else 0.0
            self._update_job(job_id, progress=round(progress, 3), matches_found=matches_found)

        try:
            pipeline.search_video(
                video_path=video_path,
                camera_id=camera_id,
                camera_location=camera_location,
                target_id=target_id,
                threshold=threshold,
                output_video_path=output_video_path,
                progress_callback=progress_callback
            )
            # Mark as completed
            self._update_job(job_id, status="completed", progress=1.0)
        except Exception as e:
            print(f"[JOB EXCEPTION] Error in background job {job_id}: {e}")
            self._update_job(job_id, status="failed", error=str(e))
