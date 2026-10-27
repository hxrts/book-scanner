"""Durable job queue: each capture is written to disk first, then processed.

Jobs that fail (network down, API error) stay in pending/ and are retried, so
nothing scanned is lost if the Pi is offline.
"""

import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)
RETRY_SECONDS = 300


class JobQueue:
    def __init__(self, data_dir, handlers, max_image_bytes):
        self.pending = Path(data_dir) / "pending"
        self.done = Path(data_dir) / "done"
        self.failed = Path(data_dir) / "failed"
        for d in (self.pending, self.done, self.failed):
            d.mkdir(parents=True, exist_ok=True)
        self.handlers = handlers  # kind -> fn(job_dict, job_dir)
        self.max_image_bytes = max_image_bytes
        self.queue = queue.Queue()
        for job_file in sorted(self.pending.glob("*/job.json")):
            self.queue.put(job_file.parent)

    def submit(self, kind, payload, files=None):
        """files: {name: bytes} written alongside the job."""
        job_dir = self.pending / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}"
        job_dir.mkdir()
        for name, data in (files or {}).items():
            (job_dir / name).write_bytes(data)
        (job_dir / "job.json").write_text(json.dumps({"kind": kind, "attempts": 0, **payload}, indent=2))
        self.queue.put(job_dir)
        return job_dir

    def _run(self, job_dir):
        job_file = job_dir / "job.json"
        job = json.loads(job_file.read_text())
        try:
            result = self.handlers[job["kind"]](job, job_dir)
            job_file.write_text(json.dumps({**job, "result": result}, indent=2, default=str))
            job_dir.rename(self.done / job_dir.name)
            log.info("job %s done", job_dir.name)
        except PermanentError as e:
            log.warning("job %s failed permanently: %s", job_dir.name, e)
            job_file.write_text(json.dumps({**job, "error": str(e)}, indent=2))
            job_dir.rename(self.failed / job_dir.name)
        except Exception as e:
            job["attempts"] += 1
            job["last_error"] = str(e)
            job_file.write_text(json.dumps(job, indent=2))
            log.exception("job %s failed (attempt %d), will retry", job_dir.name, job["attempts"])
            threading.Timer(RETRY_SECONDS, self.queue.put, args=(job_dir,)).start()

    def prune(self):
        """Delete images from the oldest finished jobs until under the cap.

        Pending jobs are never touched; job.json and ocr.txt are kept as a record.
        """
        images = sorted(
            (p for d in (self.done, self.failed) for p in d.glob("*/*.jpg")),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in images)
        for p in images:
            if total <= self.max_image_bytes:
                break
            total -= p.stat().st_size
            p.unlink()
            log.info("pruned %s", p.relative_to(p.parents[2]))

    def start(self):
        self.prune()

        def worker():
            while True:
                self._run(self.queue.get())
                self.prune()

        threading.Thread(target=worker, daemon=True, name="jobs").start()


class PermanentError(Exception):
    """A job that will never succeed (e.g. ISBN unknown to Open Library)."""
