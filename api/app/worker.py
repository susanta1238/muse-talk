"""Background worker: consumes queued jobs one at a time on the single GPU."""
import asyncio
import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional

from . import jobs
from . import storage
from .config import JOBS_DIR
from .musetalk_runner import get_runner

logger = logging.getLogger(__name__)


_queue: asyncio.Queue = asyncio.Queue()
_task: Optional[asyncio.Task] = None


def enqueue(job_id: str):
    _queue.put_nowait(job_id)


async def _process(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        logger.warning("job %s disappeared", job_id)
        return

    jobs.update_job(job_id, status="processing", progress=1, stage="starting")
    runner = get_runner()
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage.output_video_path(job_id)

    def progress_cb(pct: int, stage: str):
        jobs.update_job(job_id, progress=pct, stage=stage)

    try:
        await asyncio.to_thread(
            runner.generate,
            job["avatar_path"],
            job["audio_path"],
            str(output_path),
            str(work_dir),
            progress_cb,
        )
        jobs.update_job(
            job_id, status="completed", progress=100, stage="done",
            output_path=str(output_path),
        )
    except Exception as e:
        logger.exception("job %s failed", job_id)
        jobs.update_job(
            job_id, status="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}",
        )
    finally:
        # Clean up uploads and work dir
        try:
            if job.get("avatar_path"):
                storage.cleanup_upload(Path(job["avatar_path"]))
            if job.get("audio_path"):
                storage.cleanup_upload(Path(job["audio_path"]))
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            logger.exception("cleanup failed for job %s", job_id)


async def _loop():
    while True:
        job_id = await _queue.get()
        try:
            await _process(job_id)
        except Exception:
            logger.exception("worker loop error on job %s", job_id)
        finally:
            _queue.task_done()


def start():
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(), name="musetalk-worker")


async def stop():
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None


def requeue_pending_on_startup():
    """Re-enqueue any jobs that were 'queued' when the server last shut down."""
    for job in jobs.list_queued():
        _queue.put_nowait(job["id"])
