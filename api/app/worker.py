"""Background worker: consumes queued jobs one at a time on the single GPU.

The worker runs the heavy (blocking) MuseTalk pipeline on a **dedicated**
ThreadPoolExecutor with a single thread. This is important: if we used
asyncio.to_thread() (which uses the default loop executor that's shared
with FastAPI's sync endpoint handlers), a long-running job would starve
HTTP request handlers and the /health endpoint would appear unreachable.
"""
import asyncio
import concurrent.futures
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
# Dedicated executor — exactly one worker thread, so GPU jobs are
# serialized AND they never touch the shared asyncio default executor
# that FastAPI uses for sync endpoint handlers.
_executor: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="musetalk-job")
)


def enqueue(job_id: str):
    _queue.put_nowait(job_id)
    logger.info("job %s enqueued (queue size=%d)", job_id, _queue.qsize())


async def _process(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        logger.warning("job %s disappeared before processing", job_id)
        return

    logger.info("job %s START (avatar=%s audio=%s)", job_id,
                job["avatar_path"], job["audio_path"])
    jobs.update_job(job_id, status="processing", progress=1, stage="starting")

    runner = get_runner()
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage.output_video_path(job_id)

    def progress_cb(pct: int, stage: str):
        jobs.update_job(job_id, progress=pct, stage=stage)
        logger.info("job %s progress=%d stage=%s", job_id, pct, stage)

    loop = asyncio.get_running_loop()
    try:
        # Run the blocking pipeline on the dedicated executor.
        await loop.run_in_executor(
            _executor,
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
        logger.info("job %s DONE -> %s", job_id, output_path)
    except Exception as e:
        logger.exception("job %s FAILED", job_id)
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
    logger.info("worker loop started")
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
    try:
        _executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def requeue_pending_on_startup():
    """Re-enqueue any jobs that were 'queued' when the server last shut down."""
    n = 0
    for job in jobs.list_queued():
        _queue.put_nowait(job["id"])
        n += 1
    if n:
        logger.info("re-enqueued %d pending jobs on startup", n)
