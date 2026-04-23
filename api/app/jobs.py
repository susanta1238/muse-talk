import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional

from .config import DB_PATH

_lock = threading.Lock()
_STATUSES = ("queued", "processing", "completed", "failed")


def _conn():
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    return c


@contextmanager
def _tx():
    with _lock:
        c = _conn()
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init_db():
    with _tx() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                stage TEXT,
                error TEXT,
                avatar_path TEXT,
                audio_path TEXT,
                output_path TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )


def create_job(avatar_path: str, audio_path: str) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with _tx() as c:
        c.execute(
            "INSERT INTO jobs (id, status, progress, avatar_path, audio_path, created_at, updated_at) "
            "VALUES (?, 'queued', 0, ?, ?, ?, ?)",
            (job_id, avatar_path, audio_path, now, now),
        )
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _tx() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def update_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    output_path: Optional[str] = None,
):
    fields, values = [], []
    if status is not None:
        assert status in _STATUSES
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if output_path is not None:
        fields.append("output_path = ?")
        values.append(output_path)
    if not fields:
        return
    fields.append("updated_at = ?")
    values.append(time.time())
    values.append(job_id)
    with _tx() as c:
        c.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def list_queued() -> list:
    with _tx() as c:
        rows = c.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def reset_stuck_on_startup():
    """Mark any job stuck in 'processing' (from a previous crash) as failed."""
    with _tx() as c:
        c.execute(
            "UPDATE jobs SET status = 'failed', error = 'interrupted by server restart', updated_at = ? "
            "WHERE status = 'processing'",
            (time.time(),),
        )
