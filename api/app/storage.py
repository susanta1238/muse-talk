import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from .config import UPLOADS_DIR, VIDEOS_DIR, MAX_UPLOAD_MB

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def _safe_ext(filename: str, allowed: set[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(
            f"Unsupported file extension {ext!r}. Allowed: {sorted(allowed)}"
        )
    return ext


def _stream_to_file(src: BinaryIO, dst: Path, max_bytes: int):
    written = 0
    with open(dst, "wb") as f:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise ValueError(
                    f"Upload exceeds max size of {max_bytes // (1024 * 1024)} MB"
                )
            f.write(chunk)


def save_upload(file: UploadFile, kind: str) -> Path:
    if kind == "video":
        allowed = ALLOWED_VIDEO_EXT
    elif kind == "audio":
        allowed = ALLOWED_AUDIO_EXT
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    ext = _safe_ext(file.filename or "", allowed)
    job_key = uuid.uuid4().hex
    sub = UPLOADS_DIR / job_key
    sub.mkdir(parents=True, exist_ok=True)
    dst = sub / f"{kind}{ext}"
    _stream_to_file(file.file, dst, MAX_UPLOAD_MB * 1024 * 1024)
    return dst


def output_video_path(job_id: str) -> Path:
    return VIDEOS_DIR / f"{job_id}.mp4"


def cleanup_upload(path: Path):
    """Remove the per-upload subdir."""
    try:
        parent = path.parent
        if parent.exists() and parent.parent.name == "uploads":
            shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        pass
