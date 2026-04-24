import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import config as cfg
from .. import jobs, storage, worker
from ..services.tts import get_tts

router = APIRouter(prefix="/api/v1", tags=["generate-from-text"])
logger = logging.getLogger(__name__)


@router.post("/upload-generate-from-text")
async def upload_generate_from_text(
    avatar_video: UploadFile = File(..., description="Avatar video/image"),
    text: str = Form(..., description="Text to speak"),
    gender: str = Form("female", description="male or female (picks a default Kokoro voice)"),
    voice_id: str = Form("", description="Optional explicit Kokoro voice id (overrides gender)"),
    speed: float = Form(1.0, description="Speech rate multiplier (0.5–2.0)"),
):
    """Synthesize audio from text with Kokoro, then run lip-sync generation.

    Returns a job_id immediately. Poll GET /api/v1/job/{job_id}; download
    the resulting video from GET /api/v1/video/{job_id} once completed.
    """
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    if len(text) > cfg.TTS_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds TTS_MAX_CHARS={cfg.TTS_MAX_CHARS}",
        )
    if gender.lower() not in ("male", "female") and not voice_id:
        raise HTTPException(
            status_code=400,
            detail="gender must be 'male' or 'female' (or provide voice_id)",
        )
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=400, detail="speed must be in [0.5, 2.0]")

    # Save the avatar upload first — if it's rejected, don't waste TTS cycles.
    try:
        avatar_path = storage.save_upload(avatar_video, "video")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"avatar_video: {e}")

    # Synthesize audio synchronously — fast (~a few seconds) and lets us
    # surface TTS errors via HTTP instead of making the caller poll.
    audio_path = storage.tts_audio_path()
    try:
        get_tts().synthesize(
            text=text,
            gender=gender,
            voice_id=voice_id or None,
            out_path=audio_path,
            speed=speed,
        )
    except Exception as e:
        logger.exception("TTS synthesis failed")
        storage.cleanup_upload(avatar_path)
        storage.cleanup_upload(audio_path)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    job_id = jobs.create_job(str(avatar_path), str(audio_path))
    worker.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}
