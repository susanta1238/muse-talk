from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import jobs, storage, worker

router = APIRouter(prefix="/api/v1", tags=["generate"])


@router.post("/upload-generate-video")
async def upload_generate_video(
    avatar_video: UploadFile = File(..., description="Avatar video/image (mp4/mov/avi/mkv/webm or image)"),
    audio: UploadFile = File(..., description="Driving audio (wav/mp3/m4a/flac/ogg/aac)"),
):
    """Upload avatar media + audio and queue a lip-sync generation job.

    Returns a job_id. Poll GET /api/v1/job/{job_id} for status, then
    GET /api/v1/video/{job_id} once status == 'completed'.
    """
    try:
        avatar_path = storage.save_upload(avatar_video, "video")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"avatar_video: {e}")

    try:
        audio_path = storage.save_upload(audio, "audio")
    except ValueError as e:
        # roll back first upload
        storage.cleanup_upload(avatar_path)
        raise HTTPException(status_code=400, detail=f"audio: {e}")

    job_id = jobs.create_job(str(avatar_path), str(audio_path))
    worker.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}
