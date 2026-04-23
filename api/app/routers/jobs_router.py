from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .. import jobs as job_store

router = APIRouter(prefix="/api/v1", tags=["jobs"])


def _public_video_url(request: Request, job_id: str) -> str:
    return str(request.url_for("get_video", job_id=job_id))


@router.get("/job/{job_id}")
async def get_job_status(job_id: str, request: Request):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    resp = {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job["status"] == "completed" and job["output_path"]:
        resp["video_url"] = _public_video_url(request, job["id"])
    return resp


@router.get("/video/{job_id}", name="get_video")
async def get_video(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"video not ready (status={job['status']})",
        )
    path = Path(job["output_path"] or "")
    if not path.exists():
        raise HTTPException(status_code=410, detail="output file missing")
    return FileResponse(
        path, media_type="video/mp4", filename=f"{job_id}.mp4"
    )
