import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import jobs, worker
from .musetalk_runner import get_runner
from .routers.generate import router as generate_router
from .routers.jobs_router import router as jobs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("musetalk-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    jobs.init_db()
    jobs.reset_stuck_on_startup()

    log.info("loading MuseTalk models...")
    get_runner().load()
    log.info("models loaded")

    worker.start()
    worker.requeue_pending_on_startup()
    log.info("worker started")

    yield

    # Shutdown
    await worker.stop()


app = FastAPI(
    title="MuseTalk API",
    version="1.0",
    description="Lip-sync video generation API over MuseTalk (single GPU).",
    lifespan=lifespan,
)

if os.environ.get("CORS_ALLOW", "*"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.environ.get("CORS_ALLOW", "*")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(generate_router)
app.include_router(jobs_router)
