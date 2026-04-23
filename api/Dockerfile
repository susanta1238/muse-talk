# MuseTalk API — single GPU inference server
# Build context must be the MuseTalk project root (parent of api/):
#   docker build -f api/Dockerfile -t musetalk-api .
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        ffmpeg git git-lfs wget ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt /app/requirements.txt
COPY api/requirements-api.txt /app/api/requirements-api.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -r /app/api/requirements-api.txt \
    && pip install -U openmim \
    && mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# Copy the rest
COPY . /app

ENV STORAGE_DIR=/app/api/storage \
    MODELS_DIR=/app/models \
    USE_FLOAT16=1 \
    GPU_ID=0

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
