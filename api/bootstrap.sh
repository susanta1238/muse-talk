#!/usr/bin/env bash
# One-shot bootstrap for running the MuseTalk API on a fresh RunPod pod
# (template: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04).
#
# Usage:
#   cd /workspace/muse-talk
#   bash api/bootstrap.sh
#
# Handles:
#   1. System deps (ffmpeg)
#   2. Python deps (respecting pod's preinstalled torch)
#   3. Prebuilt mmcv wheel (avoids source build that fails on C++17)
#   4. huggingface_hub pin (avoids transformers incompat)
#   5. Model weights download (~5 GB, one-time, skipped if already present)
#
# Idempotent: safe to re-run.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================================="
echo " MuseTalk API bootstrap"
echo " Project root: $PROJECT_ROOT"
echo "=================================================================="

# ---------------------------------------------------------------- system
echo ">>> [1/6] Installing system deps (ffmpeg, git-lfs)"
apt-get update -qq
apt-get install -y -qq ffmpeg git-lfs >/dev/null

# ---------------------------------------------------------------- torch
echo ">>> [2/6] Checking preinstalled PyTorch"
TORCH_INFO=$(python -c "import torch; print(torch.__version__, torch.version.cuda or 'cpu')" 2>/dev/null || echo "missing")
echo "    torch: $TORCH_INFO"

TORCH_VERSION=$(echo "$TORCH_INFO" | awk '{print $1}' | cut -d+ -f1)
TORCH_CUDA=$(echo "$TORCH_INFO" | awk '{print $2}')

if [[ "$TORCH_VERSION" == "missing" ]]; then
    echo "    ERROR: torch is not installed on this pod."
    echo "    Use the RunPod PyTorch 2.1 template (CUDA 11.8)."
    exit 1
fi

# Pick the mmcv index URL that matches the installed torch + cuda.
TORCH_MAJOR_MINOR=$(echo "$TORCH_VERSION" | awk -F. '{print $1"."$2}')
CUDA_TAG="cu118"
case "$TORCH_CUDA" in
    11.8|11.8.*) CUDA_TAG="cu118" ;;
    12.1|12.1.*) CUDA_TAG="cu121" ;;
    12.4|12.4.*) CUDA_TAG="cu124" ;;
    cpu)         CUDA_TAG="cpu" ;;
    *)           echo "    WARNING: unknown cuda $TORCH_CUDA, assuming cu118"; CUDA_TAG="cu118" ;;
esac

MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_MAJOR_MINOR}/index.html"
echo "    mmcv wheel index: $MMCV_INDEX"

# ---------------------------------------------------------------- python deps
echo ">>> [3/6] Installing Python deps"
pip install --upgrade pip --quiet

# Install the project's requirements but DO NOT let pip rewrite torch.
# Strip torch/torchvision/torchaudio lines to avoid pulling a conflicting build.
TMP_REQS=$(mktemp)
grep -Ev '^(torch|torchvision|torchaudio)([=<>]|$)' requirements.txt > "$TMP_REQS"
pip install -r "$TMP_REQS" --quiet
rm -f "$TMP_REQS"

# Pin huggingface_hub to a version compatible with transformers==4.39.2
pip install "huggingface_hub>=0.19.3,<1.0" --quiet

# API-specific deps
pip install -r api/requirements-api.txt --quiet

# ---------------------------------------------------------------- mmcv stack
echo ">>> [4/6] Installing mm{cv,det,pose,engine} prebuilt wheels"
pip install -U openmim --quiet

# mmcv 2.1.0 has prebuilt wheels for torch 2.1 / cu118; 2.0.1 does not.
pip install mmengine --quiet
pip install "mmcv>=2.0.1,<2.2" -f "$MMCV_INDEX" --quiet || {
    echo "    Prebuilt mmcv wheel not found, falling back to mim install..."
    mim install "mmcv>=2.0.1,<2.2"
}
pip install "mmdet==3.1.0" "mmpose==1.1.0" --quiet

# ---------------------------------------------------------------- weights
echo ">>> [5/6] Downloading model weights (~5 GB, skipped if present)"
if [[ -f "models/musetalkV15/unet.pth" && -f "models/whisper/config.json" ]]; then
    echo "    Weights already present, skipping."
else
    # Use the repo's download script; it's fine with the pinned hf hub.
    if [[ -x "download_weights.sh" ]]; then
        bash download_weights.sh || true
    else
        chmod +x download_weights.sh
        bash download_weights.sh || true
    fi

    # Verify the critical files actually landed.
    MISSING=0
    for f in \
        "models/musetalkV15/unet.pth" \
        "models/musetalkV15/musetalk.json" \
        "models/sd-vae/config.json" \
        "models/whisper/config.json" \
        "models/dwpose/dw-ll_ucoco_384.pth" \
        "models/face-parse-bisent/79999_iter.pth" \
        "models/face-parse-bisent/resnet18-5c106cde.pth"
    do
        if [[ ! -f "$f" ]]; then
            echo "    MISSING: $f"
            MISSING=1
        fi
    done
    if [[ $MISSING -eq 1 ]]; then
        echo "    ERROR: some weights failed to download. Re-run the script."
        exit 1
    fi
fi

# ---------------------------------------------------------------- done
echo ">>> [6/6] Done."
echo ""
echo "=================================================================="
echo " Bootstrap complete."
echo ""
echo " To launch the API:"
echo "     bash api/run.sh"
echo ""
echo " Or in the background:"
echo "     nohup bash api/run.sh > /workspace/musetalk-api.log 2>&1 &"
echo ""
echo " API will listen on http://0.0.0.0:8000"
echo " Swagger UI:  http://<pod>:8000/docs"
echo "=================================================================="
