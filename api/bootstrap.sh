#!/usr/bin/env bash
# One-shot bootstrap for running the MuseTalk API on a fresh RunPod pod
# (template: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04).
#
# Usage:
#   cd /workspace/muse-talk
#   bash api/bootstrap.sh
#
# Handles:
#   1. System deps (ffmpeg, git-lfs)
#   2. Python deps (keeping pod's preinstalled torch)
#   3. Prebuilt mmcv wheel (avoids source build that fails on C++17)
#   4. huggingface_hub pin (transformers 4.39.2 requires <1.0)
#   5. Model weights download via hf CLI (replaces broken download_weights.sh)
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
echo ">>> [1/6] Installing system deps (ffmpeg, git-lfs, espeak-ng)"
apt-get update -qq
# espeak-ng is required by kokoro for phoneme-level TTS.
apt-get install -y -qq ffmpeg git-lfs espeak-ng >/dev/null || true

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
pip install --upgrade pip setuptools wheel --quiet

# chumpy (pulled in by mmpose) is an old package that breaks under
# modern pip's build isolation ("No module named 'pip'"). Install it
# first with --no-build-isolation so its sdist can see the parent pip.
pip install --quiet --no-build-isolation "chumpy>=0.70" || \
    pip install --quiet --no-build-isolation "numpy<1.23" "chumpy>=0.70"

# Install the project's requirements but DO NOT let pip rewrite torch.
TMP_REQS=$(mktemp)
grep -Ev '^(torch|torchvision|torchaudio)([=<>]|$)' requirements.txt > "$TMP_REQS"
pip install -r "$TMP_REQS" --quiet
rm -f "$TMP_REQS"

# API-specific deps
pip install -r api/requirements-api.txt --quiet

# Pin huggingface_hub — MUST come AFTER all installs that might upgrade it.
pip install --quiet --force-reinstall --no-deps "huggingface_hub==0.24.7"

# ---------------------------------------------------------------- mmcv stack
echo ">>> [4/6] Installing mm{cv,det,pose,engine} prebuilt wheels"
pip install mmengine --quiet
pip install "mmcv==2.1.0" -f "$MMCV_INDEX" --quiet || {
    echo "    Prebuilt mmcv wheel not found, falling back to openmim..."
    pip install -U openmim --quiet
    mim install "mmcv==2.1.0"
}
# mmdet 3.2.0 is the first release that accepts mmcv 2.1.0.
# mmpose 1.2.0 is the matching version.
pip install "mmdet==3.2.0" "mmpose==1.2.0" --quiet

# Re-pin hub in case mmdet/mmpose dragged in a newer version.
pip install --quiet --force-reinstall --no-deps "huggingface_hub==0.24.7"

# ---------------------------------------------------------------- weights
echo ">>> [5/6] Downloading model weights (~5 GB)"

WEIGHTS_OK=1
check_weights() {
    WEIGHTS_OK=1
    for f in \
        "models/musetalkV15/unet.pth" \
        "models/musetalkV15/musetalk.json" \
        "models/sd-vae/config.json" \
        "models/sd-vae/diffusion_pytorch_model.bin" \
        "models/whisper/config.json" \
        "models/whisper/pytorch_model.bin" \
        "models/whisper/preprocessor_config.json" \
        "models/dwpose/dw-ll_ucoco_384.pth" \
        "models/face-parse-bisent/79999_iter.pth" \
        "models/face-parse-bisent/resnet18-5c106cde.pth"
    do
        if [[ ! -f "$f" ]]; then
            WEIGHTS_OK=0
            echo "    missing: $f"
        fi
    done
}

check_weights
if [[ $WEIGHTS_OK -eq 1 ]]; then
    echo "    All weights already present, skipping download."
else
    mkdir -p models/musetalk models/musetalkV15 models/sd-vae models/whisper \
             models/dwpose models/face-parse-bisent models/syncnet

    # Download via huggingface_hub Python API — no CLI dependency.
    python - <<'PY'
from huggingface_hub import snapshot_download

jobs = [
    ("TMElyralab/MuseTalk",                   "models",                   None),
    ("stabilityai/sd-vae-ft-mse",             "models/sd-vae",
        ["config.json", "diffusion_pytorch_model.bin"]),
    ("openai/whisper-tiny",                   "models/whisper",
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ("yzd-v/DWPose",                          "models/dwpose",
        ["dw-ll_ucoco_384.pth"]),
    ("ByteDance/LatentSync",                  "models/syncnet",
        ["latentsync_syncnet.pt"]),
    ("ManyOtherFunctions/face-parse-bisent",  "models/face-parse-bisent",
        ["79999_iter.pth", "resnet18-5c106cde.pth"]),
]

for repo, dst, patterns in jobs:
    print(f"    [hf] {repo} -> {dst}")
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=dst,
            allow_patterns=patterns,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"        WARN: {repo} failed: {e}")
PY

    # Fallback for resnet18 face-parse which sometimes isn't on HF.
    if [[ ! -f "models/face-parse-bisent/resnet18-5c106cde.pth" ]]; then
        echo "    [curl] resnet18-5c106cde.pth"
        curl -L -o models/face-parse-bisent/resnet18-5c106cde.pth \
            https://download.pytorch.org/models/resnet18-5c106cde.pth
    fi

    check_weights
    if [[ $WEIGHTS_OK -ne 1 ]]; then
        echo "    ERROR: some weights are still missing after download."
        exit 1
    fi
fi

# ---------------------------------------------------------------- verify
echo ">>> [6/6] Verifying imports"
python - <<'PY'
import importlib, sys
for mod in ("torch", "torchvision", "transformers", "diffusers",
            "huggingface_hub", "mmcv", "mmdet", "mmpose", "mmengine",
            "fastapi", "uvicorn", "kokoro", "soundfile"):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"    ok  {mod:20s} {v}")
    except Exception as e:
        print(f"    FAIL {mod}: {e}")
        sys.exit(1)
import huggingface_hub as hh
assert hh.__version__.startswith("0."), f"huggingface_hub must be <1.0, got {hh.__version__}"
print("    hub version check passed")
PY

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
