import os
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = API_ROOT.parent

STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", API_ROOT / "storage"))
UPLOADS_DIR = STORAGE_DIR / "uploads"
VIDEOS_DIR = STORAGE_DIR / "videos"
JOBS_DIR = STORAGE_DIR / "jobs"
DB_PATH = STORAGE_DIR / "jobs.db"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
UNET_MODEL_PATH = str(MODELS_DIR / "musetalkV15" / "unet.pth")
UNET_CONFIG = str(MODELS_DIR / "musetalkV15" / "musetalk.json")
WHISPER_DIR = str(MODELS_DIR / "whisper")
VAE_TYPE = "sd-vae"

USE_FLOAT16 = os.environ.get("USE_FLOAT16", "1") == "1"
GPU_ID = int(os.environ.get("GPU_ID", "0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
FPS = int(os.environ.get("FPS", "25"))
EXTRA_MARGIN = int(os.environ.get("EXTRA_MARGIN", "10"))
PARSING_MODE = os.environ.get("PARSING_MODE", "jaw")
LEFT_CHEEK_WIDTH = int(os.environ.get("LEFT_CHEEK_WIDTH", "90"))
RIGHT_CHEEK_WIDTH = int(os.environ.get("RIGHT_CHEEK_WIDTH", "90"))
AUDIO_PAD_LEFT = int(os.environ.get("AUDIO_PAD_LEFT", "2"))
AUDIO_PAD_RIGHT = int(os.environ.get("AUDIO_PAD_RIGHT", "2"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))

# ---------------------------------------------------------------- TTS (Kokoro)
# Voice catalog: https://huggingface.co/hexgrad/Kokoro-82M
# Prefix key: a=American, b=British. Second letter: f=female, m=male.
TTS_LANG_CODE = os.environ.get("TTS_LANG_CODE", "a")  # 'a' American, 'b' British
TTS_VOICE_MALE = os.environ.get("TTS_VOICE_MALE", "am_michael")
TTS_VOICE_FEMALE = os.environ.get("TTS_VOICE_FEMALE", "af_heart")
TTS_DEFAULT_SPEED = float(os.environ.get("TTS_DEFAULT_SPEED", "1.0"))
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "5000"))

for d in (STORAGE_DIR, UPLOADS_DIR, VIDEOS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)
