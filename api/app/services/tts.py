"""Kokoro TTS service.

Lazy-loads the Kokoro pipeline on first use, keeps it in memory for
subsequent calls. Produces a 24 kHz WAV which MuseTalk's audio loader
happily re-samples to 16 kHz downstream.

Voice selection:
    gender="male"   -> cfg.TTS_VOICE_MALE   (default: am_michael)
    gender="female" -> cfg.TTS_VOICE_FEMALE (default: af_heart)
    An explicit voice_id in the request overrides gender.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config as cfg

logger = logging.getLogger(__name__)


class KokoroTTS:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipeline is not None:
            return
        from kokoro import KPipeline  # local import so failures surface at call time
        lang_code = cfg.TTS_LANG_CODE  # 'a' = American English, 'b' = British
        # Kokoro's KPipeline lazy-downloads weights on first synth call.
        self._pipeline = KPipeline(lang_code=lang_code)
        logger.info("Kokoro TTS pipeline loaded (lang_code=%s)", lang_code)

    def synthesize(
        self,
        text: str,
        gender: str = "female",
        voice_id: Optional[str] = None,
        out_path: Optional[Path] = None,
        speed: float = 1.0,
    ) -> Path:
        if out_path is None:
            raise ValueError("out_path is required")
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        voice = voice_id
        if not voice:
            if gender.lower() == "male":
                voice = cfg.TTS_VOICE_MALE
            elif gender.lower() == "female":
                voice = cfg.TTS_VOICE_FEMALE
            else:
                raise ValueError(
                    f"gender must be 'male' or 'female' (got {gender!r})"
                )

        with self._lock:
            self._load()
            import soundfile as sf

            generator = self._pipeline(text, voice=voice, speed=speed)
            chunks = []
            for _i, _ps, audio in generator:
                # audio is a torch.Tensor of shape (T,) at 24 kHz
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                chunks.append(np.asarray(audio, dtype=np.float32))

            if not chunks:
                raise RuntimeError("Kokoro produced no audio for the given text")

            audio_full = np.concatenate(chunks, axis=0)
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio_full, 24000, subtype="PCM_16")
            logger.info(
                "TTS: %d chars -> %.2fs audio (voice=%s) -> %s",
                len(text), len(audio_full) / 24000, voice, out_path,
            )
            return out_path


_instance: Optional[KokoroTTS] = None


def get_tts() -> KokoroTTS:
    global _instance
    if _instance is None:
        _instance = KokoroTTS()
    return _instance
