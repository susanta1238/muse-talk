"""Voice Activity Detection for silence gating.

MuseTalk's UNet always produces mouth motion from whatever Whisper emits,
even for silence. We use silero-vad to find silent regions in the driving
audio and, during blending, substitute the original (closed-mouth) avatar
frame for any frame that falls inside a silent window.

Design notes:
- silero-vad runs on CPU; first call lazy-loads the ~20 MB torch hub model.
- The mask is computed once per job, length == number of output frames.
- If silero-vad is unavailable (import failure), we return an all-True mask
  so behavior matches the original pipeline.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_vad_model = None
_get_speech_ts = None


def _load():
    global _vad_model, _get_speech_ts
    if _vad_model is not None:
        return True
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore
        _vad_model = load_silero_vad()
        _get_speech_ts = get_speech_timestamps
        logger.info("silero-vad loaded")
        return True
    except Exception as e:
        logger.warning("silero-vad unavailable, falling back to no gating: %s", e)
        return False


def compute_speech_mask(
    audio_path: str,
    num_frames: int,
    fps: float,
    min_speech_ms: int = 200,
    min_silence_ms: int = 250,
    threshold: float = 0.5,
    padding_frames: int = 2,
) -> np.ndarray:
    """Return a boolean mask of length num_frames. True == speech, False == silence.

    On any failure (model not available, audio decode error), returns an
    all-True mask so the pipeline degrades gracefully to its original behavior.
    """
    mask = np.ones(num_frames, dtype=bool)
    if not _load():
        return mask

    try:
        import torchaudio  # torchaudio is a MuseTalk dep already
        import torch

        waveform, sr = torchaudio.load(audio_path)
        # Mix to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # silero-vad wants 16 kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        waveform = waveform.squeeze(0)  # -> 1D tensor

        speech_windows: List[dict] = _get_speech_ts(
            waveform,
            _vad_model,
            sampling_rate=sr,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            threshold=threshold,
            return_seconds=True,
        )

        if not speech_windows:
            logger.info("VAD found no speech windows; keeping all frames as speech")
            return mask

        # Build silence mask from speech windows, then pad edges.
        mask[:] = False
        for w in speech_windows:
            start_f = int(max(0, np.floor(w["start"] * fps) - padding_frames))
            end_f = int(min(num_frames, np.ceil(w["end"] * fps) + padding_frames))
            if end_f > start_f:
                mask[start_f:end_f] = True

        speech_frames = int(mask.sum())
        logger.info(
            "VAD: %d/%d frames speech (%.1f%%), %d speech windows",
            speech_frames, num_frames,
            100.0 * speech_frames / max(1, num_frames),
            len(speech_windows),
        )
        return mask

    except Exception as e:
        logger.warning("VAD computation failed (%s); using all-True mask", e)
        return np.ones(num_frames, dtype=bool)
