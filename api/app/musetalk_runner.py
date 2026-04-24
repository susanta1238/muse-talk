"""Thin wrapper around MuseTalk inference pipeline.

Models are loaded once at process start and reused across jobs.
A lock serializes GPU access so we never run two inferences concurrently.
"""
import copy
import glob
import os
import pickle
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

# Make the MuseTalk package importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_image  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.preprocessing import (  # noqa: E402
    coord_placeholder,
    get_landmark_and_bbox,
    read_imgs,
)
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model  # noqa: E402

from . import config as cfg  # noqa: E402
from . import vad as vad_mod  # noqa: E402

_GPU_LOCK = threading.Lock()


class MuseTalkRunner:
    def __init__(self):
        self.device: Optional[torch.device] = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.fp: Optional[FaceParsing] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.weight_dtype = torch.float32
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.device = torch.device(
            f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu"
        )
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=cfg.UNET_MODEL_PATH,
            vae_type=cfg.VAE_TYPE,
            unet_config=cfg.UNET_CONFIG,
            device=self.device,
        )
        if cfg.USE_FLOAT16 and self.device.type == "cuda":
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.timesteps = torch.tensor([0], device=self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path=cfg.WHISPER_DIR)
        self.whisper = (
            WhisperModel.from_pretrained(cfg.WHISPER_DIR)
            .to(device=self.device, dtype=self.weight_dtype)
            .eval()
        )
        self.whisper.requires_grad_(False)

        self.fp = FaceParsing(
            left_cheek_width=cfg.LEFT_CHEEK_WIDTH,
            right_cheek_width=cfg.RIGHT_CHEEK_WIDTH,
        )
        self._loaded = True

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
        work_dir: str,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """Run one end-to-end inference. Returns the output mp4 path."""
        if not self._loaded:
            self.load()

        def _p(pct: int, stage: str):
            if progress_cb:
                try:
                    progress_cb(pct, stage)
                except Exception:
                    pass

        work = Path(work_dir)
        frames_dir = work / "frames"
        out_imgs_dir = work / "out_imgs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_imgs_dir.mkdir(parents=True, exist_ok=True)

        with _GPU_LOCK:
            _p(2, "extracting_frames")
            file_type = get_file_type(avatar_path)
            if file_type == "video":
                # Extract frames with ffmpeg
                cmd = [
                    "ffmpeg", "-v", "fatal", "-y", "-i", avatar_path,
                    "-start_number", "0",
                    str(frames_dir / "%08d.png"),
                ]
                subprocess.run(cmd, check=True)
                input_img_list = sorted(glob.glob(str(frames_dir / "*.png")))
                fps = get_video_fps(avatar_path) or cfg.FPS
            elif file_type == "image":
                # Duplicate single image into a short clip so cycling works
                dst = frames_dir / "00000000.png"
                shutil.copyfile(avatar_path, dst)
                input_img_list = [str(dst)]
                fps = cfg.FPS
            else:
                raise ValueError(f"Unsupported avatar file type: {avatar_path}")

            if not input_img_list:
                raise RuntimeError("No frames extracted from avatar")

            _p(10, "extracting_landmarks")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)

            _p(30, "encoding_latents")
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + cfg.EXTRA_MARGIN, frame.shape[0])
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(crop)
                input_latent_list.append(latents)

            if not input_latent_list:
                raise RuntimeError(
                    "No face detected in the avatar video/image. "
                    "Please provide media with a clearly visible face."
                )

            # Ping-pong cycle so audio longer than video loops seamlessly
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            _p(40, "encoding_audio")
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                audio_path
            )
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=cfg.AUDIO_PAD_LEFT,
                audio_padding_length_right=cfg.AUDIO_PAD_RIGHT,
            )

            _p(48, "vad")
            # Build a speech mask so silent frames keep the original mouth
            # instead of whatever Whisper hallucinates for silence.
            if cfg.VAD_ENABLED:
                speech_mask = vad_mod.compute_speech_mask(
                    audio_path=audio_path,
                    num_frames=len(whisper_chunks),
                    fps=float(fps),
                    threshold=cfg.VAD_THRESHOLD,
                    min_speech_ms=cfg.VAD_MIN_SPEECH_MS,
                    min_silence_ms=cfg.VAD_MIN_SILENCE_MS,
                    padding_frames=cfg.VAD_PADDING_FRAMES,
                )
            else:
                import numpy as _np
                speech_mask = _np.ones(len(whisper_chunks), dtype=bool)

            _p(50, "running_unet")
            video_num = len(whisper_chunks)
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=cfg.BATCH_SIZE,
                delay_frame=0,
                device=self.device,
            )
            res_frame_list = []
            total_batches = int(np.ceil(float(video_num) / cfg.BATCH_SIZE))
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                pred_latents = self.unet.model(
                    latent_batch, self.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                recon = self.vae.decode_latents(pred_latents)
                for f in recon:
                    res_frame_list.append(f)
                if total_batches:
                    _p(50 + int(30 * (i + 1) / total_batches), "running_unet")

            _p(80, "blending_frames")
            for i, res_frame in enumerate(res_frame_list):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])

                # Silence gate: during non-speech frames, keep the original
                # avatar frame so lips don't move when no one is talking.
                if i < len(speech_mask) and not speech_mask[i]:
                    cv2.imwrite(str(out_imgs_dir / f"{i:08d}.png"), ori_frame)
                    continue

                x1, y1, x2, y2 = bbox
                y2 = min(y2 + cfg.EXTRA_MARGIN, ori_frame.shape[0])
                try:
                    res_frame = cv2.resize(
                        res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                    )
                except Exception:
                    continue
                combined = get_image(
                    ori_frame, res_frame, [x1, y1, x2, y2],
                    mode=cfg.PARSING_MODE, fp=self.fp,
                )
                cv2.imwrite(str(out_imgs_dir / f"{i:08d}.png"), combined)

            _p(92, "muxing_video")
            temp_vid = work / "temp_silent.mp4"
            cmd1 = [
                "ffmpeg", "-y", "-v", "warning", "-r", str(int(fps)),
                "-f", "image2",
                "-i", str(out_imgs_dir / "%08d.png"),
                "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
                str(temp_vid),
            ]
            subprocess.run(cmd1, check=True)

            cmd2 = [
                "ffmpeg", "-y", "-v", "warning",
                "-i", audio_path,
                "-i", str(temp_vid),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(output_path),
            ]
            subprocess.run(cmd2, check=True)

            try:
                temp_vid.unlink(missing_ok=True)
                shutil.rmtree(out_imgs_dir, ignore_errors=True)
                shutil.rmtree(frames_dir, ignore_errors=True)
            except Exception:
                pass

            _p(100, "done")

        return output_path


# Singleton
_runner: Optional[MuseTalkRunner] = None


def get_runner() -> MuseTalkRunner:
    global _runner
    if _runner is None:
        _runner = MuseTalkRunner()
    return _runner
