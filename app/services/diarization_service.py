#!/usr/bin/env python3
"""
Speaker diarization ("who spoke when") with pyannote.audio.

The model is loaded lazily on first use, so deployments that do not enable
diarization never import torch/pyannote. Weights are downloaded at image build
time (the model is gated on Hugging Face; see modal_app.py) and loaded offline.
"""

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger("anot.diarization")

Turn = Tuple[float, float, str]


class DiarizationError(Exception):
    """Diarization failed; callers should fall back to an unlabelled transcript."""


class Diarizer:
    """Thread-safe wrapper around a pyannote speaker-diarization pipeline."""

    _instance: Optional["Diarizer"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "Diarizer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load(self):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationError("pyannote.audio is not installed in this deployment.") from e
        token = settings.HF_TOKEN or None
        try:
            pipeline = Pipeline.from_pretrained(settings.DIARIZE_MODEL, token=token)
        except TypeError:  # older pyannote releases name the argument differently
            pipeline = Pipeline.from_pretrained(settings.DIARIZE_MODEL, use_auth_token=token)
        if pipeline is None:
            raise DiarizationError("The diarization model could not be loaded (check access to the gated model).")
        if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        logger.info("diarization model loaded: %s", settings.DIARIZE_MODEL)
        return pipeline

    def diarize(self, waveform: np.ndarray, sample_rate: int = 16000) -> List[Turn]:
        """Return (start, end, raw_speaker_id) turns for a mono float32 waveform."""
        import torch

        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._load()
            kwargs = {"max_speakers": settings.DIARIZE_MAX_SPEAKERS} if settings.DIARIZE_MAX_SPEAKERS > 0 else {}
            tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
            try:
                result = self._pipeline({"waveform": tensor, "sample_rate": sample_rate}, **kwargs)
            except Exception as e:  # never let diarization break transcription
                raise DiarizationError("Speaker diarization failed.") from e
        annotation = getattr(result, "speaker_diarization", result)  # pyannote 4.x wraps the annotation
        return [(float(turn.start), float(turn.end), str(speaker))
                for turn, _track, speaker in annotation.itertracks(yield_label=True)]
