#!/usr/bin/env python3
"""WhisperPipeline with speaker labels (fake ASR model, fake diarizer: no models or network)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.pipelines.whisper_pipeline import WhisperPipeline
from app.services.diarization_service import DiarizationError


def _w(word, start, end):
    return {"word": " " + word, "start": start, "end": end, "probability": 0.9}


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Any pain? Um, a little.", "avg_logprob": -0.1, "low_confidence": False,
     "words": [_w("Any", 0.0, 0.5), _w("pain?", 0.5, 1.5), _w("Um,", 2.5, 2.8), _w("a", 2.8, 3.0), _w("little.", 3.0, 4.0)]},
    {"start": 5.0, "end": 8.0, "text": "Take two tablets.", "avg_logprob": -0.1, "low_confidence": False,
     "words": [_w("Take", 5.0, 5.5), _w("two", 5.5, 6.0), _w("tablets.", 6.0, 8.0)]},
]
RAW = "Any pain? Um, a little. Take two tablets."


class FakeModel:
    _last_detected_language = "en"

    def transcribe(self, wave, sample_rate, language=None):
        return RAW, [dict(s) for s in SEGMENTS], 0.1

    def get_info(self):
        return {"model_id": "fake"}


class FakeNorm:
    def normalize_text(self, text):
        return text, [], 0.0


class FakeDiarizer:
    def __init__(self, turns=None, error=None):
        self.turns, self.error = turns, error

    def diarize(self, wave, sample_rate=16000):
        if self.error:
            raise self.error
        return self.turns


def _settings(**over):
    base = dict(AUDIO_ENHANCE=False, DIARIZE=True, TERM_CORRECTION=False, STRIP_FILLERS=True,
                DICTATION_COMMANDS=True, WHISPER_PLAIN=True)
    return SimpleNamespace(**{**base, **over})


def _run(diarizer, mode="conversation", **over):
    with patch("app.pipelines.whisper_pipeline.settings", _settings(**over)):
        return WhisperPipeline(FakeModel(), FakeNorm(), diarizer=diarizer).process(
            np.zeros(16000 * 9, dtype=np.float32), 16000, normalize=True, mode=mode)


TURNS = [(0.0, 2.0, "A"), (2.0, 4.5, "B"), (4.5, 9.0, "A")]


class TestPipelineSpeakers(unittest.TestCase):
    def test_labelled_transcript_with_cleanup(self):
        out = _run(FakeDiarizer(TURNS))
        self.assertEqual(out["transcript"], "Speaker 1: Any pain?\nSpeaker 2: A little.\nSpeaker 1: Take two tablets.")
        self.assertEqual([s["speaker"] for s in out["segments"]], ["Speaker 1", "Speaker 2", "Speaker 1"])
        self.assertEqual(out["raw_transcript"], RAW)

    def test_diarization_failure_keeps_plain_transcript(self):
        out = _run(FakeDiarizer(error=DiarizationError("no model")))
        self.assertEqual(out["transcript"], "Any pain? A little. Take two tablets.")
        self.assertTrue(all("speaker" not in s for s in out["segments"]))

    def test_unexpected_diarizer_crash_is_contained(self):
        out = _run(FakeDiarizer(error=RuntimeError("boom")))
        self.assertIn("Take two tablets.", out["transcript"])

    def test_dictation_mode_skips_diarization(self):
        out = _run(FakeDiarizer(TURNS), mode="dictation")
        self.assertNotIn("Speaker", out["transcript"])

    def test_disabled_by_setting(self):
        out = _run(FakeDiarizer(TURNS), DIARIZE=False)
        self.assertNotIn("Speaker", out["transcript"])

    def test_single_speaker_has_no_prefix(self):
        out = _run(FakeDiarizer([(0.0, 9.0, "A")]))
        self.assertEqual(out["transcript"], "Any pain? A little. Take two tablets.")
        self.assertEqual({s["speaker"] for s in out["segments"]}, {"Speaker 1"})


if __name__ == "__main__":
    unittest.main()
