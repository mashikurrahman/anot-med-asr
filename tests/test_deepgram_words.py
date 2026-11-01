#!/usr/bin/env python3
"""Tests for app/services/deepgram_words.py (word timing/confidence for /v1/listen)."""

import math
import unittest

from app.services.deepgram_words import build_words, overall_confidence


class TestBuildWords(unittest.TestCase):
    def test_uses_engine_word_timings_when_tokens_match(self):
        seg = {"start": 0.0, "end": 2.0, "text": "Take two tablets.",
               "words": [{"word": " Take", "start": 0.1, "end": 0.4, "probability": 0.9},
                         {"word": " two", "start": 0.5, "end": 0.8, "probability": 0.8},
                         {"word": " tablets.", "start": 0.9, "end": 1.6, "probability": 0.7}]}
        w = build_words([seg])
        self.assertEqual([x["word"] for x in w], ["Take", "two", "tablets."])
        self.assertEqual(w[1]["start"], 0.5)
        self.assertEqual(w[2]["confidence"], 0.7)

    def test_corrected_word_keeps_engine_timing(self):
        seg = {"start": 0.0, "end": 2.0, "text": "Use albuterol",
               "words": [{"word": " Use", "start": 0.1, "end": 0.4, "probability": 0.9},
                         {"word": " albedo", "start": 0.5, "end": 1.2, "probability": 0.4}]}
        w = build_words([seg])
        self.assertEqual((w[1]["word"], w[1]["start"], w[1]["confidence"]), ("albuterol", 0.5, 0.4))

    def test_falls_back_to_proportional_timing_when_text_was_edited(self):
        seg = {"start": 10.0, "end": 12.0, "text": "Use albuterol today", "avg_logprob": -0.1,
               "words": [{"word": " Use", "start": 10.0, "end": 10.3, "probability": 0.9},
                         {"word": " albedo", "start": 10.3, "end": 11.0, "probability": 0.4}]}
        w = build_words([seg])
        self.assertEqual(len(w), 3)
        self.assertEqual(w[0]["start"], 10.0)
        self.assertEqual(w[-1]["end"], 12.0)
        self.assertTrue(all(a["end"] <= b["start"] + 1e-9 for a, b in zip(w, w[1:])))
        self.assertAlmostEqual(w[0]["confidence"], round(math.exp(-0.1), 3), places=3)

    def test_no_word_data_and_no_logprob_gives_neutral_confidence(self):
        w = build_words([{"start": 0.0, "end": 1.0, "text": "hello there"}])
        self.assertEqual(len(w), 2)
        self.assertTrue(all(0.0 <= x["confidence"] <= 1.0 for x in w))

    def test_empty_inputs(self):
        self.assertEqual(build_words([]), [])
        self.assertEqual(build_words([{"start": 0, "end": 1, "text": "  "}]), [])
        self.assertEqual(overall_confidence([]), 0.0)

    def test_overall_confidence_is_mean_and_input_not_mutated(self):
        seg = {"start": 0.0, "end": 1.0, "text": "a b", "avg_logprob": -0.5}
        before = dict(seg)
        w = build_words([seg])
        self.assertEqual(seg, before)
        self.assertAlmostEqual(overall_confidence(w), sum(x["confidence"] for x in w) / 2, places=3)


if __name__ == "__main__":
    unittest.main()
