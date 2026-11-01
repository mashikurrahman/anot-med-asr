#!/usr/bin/env python3
"""Tests for language_model/language_spans.py (English / non-English span planning)."""

import unittest

from language_model.language_spans import build_spans, english_flags, format_marker


class TestEnglishFlags(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(english_flags([0.9, 0.9, 0.05, 0.05], threshold=0.4), [True, True, False, False])

    def test_lone_non_english_window_inside_english_is_absorbed(self):
        self.assertEqual(english_flags([0.9, 0.9, 0.1, 0.9, 0.9]), [True] * 5)

    def test_lone_english_window_inside_non_english_is_absorbed(self):
        self.assertEqual(english_flags([0.1, 0.1, 0.9, 0.1, 0.1]), [False] * 5)

    def test_quiet_windows_inherit_neighbour(self):
        self.assertEqual(english_flags([0.1, None, None, 0.1]), [False] * 4)
        self.assertEqual(english_flags([None, 0.9]), [True, True])

    def test_empty_and_all_unknown(self):
        self.assertEqual(english_flags([]), [])
        self.assertEqual(english_flags([None, None]), [True, True])


class TestBuildSpans(unittest.TestCase):
    def test_merges_runs_and_clips_to_duration(self):
        spans = build_spans([True, True, False, False, True], window_sec=15.0, total_sec=70.0)
        self.assertEqual(spans, [(0.0, 30.0, True), (30.0, 60.0, False), (60.0, 70.0, True)])

    def test_empty(self):
        self.assertEqual(build_spans([], 15.0, 0.0), [])


class TestMarker(unittest.TestCase):
    def test_minutes_and_hours(self):
        self.assertEqual(format_marker(190, 340), "[Non-English speech, 3:10-5:40]")
        self.assertEqual(format_marker(3590, 3725), "[Non-English speech, 59:50-1:02:05]")


if __name__ == "__main__":
    unittest.main()
