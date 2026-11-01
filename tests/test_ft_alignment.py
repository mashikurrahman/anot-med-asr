#!/usr/bin/env python3
"""Tests for training/align_chunks.py (reference<->ASR alignment and clip planning)."""

import unittest

from training.align_chunks import align_tokens, plan_chunks, tokenize


def _hyp(words, step=0.4, gaps=None):
    """Build [(word, start, end)] with optional extra pause (seconds) after word i."""
    gaps = gaps or {}
    t, out = 0.0, []
    for i, w in enumerate(words):
        out.append((w, round(t, 2), round(t + 0.3, 2)))
        t += step + gaps.get(i, 0.0)
    return out


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Hello, World! It's 5."), ["hello", "world", "it's", "5"])

    def test_drops_empty_tokens(self):
        self.assertEqual(tokenize("-- ... ok"), ["ok"])


class TestAlign(unittest.TestCase):
    def test_exact_match_pairs_every_word(self):
        self.assertEqual(align_tokens(["a", "b", "c"], ["a", "b", "c"]), {0: 0, 1: 1, 2: 2})

    def test_deleted_hyp_word_is_unmatched_in_ref(self):
        pairs = align_tokens(["a", "uh", "b", "c"], ["a", "b", "c"])
        self.assertEqual(pairs, {0: 0, 2: 1, 3: 2})

    def test_substitution_is_not_a_match(self):
        pairs = align_tokens(["a", "x", "c"], ["a", "y", "c"])
        self.assertEqual(pairs, {0: 0, 2: 2})


class TestPlanChunks(unittest.TestCase):
    def test_short_audio_is_one_chunk(self):
        hyp = _hyp(["a"] * 10)
        chunks = plan_chunks(hyp, set(range(10)), min_sec=2, max_sec=27)
        self.assertEqual([(c.hyp_start, c.hyp_end) for c in chunks], [(0, 10)])

    def test_cuts_at_pause_between_matched_words_within_max(self):
        n = 100
        hyp = _hyp(["w"] * n, gaps={29: 1.0, 59: 1.0, 89: 1.0})  # pauses after 29, 59, 89
        chunks = plan_chunks(hyp, set(range(n)), min_sec=5, max_sec=27)
        self.assertTrue(all(c.end_sec - c.start_sec <= 27.5 for c in chunks))
        self.assertIn(chunks[0].hyp_end, (30, 60))  # cut lands on a pause
        self.assertEqual(chunks[0].hyp_end, chunks[1].hyp_start)

    def test_never_cuts_next_to_unmatched_word_when_a_matched_pause_exists(self):
        n = 100
        hyp = _hyp(["w"] * n, gaps={29: 1.0, 40: 1.0})
        matched = set(range(n)) - {30}  # word right after the first pause is unmatched
        chunks = plan_chunks(hyp, matched, min_sec=5, max_sec=27)
        self.assertNotEqual(chunks[0].hyp_end, 30)

    def test_chunks_cover_all_words_without_overlap(self):
        n = 200
        hyp = _hyp(["w"] * n, gaps={i: 0.5 for i in range(9, n, 10)})
        chunks = plan_chunks(hyp, set(range(n)), min_sec=5, max_sec=27)
        self.assertEqual(chunks[0].hyp_start, 0)
        self.assertEqual(chunks[-1].hyp_end, n)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.hyp_end, b.hyp_start)

    def test_empty_input(self):
        self.assertEqual(plan_chunks([], set()), [])


if __name__ == "__main__":
    unittest.main()
