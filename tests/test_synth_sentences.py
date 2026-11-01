#!/usr/bin/env python3
"""Tests for training/synth_sentences.py (medical-term sentence generator)."""

import unittest

from training.synth_sentences import build_term_bank, generate_sentences


class TestTermBank(unittest.TestCase):
    def test_includes_lexicon_and_extra_terms(self):
        bank = build_term_bank()
        meds = {t.lower() for t in bank["medications"]}
        for expected in ("albuterol", "augmentin", "prednisone", "solu-medrol", "xopenex"):
            self.assertIn(expected, meds)

    def test_no_empty_or_duplicate_terms_per_category(self):
        for cat, terms in build_term_bank().items():
            self.assertTrue(all(t.strip() for t in terms), cat)
            self.assertEqual(len(terms), len({t.lower() for t in terms}), cat)


class TestSentences(unittest.TestCase):
    def setUp(self):
        self.bank = build_term_bank()
        self.sents = generate_sentences(seed=7)

    def test_every_term_is_covered(self):
        text = " ".join(self.sents).lower()
        for cat, terms in self.bank.items():
            for t in terms:
                self.assertIn(t.lower(), text, f"{cat}: {t} missing")

    def test_deterministic_for_a_seed_and_varies_across_seeds(self):
        self.assertEqual(self.sents, generate_sentences(seed=7))
        self.assertNotEqual(self.sents, generate_sentences(seed=8))

    def test_sentence_shape(self):
        for s in self.sents:
            self.assertTrue(s[0].isupper(), s)
            self.assertIn(s[-1], ".?", s)
            self.assertLess(len(s.split()), 40, s)

    def test_no_duplicate_sentences(self):
        self.assertEqual(len(self.sents), len(set(self.sents)))

    def test_reasonable_volume(self):
        self.assertGreater(len(self.sents), 1200)
        self.assertLess(len(self.sents), 4000)

    def test_doses_use_digits_like_the_references(self):
        joined = " ".join(self.sents)
        self.assertRegex(joined, r"\b\d+ milligrams\b")


if __name__ == "__main__":
    unittest.main()
