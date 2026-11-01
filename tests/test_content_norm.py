#!/usr/bin/env python3
"""Tests for evaluation/content_norm.py (content-accuracy text normalization)."""

import unittest

from evaluation.content_norm import clean_for_content_wer as clean


class TestContentNorm(unittest.TestCase):
    def test_possessive_matches_spoken_plural(self):
        # Spoken, "patient's" and "patients" are identical; scoring must agree.
        self.assertEqual(clean("the patient's chart"), clean("the patients chart"))

    def test_pronoun_contraction_expands_to_is(self):
        self.assertEqual(clean("it's fine, he's here"), "it is fine he is here")

    def test_lets_expands_to_let_us(self):
        self.assertEqual(clean("let's start"), "let us start")

    def test_negations_and_modals(self):
        self.assertEqual(clean("I don't think I can't"), "i do not think i cannot")
        self.assertEqual(clean("we'll see, they've left, you're ok"), "we will see they have left you are")

    def test_curly_apostrophes(self):
        self.assertEqual(clean("it’s fine"), "it is fine")

    def test_fillers_and_backchannels_removed(self):
        self.assertEqual(clean("Uh, um, yeah okay so the knee"), "so the knee")

    def test_colloquial_forms_canonicalised(self):
        self.assertEqual(clean("I'm gonna wanna rest cause kinda tired"),
                         "i am going to want to rest because kind of tired")

    def test_medical_content_is_preserved(self):
        txt = "Prednisone 20 milligrams twice a day for COVID"
        self.assertEqual(clean(txt), "prednisone 20 milligrams twice a day for covid")

    def test_empty(self):
        self.assertEqual(clean(""), "")


if __name__ == "__main__":
    unittest.main()
