#!/usr/bin/env python3
"""Tests for language_model/term_corrector.py (conservative medical-term fixes)."""

import unittest

from language_model.term_corrector import correct_terms


class TestTermCorrector(unittest.TestCase):
    def test_known_mishearing_is_fixed(self):
        self.assertEqual(correct_terms("Do you use albedo at home?"), "Do you use albuterol at home?")

    def test_case_and_punctuation_preserved(self):
        self.assertEqual(correct_terms("Albedo, twice a day."), "Albuterol, twice a day.")

    def test_phrase_alias(self):
        self.assertEqual(correct_terms("take doxy cycling for ten days"), "take doxycycline for ten days")

    def test_correct_terms_untouched(self):
        s = "She takes albuterol and doxycycline for her heart."
        self.assertEqual(correct_terms(s), s)

    def test_close_misspelling_of_drug_fuzzy_fixed(self):
        self.assertEqual(correct_terms("started azithromicin today"), "started azithromycin today")

    def test_ordinary_words_never_changed(self):
        s = "Hearing is fine, the bacteria were gone this morning and the steroid helped."
        self.assertEqual(correct_terms(s), s)

    def test_context_rules(self):
        self.assertEqual(correct_terms("General skin ahead normal."), "General skin head normal.")
        self.assertEqual(correct_terms("Eyelid perlin discharge present."), "Eyelid purulent discharge present.")
        self.assertEqual(correct_terms("Exam, vital is within normal limit."), "Exam, vitals within normal limit.")
        self.assertEqual(correct_terms("Eyes, eyelid Perlin discharge."), "Eyes, eyelid purulent discharge.")

    def test_context_rules_leave_normal_uses(self):
        s = "Go ahead and sit. The perlin noise is unrelated, and the vital is in the chart."
        self.assertEqual(correct_terms(s), s)

    def test_empty_and_none_safe(self):
        self.assertEqual(correct_terms(""), "")

    def test_input_not_mutated_and_idempotent(self):
        s = "use albedo"
        once = correct_terms(s)
        self.assertEqual(s, "use albedo")
        self.assertEqual(correct_terms(once), once)


if __name__ == "__main__":
    unittest.main()
