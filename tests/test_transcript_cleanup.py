#!/usr/bin/env python3
"""Tests for language_model/transcript_cleanup.py (fillers and spoken-punctuation commands)."""

import unittest

from language_model.transcript_cleanup import apply_dictation_commands, strip_fillers


class TestStripFillers(unittest.TestCase):
    def test_comma_wrapped_filler(self):
        self.assertEqual(strip_fillers("Date of service, December 26, um, 2021."),
                         "Date of service, December 26, 2021.")

    def test_sentence_initial_filler_recapitalizes(self):
        self.assertEqual(strip_fillers("Exam normal. Uh, general skin normal."),
                         "Exam normal. General skin normal.")
        self.assertEqual(strip_fillers("Uh, respiratory normal."), "Respiratory normal.")

    def test_bare_filler_and_spacing(self):
        self.assertEqual(strip_fillers("in the next uh two days"), "in the next two days")

    def test_real_words_untouched(self):
        s = "Come back for the ER visit; umbrella and uhura are words, and ER means emergency room."
        self.assertEqual(strip_fillers(s), s)

    def test_empty_and_idempotent(self):
        self.assertEqual(strip_fillers(""), "")
        once = strip_fillers("a, uh, b. Um, c")
        self.assertEqual(strip_fillers(once), once)


class TestDictationCommands(unittest.TestCase):
    def test_period_enter_after_sentence(self):
        self.assertEqual(apply_dictation_commands("negative in office. Period. Enter. I recommended rest."),
                         "negative in office.\nI recommended rest.")

    def test_period_enter_after_word(self):
        self.assertEqual(apply_dictation_commands("next two to three Period. Enter. She verbalized."),
                         "next two to three.\nShe verbalized.")

    def test_spoken_period_after_comma(self):
        self.assertEqual(apply_dictation_commands("and nausea, period. Exam normal."),
                         "and nausea. Exam normal.")

    def test_spoken_comma(self):
        self.assertEqual(apply_dictation_commands("congestion, comma, sore throat, comma, headache."),
                         "congestion, sore throat, headache.")

    def test_enter_after_sentence(self):
        self.assertEqual(apply_dictation_commands("management. Enter. I provided precautions."),
                         "management.\nI provided precautions.")

    def test_real_uses_of_period_untouched(self):
        s = "Her last period was in May. The post-operative period is two weeks."
        self.assertEqual(apply_dictation_commands(s), s)

    def test_empty_and_idempotent(self):
        self.assertEqual(apply_dictation_commands(""), "")
        once = apply_dictation_commands("a. Period. Enter. b, comma, c")
        self.assertEqual(apply_dictation_commands(once), once)


if __name__ == "__main__":
    unittest.main()
