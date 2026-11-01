#!/usr/bin/env python3
"""
Product-transcript cleanup for dictation-style recordings.

- strip_fillers: removes "uh" / "um" style hesitations (and their commas).
- apply_dictation_commands: turns spoken punctuation commands into real
  punctuation ("period" -> ".", "comma" -> ",", "enter" -> new line).

Both are conservative (a bare "period" or "enter" in running text is left
alone) and are pure, idempotent string functions. The raw ASR text is kept
separately by the pipeline.
"""

import re

_FILLER = r"(?:uh+|um+|uhm+|erm|hmm+)"
_SENT_START_FILLER = re.compile(rf"(^|[.!?]\s+|\n)\s*{_FILLER}\b[,.]?\s*([a-z]?)", re.I)
_COMMA_FILLER_COMMA = re.compile(rf",\s*{_FILLER}\b\s*,", re.I)
_BARE_FILLER = re.compile(rf"\s*,?\s*(?<![\w-]){_FILLER}\b[,.]?(?!\w)", re.I)


def strip_fillers(text: str) -> str:
    """Remove hesitation fillers, fixing commas, spacing and sentence capitalization."""
    if not text:
        return text
    out = _COMMA_FILLER_COMMA.sub(",", text)

    def sentence_start(m: "re.Match[str]") -> str:
        return m.group(1) + m.group(2).upper()  # the next word now starts the sentence

    prev = None
    while prev != out:  # handles "Uh, um, ..." runs
        prev = out
        out = _SENT_START_FILLER.sub(sentence_start, out)
    out = _BARE_FILLER.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip() if out != text else out


_PERIOD_ENTER = re.compile(r"([.!?])?[\s,]*\bperiod\b[.,]?\s*\benter\b[.,]?", re.I)
_ENTER_AFTER_SENTENCE = re.compile(r"([.!?])\s+enter\b[.,]?", re.I)
_COMMA_PERIOD = re.compile(r"(\w),\s*period\b[.,]?")
_STANDALONE_PERIOD = re.compile(r"([.!?])\s+Period\.(?=\s|$)")
_SPOKEN_COMMA = re.compile(r",\s*comma\b,?", re.I)


def apply_dictation_commands(text: str) -> str:
    """Convert spoken "period"/"comma"/"enter" dictation commands to punctuation."""
    if not text:
        return text
    out = _PERIOD_ENTER.sub(lambda m: (m.group(1) or ".") + "\n", text)
    out = _ENTER_AFTER_SENTENCE.sub(lambda m: m.group(1) + "\n", out)
    out = _COMMA_PERIOD.sub(lambda m: m.group(1) + ".", out)
    out = _STANDALONE_PERIOD.sub(lambda m: m.group(1), out)
    out = _SPOKEN_COMMA.sub(",", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    return out
