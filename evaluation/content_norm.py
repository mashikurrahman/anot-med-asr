#!/usr/bin/env python3
"""
Content-accuracy text normalization.

Human references here are strict verbatim (every "uh", "um", "gonna" is written
out) while ASR output is deliberately cleaner. Scoring raw WER therefore
penalises style, not content. This module canonicalises BOTH sides identically
so the WER measures how much spoken content was captured:

  - non-lexical fillers / backchannels are dropped,
  - colloquial forms are expanded ("gonna" -> "going to"),
  - contractions are expanded, and possessives are written as the spoken plural
    form ("patient's" == "patients"), so tokenisation does not count them wrong.

Nothing clinical is altered: drug names, numbers and units pass through
unchanged.
"""

import re

# Pure non-lexical fillers + conversational backchannels with no clinical content.
FILLERS = frozenset({
    "uh", "um", "umm", "uhh", "mm", "mmm", "hmm", "hmmm", "mhm", "mmhm",
    "huh", "erm", "er", "ah", "eh", "oh", "ooh", "ha", "yeah", "yep", "yup",
    "okay", "ok", "mkay", "right", "uhhuh", "uhuh", "hm",
})

COLLOQUIAL = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of", "cause": "because",
    "cuz": "because", "coz": "because", "lemme": "let me",
    "gimme": "give me", "dunno": "do not know", "ya": "you",
}

# Words whose "'s" is the contraction of "is" rather than a possessive.
_IS_HOSTS = (
    "he|she|it|that|there|here|what|who|where|how|when|why|everything|nothing|"
    "something|today|this"
)

_CONTRACTIONS = (
    (re.compile(r"\blet's\b"), "let us"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(rf"\b({_IS_HOSTS})'s\b"), r"\1 is"),
)
_POSSESSIVE = re.compile(r"(\w)'s\b")
_NON_WORD = re.compile(r"[^\w\s]")


def clean_for_content_wer(text: str) -> str:
    """Return `text` lower-cased, contraction-expanded and filler-free."""
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = _POSSESSIVE.sub(r"\1s", text)
    text = _NON_WORD.sub(" ", text)
    words = []
    for w in text.split():
        if w in COLLOQUIAL:
            words.extend(COLLOQUIAL[w].split())
        elif w not in FILLERS:
            words.append(w)
    return " ".join(words)
