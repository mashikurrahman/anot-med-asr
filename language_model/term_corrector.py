#!/usr/bin/env python3
"""
Conservative medical-term correction for ASR output.

Fixes only mistakes that are close to unambiguous: (1) a curated table of
mishearings that are not real clinic words ("albedo" for albuterol), and
(2) a near-identical misspelling of a long drug name (edit similarity >= 0.88,
same first letter, similar length). Ordinary words are never touched, and
already-correct terms pass through unchanged. Pure function, no I/O.
"""

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

# lower-case mishearing (single word or two-word phrase) -> correct term
ALIASES: Dict[str, str] = {
    "albedo": "albuterol", "albedol": "albuterol", "albuteral": "albuterol", "albuterall": "albuterol",
    "pseudofad": "sudafed", "pseudo fed": "sudafed",
    "ezetromycin": "azithromycin", "azithromicin": "azithromycin",
    "salumedril": "Solu-Medrol", "solumedrol": "Solu-Medrol", "solu medrol": "Solu-Medrol",
    "doxy cycling": "doxycycline", "doxycycling": "doxycycline", "doxicycline": "doxycycline",
    "amoxicilin": "amoxicillin", "amoxycillin": "amoxicillin",
    "benadrill": "Benadryl",
    "kenilov": "Kenalog", "kenalov": "Kenalog", "catalog injection": "Kenalog injection",
    "augmenten": "Augmentin", "augmentan": "Augmentin",
}

# (pattern, replacement): fixes that are only safe in a specific clinical context.
CONTEXT_RULES = (
    (re.compile(r"\bahead(?=\s+normal\b)", re.I), "head"),            # "skin ahead normal"
    (re.compile(r"\bperlin(?=\s+discharge\b)", re.I), "purulent"),    # "eyelid perlin discharge"
    (re.compile(r"\bvital is(?=\s+within\b)", re.I), "vitals"),       # "vital is within normal limit"
)

FUZZY_MIN_RATIO = 0.88
FUZZY_MIN_LEN = 7
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


@lru_cache(maxsize=1)
def _drug_vocab() -> FrozenSet[str]:
    from language_model.medical_lexicon import MedicalLexicon

    words = {re.sub(r"[^a-z]", "", w.lower()) for w in MedicalLexicon().lexicon.get("medications", [])}
    return frozenset(w for w in words if len(w) >= FUZZY_MIN_LEN)


def _fuzzy_drug(word: str) -> Optional[str]:
    if len(word) < FUZZY_MIN_LEN:
        return None
    vocab = _drug_vocab()
    if word in vocab:
        return None
    best: Tuple[float, Optional[str]] = (0.0, None)
    for v in vocab:
        if v[0] != word[0] or abs(len(v) - len(word)) > 2:
            continue
        r = SequenceMatcher(None, word, v).ratio()
        if r > best[0]:
            best = (r, v)
    return best[1] if best[0] >= FUZZY_MIN_RATIO else None


def _match_case(src: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if src[:1].isupper() and repl[:1].islower() else repl


def _starts_sentence(m: "re.Match[str]") -> bool:
    before = m.string[: m.start()].rstrip()
    return not before or before[-1] in ".!?"


def correct_terms(text: str) -> str:
    """Return `text` with high-confidence medical-term mistakes fixed."""
    if not text:
        return text
    for pattern, repl in CONTEXT_RULES:
        text = pattern.sub(
            lambda m, r=repl: _match_case(m.group(), r) if _starts_sentence(m) else r, text)
    spans = [(m.start(), m.end(), m.group()) for m in _WORD.finditer(text)]
    out, last, i = [], 0, 0
    while i < len(spans):
        s, e, w = spans[i]
        repl, consumed = None, 1
        if i + 1 < len(spans):  # two-word phrase, only separated by plain spaces
            s2, e2, w2 = spans[i + 1]
            if text[e:s2].strip() == "" and "\n" not in text[e:s2]:
                repl = ALIASES.get(f"{w.lower()} {w2.lower()}")
                if repl:
                    consumed, e = 2, e2
        if repl is None:
            repl = ALIASES.get(w.lower()) or _fuzzy_drug(w.lower())
        if repl:
            out.append(text[last:s])
            out.append(_match_case(w, repl))
            last = e
        i += consumed
    out.append(text[last:])
    return "".join(out)
