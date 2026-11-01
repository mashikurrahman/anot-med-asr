#!/usr/bin/env python3
"""
Reference <-> ASR alignment and clip planning for verbatim fine-tuning.

The human transcripts have coarse timestamps (segments up to ~6 minutes), so
training clips cannot be cut from them directly. Instead the ASR's word
timestamps are used as a time scaffold: reference words that match an ASR word
inherit its time, and clips (<= ~27 s, Whisper's window is 30 s) are cut only at
pauses where both neighbouring words are matched. That makes the reference
boundary at every cut exact rather than interpolated.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Set, Tuple

PAD_SEC = 0.5          # lead + tail padding added around a clip when cutting audio
GOOD_GAP_SEC = 0.25    # pause length that counts as a clean cut point
FALLBACK_GAP_SEC = 0.1

_NON_TOKEN = re.compile(r"[^a-z0-9']")


def tokenize(text: str) -> List[str]:
    """Lower-case, punctuation-stripped tokens (empty tokens dropped)."""
    toks = (_NON_TOKEN.sub("", w) for w in (text or "").lower().replace("\u2019", "'").split())
    return [t for t in toks if t]


def align_tokens(ref: Sequence[str], hyp: Sequence[str]) -> Dict[int, int]:
    """Map ref index -> hyp index for words that match exactly (in order)."""
    matcher = SequenceMatcher(a=list(ref), b=list(hyp), autojunk=False)
    pairs: Dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs[block.a + k] = block.b + k
    return pairs


@dataclass(frozen=True)
class Chunk:
    hyp_start: int   # first ASR word index (inclusive)
    hyp_end: int     # last ASR word index + 1 (exclusive)
    start_sec: float
    end_sec: float


def _duration(words: Sequence[Tuple[str, float, float]], a: int, b: int) -> float:
    return words[b - 1][2] - words[a][1] + PAD_SEC


def plan_chunks(
    words: Sequence[Tuple[str, float, float]],
    matched: Set[int],
    min_sec: float = 8.0,
    max_sec: float = 27.0,
) -> List[Chunk]:
    """
    Greedy left-to-right clip plan over ASR words [(word, start, end), ...].

    Each clip is as long as fits in `max_sec`, ending at the latest "good" cut
    (a pause >= GOOD_GAP_SEC with both neighbouring words matched). Falls back
    to any pause, then to a forced cut, so every word lands in exactly one clip.
    """
    n = len(words)
    chunks: List[Chunk] = []
    a = 0
    while a < n:
        b_max = a + 1
        while b_max < n and _duration(words, a, b_max + 1) <= max_sec:
            b_max += 1
        if b_max >= n:
            b = n
        else:
            b = _pick_cut(words, matched, a, b_max, min_sec)
        chunks.append(Chunk(a, b, words[a][1], words[b - 1][2]))
        a = b
    return chunks


def _pick_cut(words, matched: Set[int], a: int, b_max: int, min_sec: float) -> int:
    """Latest cut index i in (a, b_max], preferring clean matched pauses."""
    good = fallback = None
    for i in range(b_max, a, -1):
        if _duration(words, a, i) < min_sec:
            break
        gap = words[i][1] - words[i - 1][2]
        if gap >= GOOD_GAP_SEC and (i - 1) in matched and i in matched:
            good = i
            break
        if fallback is None and gap >= FALLBACK_GAP_SEC:
            fallback = i
    return good or fallback or b_max
