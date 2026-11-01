#!/usr/bin/env python3
"""
Word-level timing and confidence for the Deepgram-compatible /v1/listen route.

Uses the engine's real word timestamps / probabilities whenever the final
segment has the same number of words (a corrected word keeps its timing); if
post-processing changed the word count (number normalization) it spreads the words across the segment by length and
uses the segment's average log-probability as confidence. Pure functions.
"""

import math
from typing import Any, Dict, List

_NEUTRAL_CONFIDENCE = 0.9


def _segment_confidence(seg: Dict[str, Any]) -> float:
    lp = seg.get("avg_logprob")
    return round(math.exp(min(0.0, lp)), 3) if isinstance(lp, (int, float)) else _NEUTRAL_CONFIDENCE


def _from_engine_words(tokens: List[str], engine: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"word": t, "start": round(float(e["start"]), 3), "end": round(float(e["end"]), 3),
             "confidence": round(float(e.get("probability", _NEUTRAL_CONFIDENCE)), 3)}
            for t, e in zip(tokens, engine)]


def _proportional(tokens: List[str], seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    total = sum(len(t) + 1 for t in tokens)
    conf, out, cursor = _segment_confidence(seg), [], start
    for t in tokens:
        nxt = cursor + (end - start) * (len(t) + 1) / total
        out.append({"word": t, "start": round(cursor, 3), "end": round(nxt, 3), "confidence": conf})
        cursor = nxt
    if out:
        out[-1]["end"] = round(end, 3)
    return out


def build_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return Deepgram-style word dicts (word/start/end/confidence) for all segments."""
    words: List[Dict[str, Any]] = []
    for seg in segments:
        tokens = (seg.get("text") or "").split()
        if not tokens:
            continue
        engine = seg.get("words") or []
        if engine and len(engine) == len(tokens):  # 1:1 even if a word was corrected
            words.extend(_from_engine_words(tokens, engine))
        else:
            words.extend(_proportional(tokens, seg))
    return words


def overall_confidence(words: List[Dict[str, Any]]) -> float:
    """Mean word confidence (0.0 when there are no words)."""
    return round(sum(w["confidence"] for w in words) / len(words), 3) if words else 0.0
