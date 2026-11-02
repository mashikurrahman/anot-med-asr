#!/usr/bin/env python3
"""
Combine ASR segments with speaker turns from a diarization model.

`turns` are (start_sec, end_sec, raw_speaker_id). Raw ids are renamed to
"Speaker 1", "Speaker 2", ... in order of first appearance. Segments that carry
word timings are split where the speaker changes; other segments take the
speaker with the largest overlap. Pure functions: inputs are never modified.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Turn = Tuple[float, float, str]
Segment = Dict[str, Any]


def label_map(turns: Sequence[Turn]) -> Dict[str, str]:
    """Raw speaker id -> "Speaker N", numbered by when each speaker first talks."""
    mapping: Dict[str, str] = {}
    for _start, _end, raw in sorted(turns, key=lambda t: t[0]):
        if raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
    return mapping


def _speaker_at(time: float, turns: Sequence[Turn]) -> str:
    """Speaker whose turn contains `time`, else the one whose turn is nearest."""
    best, best_dist = turns[0][2], float("inf")
    for start, end, raw in turns:
        dist = 0.0 if start <= time <= end else min(abs(time - start), abs(time - end))
        if dist < best_dist:
            best, best_dist = raw, dist
    return best


def _majority_speaker(start: float, end: float, turns: Sequence[Turn]) -> str:
    overlap: Dict[str, float] = {}
    for t_start, t_end, raw in turns:
        shared = min(end, t_end) - max(start, t_start)
        if shared > 0:
            overlap[raw] = overlap.get(raw, 0.0) + shared
    if overlap:
        return max(overlap, key=overlap.get)
    return _speaker_at((start + end) / 2.0, turns)


def _split_by_words(seg: Segment, turns: Sequence[Turn], names: Dict[str, str]) -> List[Segment]:
    runs: List[Tuple[str, List[Dict[str, Any]]]] = []
    for word in seg["words"]:
        raw = _speaker_at((float(word["start"]) + float(word["end"])) / 2.0, turns)
        if runs and runs[-1][0] == raw:
            runs[-1][1].append(word)
        else:
            runs.append((raw, [word]))
    if len(runs) == 1:
        return [{**seg, "speaker": names[runs[0][0]]}]
    pieces: List[Segment] = []
    for raw, words in runs:
        pieces.append({**seg, "speaker": names[raw], "words": words,
                       "text": "".join(w["word"] for w in words).strip(),
                       "start": round(float(words[0]["start"]), 2), "end": round(float(words[-1]["end"]), 2)})
    return pieces


def assign_speakers(segments: Sequence[Segment], turns: Sequence[Turn]) -> List[Segment]:
    """Return new segments carrying a "speaker" field (marker segments stay unlabelled)."""
    if not turns:
        return [dict(s) for s in segments]
    names = label_map(turns)
    out: List[Segment] = []
    for seg in segments:
        if seg.get("non_english"):
            out.append(dict(seg))
        elif seg.get("words"):
            out.extend(_split_by_words(seg, turns, names))
        else:
            raw = _majority_speaker(float(seg["start"]), float(seg["end"]), turns)
            out.append({**seg, "speaker": names[raw]})
    return out


def render_transcript(segments: Sequence[Segment], transform: Optional[Callable[[str], str]] = None) -> str:
    """"Speaker 1: ..." lines (consecutive turns merged). No prefix when only one speaker talks.

    `transform` (optional) is applied to each merged turn's text, so cleanup rules see whole turns."""
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    prefix = len(speakers) > 1
    lines: List[str] = []
    current: Optional[str] = None
    texts: List[str] = []

    def flush() -> None:
        nonlocal texts
        if texts:
            body = " ".join(texts)
            body = transform(body) if transform else body
            lines.append(f"{current}: {body}" if prefix and current else body)
        texts = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if seg.get("non_english") or not speaker:  # markers / unlabelled text stand alone
            flush()
            current = None
            lines.append(text)
            continue
        if speaker != current:
            flush()
            current = speaker
        texts.append(text)
    flush()
    return "\n".join(lines)
