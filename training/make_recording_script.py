#!/usr/bin/env python3
"""
Build a read-aloud script for a REAL-voice medical-term test.

Greedy set cover over sentences that are NOT in the synthetic training set
(seed 0), so the test measures whether the engine learned the *terms*, not the
sentence wording. Output: scratch_recording_script.txt (numbered lines to read,
one per sentence) and scratch_recording_script.json (reference text).
"""
import json
import sys

sys.path.insert(0, ".")

from training.synth_sentences import build_term_bank, generate_sentences  # noqa: E402

N_LINES = 150


def main() -> None:
    train = set(generate_sentences(seed=0))
    pool = [s for s in generate_sentences(seed=99) if s not in train]
    terms = {t.lower() for ts in build_term_bank().values() for t in ts}
    chosen, covered = [], set()
    for _ in range(N_LINES):
        best = max(pool, key=lambda s: len({t for t in terms if t in s.lower()} - covered))
        gain = {t for t in terms if t in best.lower()} - covered
        if not gain and len(chosen) >= 100:
            break
        chosen.append(best)
        covered |= gain
        pool.remove(best)
    with open("scratch_recording_script.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(chosen, 1):
            f.write(f"{i}. {s}\n")
    json.dump(chosen, open("scratch_recording_script.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(chosen)} lines covering {len(covered)} of {len(terms)} terms")


if __name__ == "__main__":
    main()
