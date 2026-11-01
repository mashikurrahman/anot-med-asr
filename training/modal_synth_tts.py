#!/usr/bin/env python3
"""
Generate synthetic medical-term speech (Piper TTS, CPU) into the private Modal
volume "asr-ft". No GPU, no patient data. Many US / UK / non-native-accent
voices, realistic noise / bandwidth / echo augmentation, 16 kHz FLAC clips plus
/data/manifest_synth.jsonl ({"path", "text", "voice"}).

    .venv/Scripts/modal.exe run training/modal_synth_tts.py --limit 24   # smoke test
    .venv/Scripts/modal.exe run training/modal_synth_tts.py              # full run
"""
import json
import random

import modal

from training.synth_sentences import generate_sentences

app = modal.App("asr-synth-tts")
vol = modal.Volume.from_name("asr-ft", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("piper-tts==1.8.0", "numpy<2", "scipy", "soundfile")
    .add_local_python_source("audio")
    .add_local_python_source("training")
    .add_local_python_source("language_model")
)

SINGLE = [
    "en_US-lessac-medium", "en_US-ryan-medium", "en_US-amy-medium", "en_US-joe-medium",
    "en_US-kristin-medium", "en_US-hfc_female-medium", "en_US-hfc_male-medium", "en_US-norman-medium",
    "en_US-john-medium", "en_US-bryce-medium", "en_US-kusal-medium", "en_US-mike-medium",
    "en_US-sam-medium", "en_GB-alan-medium", "en_GB-alba-medium", "en_GB-jenny_dioco-medium",
    "en_GB-northern_english_male-medium", "en_GB-cori-medium",
]
# voice -> (num_speakers, how many distinct speakers to use)
MULTI = {
    "en_US-libritts_r-medium": (904, 40), "en_GB-vctk-medium": (109, 40),
    "en_US-l2arctic-medium": (24, 24), "en_US-arctic-medium": (18, 18),
    "en_GB-aru-medium": (12, 12), "en_GB-semaine-medium": (4, 4),
}
VOICE_DIR = "/vol/tts"
OUT_DIR = "/vol/data/synth"
N_SHARDS = 8


def voice_specs(seed: int = 0):
    rng = random.Random(seed)
    specs = [(v, None) for v in SINGLE]
    for v, (n, k) in MULTI.items():
        specs += [(v, sid) for sid in rng.sample(range(n), k)]
    return specs


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def fetch_voices() -> list:
    import pathlib

    from piper.download_voices import download_voice

    pathlib.Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)
    ok = []
    for v in SINGLE + list(MULTI):
        try:
            if not pathlib.Path(f"{VOICE_DIR}/{v}.onnx").exists():
                download_voice(v, pathlib.Path(VOICE_DIR))
            ok.append(v)
        except Exception as e:  # keep going; unavailable voices are skipped
            print(f"voice {v} unavailable: {e!r}", flush=True)
    vol.commit()
    return ok


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=2)
def synth_shard(shard: int, items: list) -> list:
    import math
    import pathlib

    import numpy as np
    import soundfile as sf
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from scipy.signal import resample_poly

    from audio.enhancement import enhance_waveform
    from training.synth_audio import augment

    vol.reload()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    voices, rows = {}, []
    for i, (text, voice_name, sid, seed) in enumerate(items):
        rng = np.random.default_rng(seed)
        try:
            if voice_name not in voices:
                voices[voice_name] = PiperVoice.load(f"{VOICE_DIR}/{voice_name}.onnx")
            n_spk = getattr(voices[voice_name].config, "num_speakers", 1) or 1
            cfg = SynthesisConfig(
                speaker_id=None if sid is None else sid % n_spk, length_scale=float(rng.uniform(0.85, 1.2)),
                noise_scale=float(rng.uniform(0.5, 0.9)), noise_w_scale=float(rng.uniform(0.6, 0.9)))
            chunks = list(voices[voice_name].synthesize(text, syn_config=cfg))
            sr = chunks[0].sample_rate
            wave = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
            g = math.gcd(16000, sr)
            wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
            if not 1.0 <= wave.size / 16000 <= 28.0:
                continue
            wave = enhance_waveform(augment(wave, 16000, rng), 16000)
            rel = f"synth/s{shard:02d}_{i:05d}.flac"
            sf.write(f"/vol/data/{rel}", wave, 16000, format="FLAC")
            rows.append({"path": rel, "text": text, "voice": f"{voice_name}#{sid}"})
        except Exception as e:
            print(f"[skip] shard {shard} item {i}: {e!r}", flush=True)
    vol.commit()
    return rows


@app.function(image=image, volumes={"/vol": vol})
def write_manifest(rows: list) -> int:
    with open("/vol/data/manifest_synth.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    return len(rows)


@app.local_entrypoint()
def main(limit: int = 0):
    sentences = generate_sentences(seed=0)
    if limit:
        sentences = sentences[:limit]
    ok = set(fetch_voices.remote())
    specs = [s for s in voice_specs() if s[0] in ok]
    print(f"{len(ok)} voices, {len(specs)} speaker profiles, {len(sentences)} sentences", flush=True)
    rng = random.Random(1)
    items = [(t, *rng.choice(specs), rng.randrange(2 ** 31)) for t in sentences]
    shards = [(k, items[k::N_SHARDS]) for k in range(N_SHARDS) if items[k::N_SHARDS]]
    rows = [r for part in synth_shard.starmap(shards) for r in part]
    n = write_manifest.remote(rows)
    print(f"wrote {n} clips (of {len(sentences)} sentences)")
