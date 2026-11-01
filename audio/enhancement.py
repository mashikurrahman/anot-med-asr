#!/usr/bin/env python3
"""
Anot Medical ASR — Audio Front-End Enhancement (label-free accuracy lever)
==========================================================================
Stage 22: Deterministic, dependency-light audio clean-up applied BEFORE the ASR
engine. Targets the dominant error sources on real patient-visit recordings:

  1. Quiet / distant patient speech  -> loudness (RMS) normalization ("AGC").
  2. Low-frequency room rumble / HVAC -> FFT high-pass.
  3. DC offset from cheap mics        -> mean removal.

Pure NumPy (no scipy/librosa). It changes only the acoustic signal level and
low-frequency content; it never fabricates or removes speech content, so it
cannot alter clinical facts — it only makes existing speech easier to recognise.

All operations are safe no-ops on silence and preserve sample count and dtype.
"""

from typing import Optional

import numpy as np

EPS = 1e-9


def remove_dc(wave: np.ndarray) -> np.ndarray:
    """Remove any DC offset (constant bias) from the waveform."""
    return wave - float(np.mean(wave)) if wave.size else wave


def fft_highpass(wave: np.ndarray, sample_rate: int, cutoff_hz: float = 70.0) -> np.ndarray:
    """
    Attenuate content below `cutoff_hz` using a smooth frequency-domain mask.
    Vectorised (one FFT over the whole signal) — fine for offline batch audio.
    """
    n = wave.size
    if n == 0 or cutoff_hz <= 0:
        return wave
    # Zero-pad to a power of two: NumPy's FFT is ~20x slower on awkward lengths,
    # which made a 30-minute recording spend ~35 s in this filter alone.
    n_fft = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(wave, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    # Smooth raised-cosine transition over [cutoff/2, cutoff] to avoid ringing.
    lo, hi = cutoff_hz * 0.5, cutoff_hz
    mask = np.ones_like(freqs)
    below = freqs < lo
    trans = (freqs >= lo) & (freqs < hi)
    mask[below] = 0.0
    mask[trans] = 0.5 * (1 - np.cos(np.pi * (freqs[trans] - lo) / (hi - lo)))
    filtered = np.fft.irfft(spectrum * mask, n=n_fft)[:n]
    return filtered.astype(wave.dtype, copy=False)


def normalize_loudness(
    wave: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain: float = 20.0,
    peak_ceiling: float = 0.97,
) -> np.ndarray:
    """
    Bring the signal to a target RMS loudness (simple AGC), capped so we don't
    over-amplify near-silence, with a peak guard to prevent clipping.
    """
    if wave.size == 0:
        return wave
    rms = float(np.sqrt(np.mean(np.square(wave))))
    if rms < EPS:
        return wave  # silence: nothing to normalize
    target_lin = 10.0 ** (target_dbfs / 20.0)
    gain = min(target_lin / rms, 10.0 ** (max_gain / 20.0))
    out = wave * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(wave.dtype, copy=False)


def gain_ride(
    wave: np.ndarray,
    sample_rate: int = 16000,
    target_db: float = -26.0,
    max_boost_db: float = 25.0,
    noise_floor_db: float = -62.0,
    frame_sec: float = 0.5,
    window_sec: float = 20.0,
    step_sec: float = 2.0,
) -> np.ndarray:
    """
    Windowed AGC ("gain riding") for recordings whose level drifts (phone moved
    away from the speakers, patient turns away). A single global gain is set by
    the loudest minutes and leaves the distant stretches below the ASR's VAD and
    decoder range, which showed up as >80% word loss in the quiet half of long
    visits.

    The local speech level is the 90th-percentile frame loudness in a sliding
    window; each region is boosted (never attenuated) toward `target_db`, capped
    at `max_boost_db`. Regions whose local level is under `noise_floor_db` get no
    gain, so silence and room noise are not amplified. Gains are linearly
    interpolated between window centres to avoid audible steps.
    """
    if wave is None or wave.size == 0:
        return wave
    x = np.asarray(wave, dtype=np.float32)
    frame = max(1, int(frame_sec * sample_rate))
    n_frames = x.size // frame
    if n_frames < 2:
        return x.copy()

    frames = x[: n_frames * frame].reshape(n_frames, frame)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + EPS)

    half = max(1, int(window_sec / frame_sec / 2))
    step = max(1, int(step_sec / frame_sec))
    centres = np.arange(0, n_frames, step)
    if centres[-1] != n_frames - 1:
        centres = np.append(centres, n_frames - 1)
    gains_db = np.zeros(centres.size, dtype=np.float64)
    for k, c in enumerate(centres):
        level = float(np.percentile(frame_db[max(0, c - half): c + half + 1], 90))
        if level > noise_floor_db:
            gains_db[k] = float(np.clip(target_db - level, 0.0, max_boost_db))

    sample_pos = np.arange(x.size, dtype=np.float64)
    centre_pos = (centres + 0.5) * frame
    gain = 10.0 ** (np.interp(sample_pos, centre_pos, gains_db) / 20.0)
    out = np.clip(x * gain.astype(np.float32), -1.0, 1.0)
    return np.nan_to_num(out).astype(np.float32, copy=False)


def enhance_waveform(
    wave: np.ndarray,
    sample_rate: int = 16000,
    highpass: bool = True,
    highpass_cutoff_hz: float = 70.0,
    loudness_norm: bool = True,
    target_dbfs: float = -20.0,
    denoise: bool = False,
    gain_riding: bool = False,
) -> np.ndarray:
    """
    Apply the front-end chain. Returns a new array (input is never mutated).

    Order: DC removal -> high-pass -> optional denoise -> optional gain riding
    -> loudness normalization.
    `denoise` uses the `noisereduce` library if installed; otherwise it is a
    graceful no-op (the chain still helps via high-pass + AGC).
    """
    if wave is None or wave.size == 0:
        return wave
    out = np.asarray(wave, dtype=np.float32).copy()
    out = remove_dc(out)
    if highpass:
        out = fft_highpass(out, sample_rate, highpass_cutoff_hz)
    if denoise:
        out = _maybe_denoise(out, sample_rate)
    if gain_riding:
        out = gain_ride(out, sample_rate)
    if loudness_norm:
        out = normalize_loudness(out, target_dbfs=target_dbfs)
    # Final safety: clip to valid range and keep finite.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _maybe_denoise(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral-gate denoise via `noisereduce` if available; else no-op."""
    try:
        import noisereduce as nr  # optional dependency
    except Exception:
        return wave
    try:
        return nr.reduce_noise(y=wave, sr=sample_rate, stationary=False).astype(np.float32)
    except Exception:
        return wave
