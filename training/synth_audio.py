#!/usr/bin/env python3
"""
Augmentation that makes clean text-to-speech audio resemble real clinic
recordings (background noise, phone/handheld-mic bandwidth, room echo, level
differences). Pure NumPy, deterministic for a given RNG, never mutates inputs.
"""

import numpy as np


def add_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add broadband noise so the speech-to-noise ratio is `snr_db`."""
    power = float(np.mean(np.square(wave))) + 1e-12
    noise = rng.standard_normal(wave.size).astype(np.float32)
    noise *= np.sqrt(power / (10.0 ** (snr_db / 10.0)) / (float(np.mean(np.square(noise))) + 1e-12))
    return (wave + noise).astype(np.float32)


def bandlimit(wave: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Low-pass with a smooth FFT mask (simulates phone / distant-mic bandwidth)."""
    n_fft = 1 << (wave.size - 1).bit_length()
    spec = np.fft.rfft(wave, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mask = np.clip((cutoff_hz * 1.15 - freqs) / (cutoff_hz * 0.15), 0.0, 1.0)
    return np.fft.irfft(spec * mask, n=n_fft)[: wave.size].astype(np.float32)


def reverb(wave: np.ndarray, sample_rate: int, decay_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying impulse response (room echo)."""
    n_ir = max(2, int(decay_sec * sample_rate * 2))
    t = np.arange(n_ir) / sample_rate
    ir = rng.standard_normal(n_ir) * np.exp(-6.9 * t / max(decay_sec, 1e-3))
    ir[0] = 1.0
    ir = (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)
    n_fft = 1 << (wave.size + n_ir - 2).bit_length()
    out = np.fft.irfft(np.fft.rfft(wave, n_fft) * np.fft.rfft(ir, n_fft), n_fft)[: wave.size]
    return out.astype(np.float32)


def augment(wave: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Random chain of level change, echo, bandwidth limit and noise; returns a new array."""
    out = np.asarray(wave, dtype=np.float32).copy()
    out *= 10.0 ** (rng.uniform(-8.0, 4.0) / 20.0)
    if rng.random() < 0.35:
        out = reverb(out, sample_rate, decay_sec=float(rng.uniform(0.05, 0.3)), rng=rng)
    if rng.random() < 0.4:
        out = bandlimit(out, sample_rate, cutoff_hz=float(rng.uniform(3400, 5500)))
    if rng.random() < 0.7:
        out = add_noise(out, snr_db=float(rng.uniform(8.0, 30.0)), rng=rng)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out *= 0.99 / peak
    return np.nan_to_num(out).astype(np.float32)
