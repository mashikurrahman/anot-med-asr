#!/usr/bin/env python3
"""
Tests for the label-free audio front-end (audio/enhancement.py).

Focus: the enhancement is safe (no NaNs/clipping blowups, preserves length,
no-op on silence) and actually does its job (quiet speech amplified, DC/rumble
reduced) without fabricating content.
"""

import unittest

import numpy as np

from audio.enhancement import (
    enhance_waveform,
    fft_highpass,
    gain_ride,
    normalize_loudness,
    remove_dc,
)

SR = 16000


def _tone(freq, dur=1.0, amp=0.1, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestAudioEnhancement(unittest.TestCase):
    def test_silence_is_noop_and_safe(self):
        silence = np.zeros(SR, dtype=np.float32)
        out = enhance_waveform(silence, SR)
        self.assertEqual(out.shape, silence.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.0, places=6)

    def test_length_and_range_preserved(self):
        sig = _tone(300, amp=0.05) + 0.01 * np.random.randn(SR).astype(np.float32)
        out = enhance_waveform(sig, SR)
        self.assertEqual(out.shape, sig.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_quiet_speech_is_amplified(self):
        quiet = _tone(300, amp=0.01)  # very quiet
        out = normalize_loudness(quiet, target_dbfs=-20.0)
        self.assertGreater(float(np.sqrt(np.mean(out**2))),
                           float(np.sqrt(np.mean(quiet**2))))

    def test_highpass_attenuates_low_frequency(self):
        low = _tone(30, amp=0.3)   # rumble below cutoff
        high = _tone(1000, amp=0.3)
        mix = low + high
        out = fft_highpass(mix, SR, cutoff_hz=70.0)
        # Low-frequency energy should drop substantially.
        self.assertLess(float(np.sqrt(np.mean(out**2))), float(np.sqrt(np.mean(mix**2))))

    def test_remove_dc(self):
        sig = _tone(300, amp=0.1) + 0.2  # add DC offset
        out = remove_dc(sig)
        self.assertAlmostEqual(float(np.mean(out)), 0.0, places=4)

    def test_does_not_amplify_pure_silence(self):
        out = normalize_loudness(np.zeros(SR, dtype=np.float32))
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.0, places=6)

def _speechlike(dur, amp, seed):
    """Band-limited noise bursts (0.4s on / 0.2s off) standing in for speech."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(dur * SR), dtype=np.float32)
    for start in np.arange(0, dur - 0.4, 0.6):
        i = int(start * SR)
        out[i:i + int(0.4 * SR)] = amp * rng.standard_normal(int(0.4 * SR))
    return out


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


class TestGainRide(unittest.TestCase):
    def test_quiet_section_is_lifted_toward_loud_section(self):
        loud = _speechlike(60, 0.1, 1)
        quiet = _speechlike(60, 0.003, 2)  # ~30 dB lower
        out = gain_ride(np.concatenate([loud, quiet]), SR)
        gap_before = _rms_db(loud) - _rms_db(quiet)
        gap_after = _rms_db(out[: loud.size]) - _rms_db(out[loud.size:])
        self.assertLess(gap_after, gap_before / 2)

    def test_does_not_amplify_silence_or_noise_floor(self):
        floor = (1e-5 * np.random.default_rng(3).standard_normal(SR * 30)).astype(np.float32)
        out = gain_ride(floor, SR)
        self.assertLess(_rms_db(out), -60.0)

    def test_preserves_shape_dtype_and_input(self):
        sig = _speechlike(20, 0.05, 4)
        before = sig.copy()
        out = gain_ride(sig, SR)
        self.assertEqual(out.shape, sig.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(sig, before)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_empty_and_short_inputs_are_safe(self):
        self.assertEqual(gain_ride(np.zeros(0, dtype=np.float32), SR).size, 0)
        short = _tone(300, dur=0.2)
        self.assertEqual(gain_ride(short, SR).shape, short.shape)

    def test_wired_into_enhance_waveform_only_when_requested(self):
        loud = _speechlike(40, 0.1, 5)
        quiet = _speechlike(40, 0.003, 6)
        sig = np.concatenate([loud, quiet])
        base = enhance_waveform(sig, SR)
        rode = enhance_waveform(sig, SR, gain_riding=True)
        self.assertGreater(_rms_db(rode[loud.size:]), _rms_db(base[loud.size:]) + 6)


if __name__ == "__main__":
    unittest.main()
