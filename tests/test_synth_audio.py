#!/usr/bin/env python3
"""Tests for training/synth_audio.py (augmentation that makes TTS audio realistic)."""

import unittest

import numpy as np

from training.synth_audio import add_noise, augment, bandlimit, reverb

SR = 16000


def _speechlike(seconds=3.0, amp=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


class TestAugment(unittest.TestCase):
    def test_add_noise_hits_target_snr(self):
        x = _speechlike()
        out = add_noise(x, snr_db=10.0, rng=np.random.default_rng(1))
        noise_db = _rms_db(out - x)
        self.assertAlmostEqual(_rms_db(x) - noise_db, 10.0, delta=1.0)

    def test_bandlimit_removes_high_frequencies(self):
        x = _speechlike()
        out = bandlimit(x, SR, cutoff_hz=3400)
        spec_in = np.abs(np.fft.rfft(x))
        spec_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(x.size, 1 / SR)
        hi = freqs > 5000
        self.assertLess(spec_out[hi].mean(), 0.05 * spec_in[hi].mean())

    def test_reverb_preserves_length_and_is_finite(self):
        x = _speechlike()
        out = reverb(x, SR, decay_sec=0.2, rng=np.random.default_rng(2))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_augment_is_deterministic_bounded_and_keeps_shape(self):
        x = _speechlike()
        a = augment(x, SR, np.random.default_rng(5))
        b = augment(x, SR, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, x.shape)
        self.assertEqual(a.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(a))), 1.0)

    def test_augment_does_not_mutate_input(self):
        x = _speechlike()
        before = x.copy()
        augment(x, SR, np.random.default_rng(6))
        np.testing.assert_array_equal(x, before)


if __name__ == "__main__":
    unittest.main()
