from __future__ import annotations

import numpy as np
import pytest

SR = 16000


def _vibrato_sine(freq_hz: float, duration_sec: float, sr: int = SR) -> np.ndarray:
    """A synthetic "voice-like" tone: a fundamental plus a couple of
    harmonics with a slight natural vibrato, so WORLD has something
    voiced (f0 > 0) to analyze."""
    t = np.arange(int(duration_sec * sr)) / sr
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)
    inst_freq = freq_hz * vibrato
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    y = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    window = np.ones_like(y)
    fade = min(len(y) // 10, int(0.01 * sr))
    if fade > 0:
        window[:fade] = np.linspace(0, 1, fade)
        window[-fade:] = np.linspace(1, 0, fade)
    return (y * window).astype(np.float64)


@pytest.fixture
def sine_source():
    """A 2-second 220Hz tone, as if a singer held one stable vowel."""
    return _vibrato_sine(220.0, 2.0), SR


@pytest.fixture
def multi_tone_source():
    """Three back-to-back tones at different pitches, separated by silence,
    simulating three distinct sung syllables."""
    parts = []
    for freq in (196.0, 247.0, 294.0):
        parts.append(_vibrato_sine(freq, 0.4))
        parts.append(np.zeros(int(0.15 * SR), dtype=np.float64))
    return np.concatenate(parts), SR
