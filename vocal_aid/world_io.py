"""WORLD vocoder wrapper: decomposition and resynthesis.

Why WORLD (and not PSOLA / plain resampling):
WORLD splits a voice signal into three independent layers -- F0 (pitch),
a spectral envelope SP (timbre/formants), and aperiodicity AP (breathiness).
Because the layers are independent, this engine can replace *only* F0 with
the target note frequency while leaving SP and AP untouched. That keeps the
singer's timbre and formants intact regardless of how far the pitch moves.
PSOLA-style or simple resampling pitch-shifters move the spectral envelope
together with the pitch (formant shift), which is what causes the classic
"chipmunk" effect on upward shifts and a "muffled giant" effect on downward
shifts. Decompose/recombine sidesteps that class of artifact entirely.
"""
from __future__ import annotations

import logging

import numpy as np
import pyworld as pw
import soundfile as sf

from .types import WorldFeatures

logger = logging.getLogger(__name__)

DEFAULT_FRAME_PERIOD_MS = 5.0
DEFAULT_F0_FLOOR = 71.0
DEFAULT_F0_CEIL = 800.0


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float64 PCM, return (samples, sample_rate)."""
    x, fs = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def save_audio(path: str, x: np.ndarray, fs: int) -> None:
    """Write float64 PCM samples to a WAV file (or other soundfile format)."""
    sf.write(path, x.astype(np.float64), fs)


def decompose(
    x: np.ndarray,
    fs: int,
    frame_period_ms: float = DEFAULT_FRAME_PERIOD_MS,
    f0_floor: float = DEFAULT_F0_FLOOR,
    f0_ceil: float = DEFAULT_F0_CEIL,
) -> WorldFeatures:
    """Decompose a waveform into WORLD's f0/sp/ap representation."""
    f0, t = pw.dio(
        x,
        fs,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=frame_period_ms,
    )
    f0 = pw.stonemask(x, f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    logger.debug(
        "WORLD decompose: %d frames, frame_period=%.2fms", f0.shape[0], frame_period_ms
    )
    return WorldFeatures(f0=f0, sp=sp, ap=ap, fs=fs, frame_period_ms=frame_period_ms)


def synthesize(features: WorldFeatures) -> np.ndarray:
    """Resynthesize a waveform from WORLD f0/sp/ap frames."""
    y = pw.synthesize(
        np.ascontiguousarray(features.f0),
        np.ascontiguousarray(features.sp),
        np.ascontiguousarray(features.ap),
        features.fs,
        features.frame_period_ms,
    )
    return y
