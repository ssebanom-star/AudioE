"""F0 flattening: bend a segment's pitch contour toward its target note.

Pure step-function flattening sounds robotic, so each segment's F0 is
reshaped in three layers before being blended back with the source:

1. Portamento -- the first `portamento_ms` glide linearly from the previous
   note's ending frequency (or the segment's own starting pitch) up/down to
   the target note frequency, instead of jumping instantly.
2. Sustain -- held at the target note frequency for the remainder of the
   segment.
3. Optional vibrato -- a sinusoidal modulation layered on top of the
   sustain portion only (starts at zero phase so it doesn't click).

`flatten` (0..1) then blends this shaped curve back with the untouched
source F0: 1.0 is fully locked to the note, lower values let some of the
original pitch wobble show through. Unvoiced frames (source f0 == 0) are
always left at 0 -- flattening never invents pitch where there was none.
"""
from __future__ import annotations

import numpy as np


def flatten_pitch(
    f0: np.ndarray,
    target_freq: float,
    frame_period_ms: float,
    flatten: float = 1.0,
    portamento_ms: float = 30.0,
    vibrato_rate_hz: float = 0.0,
    vibrato_depth_semitones: float = 0.0,
    prev_freq: float | None = None,
) -> np.ndarray:
    """Reshape a segment's F0 contour toward `target_freq`.

    Returns a new array the same length as `f0`; unvoiced (f0==0) frames
    are preserved as 0 regardless of `flatten`/`target_freq`.
    """
    n = f0.shape[0]
    if n == 0 or target_freq <= 0:
        return f0.copy()

    voiced = f0 > 0
    frame_period_sec = frame_period_ms / 1000.0
    portamento_frames = min(int(round(portamento_ms / frame_period_ms)), n)

    if prev_freq and prev_freq > 0:
        start_freq = prev_freq
    else:
        voiced_idx = np.flatnonzero(voiced)
        start_freq = f0[voiced_idx[0]] if voiced_idx.size else target_freq

    shaped = np.full(n, target_freq, dtype=np.float64)
    if portamento_frames > 1:
        shaped[:portamento_frames] = np.linspace(start_freq, target_freq, portamento_frames)
    elif portamento_frames == 1:
        shaped[0] = start_freq

    if vibrato_rate_hz > 0 and vibrato_depth_semitones > 0:
        frame_idx = np.arange(n)
        t_sustain = np.clip(frame_idx - portamento_frames, 0, None) * frame_period_sec
        vibrato_ratio = 2.0 ** (
            (vibrato_depth_semitones / 12.0) * np.sin(2.0 * np.pi * vibrato_rate_hz * t_sustain)
        )
        sustain_mask = frame_idx >= portamento_frames
        shaped = np.where(sustain_mask, shaped * vibrato_ratio, shaped)

    flatten = float(np.clip(flatten, 0.0, 1.0))
    blended = flatten * shaped + (1.0 - flatten) * f0
    return np.where(voiced, blended, 0.0)
