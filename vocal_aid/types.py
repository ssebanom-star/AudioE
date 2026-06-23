"""Shared data structures for the vocal pitch-correction pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Note:
    """A single melody note (or rest) in absolute time."""

    start_sec: float
    duration_sec: float
    midi_number: int | None  # None means rest / silence
    velocity: int = 100

    @property
    def is_rest(self) -> bool:
        return self.midi_number is None

    @property
    def frequency_hz(self) -> float:
        """440 * 2**((n-69)/12); 0.0 for rests."""
        if self.midi_number is None:
            return 0.0
        return 440.0 * 2.0 ** ((self.midi_number - 69) / 12.0)


@dataclass
class WorldFeatures:
    """WORLD-domain decomposition of a source signal.

    f0: (n_frames,) fundamental frequency per frame, 0.0 = unvoiced.
    sp: (n_frames, fft_size//2+1) spectral envelope.
    ap: (n_frames, fft_size//2+1) aperiodicity.
    """

    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    fs: int
    frame_period_ms: float

    @property
    def n_frames(self) -> int:
        return self.f0.shape[0]


@dataclass
class Segment:
    """One note's worth of WORLD frames, ready for pitch-flattening + synthesis.

    Produced by a MappingStrategy from source WorldFeatures + a Note.
    `f0`/`sp`/`ap` are already sliced from the source (not yet time-stretched
    or pitch-flattened) -- those steps happen later in the shared pipeline.
    """

    note: Note
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    is_silence: bool = False
