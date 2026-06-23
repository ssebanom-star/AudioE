"""Mapping strategy interface.

A MappingStrategy decides *which source frames* feed *which melody note*.
Everything downstream of `map()` (time-stretch, F0 flattening, WORLD
resynthesis) is shared by every strategy -- they only ever produce a plain
list of Segment objects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Note, Segment, WorldFeatures

SILENCE_SP_FLOOR = 1e-6


def make_silence_segment(note: Note, fft_bins: int) -> Segment:
    """Build a one-frame silent Segment for a rest note.

    f0=0 marks the frame unvoiced; a near-zero spectral envelope keeps
    resynthesized energy negligible regardless of aperiodicity.
    """
    f0 = np.zeros(1, dtype=np.float64)
    sp = np.full((1, fft_bins), SILENCE_SP_FLOOR, dtype=np.float64)
    ap = np.ones((1, fft_bins), dtype=np.float64)
    return Segment(note=note, f0=f0, sp=sp, ap=ap, is_silence=True)


class MappingStrategy(ABC):
    """Maps source WORLD frames onto melody notes, producing Segments."""

    @abstractmethod
    def map(self, source: WorldFeatures, notes: list[Note]) -> list[Segment]:
        """Return one Segment per note, in the same order as `notes`."""
        raise NotImplementedError
