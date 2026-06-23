"""Mode B: single stable sample, reused for every note.

Picks the longest contiguous voiced run in the source (the most likely
"stable vowel") and reuses that same slice of frames for every melody note,
ignoring lyric/syllable content entirely. This is the most robust mode --
there is no boundary-detection step that can go wrong -- at the cost of
flattening all lyrics into one repeated timbre.
"""
from __future__ import annotations

import logging

import numpy as np

from ..types import Note, Segment, WorldFeatures
from .base import MappingStrategy, make_silence_segment

logger = logging.getLogger(__name__)


def _longest_voiced_run(f0: np.ndarray) -> tuple[int, int]:
    """Return (start, end) frame indices of the longest run where f0 > 0."""
    voiced = f0 > 0
    if not voiced.any():
        return 0, f0.shape[0]

    best_start, best_len = 0, 0
    run_start = None
    for i, v in enumerate(voiced):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            run_len = i - run_start
            if run_len > best_len:
                best_start, best_len = run_start, run_len
            run_start = None
    if run_start is not None:
        run_len = len(voiced) - run_start
        if run_len > best_len:
            best_start, best_len = run_start, run_len
    return best_start, best_start + best_len


class SingleSampleStrategy(MappingStrategy):
    """Reuse one stable vowel slice from the source for every note."""

    def map(self, source: WorldFeatures, notes: list[Note]) -> list[Segment]:
        start, end = _longest_voiced_run(source.f0)
        if end <= start:
            logger.warning("No voiced frames found in source; using entire signal")
            start, end = 0, source.n_frames
        logger.info(
            "Mode B: reusing source frames [%d:%d] (%.0fms) for all %d notes",
            start,
            end,
            (end - start) * source.frame_period_ms,
            len(notes),
        )

        fft_bins = source.sp.shape[1]
        segments: list[Segment] = []
        for note in notes:
            if note.is_rest:
                segments.append(make_silence_segment(note, fft_bins))
            else:
                segments.append(
                    Segment(
                        note=note,
                        f0=source.f0[start:end].copy(),
                        sp=source.sp[start:end].copy(),
                        ap=source.ap[start:end].copy(),
                    )
                )
        return segments
