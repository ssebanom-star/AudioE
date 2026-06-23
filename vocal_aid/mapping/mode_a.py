"""Mode A: automatic syllable/utterance segmentation, mapped 1:1 to notes.

Boundaries are detected from a combination of frame energy (mean of the
WORLD spectral envelope per frame) and voicing (WORLD's own f0>0 decision),
which is a simple energy+VAD scheme, not a trained segmenter. Short gaps are
bridged (a glottal catch inside a sustained vowel shouldn't split a
syllable) and short blips are dropped (noise, not a real utterance).

Limitations (see README): boundary detection is heuristic and will
mis-segment on breathy onsets, sibilants, portamento glides between
syllables, or noisy recordings. There is no lyric/phoneme awareness --
this only finds "something was vocalized here" boundaries.

Count-mismatch rule (spec-mandated, fixed behavior):
- fewer utterances than notes -> the last detected utterance is repeated
  for the remaining notes.
- more utterances than notes -> extra utterances are truncated (dropped).
"""
from __future__ import annotations

import logging

import numpy as np

from ..types import Note, Segment, WorldFeatures
from .base import MappingStrategy, make_silence_segment

logger = logging.getLogger(__name__)

DEFAULT_ENERGY_RATIO = 0.05
DEFAULT_MIN_SILENCE_MS = 100.0
DEFAULT_MIN_SEGMENT_MS = 60.0


def _detect_utterances(
    source: WorldFeatures,
    energy_ratio: float = DEFAULT_ENERGY_RATIO,
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    min_segment_ms: float = DEFAULT_MIN_SEGMENT_MS,
) -> list[tuple[int, int]]:
    """Energy+voicing based syllable/utterance boundary detection."""
    frame_energy = source.sp.mean(axis=1)
    peak = frame_energy.max() if frame_energy.size else 0.0
    threshold = peak * energy_ratio
    active = (frame_energy > threshold) & (source.f0 > 0)

    runs: list[tuple[int, int]] = []
    run_start = None
    for i, a in enumerate(active):
        if a and run_start is None:
            run_start = i
        elif not a and run_start is not None:
            runs.append((run_start, i))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(active)))

    if not runs:
        return []

    min_silence_frames = max(1, round(min_silence_ms / source.frame_period_ms))
    merged: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= min_silence_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    min_segment_frames = max(1, round(min_segment_ms / source.frame_period_ms))
    return [(s, e) for s, e in merged if (e - s) >= min_segment_frames]


class AutoSyllableStrategy(MappingStrategy):
    """Auto-detect utterances in the source and map them 1:1 to notes."""

    def __init__(
        self,
        energy_ratio: float = DEFAULT_ENERGY_RATIO,
        min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
        min_segment_ms: float = DEFAULT_MIN_SEGMENT_MS,
    ) -> None:
        self.energy_ratio = energy_ratio
        self.min_silence_ms = min_silence_ms
        self.min_segment_ms = min_segment_ms

    def map(self, source: WorldFeatures, notes: list[Note]) -> list[Segment]:
        utterances = _detect_utterances(
            source, self.energy_ratio, self.min_silence_ms, self.min_segment_ms
        )
        voiced_notes = [n for n in notes if not n.is_rest]
        fft_bins = source.sp.shape[1]

        if not utterances:
            logger.warning(
                "Mode A: no utterances detected; falling back to entire source for every note"
            )
            utterances = [(0, source.n_frames)]

        logger.info(
            "Mode A: detected %d utterance(s) for %d voiced note(s)",
            len(utterances),
            len(voiced_notes),
        )
        if len(utterances) < len(voiced_notes):
            logger.warning(
                "Mode A: %d utterances < %d notes; repeating last utterance for the remainder",
                len(utterances),
                len(voiced_notes),
            )
        elif len(utterances) > len(voiced_notes):
            logger.warning(
                "Mode A: %d utterances > %d notes; truncating extra utterances",
                len(utterances),
                len(voiced_notes),
            )

        segments: list[Segment] = []
        for i, note in enumerate(notes):
            if note.is_rest:
                segments.append(make_silence_segment(note, fft_bins))
                continue
            note_idx = sum(1 for n in notes[: i + 1] if not n.is_rest) - 1
            utt_idx = min(note_idx, len(utterances) - 1)
            start, end = utterances[utt_idx]
            segments.append(
                Segment(
                    note=note,
                    f0=source.f0[start:end].copy(),
                    sp=source.sp[start:end].copy(),
                    ap=source.ap[start:end].copy(),
                )
            )
        return segments
