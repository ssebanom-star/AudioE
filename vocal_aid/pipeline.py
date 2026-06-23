"""End-to-end orchestration shared by every mapping mode.

decompose -> mapping strategy -> [shared: stretch + flatten per segment] ->
concatenate -> synthesize -> write wav.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import world_io
from .mapping import STRATEGIES
from .melody import parse_melody
from .pitch import flatten_pitch
from .stretch import time_stretch_frames
from .types import Note, Segment, WorldFeatures

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    mode: str = "b"
    flatten: float = 1.0
    portamento_ms: float = 30.0
    vibrato_rate_hz: float = 0.0
    vibrato_depth_semitones: float = 0.0
    preserve_attack_ms: float = 0.0
    frame_period_ms: float = world_io.DEFAULT_FRAME_PERIOD_MS


def note_target_frames(note: Note, frame_period_ms: float) -> int:
    return max(1, round(note.duration_sec * 1000.0 / frame_period_ms))


def process_segment(
    segment: Segment,
    config: PipelineConfig,
    frame_period_ms: float,
    prev_freq: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-stretch a segment to its note's length, then flatten its pitch."""
    target_frames = note_target_frames(segment.note, frame_period_ms)
    preserve_frames = (
        0 if segment.is_silence else max(0, round(config.preserve_attack_ms / frame_period_ms))
    )
    f0, sp, ap = time_stretch_frames(
        segment.f0,
        segment.sp,
        segment.ap,
        target_frames,
        preserve_attack_frames=preserve_frames,
        preserve_release_frames=preserve_frames,
    )
    if segment.is_silence:
        return f0, sp, ap

    f0 = flatten_pitch(
        f0,
        target_freq=segment.note.frequency_hz,
        frame_period_ms=frame_period_ms,
        flatten=config.flatten,
        portamento_ms=config.portamento_ms,
        vibrato_rate_hz=config.vibrato_rate_hz,
        vibrato_depth_semitones=config.vibrato_depth_semitones,
        prev_freq=prev_freq,
    )
    return f0, sp, ap


def run(source_path: str, melody_path: str, out_path: str, config: PipelineConfig) -> None:
    x, fs = world_io.load_audio(source_path)
    source = world_io.decompose(x, fs, frame_period_ms=config.frame_period_ms)

    notes = parse_melody(melody_path)
    logger.info("Parsed %d note(s)/rest(s) from %s", len(notes), melody_path)

    if config.mode not in STRATEGIES:
        raise ValueError(f"Unknown mode {config.mode!r}; choose from {sorted(STRATEGIES)}")
    strategy = STRATEGIES[config.mode]()
    segments = strategy.map(source, notes)

    f0_parts: list[np.ndarray] = []
    sp_parts: list[np.ndarray] = []
    ap_parts: list[np.ndarray] = []
    prev_freq: float | None = None
    for segment in segments:
        f0, sp, ap = process_segment(segment, config, source.frame_period_ms, prev_freq)
        f0_parts.append(f0)
        sp_parts.append(sp)
        ap_parts.append(ap)
        prev_freq = None if segment.is_silence else segment.note.frequency_hz

    out_features = WorldFeatures(
        f0=np.concatenate(f0_parts, axis=0),
        sp=np.concatenate(sp_parts, axis=0),
        ap=np.concatenate(ap_parts, axis=0),
        fs=source.fs,
        frame_period_ms=source.frame_period_ms,
    )
    y = world_io.synthesize(out_features)
    world_io.save_audio(out_path, y, source.fs)
    logger.info("Wrote %s (%.2fs)", out_path, len(y) / source.fs)
