"""Melody parsing: MIDI / MusicXML -> ordered list of Note (with rests)."""
from __future__ import annotations

import logging
import os

import mido

from .types import Note

logger = logging.getLogger(__name__)

_REST_GAP_EPS_SEC = 1e-4  # ignore gaps smaller than this when inserting rests


def midi_note_to_hz(n: int) -> float:
    """440 * 2**((n-69)/12)."""
    return 440.0 * 2.0 ** ((n - 69) / 12.0)


def _insert_rests(notes: list[Note]) -> list[Note]:
    """Fill silent gaps between consecutive notes with explicit rest Notes.

    Keeps the absolute timeline intact so concatenated segments reproduce
    the original melody's rhythm, not just a back-to-back run of notes.
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n.start_sec)
    filled: list[Note] = []
    cursor = 0.0
    for note in notes:
        gap = note.start_sec - cursor
        if gap > _REST_GAP_EPS_SEC:
            filled.append(Note(start_sec=cursor, duration_sec=gap, midi_number=None))
        filled.append(note)
        cursor = note.start_sec + note.duration_sec
    return filled


def parse_midi(path: str) -> list[Note]:
    """Parse a MIDI file into an absolute-time Note list (monophonic melody).

    Tracks are merged in playback order (mido does this automatically when
    iterating a MidiFile directly), which assumes the melody is monophonic --
    overlapping notes across tracks are not a supported input.
    """
    midi_file = mido.MidiFile(path)
    notes: list[Note] = []
    active: dict[tuple[int, int], tuple[float, int]] = {}  # (channel, note) -> (start, velocity)
    t = 0.0
    for msg in midi_file:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start, velocity = active.pop(key)
                duration = t - start
                if duration > 0:
                    notes.append(
                        Note(
                            start_sec=start,
                            duration_sec=duration,
                            midi_number=msg.note,
                            velocity=velocity,
                        )
                    )
                else:
                    logger.warning("Dropping zero-duration MIDI note at t=%.3f", start)
    if active:
        logger.warning("%d MIDI note(s) never received note_off; dropped", len(active))
    if not notes:
        raise ValueError(f"No notes found in MIDI file: {path}")
    notes.sort(key=lambda n: n.start_sec)
    return _insert_rests(notes)


def _tempo_offset_to_seconds(offset_ql: float, tempo_events: list[tuple[float, float]]) -> float:
    """Convert a quarterLength offset to seconds given piecewise-constant tempo."""
    seconds = 0.0
    for i, (seg_start_ql, bpm) in enumerate(tempo_events):
        if offset_ql <= seg_start_ql:
            break
        seg_end_ql = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else float("inf")
        span_ql = min(offset_ql, seg_end_ql) - seg_start_ql
        seconds += span_ql * (60.0 / bpm)
        if offset_ql <= seg_end_ql:
            break
    return seconds


def parse_musicxml(path: str) -> list[Note]:
    """Parse a MusicXML file into an absolute-time Note list."""
    import music21

    score = music21.converter.parse(path)
    flat = score.flatten()

    tempo_events = sorted(
        {(mm.offset, mm.number) for mm in flat.getElementsByClass(music21.tempo.MetronomeMark)}
    )
    if not tempo_events or tempo_events[0][0] > 0.0:
        tempo_events = [(0.0, 120.0)] + tempo_events

    notes: list[Note] = []
    for el in flat.notesAndRests:
        start_sec = _tempo_offset_to_seconds(float(el.offset), tempo_events)
        end_sec = _tempo_offset_to_seconds(
            float(el.offset) + float(el.duration.quarterLength), tempo_events
        )
        duration_sec = end_sec - start_sec
        if duration_sec <= 0:
            continue
        if el.isRest:
            midi_number = None
        elif el.isChord:
            midi_number = int(el.pitches[-1].midi)  # top note of chord
        else:
            midi_number = int(el.pitch.midi)
        notes.append(Note(start_sec=start_sec, duration_sec=duration_sec, midi_number=midi_number))

    if not notes:
        raise ValueError(f"No notes found in MusicXML file: {path}")
    notes.sort(key=lambda n: n.start_sec)
    return _insert_rests(notes)


def parse_melody(path: str) -> list[Note]:
    """Dispatch to the MIDI or MusicXML parser based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mid", ".midi"):
        return parse_midi(path)
    if ext in (".xml", ".musicxml", ".mxl"):
        return parse_musicxml(path)
    raise ValueError(f"Unsupported melody file extension: {ext}")
