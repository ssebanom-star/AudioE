from __future__ import annotations

import mido
import pytest

from vocal_aid.melody import midi_note_to_hz, parse_midi


def test_midi_note_to_hz_reference_points():
    assert midi_note_to_hz(69) == pytest.approx(440.0)
    assert midi_note_to_hz(57) == pytest.approx(220.0)
    assert midi_note_to_hz(81) == pytest.approx(880.0)


def _write_test_midi(path: str) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    # Note 60 for one beat, silent gap of half a beat, then note 64 for one beat.
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=64, velocity=100, time=240))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    mid.save(path)


def test_parse_midi_extracts_notes_and_rest_gap(tmp_path):
    path = str(tmp_path / "test.mid")
    _write_test_midi(path)
    notes = parse_midi(path)

    pitched = [n for n in notes if not n.is_rest]
    rests = [n for n in notes if n.is_rest]

    assert len(pitched) == 2
    assert pitched[0].midi_number == 60
    assert pitched[1].midi_number == 64
    assert pitched[0].start_sec == pytest.approx(0.0, abs=1e-6)
    assert pitched[0].duration_sec == pytest.approx(0.5, abs=1e-3)  # 1 beat @ 120bpm
    assert len(rests) == 1
    assert rests[0].duration_sec == pytest.approx(0.25, abs=1e-3)


def test_parse_midi_missing_file_raises():
    with pytest.raises(Exception):
        parse_midi("/nonexistent/path.mid")
