from __future__ import annotations

from vocal_aid import world_io
from vocal_aid.mapping import AutoSyllableStrategy, SingleSampleStrategy
from vocal_aid.types import Note


def _notes(n, midi=60):
    return [Note(start_sec=i * 0.4, duration_sec=0.4, midi_number=midi) for i in range(n)]


def test_mode_b_returns_one_segment_per_note(sine_source):
    x, fs = sine_source
    features = world_io.decompose(x, fs)
    notes = _notes(5)
    segments = SingleSampleStrategy().map(features, notes)
    assert len(segments) == 5
    assert all(not s.is_silence for s in segments)


def test_mode_b_reuses_identical_source_slice_for_every_note(sine_source):
    x, fs = sine_source
    features = world_io.decompose(x, fs)
    notes = _notes(4)
    segments = SingleSampleStrategy().map(features, notes)
    for seg in segments[1:]:
        assert seg.f0.shape == segments[0].f0.shape
        assert (seg.f0 == segments[0].f0).all()


def test_mode_b_rest_note_produces_silence_segment(sine_source):
    x, fs = sine_source
    features = world_io.decompose(x, fs)
    notes = [Note(start_sec=0.0, duration_sec=0.2, midi_number=None)]
    segments = SingleSampleStrategy().map(features, notes)
    assert segments[0].is_silence
    assert (segments[0].f0 == 0).all()


def test_mode_a_detects_three_distinct_utterances(multi_tone_source):
    x, fs = multi_tone_source
    features = world_io.decompose(x, fs)
    notes = _notes(3)
    segments = AutoSyllableStrategy().map(features, notes)
    assert len(segments) == 3
    means = [float(s.f0[s.f0 > 0].mean()) for s in segments]
    # the three source tones are 196/247/294 Hz, clearly separable
    assert means[0] < means[1] < means[2]


def test_mode_a_repeats_last_utterance_when_fewer_than_notes(multi_tone_source):
    x, fs = multi_tone_source
    features = world_io.decompose(x, fs)
    notes = _notes(5)  # 3 utterances detected, 5 notes requested
    segments = AutoSyllableStrategy().map(features, notes)
    assert len(segments) == 5
    last_mean = float(segments[2].f0[segments[2].f0 > 0].mean())
    fourth_mean = float(segments[3].f0[segments[3].f0 > 0].mean())
    fifth_mean = float(segments[4].f0[segments[4].f0 > 0].mean())
    assert fourth_mean == fifth_mean == last_mean


def test_mode_a_truncates_extra_utterances_when_more_than_notes(multi_tone_source):
    x, fs = multi_tone_source
    features = world_io.decompose(x, fs)
    notes = _notes(2)  # 3 utterances detected, only 2 notes requested
    segments = AutoSyllableStrategy().map(features, notes)
    assert len(segments) == 2
    means = [float(s.f0[s.f0 > 0].mean()) for s in segments]
    assert means[0] < means[1]  # first two utterances (196Hz, 247Hz), third dropped


def test_mode_a_skips_rests_when_indexing_utterances(multi_tone_source):
    x, fs = multi_tone_source
    features = world_io.decompose(x, fs)
    notes = [
        Note(start_sec=0.0, duration_sec=0.2, midi_number=None),  # rest
        Note(start_sec=0.2, duration_sec=0.4, midi_number=60),
        Note(start_sec=0.6, duration_sec=0.4, midi_number=62),
    ]
    segments = AutoSyllableStrategy().map(features, notes)
    assert segments[0].is_silence
    assert not segments[1].is_silence
    assert not segments[2].is_silence
    mean1 = float(segments[1].f0[segments[1].f0 > 0].mean())
    mean2 = float(segments[2].f0[segments[2].f0 > 0].mean())
    assert mean1 < mean2
