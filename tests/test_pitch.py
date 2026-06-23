from __future__ import annotations

import numpy as np

from vocal_aid import world_io
from vocal_aid.pitch import flatten_pitch


def test_flatten_locks_voiced_frames_to_target_within_tolerance(sine_source):
    """flatten=1.0 should pull the *unvoiced-excluded* F0 mean to the target,
    not just nudge it -- verified by resynthesizing and re-extracting F0."""
    x, fs = sine_source
    features = world_io.decompose(x, fs)

    target_freq = 330.0
    flat_f0 = flatten_pitch(
        features.f0,
        target_freq=target_freq,
        frame_period_ms=features.frame_period_ms,
        flatten=1.0,
        portamento_ms=30.0,
    )

    resynth = world_io.synthesize(
        world_io.WorldFeatures(
            f0=flat_f0, sp=features.sp, ap=features.ap, fs=fs, frame_period_ms=features.frame_period_ms
        )
    )
    re_decomposed = world_io.decompose(resynth, fs)

    # Skip the portamento ramp and any edge frames; check the sustained middle.
    portamento_frames = round(30.0 / features.frame_period_ms)
    sustain = re_decomposed.f0[portamento_frames + 5 : -5]
    voiced = sustain[sustain > 0]
    assert voiced.size > 0
    measured = float(np.mean(voiced))
    assert abs(measured - target_freq) / target_freq < 0.03


def test_flatten_zero_keeps_original_pitch(sine_source):
    """flatten=0.0 should leave voiced F0 essentially untouched (besides the
    forced portamento entry, which a flatten of 0 should also neutralize to
    the source's own starting pitch)."""
    x, fs = sine_source
    features = world_io.decompose(x, fs)

    flat_f0 = flatten_pitch(
        features.f0,
        target_freq=440.0,
        frame_period_ms=features.frame_period_ms,
        flatten=0.0,
        portamento_ms=30.0,
    )
    voiced_mask = features.f0 > 0
    np.testing.assert_allclose(flat_f0[voiced_mask], features.f0[voiced_mask], rtol=1e-6)


def test_unvoiced_frames_are_never_flattened():
    f0 = np.array([0.0, 0.0, 200.0, 210.0, 0.0, 0.0])
    out = flatten_pitch(f0, target_freq=300.0, frame_period_ms=5.0, flatten=1.0, portamento_ms=0.0)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[4] == 0.0
    assert out[5] == 0.0


def test_rest_note_target_freq_zero_returns_copy():
    f0 = np.array([0.0, 0.0, 0.0])
    out = flatten_pitch(f0, target_freq=0.0, frame_period_ms=5.0, flatten=1.0)
    np.testing.assert_array_equal(out, f0)


def test_portamento_ramps_from_prev_freq():
    f0 = np.full(20, 250.0)
    out = flatten_pitch(
        f0,
        target_freq=300.0,
        frame_period_ms=5.0,
        flatten=1.0,
        portamento_ms=25.0,  # 5 frames
        prev_freq=200.0,
    )
    assert abs(out[0] - 200.0) < 1.0
    n_ramp = round(25.0 / 5.0)
    assert abs(out[n_ramp] - 300.0) < 1.0
    assert np.all(out[n_ramp:] == out[n_ramp])


def test_vibrato_modulates_sustain_around_target():
    f0 = np.full(200, 250.0)
    out = flatten_pitch(
        f0,
        target_freq=300.0,
        frame_period_ms=5.0,
        flatten=1.0,
        portamento_ms=0.0,
        vibrato_rate_hz=5.0,
        vibrato_depth_semitones=1.0,
    )
    assert out.max() > 300.0
    assert out.min() < 300.0
    assert abs(float(np.mean(out)) - 300.0) < 5.0
