from __future__ import annotations

import numpy as np
import pytest

from vocal_aid.stretch import time_stretch_frames


def _frames(n, fft_bins=5):
    f0 = np.linspace(100.0, 200.0, n)
    sp = np.tile(np.linspace(1.0, 2.0, fft_bins), (n, 1)) + np.arange(n).reshape(-1, 1)
    ap = np.full((n, fft_bins), 0.1)
    return f0, sp, ap


@pytest.mark.parametrize("n_src,target", [(10, 20), (20, 10), (5, 5), (1, 8), (8, 1)])
def test_output_length_matches_target(n_src, target):
    f0, sp, ap = _frames(n_src)
    out_f0, out_sp, out_ap = time_stretch_frames(f0, sp, ap, target)
    assert out_f0.shape[0] == target
    assert out_sp.shape == (target, sp.shape[1])
    assert out_ap.shape == (target, ap.shape[1])


def test_zero_target_frames_returns_empty():
    f0, sp, ap = _frames(10)
    out_f0, out_sp, out_ap = time_stretch_frames(f0, sp, ap, 0)
    assert out_f0.shape[0] == 0
    assert out_sp.shape[0] == 0
    assert out_ap.shape[0] == 0


def test_empty_source_raises():
    f0, sp, ap = _frames(0)
    with pytest.raises(ValueError):
        time_stretch_frames(f0, sp, ap, 10)


def test_endpoints_preserved_when_stretching():
    f0, sp, ap = _frames(10)
    out_f0, _, _ = time_stretch_frames(f0, sp, ap, 30)
    assert out_f0[0] == pytest.approx(f0[0])
    assert out_f0[-1] == pytest.approx(f0[-1])


def test_preserve_attack_release_holds_edges_fixed_when_room_available():
    """When the target is long enough, the attack/release windows should be
    resampled 1:1 (i.e. literally unchanged), and only the sustain
    in-between should reflect the stretch."""
    n_src = 20
    f0, sp, ap = _frames(n_src)
    attack, release = 4, 4
    target = 40  # plenty of room for fixed 4+4 plus a stretched sustain

    out_f0, _, _ = time_stretch_frames(
        f0, sp, ap, target, preserve_attack_frames=attack, preserve_release_frames=release
    )
    np.testing.assert_allclose(out_f0[:attack], f0[:attack])
    np.testing.assert_allclose(out_f0[-release:], f0[-release:])
    assert out_f0.shape[0] == target


def test_preserve_attack_shrinks_gracefully_when_target_too_small():
    n_src = 20
    f0, sp, ap = _frames(n_src)
    out_f0, out_sp, out_ap = time_stretch_frames(
        f0, sp, ap, 3, preserve_attack_frames=10, preserve_release_frames=10
    )
    assert out_f0.shape[0] == 3
