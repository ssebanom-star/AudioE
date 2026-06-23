"""Frame-domain time-stretching of WORLD features (no external DSP library).

Stretching is done by resampling along the frame axis with linear
interpolation. An optional attack/release "preserve" window can be carved
out of both ends of a segment; those windows are resampled 1:1 onto an
equal-length target window (i.e. left effectively unstretched), while only
the sustain (middle) portion absorbs the length change. This reduces the
"vowel smear" you get from naively stretching an entire syllable, since
consonant-adjacent transients at the edges keep their original shape.
"""
from __future__ import annotations

import numpy as np


def _resample_axis0(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample `arr` along axis 0 to exactly `n_out` frames."""
    n_in = arr.shape[0]
    if n_out <= 0:
        return arr[:0]
    if n_in == 0:
        return np.zeros((n_out, *arr.shape[1:]), dtype=arr.dtype)
    if n_in == 1:
        return np.repeat(arr, n_out, axis=0)

    src_idx = np.linspace(0, n_in - 1, n_out)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (src_idx - lo).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[lo] * (1 - frac) + arr[hi] * frac


def time_stretch_frames(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    target_n_frames: int,
    preserve_attack_frames: int = 0,
    preserve_release_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a (f0, sp, ap) frame triple to `target_n_frames` frames.

    If `preserve_attack_frames`/`preserve_release_frames` are > 0, those many
    frames at the start/end are resampled 1:1 onto an equal-size window
    (effectively held fixed) and only the remaining sustain frames stretch.
    """
    n_src = f0.shape[0]
    if n_src == 0:
        raise ValueError("Cannot time-stretch an empty source segment")
    if target_n_frames <= 0:
        return f0[:0], sp[:0], ap[:0]

    attack = min(preserve_attack_frames, n_src)
    release = min(preserve_release_frames, max(n_src - attack, 0))
    sustain_src = n_src - attack - release

    attack_tgt = min(attack, target_n_frames)
    release_tgt = min(release, max(target_n_frames - attack_tgt, 0))
    sustain_tgt = target_n_frames - attack_tgt - release_tgt

    splits = [
        (0, attack, attack_tgt),
        (attack, attack + sustain_src, sustain_tgt),
        (attack + sustain_src, n_src, release_tgt),
    ]

    f0_parts, sp_parts, ap_parts = [], [], []
    for start, end, n_out in splits:
        f0_parts.append(_resample_axis0(f0[start:end], n_out))
        sp_parts.append(_resample_axis0(sp[start:end], n_out))
        ap_parts.append(_resample_axis0(ap[start:end], n_out))

    f0_out = np.concatenate(f0_parts, axis=0)
    sp_out = np.concatenate(sp_parts, axis=0)
    ap_out = np.concatenate(ap_parts, axis=0)
    return f0_out, sp_out, ap_out
