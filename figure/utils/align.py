# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional

import numpy as np
import json
import os

AlignMode = Literal["interval_pre", "interval_warp"]
PadMode = Literal["edge", "zero", "nan"]



# =========================================================

def _slice_with_pad(
    X: np.ndarray,          # (n_neuron, T)
    t0: int,
    t1: int,
    pad_mode: PadMode = "edge",
) -> np.ndarray:
    """
    Return X[:, t0:t1] but ALWAYS length (t1-t0) by padding if out-of-bounds.
    Padding policy:
      - edge: replicate boundary value
      - zero: pad zeros
      - nan:  pad NaNs
    """
    X = np.asarray(X, float)
    n_neuron, T = X.shape
    t0 = int(t0); t1 = int(t1)
    L = max(0, t1 - t0)
    if L == 0:
        return np.zeros((n_neuron, 0), dtype=float)

    # in-bounds part
    a0 = max(0, t0)
    a1 = min(T, t1)
    core = X[:, a0:a1]
    core_L = core.shape[1]

    if core_L == L:
        return core

    # padding needed
    left_pad = max(0, 0 - t0)
    right_pad = max(0, t1 - T)

    if pad_mode == "zero":
        pad_left = np.zeros((n_neuron, left_pad), float)
        pad_right = np.zeros((n_neuron, right_pad), float)
    elif pad_mode == "nan":
        pad_left = np.full((n_neuron, left_pad), np.nan, float)
        pad_right = np.full((n_neuron, right_pad), np.nan, float)
    else:  # "edge"
        # if core is empty, fall back to zeros
        if core_L == 0:
            pad_left = np.zeros((n_neuron, left_pad), float)
            pad_right = np.zeros((n_neuron, right_pad), float)
        else:
            left_val = core[:, [0]]
            right_val = core[:, [-1]]
            pad_left = np.repeat(left_val, left_pad, axis=1)
            pad_right = np.repeat(right_val, right_pad, axis=1)

    return np.concatenate([pad_left, core, pad_right], axis=1)


def _resample_to_bins(
    seg: np.ndarray,        # (n_neuron, L)
    bins: int,
    pad_mode: PadMode = "edge",
) -> np.ndarray:
    """
    Linear interpolation resample to fixed bins.
    If L<2, pads to length 2 first (edge/zero/nan), then resamples.
    """
    seg = np.asarray(seg, float)
    n_neuron, L = seg.shape
    bins = int(bins)
    if bins <= 0:
        return np.zeros((n_neuron, 0), float)

    if L < 2:
        # make it at least 2 points
        seg2 = _slice_with_pad(seg, 0, 2, pad_mode=pad_mode)
        seg = seg2
        L = 2

    x_old = np.linspace(0.0, 1.0, L)
    x_new = np.linspace(0.0, 1.0, bins)

    out = np.empty((n_neuron, bins), float)
    for nn in range(n_neuron):
        y = seg[nn]
        # handle NaN if any
        m = np.isfinite(y)
        if m.sum() < 2:
            if pad_mode == "nan":
                out[nn] = np.nan
            else:
                out[nn] = 0.0
            continue
        idx = np.arange(L)
        y2 = y.copy()
        y2[~m] = np.interp(idx[~m], idx[m], y[m])
        out[nn] = np.interp(x_new, x_old, y2)
    return out



# =========================================================

def compute_min_true_interval_steps(
    sample_onset: np.ndarray,    # (n_cond,n_trial,n_pos)
    stim_len_steps: int,
    cond_indices: Tuple[int, ...] = (2, 3),
) -> int:
    """
    TRUE interval length = onset[k+1] - (onset[k] + stim_len_steps)
    Take minimum across specified conditions (default: arr conds 2 and 3).
    """
    sample_onset = np.asarray(sample_onset, int)
    stim_len_steps = int(stim_len_steps)

    n_cond, n_trial, n_pos = sample_onset.shape
    n_int = n_pos - 1

    vals = []
    for c in cond_indices:
        for tr in range(n_trial):
            for k in range(n_int):
                t0_true = int(sample_onset[c, tr, k]) + stim_len_steps
                t1_true = int(sample_onset[c, tr, k+1])
                L = t1_true - t0_true
                if L > 0:
                    vals.append(L)

    if len(vals) == 0:
        return 1
    return int(np.min(vals))


def resolve_pre_len_steps(
    sample_onset: np.ndarray,
    stim_len_steps: int,
    proportion: float = 0.5,
    pre_len_steps: Optional[int] = None,
    cond_indices_for_min: Tuple[int, ...] = (2, 3),
    min_clip: int = 1,
) -> int:
    """
    pre_len_steps = floor(min_true_interval * proportion) if not provided.
    """
    if pre_len_steps is not None:
        return max(int(min_clip), int(pre_len_steps))

    min_L = compute_min_true_interval_steps(sample_onset, stim_len_steps, cond_indices_for_min)
    L = int(np.floor(float(min_L) * float(proportion)))
    L = max(int(min_clip), L)
    return L


# =========================================================
#  Alignment: rebuild only Sample stage, then concat other phases
# =========================================================

@dataclass
class AlignConfig:
    stim_len_steps: int = 10

    align_mode: AlignMode = "interval_pre"

    # for interval_pre
    pre_len_steps: Optional[int] = None
    pre_len_proportion: float = 0.5
    min_interval_conds: Tuple[int, ...] = (2, 3)
    pre_len_min_clip: int = 1

    # for interval_warp
    warp_bins: int = 30

    # padding to prevent NaN blank columns in heatmaps
    pad_mode: PadMode = "edge"

    # diagnostics
    verbose: bool = True


def compute_phase_aligned(
    phase_orig: Dict[str, Tuple[int, int]],
    n_pos: int,
    stim_len_steps: int,
    align_mode: AlignMode,
    pre_len_steps: int,
    warp_bins: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Deterministic new phase indices after rebuilding Sample stage.
    Fix/Delay/Test/Response keep original lengths; only Sample length changes.
    """
    # original phase lengths
    def _len(name: str) -> int:
        s, e = phase_orig[name]
        return int(e - s)

    L_fix = _len("Fixation")
    L_dly = _len("Delay")
    L_tst = _len("Test")
    L_rsp = _len("Response")

    n_int = n_pos - 1
    if align_mode == "interval_pre":
        L_sample = n_pos * stim_len_steps + n_int * pre_len_steps
    else:
        L_sample = n_pos * stim_len_steps + n_int * warp_bins

    phase_aligned = {
        "Fixation": (0, L_fix),
        "Sample": (L_fix, L_fix + L_sample),
        "Delay": (L_fix + L_sample, L_fix + L_sample + L_dly),
        "Test": (L_fix + L_sample + L_dly, L_fix + L_sample + L_dly + L_tst),
        "Response": (L_fix + L_sample + L_dly + L_tst, L_fix + L_sample + L_dly + L_tst + L_rsp),
    }
    return phase_aligned


def align_one_trial(
    X: np.ndarray,                         # (n_neuron, T)
    onsets: np.ndarray,                    # (n_pos,)
    phase_orig: Dict[str, Tuple[int, int]],
    cfg: AlignConfig,
    pre_len_steps_used: int,
) -> np.ndarray:
    """
    Build aligned single-trial full axis:
      Fixation (orig) + rebuilt Sample (stim + aligned intervals) + Delay + Test + Response (orig)
    """
    X = np.asarray(X, float)
    on = np.asarray(onsets, int)
    n_neuron, T = X.shape
    n_pos = int(on.size)
    n_int = n_pos - 1

    # --- slice original phases ---
    fix_s, fix_e = phase_orig["Fixation"]
    dly_s, dly_e = phase_orig["Delay"]
    tst_s, tst_e = phase_orig["Test"]
    rsp_s, rsp_e = phase_orig["Response"]

    seg_fix = _slice_with_pad(X, fix_s, fix_e, pad_mode=cfg.pad_mode)
    seg_dly = _slice_with_pad(X, dly_s, dly_e, pad_mode=cfg.pad_mode)
    seg_tst = _slice_with_pad(X, tst_s, tst_e, pad_mode=cfg.pad_mode)
    seg_rsp = _slice_with_pad(X, rsp_s, rsp_e, pad_mode=cfg.pad_mode)

    # --- rebuild Sample stage ---
    parts = []
    stimL = int(cfg.stim_len_steps)

    for k in range(n_pos):
        t0 = int(on[k])
        t1 = t0 + stimL
        seg_stim = _slice_with_pad(X, t0, t1, pad_mode=cfg.pad_mode)  # fixed stimL
        parts.append(seg_stim)

        if k < n_pos - 1:
            t_next = int(on[k + 1])
            t0_true = int(on[k] + stimL)
            t1_true = t_next

            # enforce TRUE interval bounds; pad if weird
            if cfg.align_mode == "interval_pre":
                Lpre = int(pre_len_steps_used)
                # last Lpre within TRUE interval
                t1i = t1_true
                t0i = max(t1i - Lpre, t0_true)
                seg = _slice_with_pad(X, t0i, t1i, pad_mode=cfg.pad_mode)  # length <= Lpre by construction
                # left pad to Lpre to keep fixed length
                if seg.shape[1] < Lpre:
                    if cfg.pad_mode == "nan":
                        pad = np.full((n_neuron, Lpre - seg.shape[1]), np.nan, float)
                    elif cfg.pad_mode == "zero":
                        pad = np.zeros((n_neuron, Lpre - seg.shape[1]), float)
                    else:
                        # edge pad on left using first col of seg (or zeros if seg empty)
                        if seg.shape[1] == 0:
                            pad = np.zeros((n_neuron, Lpre), float)
                            seg = np.zeros((n_neuron, 0), float)
                        else:
                            pad = np.repeat(seg[:, [0]], Lpre - seg.shape[1], axis=1)
                    seg = np.concatenate([pad, seg], axis=1)
                parts.append(seg)

            else:  # interval_warp
                bins = int(cfg.warp_bins)
                # TRUE interval segment
                seg_true = _slice_with_pad(X, t0_true, t1_true, pad_mode=cfg.pad_mode)
                seg_w = _resample_to_bins(seg_true, bins=bins, pad_mode=cfg.pad_mode)
                parts.append(seg_w)

    seg_sample = np.concatenate(parts, axis=1) if len(parts) else np.zeros((n_neuron, 0), float)

    # --- concatenate full aligned trial ---
    aligned = np.concatenate([seg_fix, seg_sample, seg_dly, seg_tst, seg_rsp], axis=1)
    return aligned


def align_data_full_trial(
    data: np.ndarray,                         # (n_cond,n_trial,n_neuron,n_time)
    sample_onset: np.ndarray,                 # (n_cond,n_trial,n_pos)
    phase_orig: Dict[str, Tuple[int, int]],    # indices in ORIGINAL trial axis
    cfg: AlignConfig,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Main entry.
    Returns:
      aligned_data: (n_cond,n_trial,n_neuron,T_aligned)
      phase_aligned: dict with new indices on aligned axis
    """
    data = np.asarray(data, float)
    sample_onset = np.asarray(sample_onset, int)

    n_cond, n_trial, n_neuron, n_time = data.shape
    _, _, n_pos = sample_onset.shape

    # resolve pre_len_steps
    pre_len_steps_used = resolve_pre_len_steps(
        sample_onset=sample_onset,
        stim_len_steps=cfg.stim_len_steps,
        proportion=cfg.pre_len_proportion,
        pre_len_steps=cfg.pre_len_steps,
        cond_indices_for_min=cfg.min_interval_conds,
        min_clip=cfg.pre_len_min_clip,
    )

    phase_aligned = compute_phase_aligned(
        phase_orig=phase_orig,
        n_pos=n_pos,
        stim_len_steps=cfg.stim_len_steps,
        align_mode=cfg.align_mode,
        pre_len_steps=pre_len_steps_used,
        warp_bins=cfg.warp_bins,
    )

    T_aligned = phase_aligned["Response"][1]

    if cfg.verbose:
        print(f"[align] mode={cfg.align_mode} | stim_len_steps={cfg.stim_len_steps} | "
              f"pre_len_steps_used={pre_len_steps_used} | warp_bins={cfg.warp_bins} | pad_mode={cfg.pad_mode}")
        print(f"[align] T_aligned={T_aligned} | phase_aligned={phase_aligned}")

    aligned_data = np.empty((n_cond, n_trial, n_neuron, T_aligned), dtype=float)

    # diagnostics: check if any trial length deviates (should not)
    bad = 0
    for c in range(n_cond):
        for tr in range(n_trial):
            X = data[c, tr]  # (n_neuron,n_time)
            on = sample_onset[c, tr]
            a = align_one_trial(X, on, phase_orig, cfg, pre_len_steps_used)
            if a.shape[1] != T_aligned:
                bad += 1
                # hard enforce length by trimming/padding (should be extremely rare)
                if a.shape[1] > T_aligned:
                    a = a[:, :T_aligned]
                else:
                    # right pad
                    padL = T_aligned - a.shape[1]
                    if cfg.pad_mode == "nan":
                        pad = np.full((n_neuron, padL), np.nan, float)
                    elif cfg.pad_mode == "zero":
                        pad = np.zeros((n_neuron, padL), float)
                    else:
                        pad = np.repeat(a[:, [-1]], padL, axis=1) if a.shape[1] > 0 else np.zeros((n_neuron, padL), float)
                    a = np.concatenate([a, pad], axis=1)
            aligned_data[c, tr] = a

    if cfg.verbose:
        if bad == 0:
            print("[align] All trials produced consistent aligned length. No tail-padding was needed.")
        else:
            print(f"[align][WARN] {bad} trials had inconsistent lengths; trim/pad applied as fallback.")

    return aligned_data, phase_aligned




