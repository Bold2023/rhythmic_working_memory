# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_1samp, shapiro, wilcoxon, ttest_ind, ttest_rel
from scipy.stats import pearsonr

from utils.activity_loader import mask_identity_neurons




WindowMode = Literal["stim_pos", "stim_id", "interval_index"]
FDRScope = Literal["global", "within_window", "none"]


# =========================================================
# 0) Plot style
# =========================================================

def set_style(fontsize: int = 9):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.0,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================================================
# 1) Preprocess + FDR utilities
# =========================================================

def zscore_data_across_time(data: np.ndarray) -> np.ndarray:
    """Z-score per neuron across all (trial,time) samples."""
    xz = []
    for data_c in data:
        n_trial, n_neuron, n_time = data_c.shape
        xc = data_c.transpose(0, 2, 1).reshape(-1, n_neuron)
        mu = np.nanmean(xc, axis=0, keepdims=True)
        sd = np.nanstd(xc, axis=0, ddof=1, keepdims=True) + 1e-12
        xzc = (xc - mu) / sd
        xz.append(xzc.reshape(n_trial, n_time, n_neuron).transpose(0, 2, 1))
    return xz


def bh_fdr(p: np.ndarray) -> np.ndarray: 
    """Benjamini–Hochberg FDR adjusted p-values. p can contain NaNs."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if m.sum() == 0:
        return out
    pv = p[m]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    n = pv_sorted.size
    q = pv_sorted * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out[m] = q[inv]
    return out


# =========================================================
# 2) Window extraction
# =========================================================

def extract_stim_resp_pos(
    data: List[np.ndarray],              # (cond) (trial, neuron, time)
    sample_onset: List[np.ndarray],      # (cond, trial, pos)
    stim_len_steps: int,
) -> List[np.ndarray]:
    """Return: (cond, trial, pos, neuron) mean activity in stim window."""
    sample_onset = [np.asarray(so, int) for so in sample_onset]

    n_cond = len(data)
    _, n_pos = sample_onset[0].shape

    out = []
    for c in range(n_cond):
        n_trial, n_neuron, n_time = data[c].shape
        out_c = np.full((n_trial, n_pos, n_neuron), np.nan, float)
        for tr in range(n_trial):
            for k in range(n_pos):
                t0 = int(sample_onset[c][tr, k])
                t1 = t0 + stim_len_steps
                if t0 < 0 or t1 > n_time:
                    continue
                out_c[tr, k, :] = np.nanmean(data[c][tr, :, t0:t1], axis=-1)
        out.append(out_c)
    return out


def extract_interval_resp_pre(
    data: List[np.ndarray],              # (cond) (trial, neuron, time)
    sample_onset: List[np.ndarray],      # (cond, trial, pos)
    stim_len_steps: int,
    pre_len_steps: int,
    pad_short: bool = True,
) -> List[np.ndarray]:
    """
    TRUE interval: [onset_k+stim_len, onset_{k+1})
    pre window: last pre_len_steps within TRUE interval (clamped).
    Return: (cond, trial, interval_k, neuron)
    """
    sample_onset = [np.asarray(so, int) for so in sample_onset]

    n_cond = len(data)
    _, n_pos = sample_onset[0].shape
    n_int = n_pos - 1
    
    out = []
    for c in range(n_cond):
        n_trial, n_neuron, n_time = data[c].shape
        out_c = np.full((n_trial, n_int, n_neuron), np.nan, float)
        for tr in range(n_trial):
            for k in range(n_int):
                t_k = int(sample_onset[c][tr, k])
                t_next = int(sample_onset[c][tr, k+1])
                t0_true = t_k + stim_len_steps
                t1_true = t_next
                if t0_true < 0 or t1_true > n_time or t1_true <= t0_true:
                    continue

                t1 = t1_true
                t0 = max(t1 - pre_len_steps, t0_true)  # clamp into TRUE interval
                seg = data[c][tr, :, t0:t1]
                if seg.shape[1] == 0:
                    continue

                if pad_short and seg.shape[1] < pre_len_steps:
                    pad = np.full((n_neuron, pre_len_steps - seg.shape[1]), np.nan, float)
                    seg = np.concatenate([pad, seg], axis=1)

                out_c[tr, k, :] = np.nanmean(seg, axis=-1)
        out.append(out_c)
    return out


# =========================================================
# 3) η² computation (per neuron)
# =========================================================

def _eta2_from_groups(groups: List[np.ndarray]) -> Tuple[float, float]:
    valid = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() >= 2]
    if len(valid) < 2:
        return np.nan, np.nan
    try:
        _, p = f_oneway(*valid)
    except Exception:
        return np.nan, np.nan

    allv = np.concatenate(valid, axis=0)
    gm = np.nanmean(allv)
    ss_between = 0.0
    for g in valid:
        mu = np.nanmean(g)
        ss_between += g.size * (mu - gm) ** 2
    ss_total = np.nansum((allv - gm) ** 2)
    eta2 = ss_between / (ss_total + 1e-12)
    return float(eta2), float(p)


def eta2_stim_position_per_neuron(stim_resp_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stim_resp_pos = np.asarray(stim_resp_pos, float)
    n_trial, n_pos, n_neuron = stim_resp_pos.shape
    eta2 = np.full(n_neuron, np.nan, float)
    p = np.full(n_neuron, np.nan, float)
    for n in range(n_neuron):
        groups = [stim_resp_pos[:, k, n] for k in range(n_pos)]
        eta2[n], p[n] = _eta2_from_groups(groups)
    return eta2, p


def eta2_interval_index_per_neuron(int_resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    int_resp = np.asarray(int_resp, float)
    n_trial, n_int, n_neuron = int_resp.shape
    eta2 = np.full(n_neuron, np.nan, float)
    p = np.full(n_neuron, np.nan, float)
    for n in range(n_neuron):
        groups = [int_resp[:, k, n] for k in range(n_int)]
        eta2[n], p[n] = _eta2_from_groups(groups)
    return eta2, p


def eta2_identity_per_neuron(
    stim_resp_pos: np.ndarray,     # (trial, pos, neuron)
    sample_id: np.ndarray,         # (trial, pos)
    residualize_by_pos: bool = True,
    n_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stim_resp_pos = np.asarray(stim_resp_pos, float)
    sample_id = np.asarray(sample_id, int)
    n_trial, n_pos, n_neuron = stim_resp_pos.shape

    X = stim_resp_pos.copy()
    if residualize_by_pos:
        pos_mean = np.nanmean(X, axis=0, keepdims=True)
        X = X - pos_mean

    y = sample_id.reshape(-1)
    Xf = X.reshape(-1, n_neuron)

    if n_id is None:
        n_id = int(np.nanmax(sample_id)) + 1

    eta2 = np.full(n_neuron, np.nan, float)
    p = np.full(n_neuron, np.nan, float)
    for n in range(n_neuron):
        r = Xf[:, n]
        groups = [r[y == i] for i in range(n_id)]
        eta2[n], p[n] = _eta2_from_groups(groups)
    return eta2, p


# =========================================================
# 4) Paired tests across neurons (Δ vs 0; modulation)
# =========================================================

def one_sample_test(vals: np.ndarray, normal_alpha: float = 0.05, use_shapiro: bool = True) -> Dict[str, float]:
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n < 3:
        return {"p_raw": np.nan, "dz": np.nan, "n": n, "method": "insufficient"}

    dz = float(np.mean(vals) / (np.std(vals, ddof=1) + 1e-12))
    p_norm = 0.0
    if use_shapiro and 3 <= n <= 5000:
        try:
            _, p_norm = shapiro(vals)
        except Exception:
            p_norm = 0.0

    if p_norm >= normal_alpha:
        _, p = ttest_1samp(vals, 0.0)
        return {"p_raw": float(p), "dz": dz, "n": n, "method": "t_1samp"}
    else:
        try:
            _, p = wilcoxon(vals)
            return {"p_raw": float(p), "dz": dz, "n": n, "method": "wilcoxon"}
        except Exception:
            return {"p_raw": 1.0, "dz": dz, "n": n, "method": "wilcoxon_failed"}


# =========================================================
# 5) p formatting
# =========================================================

def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NaN"
    if p < 1e-4:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# =========================================================
# 6) Selectivity categories (for scatter colors)
# =========================================================

@dataclass
class SelMasks:
    arr_sel: np.ndarray
    rhy_sel: np.ndarray
    both: np.ndarray
    arr_only: np.ndarray
    rhy_only: np.ndarray
    neither: np.ndarray


def compute_selective_masks(
    eta_arr: np.ndarray,
    p_arr: np.ndarray,
    eta_rhy: np.ndarray,
    p_rhy: np.ndarray,
    eta_thr: float = 0.05,
    p_alpha: float = 0.01,
    use_fdr: bool = True,
) -> SelMasks:
    eta_arr = np.asarray(eta_arr, float)
    eta_rhy = np.asarray(eta_rhy, float)
    p_arr = np.asarray(p_arr, float)
    p_rhy = np.asarray(p_rhy, float)

    p_arr_use = bh_fdr(p_arr) if use_fdr else p_arr
    p_rhy_use = bh_fdr(p_rhy) if use_fdr else p_rhy

    arr_sel = np.isfinite(eta_arr) & np.isfinite(p_arr_use) & (eta_arr >= eta_thr) & (p_arr_use < p_alpha)
    rhy_sel = np.isfinite(eta_rhy) & np.isfinite(p_rhy_use) & (eta_rhy >= eta_thr) & (p_rhy_use < p_alpha)

    both = arr_sel & rhy_sel
    arr_only = arr_sel & (~rhy_sel)
    rhy_only = rhy_sel & (~arr_sel)
    neither = (~arr_sel) & (~rhy_sel) & np.isfinite(eta_arr) & np.isfinite(eta_rhy)

    return SelMasks(arr_sel=arr_sel, rhy_sel=rhy_sel, both=both, arr_only=arr_only, rhy_only=rhy_only, neither=neither)


# =========================================================
# 7) Plot 1: scatter η² (arr vs rhy), single figure
# =========================================================



# --- overlay: median dot (journal-friendly) ---
def _median_dot_ax(ax, x, v, vert=True):
    if v.size == 0:
        return
    med = float(np.nanmedian(v))
    if vert:
        p = ([x], [med])
    else:
        p = ([med], [x])
    ax.scatter(
        *p,
        s=28,
        facecolor="white",
        edgecolor="0.25",
        linewidth=0.9,
        zorder=4
    )


def _annotate_bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=0.9, color="0.2")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9, color="0.1")


def plot_eta2_scatter(
    eta_arr: np.ndarray,
    eta_rhy: np.ndarray,
    sel: SelMasks,
    title: str,
    save_path: Optional[str] = None,
    save_dpi: int = 300,
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
):

    eta_arr = np.asarray(eta_arr, float)
    eta_rhy = np.asarray(eta_rhy, float)

    m = np.isfinite(eta_arr) & np.isfinite(eta_rhy)
    if xerr is not None:
        xerr = np.asarray(xerr, float)
        m = m & np.isfinite(xerr)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        m = m & np.isfinite(yerr)

    n_total = int(m.sum())
    if n_total == 0:
        print(f"[plot_eta2_scatter_by_change] No finite points for: {title}")
        return

    lim = 1.0

    fig = plt.figure(figsize=(6, 6), dpi=(save_dpi if save_path else 120))
    gs = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    ax = fig.add_subplot(gs[0, 0])

    s = 40

    legal_indices = [idx for idx in range(eta_arr.size) if m[idx]]
    
    ax.scatter(
        eta_arr[legal_indices], eta_rhy[legal_indices],
        s=s, c='grey', linewidth=1.2,
        alpha=0.6, zorder=7,
    )

    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=0.9, color="0.35", zorder=2)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("η² (arrhythmic)")
    ax.set_ylabel("η² (rhythmic)")
    ax.set_title(title, pad=5)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])


    _despine(ax)


    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()




def plot_eta2_scatter_by_change(
    eta_arr: np.ndarray,
    eta_rhy: np.ndarray,
    sel: SelMasks,
    title: str,
    save_path: Optional[str] = None,
    save_dpi: int = 300,
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,

    marked_indices: Optional[List[int]] = None,
):
    colors = {
        "Neither": "#1f77b4",
        "Arr-only": "#ff7f0e",
        "Rhy-only": "#2ca02c",
        "Both": "#d62728",
    }

    eta_arr = np.asarray(eta_arr, float)
    eta_rhy = np.asarray(eta_rhy, float)

    m = np.isfinite(eta_arr) & np.isfinite(eta_rhy)
    if xerr is not None:
        xerr = np.asarray(xerr, float)
        m = m & np.isfinite(xerr)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        m = m & np.isfinite(yerr)

    n_total = int(m.sum())
    if n_total == 0:
        print(f"[plot_eta2_scatter_by_change] No finite points for: {title}")
        return

    x_use = eta_arr[m]
    y_use = eta_rhy[m]
    # hi = float(np.nanpercentile(np.concatenate([x_use, y_use]), 99))
    lim = 1.0

    fig = plt.figure(figsize=(6, 6), dpi=(save_dpi if save_path else 120))
    gs = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    ax = fig.add_subplot(gs[0, 0])

    s = 40

    marked_indices = [idx for idx in marked_indices if m[idx]]
    unmarked_indices = [idx for idx in range(eta_arr.size) if m[idx] and (idx not in marked_indices)]
    n_marked = len(marked_indices)
    n_unmarked = len(unmarked_indices)
    
    ax.scatter(
        eta_arr[marked_indices], eta_rhy[marked_indices],
        s=s, c='g', linewidth=1.2,
        alpha=0.6, zorder=7,
    )
    ax.scatter(
        eta_arr[unmarked_indices], eta_rhy[unmarked_indices],
        s=s, c='r', linewidth=1.2,
        alpha=0.6, zorder=7,
    )

    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=0.9, color="0.35", zorder=2)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("η² (arrhythmic)")
    ax.set_ylabel("η² (rhythmic)")
    ax.set_title(title, pad=5)


    colors = ["r", "g"]

    parts = ax.boxplot(
        [eta_rhy[unmarked_indices], eta_rhy[marked_indices]], positions=[1.05, 1.12], widths=0.05,
        showmeans=False, patch_artist=True,
    )

    f, p = ttest_ind(eta_rhy[marked_indices], eta_rhy[unmarked_indices])
    ax.text(0.9, 0.3, f"rhy f = {f:.4f}", ha="center", va="top", fontsize=9, color="0.1")
    ax.text(0.9, 0.2, f"rhy p = {p:.4f}", ha="center", va="top", fontsize=9, color="0.1")

    # bodies styling
    # for i, body in enumerate(parts["bodies"]):
    for i, body in enumerate(parts["boxes"]):
        body.set_alpha(0.4)
        body.set_edgecolor("0.25")
        body.set_linewidth(0.8)
        body.set_facecolor(colors[i])

    _median_dot_ax(ax, 1.05, eta_rhy[unmarked_indices])
    _median_dot_ax(ax, 1.12, eta_rhy[marked_indices])

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticks([0, 1], [f"Unmarked ({n_unmarked})", f"Marked ({n_marked})"])
    # ax.tick_params(axis="x", length=0)
    # ax.set_ylim(0, lim)

    
    parts = ax.boxplot(
        [eta_arr[unmarked_indices], eta_arr[marked_indices]], positions=[1.05, 1.12], widths=0.05, vert=False,
        showmeans=False, patch_artist=True,
    )

    f, p = ttest_ind(eta_arr[marked_indices],  eta_arr[unmarked_indices])
    ax.text(0.25, 0.9, f"arr f = {f:.4f}", ha="center", va="top", fontsize=9, color="0.1")
    ax.text(0.25, 0.8, f"arr p = {p:.4f}", ha="center", va="top", fontsize=9, color="0.1")

    # bodies styling
    # for i, body in enumerate(parts["bodies"]):
    for i, body in enumerate(parts["boxes"]):
        body.set_alpha(0.4)
        body.set_edgecolor("0.25")
        body.set_linewidth(0.8)
        body.set_facecolor(colors[i])

    _median_dot_ax(ax, 1.05, eta_arr[unmarked_indices], vert=False)
    _median_dot_ax(ax, 1.12, eta_arr[marked_indices], vert=False)

    # ax.set_xlim(0-0.015, lim+0.020)
    # ax.set_xticks([])
    # # ax.set_yticks([])
    # ax.set_yticks([0, 1], [f"Unmarked ({n_unmarked})", f"Marked ({n_marked})"])
    # ax.tick_params(axis="y", length=0)


    ax.set_xlim(0, lim + 0.15)
    ax.set_ylim(0, lim + 0.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])


    _despine(ax)


    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()





def plot_eta2_scatter_by_change__values(
    eta_arr: np.ndarray,
    eta_rhy: np.ndarray,
    sel: SelMasks,
    title: str,
    save_path: Optional[str] = None,
    save_dpi: int = 300,
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,

    values = None
):

    eta_arr = np.asarray(eta_arr, float)
    eta_rhy = np.asarray(eta_rhy, float)

    m = np.isfinite(eta_arr) & np.isfinite(eta_rhy)
    if xerr is not None:
        xerr = np.asarray(xerr, float)
        m = m & np.isfinite(xerr)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        m = m & np.isfinite(yerr)

    n_total = int(m.sum())
    if n_total == 0:
        print(f"[plot_eta2_scatter_by_change] No finite points for: {title}")
        return

    lim = 1.0

    fig = plt.figure(figsize=(6, 6), dpi=(save_dpi if save_path else 120))
    gs = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    ax = fig.add_subplot(gs[0, 0])

    s = 40

    correlation_x, p_value_x = pearsonr(values[m], eta_arr[m])
    correlation_y, p_value_y = pearsonr(values[m], eta_rhy[m])

    
    norm = plt.Normalize(vmin=np.min(values[m]), vmax=np.max(values[m]))

    sc = ax.scatter(
        eta_arr[m], eta_rhy[m],
        s=s, c=values[m], cmap='viridis',  # Use 'viridis' colormap
        linewidth=1.2, alpha=0.6, zorder=7,
        norm=norm  # Normalize values for color mapping
    )

    ax.text(0.7, 0.1, f"corr arr: {correlation_x:.4f} (p={p_value_x:.4f})", ha="center", va="top", fontsize=9, color="0.1")
    ax.text(0.7, 0.2, f"corr rhy: {correlation_y:.4f} (p={p_value_y:.4f})", ha="center", va="top", fontsize=9, color="0.1")

    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("η² (arrhythmic)")
    ax.set_ylabel("η² (rhythmic)")
    ax.set_xlim(0, lim + 0.15)
    ax.set_ylim(0, lim + 0.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    cbar = plt.colorbar(sc)
    cbar.set_label('Values')

    _despine(ax)


    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()





# =========================================================
# 8) Plot 2: Δη² violin (change0 & change3) single figure
# =========================================================


def plot_delta_eta2_violin(
    d0: np.ndarray,
    d3: np.ndarray,
    stat0: Dict[str, float],
    stat3: Dict[str, float],
    stat_mod: Dict[str, float],
    title: str,
    fdr_note: str,
    save_path: Optional[str] = None,
    save_dpi: int = 300,
    marked_indices: Optional[List[int]] = None,
):
    d0 = np.asarray(d0, float)
    d3 = np.asarray(d3, float)
    d0v = d0[np.isfinite(d0)]
    d3v = d3[np.isfinite(d3)]

    # --- figure ---
    fig = plt.figure(figsize=(3.6, 2.8), dpi=(save_dpi if save_path else 200))
    ax = plt.gca()

    # --- colors (match your previous grayscale-ish style; keep subtle) ---
    fill0 = "0.4"
    fill3 = "0.4"
    edge = "0.25"

    # --- violin  ---
    parts = ax.violinplot(
        # [d0v, d3v],
        # positions=[0, 1],
        [d0v],
        positions=[0],
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False,   # we will draw median ourselves for better control
    )

    print(f"[plot_delta_eta2_violin]")
    print(f"    n      = {d0v.size}")
    print(f"    mean   = {np.nanmean(d0v)}")
    print(f"    median = {np.nanmedian(d0v)}")
    print(f"    std    = {np.nanstd(d0v)}")
    print(f"    stat0  = {stat0}")

    # bodies styling
    for i, body in enumerate(parts["bodies"]):
        body.set_alpha(0.22)
        body.set_edgecolor(edge)
        body.set_linewidth(0.8)
        body.set_facecolor(fill0 if i == 0 else fill3)

    # --- overlay: median dot (journal-friendly) ---
    def _median_dot(x, v):
        if v.size == 0:
            return
        med = float(np.nanmedian(v))
        ax.scatter(
            [x], [med],
            s=28,
            facecolor="white",
            edgecolor=edge,
            linewidth=0.9,
            zorder=4
        )

    _median_dot(0, d0v)
    # _median_dot(1, d3v)

    # --- overlay: beeswarm-like jitter scatter (lightweight) ---
    rng = np.random.default_rng(0)
    # smaller jitter + smaller points for N~100
    j0 = 0 + rng.uniform(-0.10, 0.10, size=d0v.size)
    # j3 = 1 + rng.uniform(-0.10, 0.10, size=d3v.size)
    ax.scatter(j0, d0v, s=10, color="0.15", alpha=0.30, edgecolor="none", zorder=2)
    # ax.scatter(j3, d3v, s=10, color="0.15", alpha=0.30, edgecolor="none", zorder=2)
    
    if marked_indices is not None:      
        parts = ax.violinplot(
            [d0v[marked_indices]],
            positions=[1],
            widths=0.72,
            showmeans=False,
            showextrema=False,
            showmedians=False,   # we will draw median ourselves for better control
        )  
        for i, body in enumerate(parts["bodies"]):
            body.set_alpha(0.22)
            body.set_edgecolor(edge)
            body.set_linewidth(0.8)
            body.set_facecolor(fill0 if i == 0 else fill3)
        _median_dot(1, d0v[marked_indices])
        ax.scatter(
            j0[marked_indices], d0v[marked_indices],
            marker='x', s=13, c='k', linewidth=1.2,
            alpha=1.0, zorder=7,
        )
        p_pred = one_sample_test(d0v[marked_indices])["p_raw"]

        _, p_diff = ttest_ind(d0v[marked_indices], d3v[marked_indices], nan_policy='omit')

    # --- baseline ---
    ax.axhline(0, linestyle="--", linewidth=0.9, color="0.35", zorder=1)

    # --- axes labels ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all", "predictive"])
    ax.set_ylabel("Δη² = η²(rhythmic) − η²(arrhythmic)")
    ax.set_title(title, pad=5)

    _despine(ax)

    # --- robust y-limits (avoid too much whitespace) ---
    allv = np.concatenate([d0v, d3v]) if (d0v.size + d3v.size) else np.array([0.0])
    if allv.size == 0:
        allv = np.array([0.0])

    # robust limits based on percentiles
    p1, p99 = np.nanpercentile(allv, [1, 99]) if allv.size >= 5 else (float(np.min(allv)), float(np.max(allv)))
    if np.isclose(p1, p99):
        p1 -= 0.1
        p99 += 0.1

    pad = 0.22 * (p99 - p1 + 1e-12)
    ylo = p1 - 0.6 * pad
    yhi = p99 + 1.4 * pad
    ax.set_ylim(ylo, yhi)

    # --- p-value annotations (journal style) ---
    span = yhi - ylo
    # group vs 0: place slightly above each violin top region
    y_text = yhi - 0.22 * span
    txt0 = f"p={_format_p(stat0['p_fdr'])} {_sig_stars(stat0['p_fdr'])}"
    txt3 = f"p={_format_p(p_pred)} {_sig_stars(p_pred)}"
    ax.text(0, y_text, txt0, ha="center", va="bottom", fontsize=9, color="0.10")
    ax.text(1, y_text, txt3, ha="center", va="bottom", fontsize=9, color="0.10")

    # modulation bracket: put above the two texts
    y_br = yhi - 0.12 * span
    h = 0.035 * span
    txtm = f"p={_format_p(p_diff)} {_sig_stars(p_diff)}"
    _annotate_bracket(ax, 0, 1, y_br, txtm, h=h)

    # --- optional n  ---
    # ax.text(
    #     0.02, 0.02,
    #     f"n(change0)={d0v.size}, n(change3)={d3v.size}",
    #     transform=ax.transAxes,
    #     ha="left", va="bottom",
    #     fontsize=9, color="0.25"
    # )

    fig.tight_layout(pad=0.4)
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()





def plot_marked_eta2_violin(
    d0: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    save_dpi: int = 300,
    marked_indices: Optional[List[int]] = [],
):

    d0 = np.asarray(d0, float)
    nan_indices = np.where(~np.isfinite(d0))[0].tolist()

    fig = plt.figure(figsize=(3.6, 2.8), dpi=(save_dpi if save_path else 200))
    ax = plt.gca()

    fill0 = "0.4"
    fill3 = "0.4"
    edge = "0.25"
    colors = ["r", "g"]

    marked_indices = [i for i in marked_indices if i not in nan_indices]
    unmarked_indices = [i for i in range(d0.size) if (i not in marked_indices) and (i not in nan_indices)]

    # --- violin  ---
    parts = ax.violinplot(
        [d0[unmarked_indices], d0[marked_indices]], positions=[0, 1], widths=0.72,
        showmeans=False, showextrema=False, showmedians=False, 
    )

    print(f"[plot_marker_eta2_violin]")
    print(f"    n      = {d0[unmarked_indices].size:03d} | {d0[marked_indices].size:03d}")
    print(f"    mean   = {np.mean(d0[unmarked_indices]):.3f} | {np.mean(d0[marked_indices]):.3f}")
    print(f"    median = {np.median(d0[unmarked_indices]):.3f} | {np.median(d0[marked_indices]):.3f}")
    print(f"    std    = {np.std(d0[unmarked_indices]):.3f} | {np.std(d0[marked_indices]):.3f}")

    # bodies styling
    for i, body in enumerate(parts["bodies"]):
        body.set_alpha(0.22)
        body.set_edgecolor(edge)
        body.set_linewidth(0.8)
        body.set_facecolor(colors[i])

    # --- overlay: median dot (journal-friendly) ---
    def _median_dot(x, v):
        if v.size == 0:
            return
        med = float(np.nanmedian(v))
        ax.scatter(
            [x], [med],
            s=28,
            facecolor="white",
            edgecolor=edge,
            linewidth=0.9,
            zorder=4
        )

    _median_dot(0, d0[unmarked_indices])
    _median_dot(1, d0[marked_indices])

    rng = np.random.default_rng(0)
    j_l = 0 + rng.uniform(-0.10, 0.10, size=len(unmarked_indices))
    j_r = 1 + rng.uniform(-0.10, 0.10, size=len(marked_indices))
    ax.scatter(j_l, d0[unmarked_indices], s=10, color="0.15", alpha=0.30, edgecolor="none", zorder=2)
    ax.scatter(j_r, d0[marked_indices], s=10, color="0.15", alpha=0.30, edgecolor="none", zorder=2)

    p_l = one_sample_test(d0[unmarked_indices])["p_raw"]
    p_r = one_sample_test(d0[marked_indices])["p_raw"]
    _, p_diff = ttest_ind(d0[unmarked_indices], d0[marked_indices], nan_policy='omit')

    # --- baseline ---
    ax.axhline(0, linestyle="--", linewidth=0.9, color="0.35", zorder=1)

    # --- axes labels ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-X", "X"])
    ax.set_ylabel("Δη² (rhythmic - arrhythmic)")
    ax.set_title(title, pad=5)

    _despine(ax)

    # --- robust y-limits (avoid too much whitespace) ---
    allv = d0
    if allv.size == 0:
        allv = np.array([0.0])

    # robust limits based on percentiles
    p1, p99 = np.nanpercentile(allv, [1, 99]) if allv.size >= 5 else (float(np.min(allv)), float(np.max(allv)))
    if np.isclose(p1, p99):
        p1 -= 0.1
        p99 += 0.1

    pad = 0.22 * (p99 - p1 + 1e-12)
    ylo = p1 - 0.6 * pad
    yhi = p99 + 1.4 * pad
    ax.set_ylim(ylo, yhi)

    # --- p-value annotations (journal style) ---
    span = yhi - ylo
    y_text = yhi - 0.22 * span
    txt_l = f"p {_format_p(p_l)} {_sig_stars(p_l)}"
    txt_r = f"p {_format_p(p_r)} {_sig_stars(p_r)}"
    ax.text(0, y_text, txt_l, ha="center", va="bottom", fontsize=9, color="0.10")
    ax.text(1, y_text, txt_r, ha="center", va="bottom", fontsize=9, color="0.10")

    # modulation bracket: put above the two texts
    y_br = yhi - 0.12 * span
    h = 0.035 * span
    txtm = f"p {_format_p(p_diff)} {_sig_stars(p_diff)}"
    _annotate_bracket(ax, 0, 1, y_br, txtm, h=h)

    fig.tight_layout(pad=0.4)
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()




# =========================================================
# 9) Main pipeline
# =========================================================

@dataclass
class SelectivityConfig:
    stim_len_steps: int = 10
    zscore: bool = True
    pre_len_steps: int = 10

    do_identity: bool = True
    residualize_id_by_pos: bool = True

    # tests
    normal_alpha: float = 0.05
    use_shapiro: bool = True

    # test-level multiple-comparison control (reported tests)
    test_fdr_scope: FDRScope = "global"   # global recommended for manuscript

    # selectivity definition (for scatter point coloring)
    sel_eta2_thr: float = 0.05
    sel_p_alpha: float = 0.01
    sel_use_fdr: bool = True

    # output save
    out_dir: Optional[str] = None
    save_dpi: int = 300
    fontsize: int = 9




def compute_eta2_for_trial_batch(
    data: List[np.ndarray],             # (n_trial, n_neuron, n_time)
    sample_onset: List[np.ndarray],     # (n_trial, n_pos)
    sample_id: List[np.ndarray],        # (n_trial, n_pos)
    cfg: SelectivityConfig,
    n_id: Optional[int] = 6,
):
    set_style(cfg.fontsize)

    data = [data]
    sample_onset = [sample_onset]
    sample_id = [sample_id]

    data_use = zscore_data_across_time(data) if cfg.zscore else data

    # windows
    stim_resp_pos = extract_stim_resp_pos(data_use, sample_onset, cfg.stim_len_steps)
    int_resp_pre = extract_interval_resp_pre(data_use, sample_onset, cfg.stim_len_steps, cfg.pre_len_steps, pad_short=True)

    # compute eta2 and neuron-wise p for each condition
    # stim-pos
    eta_stim_pos, p_stim_pos = eta2_stim_position_per_neuron(stim_resp_pos[0])

    # interval-index
    eta_int_idx, p_int_idx = eta2_interval_index_per_neuron(int_resp_pre[0])

    # stim-id
    if cfg.do_identity:
        eta_stim_id, p_stim_id = eta2_identity_per_neuron(
            stim_resp_pos[0], sample_id=sample_id[0],
            residualize_by_pos=cfg.residualize_id_by_pos, n_id=n_id
        )

    # define windows
    windows: List[Tuple[str, Dict[int, np.ndarray], Dict[int, np.ndarray]]] = [
        ("Stim-window POSITION η²", eta_stim_pos, p_stim_pos),
        ("Interval-window INDEX η² (pre)", eta_int_idx, p_int_idx),
    ]
    if cfg.do_identity:
        tag = "Stim-window IDENTITY η²" if cfg.residualize_id_by_pos else "Stim-window IDENTITY η² (raw)"
        windows.append((tag, eta_stim_id, p_stim_id))
    
    return windows





def run_selectivity_effect_analysis(
    data: List[np.ndarray],             # (4) (n_trial, n_neuron, n_time)
    sample_onset: List[np.ndarray],     # (4, n_trial, n_pos)
    sample_id: List[np.ndarray],        # (n_trial, n_pos)
    cfg: SelectivityConfig,
    marked_indices: Optional[List[int]] = None,
    values = None,
):
    set_style(cfg.fontsize)

    for idx, single_conditional_data in enumerate(data):
        data[idx], _ = mask_identity_neurons(single_conditional_data)

    if len(data) != 4:
        raise ValueError("data must have 4 conditions: [rhy0, rhy3, arr0, arr3].")

    data_use = zscore_data_across_time(data) if cfg.zscore else data

    # windows
    stim_resp_pos = extract_stim_resp_pos(data_use, sample_onset, cfg.stim_len_steps)
    int_resp_pre = extract_interval_resp_pre(data_use, sample_onset, cfg.stim_len_steps, cfg.pre_len_steps, pad_short=True)

    # compute eta2 and neuron-wise p for each condition
    # stim-pos
    eta_stim_pos = {}
    p_stim_pos = {}
    for c in range(4):
        eta_stim_pos[c], p_stim_pos[c] = eta2_stim_position_per_neuron(stim_resp_pos[c])

    # interval-index
    eta_int_idx = {}
    p_int_idx = {}
    for c in range(4):
        eta_int_idx[c], p_int_idx[c] = eta2_interval_index_per_neuron(int_resp_pre[c])

    # stim-id
    eta_stim_id, p_stim_id = {}, {}
    if cfg.do_identity:
        for c in range(4):
            n_id = int(np.nanmax(sample_id[c])) + 1
            eta_stim_id[c], p_stim_id[c] = eta2_identity_per_neuron(
                stim_resp_pos[c], sample_id=sample_id[c],
                residualize_by_pos=cfg.residualize_id_by_pos, n_id=n_id
            )

    # define windows
    windows: List[Tuple[str, Dict[int, np.ndarray], Dict[int, np.ndarray]]] = [
        ("Stim-window POSITION η²", eta_stim_pos, p_stim_pos),
        ("Interval-window INDEX η² (pre)", eta_int_idx, p_int_idx),
    ]
    if cfg.do_identity:
        tag = "Stim-window IDENTITY η²" if cfg.residualize_id_by_pos else "Stim-window IDENTITY η² (raw)"
        windows.append((tag, eta_stim_id, p_stim_id))

    # ---- collect test p for optional test-level global FDR ----
    # Each window contributes 3 reported tests:
    #   change0: d0 vs 0
    #   change3: d3 vs 0
    #   modulation: (d3-d0) vs 0
    test_registry = []  # (win_name, key, p_raw)
    interim = {}

    for win_name, eta_map, _p_map in windows:
        d0 = eta_map[0] - eta_map[2]   # (rhy0 - arr0)
        d3 = eta_map[1] - eta_map[3]   # (rhy3 - arr3)
        dd = d3 - d0                   # modulation per neuron

        stat0 = one_sample_test(d0, cfg.normal_alpha, cfg.use_shapiro)
        stat3 = one_sample_test(d3, cfg.normal_alpha, cfg.use_shapiro)
        statm = one_sample_test(dd, cfg.normal_alpha, cfg.use_shapiro)

        # placeholders
        stat0["p_fdr"] = np.nan
        stat3["p_fdr"] = np.nan
        statm["p_fdr"] = np.nan

        interim[win_name] = {
            "eta": eta_map,
            "p_eta": _p_map,
            "d0": d0, "d3": d3, "dd": dd,
            "stat0": stat0, "stat3": stat3, "statm": statm,
        }

        test_registry += [
            (win_name, "change0", stat0["p_raw"]),
            (win_name, "change3", stat3["p_raw"]),
            (win_name, "modulation", statm["p_raw"]),
        ]

    # ---- apply test-level FDR ----
    if cfg.test_fdr_scope == "none":
        for win_name in interim:
            interim[win_name]["stat0"]["p_fdr"] = interim[win_name]["stat0"]["p_raw"]
            interim[win_name]["stat3"]["p_fdr"] = interim[win_name]["stat3"]["p_raw"]
            interim[win_name]["statm"]["p_fdr"] = interim[win_name]["statm"]["p_raw"]

        fdr_note_global = "No multiple-comparison correction applied."
    elif cfg.test_fdr_scope == "within_window":
        for win_name in interim:
            p_raws = np.array([
                interim[win_name]["stat0"]["p_raw"],
                interim[win_name]["stat3"]["p_raw"],
                interim[win_name]["statm"]["p_raw"],
            ], float)
            p_adj = bh_fdr(p_raws)
            interim[win_name]["stat0"]["p_fdr"] = float(p_adj[0]) if np.isfinite(p_adj[0]) else np.nan
            interim[win_name]["stat3"]["p_fdr"] = float(p_adj[1]) if np.isfinite(p_adj[1]) else np.nan
            interim[win_name]["statm"]["p_fdr"] = float(p_adj[2]) if np.isfinite(p_adj[2]) else np.nan

        fdr_note_global = "Benjamini–Hochberg FDR correction applied within each window (N=3)."
    elif cfg.test_fdr_scope == "global":
        p_raws = np.array([t[2] for t in test_registry], float)
        p_adj = bh_fdr(p_raws)
        for (win_name, key, _), padj in zip(test_registry, p_adj):
            if key == "change0":
                interim[win_name]["stat0"]["p_fdr"] = float(padj) if np.isfinite(padj) else np.nan
            elif key == "change3":
                interim[win_name]["stat3"]["p_fdr"] = float(padj) if np.isfinite(padj) else np.nan
            elif key == "modulation":
                interim[win_name]["statm"]["p_fdr"] = float(padj) if np.isfinite(padj) else np.nan

        fdr_note_global = f"Benjamini–Hochberg FDR correction applied globally across all reported tests (N={len(test_registry)})."
    else:
        raise ValueError(f"Unknown test_fdr_scope: {cfg.test_fdr_scope}")

    # ---- plotting (separate figures) ----
    if cfg.out_dir is not None:
        os.makedirs(cfg.out_dir, exist_ok=True)

    results = {}

    for win_name, eta_map, p_map in windows:

        print(win_name)

        d0 = interim[win_name]["d0"]
        d3 = interim[win_name]["d3"]

        stat0 = interim[win_name]["stat0"]
        stat3 = interim[win_name]["stat3"]
        statm = interim[win_name]["statm"]

        # selection masks for scatter coloring (per change)
        # change0 uses cond 2 (arr0) vs 0 (rhy0)
        sel0 = compute_selective_masks(
            eta_arr=eta_map[2], p_arr=p_map[2],
            eta_rhy=eta_map[0], p_rhy=p_map[0],
            eta_thr=cfg.sel_eta2_thr, p_alpha=cfg.sel_p_alpha, use_fdr=cfg.sel_use_fdr
        )
        # change3 uses cond 3 (arr3) vs 1 (rhy3)
        sel3 = compute_selective_masks(
            eta_arr=eta_map[3], p_arr=p_map[3],
            eta_rhy=eta_map[1], p_rhy=p_map[1],
            eta_thr=cfg.sel_eta2_thr, p_alpha=cfg.sel_p_alpha, use_fdr=cfg.sel_use_fdr
        )


        results[win_name] = interim[win_name]
        results[win_name]["sel_change0"] = sel0
        results[win_name]["sel_change3"] = sel3

    return results, windows















