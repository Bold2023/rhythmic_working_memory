from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway


FDRScope = str


def set_style(fontsize: int = 9):
    plt.rcParams.update(
        {
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
        }
    )


def zscore_data_across_time(data: np.ndarray) -> np.ndarray:
    xz = []
    for data_c in data:
        n_trial, n_neuron, n_time = data_c.shape
        xc = data_c.transpose(0, 2, 1).reshape(-1, n_neuron)
        mu = np.nanmean(xc, axis=0, keepdims=True)
        sd = np.nanstd(xc, axis=0, ddof=1, keepdims=True) + 1e-12
        xzc = (xc - mu) / sd
        xz.append(xzc.reshape(n_trial, n_time, n_neuron).transpose(0, 2, 1))
    return xz


def extract_stim_resp_pos(
    data: List[np.ndarray],
    sample_onset: List[np.ndarray],
    stim_len_steps: int,
) -> List[np.ndarray]:
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
    data: List[np.ndarray],
    sample_onset: List[np.ndarray],
    stim_len_steps: int,
    pre_len_steps: int,
    pad_short: bool = True,
) -> List[np.ndarray]:
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
                t_next = int(sample_onset[c][tr, k + 1])
                t0_true = t_k + stim_len_steps
                t1_true = t_next
                if t0_true < 0 or t1_true > n_time or t1_true <= t0_true:
                    continue

                t1 = t1_true
                t0 = max(t1 - pre_len_steps, t0_true)
                seg = data[c][tr, :, t0:t1]
                if seg.shape[1] == 0:
                    continue

                if pad_short and seg.shape[1] < pre_len_steps:
                    pad = np.full((n_neuron, pre_len_steps - seg.shape[1]), np.nan, float)
                    seg = np.concatenate([pad, seg], axis=1)

                out_c[tr, k, :] = np.nanmean(seg, axis=-1)
        out.append(out_c)
    return out


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
    _, n_pos, n_neuron = stim_resp_pos.shape
    eta2 = np.full(n_neuron, np.nan, float)
    p = np.full(n_neuron, np.nan, float)
    for n in range(n_neuron):
        groups = [stim_resp_pos[:, k, n] for k in range(n_pos)]
        eta2[n], p[n] = _eta2_from_groups(groups)
    return eta2, p


def eta2_interval_index_per_neuron(int_resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    int_resp = np.asarray(int_resp, float)
    _, n_int, n_neuron = int_resp.shape
    eta2 = np.full(n_neuron, np.nan, float)
    p = np.full(n_neuron, np.nan, float)
    for n in range(n_neuron):
        groups = [int_resp[:, k, n] for k in range(n_int)]
        eta2[n], p[n] = _eta2_from_groups(groups)
    return eta2, p


def eta2_identity_per_neuron(
    stim_resp_pos: np.ndarray,
    sample_id: np.ndarray,
    residualize_by_pos: bool = True,
    n_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stim_resp_pos = np.asarray(stim_resp_pos, float)
    sample_id = np.asarray(sample_id, int)
    _, _, n_neuron = stim_resp_pos.shape

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


@dataclass
class SelectivityConfig:
    stim_len_steps: int = 10
    zscore: bool = True
    pre_len_steps: int = 10

    do_identity: bool = True
    residualize_id_by_pos: bool = True

    normal_alpha: float = 0.05
    use_shapiro: bool = True

    test_fdr_scope: FDRScope = "global"

    sel_eta2_thr: float = 0.05
    sel_p_alpha: float = 0.01
    sel_use_fdr: bool = True

    out_dir: Optional[str] = None
    save_dpi: int = 300
    fontsize: int = 9


def compute_eta2_for_trial_batch(
    data: List[np.ndarray],
    sample_onset: List[np.ndarray],
    sample_id: List[np.ndarray],
    cfg: SelectivityConfig,
    n_id: Optional[int] = 6,
):
    set_style(cfg.fontsize)

    data = [data]
    sample_onset = [sample_onset]
    sample_id = [sample_id]

    data_use = zscore_data_across_time(data) if cfg.zscore else data

    stim_resp_pos = extract_stim_resp_pos(data_use, sample_onset, cfg.stim_len_steps)
    int_resp_pre = extract_interval_resp_pre(
        data_use,
        sample_onset,
        cfg.stim_len_steps,
        cfg.pre_len_steps,
        pad_short=True,
    )

    eta_stim_pos, p_stim_pos = eta2_stim_position_per_neuron(stim_resp_pos[0])
    eta_int_idx, p_int_idx = eta2_interval_index_per_neuron(int_resp_pre[0])

    windows = [
        ("Stim-window POSITION eta2", eta_stim_pos, p_stim_pos),
        ("Interval-window INDEX eta2 (pre)", eta_int_idx, p_int_idx),
    ]

    if cfg.do_identity:
        eta_stim_id, p_stim_id = eta2_identity_per_neuron(
            stim_resp_pos[0],
            sample_id=sample_id[0],
            residualize_by_pos=cfg.residualize_id_by_pos,
            n_id=n_id,
        )
        tag = (
            "Stim-window IDENTITY eta2"
            if cfg.residualize_id_by_pos
            else "Stim-window IDENTITY eta2 (raw)"
        )
        windows.append((tag, eta_stim_id, p_stim_id))

    return windows
