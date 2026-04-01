#!/usr/bin/env python3
"""
experiment_controllers.py  —  Compare N IBVS control laws comprehensively.

Usage
-----
Two controllers, one run each (backwards-compatible):
    python experiment_controllers.py ctrl1.csv ctrl2.csv

Multiple runs per controller (use -- as group separator):
    python experiment_controllers.py sh1.csv sh2.csv sh3.csv -- ph1.csv ph2.csv ph3.csv

Three or more controller groups:
    python experiment_controllers.py a1.csv a2.csv -- b1.csv b2.csv -- c1.csv c2.csv

Single file (view only):
    python experiment_controllers.py run.csv

Rules
-----
  • '--' separates controller groups; files within a group = repeated runs
  • Two bare files (no '--') → two single-run groups (backwards-compatible)
  • One file, no '--' → single group, single run

All plots are windowed: starts at first detection, stops at 45 s OR when the
bottle is not detected for MAX_LOSS_PLOT consecutive frames (whichever first).

Figures saved to ctrl_comparison/ beside the first CSV
-------------------------------------------------------
  fig1_feature_convergence.png  — μ, ℓ vs time (overlaid)
  fig2_feature_errors.png       — e_μ, e_ℓ errors (overlaid)
  fig3_body_velocities.png      — vx, vy, ωz body velocities (overlaid)
  fig4_wheel_rpm.png            — per-wheel RPM tracking  [if logged]
  fig5_velocity_energy.png      — ‖ξ_b(t)‖² instantaneous + E(t)=∫‖ξ_b‖²dt cumulative
  fig6_error_norm.png           — ‖e‖ norm (overlaid)
  fig7_feature_trajectory.png   — μ × ℓ phase portrait
  fig8_combined_table.png       — controller parameters + performance metrics table
  fig10_hamiltonian.png         — H_d(t) Lyapunov function + ΔH_d passivity certificate
"""
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Allow ibvs_utils to live in the same dir OR one level up (Downloads/)
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ibvs_utils import (
    load_log, col, time_from_detection,
    mae, apply_style,
)

# ── Convergence thresholds ────────────────────────────────────────────────────
CONV_THRESH_MU   = 0.01   # |e_μ| band for settling time  [–]
CONV_THRESH_ELL  = 0.01   # |e_ℓ| band for settling time  [–]
SUSTAINED_FRAMES = 5      # consecutive frames within band
RISE_FRAC        = 0.10   # rise time: first time |e| < 10 % of peak |e|

# ── Plot time window ──────────────────────────────────────────────────────────
MAX_TIME_PLOT = 45.0   # seconds after first detection
MAX_LOSS_PLOT = 20     # consecutive lost-detection frames → end window

# ── Wheel geometry defaults ───────────────────────────────────────────────────
WHEEL_RADIUS = 0.060   # m
HALF_LENGTH  = 0.093   # m
HALF_WIDTH   = 0.087   # m
WHEEL_IDXS   = [1, 2, 3, 4]
WHEEL_LABELS = ['FL (W1)', 'BL (W2)', 'FR (W3)', 'BR (W4)']
WHEEL_COLORS = ['#E63946', '#F1C40F', '#2ECC71', '#9B59B6']

# Colour palette — up to 8 groups; cycles if more are provided
CTRL_COLORS = [
    '#1f77b4',   # blue
    '#d62728',   # red
    '#2ca02c',   # green
    '#ff7f0e',   # orange
    '#9467bd',   # purple
    '#8c564b',   # brown
    '#17becf',   # cyan
    '#e377c2',   # pink
]
LSTYLES = ['-', '--', '-.', ':']       # cycles for B&W printing

# ── Gain/parameter keys shown in the parameters table (fig 8) ────────────────
GAIN_KEYS = [
    ('K_MU',       'K_μ  (proportional)'),
    ('K_ELL',      'K_ℓ  (proportional)'),
    ('KI_MU',      'K_i,μ  (integral)'),
    ('KI_ELL',     'K_i,ℓ  (integral)'),
    ('KD_MU',      'K_d,μ  (damping)'),
    ('KD_ELL',     'K_d,ℓ  (damping)'),
    ('LEAK',       'λ_leak  (leaky integrator)'),
    ('LAMBDA_DLS',      'λ_DLS  (pinv regularisation)'),
    ('K_AW',            'K_aw  (anti-windup)'),
    ('ENABLE_INTEGRAL', 'Integral enabled  (1/0)'),
    ('MAX_INT_U',  'Max integrator μ'),
    ('MAX_INT_ELL','Max integrator ℓ'),
    ('VX_MAX',     'v_x,max  [m/s]'),
    ('WZ_MAX',     'ω_z,max  [rad/s]'),
    ('ELL_DES',    'ℓ_des  (setpoint)'),
    ('MU_DES',     'μ_des  (setpoint)'),
    ('LAMBDA_U',   'λ_u  (camera model)'),
    ('OBJ_HEIGHT', 'H_obj  [m]'),
]


# ── Utility helpers ───────────────────────────────────────────────────────────
def _safe_float(meta, key, default):
    try:
        s = str(meta.get(key, default))
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        return float(m.group(0)) if m else float(default)
    except Exception:
        return float(default)


def _rise_time(t, error, frac=RISE_FRAC):
    """First time |error| < frac * peak_error (peak from first 25 % of data)."""
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = ~np.isnan(t) & ~np.isnan(error)
    t, error = t[valid], error[valid]
    if len(error) == 0:
        return float('nan')
    n_ref = max(1, len(error) // 4)
    ref = float(np.nanmax(np.abs(error[:n_ref])))
    if ref < 1e-9:
        return 0.0
    idx = np.where(np.abs(error) < frac * ref)[0]
    return float(t[idx[0]] - t[0]) if len(idx) else float('nan')


def _rise_val(t, error, frac=RISE_FRAC):
    """Error value at the rise time moment (first time |e| < frac * peak)."""
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = ~np.isnan(t) & ~np.isnan(error)
    t, error = t[valid], error[valid]
    if len(error) == 0:
        return float('nan')
    n_ref = max(1, len(error) // 4)
    ref = float(np.nanmax(np.abs(error[:n_ref])))
    if ref < 1e-9:
        return float(error[0]) if len(error) else float('nan')
    idx = np.where(np.abs(error) < frac * ref)[0]
    return float(error[idx[0]]) if len(idx) else float('nan')


def _settling_time(t, error, threshold, n_frames=SUSTAINED_FRAMES):
    """First time |error| stays < threshold for n_frames consecutive frames."""
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = ~np.isnan(t) & ~np.isnan(error)
    t, error = t[valid], error[valid]
    if len(error) == 0:
        return float('nan')
    within = np.abs(error) < threshold
    for i in range(len(within) - n_frames + 1):
        if np.all(within[i:i + n_frames]):
            return float(t[i] - t[0])
    return float('nan')


def _compute_t_end(t_rel, detected, max_t=MAX_TIME_PLOT, max_lost=MAX_LOSS_PLOT):
    """
    End of the plot window: min(max_t, time of first long detection gap after t=0).
    A 'long gap' is max_lost consecutive rows with detected==0.
    """
    consec = 0
    for i in range(len(detected)):
        if t_rel[i] < 0:
            continue
        if detected[i] == 0:
            consec += 1
            if consec >= max_lost:
                return min(max_t, float(t_rel[i]))
        else:
            consec = 0
    return max_t


def _fwd_kinematics(rpm1, rpm2, rpm3, rpm4, r, l, w):
    """O-type mecanum FK: four RPM arrays → vx, vy, wz."""
    scale = 2.0 * np.pi / 60.0
    u1, u2, u3, u4 = rpm1 * scale, rpm2 * scale, rpm3 * scale, rpm4 * scale
    lw = l + w
    vx = r / 4.0 * (u1 + u2 + u3 + u4)
    vy = r / 4.0 * (-u1 + u2 + u3 - u4)
    wz = r / (4.0 * lw) * (-u1 - u2 + u3 + u4)
    return vx, vy, wz


def _dt_array(t):
    dt = np.diff(t, prepend=t[0])
    dt[0] = float(np.median(dt[1:])) if len(dt) > 2 else (t[1] - t[0] if len(t) > 1 else 1.0)
    return np.clip(dt, 1e-6, 1.0)


# ── Process one log file ──────────────────────────────────────────────────────
def _process(path):
    meta, data = load_log(path)
    t_rel = time_from_detection(data)
    t_end = _compute_t_end(t_rel, data['detected'])

    # Detection-only rows within the time window
    win_mask = (data['detected'] == 1) & (t_rel <= t_end)
    det_t = t_rel[win_mask]
    det   = {k: v[win_mask] for k, v in data.items()}

    if len(det_t) == 0:
        raise RuntimeError(f'No detection frames within window (t_end={t_end:.1f} s)')

    mu_d  = col(det, 'mu')
    ell_d = col(det, 'ell')
    e_mu  = col(det, 'e_mu')
    e_ell = col(det, 'e_ell')

    mu_des_arr = col(det, 'mu_des')
    mu_des = float(np.nanmean(mu_des_arr)) if not np.all(np.isnan(mu_des_arr)) else 0.0
    ell_des_arr = col(det, 'ell_des')
    ell_des = float(np.nanmean(ell_des_arr)) if not np.all(np.isnan(ell_des_arr)) else float('nan')

    if np.all(np.isnan(e_mu)):
        e_mu = mu_d - mu_des
    if np.all(np.isnan(e_ell)):
        e_ell = ell_d - (ell_des if not np.isnan(ell_des) else float(np.nanmean(ell_d)))

    vx_cmd = col(det, 'vx_cmd')
    vy_cmd = col(det, 'vy_cmd')
    wz_cmd = col(det, 'wz_cmd')

    # ── Pixel-space auto-detection ────────────────────────────────────────────
    img_w      = int(_safe_float(meta, 'IMG_WIDTH',  0)) or None
    img_h      = int(_safe_float(meta, 'IMG_HEIGHT', 0)) or None
    is_pixel   = img_w is not None and img_h is not None
    thresh_mu  = CONV_THRESH_MU  * img_w if is_pixel else CONV_THRESH_MU
    thresh_ell = CONV_THRESH_ELL * img_h if is_pixel else CONV_THRESH_ELL
    units_mu   = 'px' if is_pixel else '–'
    units_ell  = 'px' if is_pixel else 'norm'

    rt_mu  = _rise_time(det_t, e_mu)
    rt_ell = _rise_time(det_t, e_ell)
    rv_mu  = _rise_val(det_t, e_mu)
    rv_ell = _rise_val(det_t, e_ell)
    st_mu  = _settling_time(det_t, e_mu,  thresh_mu)
    st_ell = _settling_time(det_t, e_ell, thresh_ell)
    bw_mu  = 0.35 / rt_mu  if not np.isnan(rt_mu)  and rt_mu > 0 else float('nan')
    bw_ell = 0.35 / rt_ell if not np.isnan(rt_ell) and rt_ell > 0 else float('nan')
    e_norm = np.sqrt(e_mu ** 2 + e_ell ** 2)

    def _ss_mae(t, error, t_settle):
        """MAE computed only over samples after the settling time."""
        if np.isnan(t_settle):
            return float('nan')
        mask = (t - t[0]) >= t_settle
        return mae(error[mask]) if np.any(mask) else float('nan')

    ss_mae_mu  = _ss_mae(det_t, e_mu,  st_mu)
    ss_mae_ell = _ss_mae(det_t, e_ell, st_ell)

    has_wheel = all(
        f'rpm{i}_t' in det and f'rpm{i}' in det and f'duty{i}' in det
        for i in WHEEL_IDXS
    )
    l2_energy  = float('nan')
    l2_instant = np.full(len(det_t), np.nan)
    l2_cumul   = np.full(len(det_t), np.nan)
    wheel_rmse = [float('nan')] * 4
    wheel_tgt  = [np.full(len(det_t), np.nan)] * 4
    wheel_act  = [np.full(len(det_t), np.nan)] * 4
    wheel_duty = [np.full(len(det_t), np.nan)] * 4
    vx_act = np.full(len(det_t), np.nan)
    vy_act = np.full(len(det_t), np.nan)
    wz_act = np.full(len(det_t), np.nan)

    if has_wheel:
        r_w = _safe_float(meta, 'WHEEL_RADIUS', WHEEL_RADIUS)
        l_w = _safe_float(meta, 'HALF_LENGTH',  HALF_LENGTH)
        w_w = _safe_float(meta, 'HALF_WIDTH',   HALF_WIDTH)
        for j, i in enumerate(WHEEL_IDXS):
            wheel_tgt[j]  = col(det, f'rpm{i}_t')
            wheel_act[j]  = col(det, f'rpm{i}')
            wheel_duty[j] = col(det, f'duty{i}')
            wheel_rmse[j] = float(np.sqrt(np.nanmean((wheel_act[j] - wheel_tgt[j]) ** 2)))
        duties     = np.column_stack(wheel_duty)
        l2_instant = np.nanmean(duties ** 2, axis=1)
        dt         = _dt_array(det_t)
        l2_cumul   = np.cumsum(l2_instant * dt)
        l2_energy  = float(l2_cumul[-1])
        vx_act, vy_act, wz_act = _fwd_kinematics(
            wheel_act[0], wheel_act[1], wheel_act[2], wheel_act[3], r_w, l_w, w_w)

    # ── Velocity energy / actuation cost ─────────────────────────────────────
    # Use actual FK body velocities so the cost reflects true robot motion,
    # independent of VX_MAX / WZ_MAX / MAX_RPM saturation settings.
    # Falls back to commanded velocities only when wheel data is absent.
    dt_arr  = _dt_array(det_t)
    if has_wheel:
        xi_b_sq = (np.nan_to_num(vx_act) ** 2
                   + np.nan_to_num(vy_act) ** 2
                   + np.nan_to_num(wz_act) ** 2)
    else:
        xi_b_sq = (np.nan_to_num(vx_cmd) ** 2
                   + np.nan_to_num(vy_cmd) ** 2
                   + np.nan_to_num(wz_cmd) ** 2)
    E_cumul = np.cumsum(xi_b_sq * dt_arr)
    E_total = float(E_cumul[-1]) if len(E_cumul) > 0 else float('nan')

    # ── H_d(t) — closed-loop Hamiltonian ─────────────────────────────────────
    # Priority order for H_d:
    #   1. PH controllers log H/dH directly         → most accurate
    #   2. IBVS pixel logs z_mu/z_ell (integral)   → accurate (uses real clamped z)
    #   3. Legacy SH — no z logged                  → numerically reconstructed
    H_val      = col(det, 'H')
    dH_val     = col(det, 'dH')
    z_mu_col   = col(det, 'z_mu')
    z_ell_col  = col(det, 'z_ell')
    if not np.all(np.isnan(H_val)):
        # Case 1: PH — H logged directly
        H_d     = H_val
        dH_d    = dH_val
        h_source = 'direct'
    else:
        kp_mu  = _safe_float(meta, 'K_MU',  1.0)
        kp_ell = _safe_float(meta, 'K_ELL', 0.28)
        ki_mu  = _safe_float(meta, 'KI_MU',  0.1)
        ki_ell = _safe_float(meta, 'KI_ELL', 0.001)
        if not np.all(np.isnan(z_mu_col)):
            # Case 2: IBVS px — integral state logged directly
            z_mu_r  = np.nan_to_num(z_mu_col)
            z_ell_r = np.nan_to_num(z_ell_col)
            h_source = 'logged_z'
        else:
            # Case 3: legacy SH CSV — reconstruct z by forward Euler integration
            e_mu_c  = np.nan_to_num(e_mu)
            e_ell_c = np.nan_to_num(e_ell)
            z_mu_r  = np.zeros(len(det_t))
            z_ell_r = np.zeros(len(det_t))
            for _i in range(1, len(det_t)):
                z_mu_r[_i]  = z_mu_r[_i-1]  + e_mu_c[_i-1]  * dt_arr[_i-1]
                z_ell_r[_i] = z_ell_r[_i-1] + e_ell_c[_i-1] * dt_arr[_i-1]
            h_source = 'reconstructed'
        H_d_P = 0.5 * (kp_mu * e_mu ** 2 + kp_ell * e_ell ** 2)
        H_d_I = (0.5 * (z_mu_r  ** 2 / ki_mu  if ki_mu  > 1e-9 else np.zeros(len(det_t)))
               + 0.5 * (z_ell_r ** 2 / ki_ell if ki_ell > 1e-9 else np.zeros(len(det_t))))
        H_d  = H_d_P + H_d_I
        dH_d = np.diff(H_d, prepend=H_d[0])

    ctrl_name = meta.get('controller', os.path.basename(path))

    return {
        'ctrl_name':  ctrl_name,
        'meta':       meta,
        'det_t':      det_t,
        't_end':      t_end,
        'mu':         mu_d,
        'ell':        ell_d,
        'e_mu':       e_mu,
        'e_ell':      e_ell,
        'e_norm':     e_norm,
        'mu_des':     mu_des,
        'ell_des':    ell_des,
        'vx_cmd':     vx_cmd,
        'vy_cmd':     vy_cmd,
        'wz_cmd':     wz_cmd,
        'vx_act':     vx_act,
        'vy_act':     vy_act,
        'wz_act':     wz_act,
        'rt_mu':      rt_mu,
        'rt_ell':     rt_ell,
        'rv_mu':      rv_mu,
        'rv_ell':     rv_ell,
        'st_mu':      st_mu,
        'st_ell':     st_ell,
        'bw_mu':      bw_mu,
        'bw_ell':     bw_ell,
        'mae_mu':        mae(e_mu),
        'mae_ell':       mae(e_ell),
        'ss_mae_mu':     ss_mae_mu,
        'ss_mae_ell':    ss_mae_ell,
        'thresh_mu':     thresh_mu,
        'thresh_ell':    thresh_ell,
        'units_mu':      units_mu,
        'units_ell':     units_ell,
        'max_wz':     float(np.nanmax(np.abs(wz_act))) if has_wheel else float(np.nanmax(np.abs(wz_cmd))),
        'max_vx':     float(np.nanmax(np.abs(vx_act))) if has_wheel else float(np.nanmax(np.abs(vx_cmd))),
        'l2_energy':  l2_energy,
        'l2_instant': l2_instant,
        'l2_cumul':   l2_cumul,
        'wheel_rmse': wheel_rmse,
        'wheel_tgt':  wheel_tgt,
        'wheel_act':  wheel_act,
        'wheel_duty': wheel_duty,
        'has_wheel':  has_wheel,
        'n_det':      int(np.sum(data['detected'] == 1)),
        'duration':   float(det_t[-1] - det_t[0]) if len(det_t) > 1 else float('nan'),
        'xi_b_sq':    xi_b_sq,
        'E_cumul':    E_cumul,
        'E_total':    E_total,
        'H_val':      H_val,      # raw logged column (NaN for non-PH)
        'H_d':        H_d,       # PH: direct; IBVS-px: from logged z; SH: reconstructed
        'dH_d':       dH_d,
        'h_source':   h_source,  # 'direct' | 'logged_z' | 'reconstructed'
    }


# ── Plot helper: draw all runs in a group on one axis ────────────────────────
def _plot_runs(ax, group, key, j, label=None, **extra):
    """
    Plot `r[key]` vs `r['det_t']` for every run in `group`.
    Multiple runs are drawn thin+semi-transparent; single run is full opacity.
    """
    c     = CTRL_COLORS[j % len(CTRL_COLORS)]
    ls    = LSTYLES[j % len(LSTYLES)]
    n     = len(group)
    alpha = extra.pop('alpha', 0.45 if n > 1 else 1.0)
    lw    = extra.pop('linewidth', 1.0 if n > 1 else 1.8)

    for k, r in enumerate(group):
        lbl = (label or r['ctrl_name']) if k == 0 else '_nolegend_'
        ax.plot(r['det_t'], r[key],
                color=c, linewidth=lw, linestyle=ls, alpha=alpha,
                label=lbl, **extra)


# ── Table data builders ───────────────────────────────────────────────────────
def _gains_table(groups):
    """
    Returns (row_labels, col_labels, cell_text) for the gains/parameters table.
    Only rows where at least one controller has the key are included.
    """
    col_labels = [g[0]['ctrl_name'] for g in groups]
    rows = []
    for key, label in GAIN_KEYS:
        vals = [g[0]['meta'].get(key, None) for g in groups]
        if any(v is not None for v in vals):
            rows.append((label, [str(v) if v is not None else '—' for v in vals]))
    return [r[0] for r in rows], col_labels, [r[1] for r in rows]


def _metric_str(group, key, decimals=3):
    """'value' or 'mean ± std' across runs (skips NaN)."""
    vals = [r[key] for r in group if not (isinstance(r[key], float) and np.isnan(r[key]))]
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.{decimals}f}'
    return f'{np.mean(vals):.{decimals}f} ± {np.std(vals):.{decimals}f}'


def _metrics_table(groups):
    """Returns (row_labels, col_labels, cell_text) for the metrics table."""
    col_labels = [g[0]['ctrl_name'] for g in groups]

    rows_spec = [
        ('det',   'Det. frames (avg)',    lambda g: str(int(np.mean([r['n_det'] for r in g])))),
        ('dur',   'Duration [s]',         lambda g: _metric_str(g, 'duration')),
        (None,    '─── μ ───',            lambda _: ''),
        ('rt_mu', 'T_r(μ) [s]',           lambda g: _metric_str(g, 'rt_mu')),
        ('rv_mu', 'e_μ at T_r',           lambda g: _metric_str(g, 'rv_mu')),
        ('st_mu', 'T_s(μ) [s]',           lambda g: _metric_str(g, 'st_mu')),
        ('bw_mu', 'BW(μ) ≈ 0.35/T_r [Hz]', lambda g: _metric_str(g, 'bw_mu')),
        ('m_mu',   'MAE(μ)',               lambda g: _metric_str(g, 'mae_mu')),
        ('ssm_mu', 'SS-MAE(μ)  [t≥T_s]', lambda g: _metric_str(g, 'ss_mae_mu')),
        (None,    '─── ℓ ───',            lambda _: ''),
        ('rt_el', 'T_r(ℓ) [s]',           lambda g: _metric_str(g, 'rt_ell')),
        ('rv_el', 'e_ℓ at T_r',           lambda g: _metric_str(g, 'rv_ell')),
        ('st_el', 'T_s(ℓ) [s]',           lambda g: _metric_str(g, 'st_ell')),
        ('bw_el', 'BW(ℓ) ≈ 0.35/T_r [Hz]', lambda g: _metric_str(g, 'bw_ell')),
        ('m_el',  'MAE(ℓ)',               lambda g: _metric_str(g, 'mae_ell')),
        ('ssm_el','SS-MAE(ℓ)  [t≥T_s]',  lambda g: _metric_str(g, 'ss_mae_ell')),
        (None,    '─── velocities / actuation cost ───', lambda _: ''),
        ('mwz',   'Max |ωz| [rad/s]',                  lambda g: _metric_str(g, 'max_wz')),
        ('mvx',   'Max |vx| [m/s]',                    lambda g: _metric_str(g, 'max_vx')),
        ('Etot',  'E = int||xi_b||^2 dt  [(m/s)^2 s]',  lambda g: _metric_str(g, 'E_total')),
    ]
    row_labels = [s[1] for s in rows_spec]
    cell_text  = [[s[2](g) for g in groups] for s in rows_spec]
    sep_rows   = {i for i, s in enumerate(rows_spec) if s[0] is None}
    return row_labels, col_labels, cell_text, sep_rows


def _draw_table(fig, row_labels, col_labels, cell_text, title,
                sep_rows=None, col_colors=None):
    """Draw a matplotlib table on a new figure; return the figure."""
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle(title, fontsize=13, fontweight='bold')

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='right',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Colour header row
    for j, _ in enumerate(col_labels):
        clr = (col_colors[j] if col_colors else CTRL_COLORS[j % len(CTRL_COLORS)]) + '55'
        tbl[(0, j)].set_facecolor(clr)
        tbl[(0, j)].set_text_props(fontweight='bold')

    # Light-grey separator rows
    if sep_rows:
        for i in sep_rows:
            for jj in range(-1, len(col_labels)):
                tbl[(i + 1, jj)].set_facecolor('#eeeeee')
                tbl[(i + 1, jj)].set_text_props(color='#666666', style='italic')

    return fig


def _combined_table(groups):
    """Gains + metrics in one table."""
    row_g, col_g, cell_g = _gains_table(groups)
    row_m, col_m, cell_m, sep_m = _metrics_table(groups)
    if not row_g:
        return row_m, col_m, cell_m, sep_m
    n_g = len(row_g)
    sep_row = '─── Performance Metrics ───'
    row_labels = row_g + [sep_row] + row_m
    cell_text  = cell_g + [[''] * len(groups)] + cell_m
    sep_rows   = {n_g} | {n_g + 1 + i for i in sep_m}
    return row_labels, col_g, cell_text, sep_rows


# ── Group parser ──────────────────────────────────────────────────────────────
def _parse_groups(paths):
    """
    Split a flat list of paths on '--' separators.

    Rules
    -----
    '--' present   → one group per segment (N groups)
    2 bare files   → two single-run groups  (backwards-compatible)
    1 bare file    → one group, one run
    3+ bare files  → error (ambiguous; require '--' to separate groups)
    """
    if '--' in paths:
        groups, current = [], []
        for p in paths:
            if p == '--':
                if current:
                    groups.append(current)
                current = []
            else:
                current.append(p)
        if current:
            groups.append(current)
        empty = [i for i, g in enumerate(groups) if not g]
        if empty:
            print(f'ERROR: empty group(s) at position(s) {empty} — '
                  'check for consecutive -- separators.')
            sys.exit(1)
        return groups
    elif len(paths) == 1:
        return [paths]
    elif len(paths) == 2:
        return [[paths[0]], [paths[1]]]   # backwards-compatible
    else:
        print('ERROR: pass exactly 1 or 2 CSV files, OR use -- to separate groups.\n'
              'Example (3 SH runs vs 3 PH runs):\n'
              '  python experiment_controllers.py sh1.csv sh2.csv sh3.csv '
              '-- ph1.csv ph2.csv ph3.csv')
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(paths):
    group_paths = _parse_groups(paths)

    apply_style()

    groups = []
    for gp in group_paths:
        g = []
        for path in gp:
            try:
                r = _process(path)
                g.append(r)
                print(f'  Loaded  {os.path.basename(path):45s}'
                      f'→ {r["ctrl_name"]}  '
                      f'({len(r["det_t"])} frames, window {r["t_end"]:.1f} s)')
            except Exception as e:
                print(f'  [error] {path}: {e}')
        if g:
            groups.append(g)

    if not groups:
        sys.exit(1)

    n_groups = len(groups)
    out_dir = Path(paths[0]).parent / 'ctrl_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f'  Saved → {p.name}')
        plt.close(fig)

    print(f'\nGenerating 9 figures ...')

    # ── Fig 1: Feature convergence — μ and ℓ, overlaid ───────────────────────
    fig1, (ax_mu, ax_ell) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for j, group in enumerate(groups):
        _plot_runs(ax_mu,  group, 'mu',  j)
        _plot_runs(ax_ell, group, 'ell', j)

    _r0    = groups[0][0]
    mu_des = _r0['mu_des']
    ax_mu.axhline(mu_des, color='gray', linestyle='--', linewidth=1.0,
                  label=f'μ_des = {mu_des:.2f}')
    ax_mu.axhspan(mu_des - _r0['thresh_mu'], mu_des + _r0['thresh_mu'],
                  alpha=0.10, color='green', zorder=0, label=f'±{_r0["thresh_mu"]:.3g} band')

    ell_des_vals = [g[0]['ell_des'] for g in groups if not np.isnan(g[0]['ell_des'])]
    if ell_des_vals:
        ell_des = ell_des_vals[0]
        ax_ell.axhline(ell_des, color='gray', linestyle='--', linewidth=1.0,
                       label='_nolegend_')
        ax_ell.axhspan(ell_des - _r0['thresh_ell'], ell_des + _r0['thresh_ell'],
                       alpha=0.10, color='green', zorder=0, label='_nolegend_')

    for ax, ylbl in [
        (ax_mu,  f'μ  [{_r0["units_mu"]}]'),
        (ax_ell, f'ℓ  [{_r0["units_ell"]}]'),
    ]:
        ax.set_ylabel(ylbl)
        ax.set_xlabel('Time since first detection [s]')
        ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = ax_mu.get_legend_handles_labels()
    fig1.legend(handles, labels, fontsize=8, loc='upper right',
                bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig1, 'fig1_feature_convergence.png')

    # ── Fig 2: Feature errors — overlaid, NO settling/rise markers ───────────
    fig2, (ax_emu, ax_eell) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for j, group in enumerate(groups):
        _plot_runs(ax_emu,  group, 'e_mu',  j)
        _plot_runs(ax_eell, group, 'e_ell', j)

    _r0 = groups[0][0]
    for k, (ax, thresh, ylbl) in enumerate([
        (ax_emu,  _r0['thresh_mu'],  f'e_μ  [{_r0["units_mu"]}]'),
        (ax_eell, _r0['thresh_ell'], f'e_ℓ  [{_r0["units_ell"]}]'),
    ]):
        ax.axhline(0, color='k', linewidth=0.7)
        ax.axhspan(-thresh, thresh, alpha=0.10, color='green', zorder=0,
                   label=f'±{thresh:.3g} conv. band' if k == 0 else '_nolegend_')
        ax.set_ylabel(ylbl)
        ax.set_xlabel('Time since first detection [s]')
        ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = ax_emu.get_legend_handles_labels()
    fig2.legend(handles, labels, fontsize=8, loc='upper right',
                bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig2, 'fig2_feature_errors.png')

    # ── Fig 3: Body velocity — vx, vy and ωz OVERLAID ───────────────────────
    fig3, (ax_vx, ax_vy, ax_wz) = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

    for j, group in enumerate(groups):
        c  = CTRL_COLORS[j % len(CTRL_COLORS)]
        ls = LSTYLES[j % len(LSTYLES)]
        n  = len(group)
        alpha = 0.50 if n > 1 else 1.0
        lw    = 1.0  if n > 1 else 1.8

        for k, r in enumerate(group):
            if not r['has_wheel'] or np.all(np.isnan(r['vx_act'])):
                continue
            lbl = r['ctrl_name'] if k == 0 else '_nolegend_'
            ax_vx.plot(r['det_t'], r['vx_act'], color=c, lw=lw, ls=ls,
                       alpha=alpha, label=lbl)
            ax_vy.plot(r['det_t'], r['vy_act'], color=c, lw=lw, ls=ls,
                       alpha=alpha, label='_nolegend_')
            ax_wz.plot(r['det_t'], r['wz_act'], color=c, lw=lw, ls=ls,
                       alpha=alpha, label='_nolegend_')

    for ax, ylbl in [
        (ax_vx, 'vx  [m/s]'),
        (ax_vy, 'vy  [m/s]'),
        (ax_wz, 'ωz  [rad/s]'),
    ]:
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_ylabel(ylbl)
        ax.set_xlabel('Time since first detection [s]')
        ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = ax_vx.get_legend_handles_labels()
    fig3.legend(handles, labels, fontsize=8, loc='upper right',
                bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig3, 'fig3_body_velocities.png')

    # ── Fig 4: Per-wheel RPM (one column per controller group) ───────────────
    if any(r['has_wheel'] for g in groups for r in g):
        fig4, axes = plt.subplots(4, n_groups, figsize=(7 * n_groups, 14), squeeze=False)

        for j, group in enumerate(groups):
            n = len(group)
            alpha_r = 0.45 if n > 1 else 1.0
            lw_r    = 1.0  if n > 1 else 1.5

            for row, (lbl, wc) in enumerate(zip(WHEEL_LABELS, WHEEL_COLORS)):
                ax = axes[row, j]
                for k, r in enumerate(group):
                    if not r['has_wheel']:
                        continue
                    ax.plot(r['det_t'], r['wheel_act'][row], '-',
                            color=wc, lw=lw_r, alpha=alpha_r,
                            label=r['ctrl_name'] if k == 0 else '_nolegend_')

                ax.set_title(lbl, fontsize=9)
                ax.set_ylabel('RPM')
                ax.grid(True, linestyle=':', alpha=0.5)
                if row == 3:
                    ax.set_xlabel('Time since first detection [s]')

        all_handles, all_labels = [], []
        for j in range(n_groups):
            h, l = axes[0, j].get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
        fig4.legend(all_handles, all_labels, fontsize=7, loc='upper right',
                    bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
        plt.tight_layout()
        _save(fig4, 'fig4_wheel_rpm.png')

    # ── Fig 5: Velocity energy — ‖ξ_b(t)‖² and E(t) = ∫‖ξ_b‖² dt ─────────
    fig5, (ax_xi, ax_E) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for j, group in enumerate(groups):
        _plot_runs(ax_xi, group, 'xi_b_sq', j)
        _plot_runs(ax_E,  group, 'E_cumul', j)

    for ax, ylbl in [
        (ax_xi, '||xi_b||^2  [(m/s)^2 + (rad/s)^2]'),
        (ax_E,  'E(t)  [(m/s)^2 s]'),
    ]:
        ax.set_ylabel(ylbl)
        ax.set_xlabel('Time since first detection [s]')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = ax_xi.get_legend_handles_labels()
    fig5.legend(handles, labels, fontsize=8, loc='upper right',
                bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig5, 'fig5_velocity_energy.png')

    # ── Fig 6: Error norm — OVERLAID, no settling lines ──────────────────────
    _r0 = groups[0][0]
    thresh_norm = float(np.sqrt(_r0['thresh_mu'] ** 2 + _r0['thresh_ell'] ** 2))
    fig6, ax6 = plt.subplots(figsize=(10, 5))

    for j, group in enumerate(groups):
        _plot_runs(ax6, group, 'e_norm', j)

    ax6.axhline(thresh_norm, color='green', linestyle='--', linewidth=1.2,
                label=f'Threshold {thresh_norm:.3f}')
    ax6.axhspan(0, thresh_norm, alpha=0.07, color='green', zorder=0)
    ax6.set_xlabel('Time since first detection [s]')
    ax6.set_ylabel('||e||')
    ax6.set_ylim(bottom=0)
    ax6.legend(fontsize=8)
    ax6.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _save(fig6, 'fig6_error_norm.png')

    # ── Fig 7: Feature-space trajectory (μ × ℓ) ──────────────────────────────
    fig7, ax7 = plt.subplots(figsize=(8, 7))

    for j, group in enumerate(groups):
        c  = CTRL_COLORS[j % len(CTRL_COLORS)]
        ls = LSTYLES[j % len(LSTYLES)]
        n  = len(group)
        alpha = 0.50 if n > 1 else 0.85

        for k, r in enumerate(group):
            valid = ~np.isnan(r['mu']) & ~np.isnan(r['ell'])
            mu_v, ell_v = r['mu'][valid], r['ell'][valid]
            if len(mu_v) == 0:
                continue
            lbl = r['ctrl_name'] if k == 0 else '_nolegend_'
            ax7.plot(mu_v, ell_v, color=c, lw=1.2, ls=ls, alpha=alpha, label=lbl)

            # Direction arrows only on first run
            if k == 0:
                for frac in (0.33, 0.66):
                    idx = int(frac * len(mu_v))
                    if idx + 1 < len(mu_v):
                        ax7.annotate('',
                                     xy=(mu_v[idx + 1], ell_v[idx + 1]),
                                     xytext=(mu_v[idx], ell_v[idx]),
                                     arrowprops=dict(arrowstyle='->', color=c, lw=1.6))

            ax7.scatter(mu_v[0],  ell_v[0],  marker='o', s=70, color=c, zorder=5,
                        edgecolors='black', lw=0.6,
                        label='Start' if k == 0 else '_nolegend_')
            ax7.scatter(mu_v[-1], ell_v[-1], marker='X', s=70, color=c, zorder=5,
                        edgecolors='black', lw=0.6,
                        label='End' if k == 0 else '_nolegend_')

    sp_mu = groups[0][0]['mu_des']
    sp_ell_vals = [g[0]['ell_des'] for g in groups if not np.isnan(g[0]['ell_des'])]
    if sp_ell_vals:
        sp_ell = sp_ell_vals[0]
        ax7.scatter(sp_mu, sp_ell, marker='*', s=350, color='gold',
                    edgecolors='black', lw=0.8, zorder=6, label='Setpoint')
        ax7.add_patch(mpatches.Ellipse(
            (sp_mu, sp_ell), width=2 * CONV_THRESH_MU, height=2 * CONV_THRESH_ELL,
            fill=True, alpha=0.15, color='green', zorder=3))
        ax7.add_patch(mpatches.Ellipse(
            (sp_mu, sp_ell), width=2 * CONV_THRESH_MU, height=2 * CONV_THRESH_ELL,
            fill=False, edgecolor='green', linestyle='--', lw=1.2, zorder=4))

    ax7.set_xlabel('μ  [–]', fontsize=11)
    ax7.set_ylabel('ℓ  [norm]', fontsize=11)
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _save(fig7, 'fig7_feature_trajectory.png')

    # ── Fig 8: Combined parameters + metrics table ───────────────────────────
    row_c, col_c, cell_c, sep_c = _combined_table(groups)
    fig8_h = max(5.0, 0.38 * len(row_c) + 1.5)
    fig8 = plt.figure(figsize=(4 + 3.5 * n_groups, fig8_h))
    _draw_table(fig8, row_c, col_c, cell_c,
                'Parameters & Performance Metrics  (mean ± std for multiple runs)',
                sep_rows=sep_c)
    plt.tight_layout()
    _save(fig8, 'fig8_combined_table.png')

    # ── Fig 10: H_d(t) — passivity verification (primary energy metric) ───────
    # H_d(t) = ½ e^T Kp e + ½ z^T Ki⁻¹ z  (Muñoz-Arias et al., eq. 30)
    # PH guarantee (Theorem 1, Kd > 0): Ḣ_d ≤ 0 always.
    # SH law: H_d reconstructed numerically — may be non-monotonic.
    fig10, (ax_H, ax_dH) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for j, group in enumerate(groups):
        c  = CTRL_COLORS[j % len(CTRL_COLORS)]
        ls = LSTYLES[j % len(LSTYLES)]
        n  = len(group)
        alpha = 0.45 if n > 1 else 1.0
        lw    = 1.0  if n > 1 else 1.8

        for k, r in enumerate(group):
            src = r.get('h_source', 'reconstructed')
            tag = {'direct': '', 'logged_z': '  (from logged z)', 'reconstructed': '  (reconstructed)'}.get(src, '')
            lbl = (r['ctrl_name'] + tag) if k == 0 else '_nolegend_'
            ax_H.plot(r['det_t'],  r['H_d'],  color=c, lw=lw, ls=ls,
                      alpha=alpha, label=lbl)
            ax_dH.plot(r['det_t'], r['dH_d'], color=c, lw=lw, ls=ls,
                       alpha=alpha, label=lbl if k == 0 else '_nolegend_')

    ax_H.set_ylabel('$H_d(t)$  [–]')
    ax_H.set_xlabel('Time since first detection [s]')
    ax_H.set_ylim(bottom=0)
    ax_H.grid(True, linestyle=':', alpha=0.5)

    ax_dH.axhline(0, color='k', linewidth=1.0, zorder=3, label='$\\Delta H_d = 0$')
    ax_dH.axhspan(-1e9, 0, alpha=0.08, color='green', zorder=0,
                  label='$\\Delta H_d \\leq 0$  (passive)')
    ax_dH.set_ylabel('$\\Delta H_d$ per frame  [–]')
    ax_dH.set_xlabel('Time since first detection [s]')
    ax_dH.grid(True, linestyle=':', alpha=0.5)

    h1, l1 = ax_H.get_legend_handles_labels()
    h2, l2 = ax_dH.get_legend_handles_labels()
    fig10.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right',
                 bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig10, 'fig10_hamiltonian.png')

    # ── Console summary ───────────────────────────────────────────────────────
    W   = 24
    SEP = '=' * (28 + W * n_groups)
    sep = '-' * (28 + W * n_groups)

    def _row(label, vals):
        print(f"  {label:<26}" + ''.join(f"  {v:>{W}}" for v in vals))

    print('\n' + SEP)
    print(f"  {'Metric':<26}" +
          ''.join(f"  {g[0]['ctrl_name'][:W]:>{W}}" for g in groups))
    print(SEP)
    _row('Runs',              [str(len(g)) for g in groups])
    _row('Det. frames (avg)', [str(int(np.mean([r['n_det'] for r in g]))) for g in groups])
    _row('Duration [s]',      [_metric_str(g, 'duration') for g in groups])
    print(sep)
    _row('T_r(μ) [s]',        [_metric_str(g, 'rt_mu')  for g in groups])
    _row('T_s(μ) [s]',        [_metric_str(g, 'st_mu')  for g in groups])
    _row('BW(μ) [Hz]',        [_metric_str(g, 'bw_mu')  for g in groups])
    _row('MAE(μ)',             [_metric_str(g, 'mae_mu')     for g in groups])
    _row('SS-MAE(μ) [t≥T_s]', [_metric_str(g, 'ss_mae_mu') for g in groups])
    print(sep)
    _row('T_r(ℓ) [s]',        [_metric_str(g, 'rt_ell')    for g in groups])
    _row('T_s(ℓ) [s]',        [_metric_str(g, 'st_ell')    for g in groups])
    _row('BW(ℓ) [Hz]',        [_metric_str(g, 'bw_ell')    for g in groups])
    _row('MAE(ℓ)',             [_metric_str(g, 'mae_ell')   for g in groups])
    _row('SS-MAE(ℓ) [t≥T_s]', [_metric_str(g, 'ss_mae_ell') for g in groups])
    print(sep)
    _row('Max |ωz| [rad/s]',  [_metric_str(g, 'max_wz')    for g in groups])
    _row('Max |vx| [m/s]',    [_metric_str(g, 'max_vx')    for g in groups])
    _row('L² energy [%²·s]',  [_metric_str(g, 'l2_energy') for g in groups])
    print(SEP + '\n')
    print(f'  Figures saved to: {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
