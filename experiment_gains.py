#!/usr/bin/env python3
"""
experiment_gains.py  —  Compare multiple gain configurations (Kp, Ki, PH terms).

Usage
-----
    python3 experiment_gains.py run1.csv run2.csv run3.csv [...]

All plots are windowed: starts at first detection, stops at 45 s OR when the
bottle is not detected for MAX_LOSS_PLOT consecutive frames (same as the other
analysis scripts).

Figures saved to gains_comparison/ beside the first CSV
--------------------------------------------------------
  fig1_error_convergence.png  — μ and ℓ error vs time (all runs overlaid)
  fig2_combined_table.png     — controller parameters + performance metrics (combined table)

Gain keys read from CSV headers (explicit form preferred):
    # K_MU=0.5      # K_ELL=1.0
    # KI_MU=0.01    # KI_ELL=0.4
Also parses compact forms:  # Kp=diag(0.5,1.0)  # Ki=diag(0.01,0.4)
Port-Hamiltonian variables:  KD_MU, KD_ELL, LEAK, LAMBDA_DLS, K_AW
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

# ── Time window (mirrors experiment_controllers.py) ───────────────────────────
MAX_TIME_PLOT = 45.0   # seconds after first detection
MAX_LOSS_PLOT = 20     # consecutive lost-detection frames → end window

# ── Convergence thresholds ────────────────────────────────────────────────────
CONV_THRESH_MU   = 0.01
CONV_THRESH_ELL  = 0.01
SUSTAINED_FRAMES = 5
RISE_FRAC        = 0.10

# ── Gain/parameter keys shown in the table (fig 2) ───────────────────────────
GAIN_KEYS = [
    ('K_MU',        'Kp_μ  (proportional)'),
    ('K_ELL',       'Kp_ℓ  (proportional)'),
    ('KI_MU',       'Ki_μ  (integral)'),
    ('KI_ELL',      'Ki_ℓ  (integral)'),
    ('KD_MU',       'Kd_μ  (damping / pH)'),
    ('KD_ELL',      'Kd_ℓ  (damping / pH)'),
    ('LEAK',        'λ_leak  (leaky integrator / pH)'),
    ('LAMBDA_DLS',  'λ_DLS  (pinv regularisation / pH)'),
    ('K_AW',        'K_aw  (anti-windup)'),
    ('MAX_INT_U',   'Max integrator μ'),
    ('MAX_INT_ELL', 'Max integrator ℓ'),
    ('ELL_SCALE_MIN','ℓ scale min  (depth-weighted)'),
    ('ELL_SCALE_MAX','ℓ scale max  (depth-weighted)'),
    ('VX_MAX',      'v_x,max  [m/s]'),
    ('WZ_MAX',      'ω_z,max  [rad/s]'),
    ('ELL_DES',     'ℓ_des  (setpoint)'),
    ('MU_DES',      'μ_des  (setpoint)'),
    ('LAMBDA_U',    'λ_u  (camera model)'),
    ('OBJ_HEIGHT',  'H_obj  [m]'),
]

GAIN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ── CSV loading ───────────────────────────────────────────────────────────────
def _load_log(path):
    """Load IBVS log CSV with #KEY=VALUE metadata header lines."""
    meta = {}
    clean_lines = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('"#'):
                line = line.strip('"')
            if line.startswith('#'):
                m = re.match(r'#\s*([\w_]+)\s*=\s*(.+)\s*$', line)
                if m:
                    k = m.group(1).strip()
                    v = m.group(2).strip()
                    try:
                        meta[k] = float(v)
                    except Exception:
                        meta[k] = v
                clean_lines.append(line)
            elif line:
                clean_lines.append(line)

    csv_text = '\n'.join(l for l in clean_lines if not l.startswith('#'))
    df = pd.read_csv(StringIO(csv_text))
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
    data = {c: df[c].values for c in df.columns}

    # Detection flag normalisation
    if 'detected' not in data:
        for src in ('stable', 'bottle_detected', 'bottle_seen'):
            if src in data:
                data['detected'] = (data[src] == 1).astype(int)
                break
        else:
            data['detected'] = np.ones(len(data.get('t', [])), dtype=int)

    return meta, data


# ── Time windowing ────────────────────────────────────────────────────────────
def _compute_t_end(t_rel, detected, max_t=MAX_TIME_PLOT, max_lost=MAX_LOSS_PLOT):
    """End of plot window: min(max_t, first long detection gap after t=0)."""
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


def _apply_window(data):
    """
    Return (det_t, det_data, t_end).
    det_t and det_data contain only detected==1 rows within the time window,
    with time zeroed at the first detection.
    """
    t   = data.get('t', np.array([]))
    det = data.get('detected', np.ones(len(t), dtype=int))

    first = np.where(det == 1)[0]
    if len(first) == 0:
        return np.array([]), {}, MAX_TIME_PLOT

    t0    = t[first[0]]
    t_rel = t - t0
    t_end = _compute_t_end(t_rel, det)

    win   = (det == 1) & (t_rel <= t_end)
    det_t = t_rel[win]
    det_d = {k: v[win] for k, v in data.items()}
    return det_t, det_d, t_end


# ── Metrics ───────────────────────────────────────────────────────────────────
def _rise_time(t, error, frac=RISE_FRAC):
    t     = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = ~np.isnan(t) & ~np.isnan(error)
    t, error = t[valid], error[valid]
    if len(error) == 0:
        return float('nan')
    n_ref = max(1, len(error) // 4)
    ref   = float(np.nanmax(np.abs(error[:n_ref])))
    if ref < 1e-9:
        return 0.0
    idx = np.where(np.abs(error) < frac * ref)[0]
    return float(t[idx[0]] - t[0]) if len(idx) else float('nan')


def _rise_val(t, error, frac=RISE_FRAC):
    """Error value at the rise time moment (first time |e| < frac * peak)."""
    t     = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = ~np.isnan(t) & ~np.isnan(error)
    t, error = t[valid], error[valid]
    if len(error) == 0:
        return float('nan')
    n_ref = max(1, len(error) // 4)
    ref   = float(np.nanmax(np.abs(error[:n_ref])))
    if ref < 1e-9:
        return float(error[0]) if len(error) else float('nan')
    idx = np.where(np.abs(error) < frac * ref)[0]
    return float(error[idx[0]]) if len(idx) else float('nan')


def _settling_time(t, error, threshold, n_frames=SUSTAINED_FRAMES):
    t     = np.asarray(t, dtype=float)
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


def _mae(error):
    e = np.asarray(error, dtype=float)
    v = e[~np.isnan(e)]
    return float(np.mean(np.abs(v))) if len(v) else float('nan')


def _ss_mae(t, error, t_settle):
    """MAE over samples at t >= t_settle (steady-state only)."""
    if np.isnan(t_settle):
        return float('nan')
    t     = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    mask  = (t - t[0]) >= t_settle
    return _mae(error[mask]) if np.any(mask) else float('nan')


def _fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{decimals}f}'


# ── Gain extraction ───────────────────────────────────────────────────────────
def _meta_any(meta, *keys):
    """Return first key found in meta (case-insensitive)."""
    for k in keys:
        if k in meta:
            return meta[k]
        kl = k.lower()
        for mk, mv in meta.items():
            if mk.lower() == kl:
                return mv
    return None


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _parse_two_floats(s):
    if s is None:
        return None, None
    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', str(s))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


def extract_gains(meta):
    """Extract Kp/Ki from meta (explicit keys preferred, compact form fallback)."""
    kp_mu  = _to_float(_meta_any(meta, 'K_MU',  'Kp_mu',  'KP_MU',  'K_mu'))
    kp_ell = _to_float(_meta_any(meta, 'K_ELL', 'Kp_ell', 'KP_ELL', 'K_ell', 'K_Z'))
    ki_mu  = _to_float(_meta_any(meta, 'KI_MU', 'Ki_mu',  'ki_mu'))
    ki_ell = _to_float(_meta_any(meta, 'KI_ELL','Ki_ell', 'ki_ell', 'Ki_Z'))

    if kp_mu is None or kp_ell is None:
        a, b   = _parse_two_floats(_meta_any(meta, 'Kp', 'KP', 'K_P'))
        kp_mu  = kp_mu  if kp_mu  is not None else a
        kp_ell = kp_ell if kp_ell is not None else b

    if ki_mu is None or ki_ell is None:
        a, b   = _parse_two_floats(_meta_any(meta, 'Ki', 'KI', 'K_I'))
        ki_mu  = ki_mu  if ki_mu  is not None else a
        ki_ell = ki_ell if ki_ell is not None else b

    def _nan(x):
        return float('nan') if x is None else float(x)

    return {
        'Kp_mu':  _nan(kp_mu),
        'Kp_ell': _nan(kp_ell),
        'Ki_mu':  _nan(ki_mu),
        'Ki_ell': _nan(ki_ell),
    }



# ── Table helpers ─────────────────────────────────────────────────────────────
def _draw_table(fig, row_labels, col_labels, cell_text, title, sep_rows=None):
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
    for j in range(len(col_labels)):
        clr = GAIN_COLORS[j % len(GAIN_COLORS)] + '55'
        tbl[(0, j)].set_facecolor(clr)
        tbl[(0, j)].set_text_props(fontweight='bold')
    if sep_rows:
        for i in sep_rows:
            for jj in range(-1, len(col_labels)):
                tbl[(i + 1, jj)].set_facecolor('#eeeeee')
                tbl[(i + 1, jj)].set_text_props(color='#666666', style='italic')
    return fig


def _gains_table_data(run_names, metas):
    """One column per run, rows = gain parameters (only those present in ≥1 run)."""
    rows, visible = [], []
    for key, label in GAIN_KEYS:
        vals = [_meta_any(meta, key) for meta in metas]
        if any(v is not None for v in vals):
            visible.append(label)
            rows.append([str(v) if v is not None else '—' for v in vals])
    return visible, run_names, rows


def _metrics_table_data(run_names, metrics):
    """One column per run, rows = performance metrics."""
    rows_spec = [
        ('det',   'Det. frames',          lambda m: str(m.get('n_det', '—'))),
        ('dur',   'Duration [s]',         lambda m: _fmt(m.get('duration', float('nan')))),
        ('wind',  'Window end [s]',       lambda m: _fmt(m.get('t_end', float('nan')), 1)),
        (None,    '─── μ ───',            lambda _: ''),
        ('rt_mu', 'T_r(μ) [s]',           lambda m: _fmt(m.get('rt_mu', float('nan')))),
        ('rv_mu', 'e_μ at T_r',           lambda m: _fmt(m.get('rv_mu', float('nan')))),
        ('st_mu', 'T_s(μ) [s]',           lambda m: _fmt(m.get('st_mu', float('nan')))),
        ('bw_mu', 'BW(μ) ≈ 0.35/Tr [Hz]', lambda m: _fmt(m.get('bw_mu', float('nan')))),
        ('ma_mu',  'MAE(μ)',               lambda m: _fmt(m.get('mae_mu',    float('nan')))),
        ('ssm_mu', 'SS-MAE(μ)  [t≥T_s]', lambda m: _fmt(m.get('ss_mae_mu', float('nan')))),
        (None,    '─── ℓ ───',            lambda _: ''),
        ('rt_el', 'T_r(ℓ) [s]',           lambda m: _fmt(m.get('rt_ell', float('nan')))),
        ('rv_el', 'e_ℓ at T_r',           lambda m: _fmt(m.get('rv_ell', float('nan')))),
        ('st_el', 'T_s(ℓ) [s]',           lambda m: _fmt(m.get('st_ell', float('nan')))),
        ('bw_el', 'BW(ℓ) ≈ 0.35/Tr [Hz]', lambda m: _fmt(m.get('bw_ell', float('nan')))),
        ('ma_el',  'MAE(ℓ)',               lambda m: _fmt(m.get('mae_ell',    float('nan')))),
        ('ssm_el', 'SS-MAE(ℓ)  [t≥T_s]', lambda m: _fmt(m.get('ss_mae_ell', float('nan')))),
    ]
    row_labels = [s[1] for s in rows_spec]
    cell_text  = [[s[2](m) for m in metrics] for s in rows_spec]
    sep_rows   = {i for i, s in enumerate(rows_spec) if s[0] is None}
    return row_labels, run_names, cell_text, sep_rows


def _combined_table_data(run_names, metas, metrics):
    """Gains + metrics in one table; handles missing gain rows gracefully."""
    g_labels, _, g_cells = _gains_table_data(run_names, metas)
    m_labels, _, m_cells, m_sep = _metrics_table_data(run_names, metrics)
    if not g_labels:
        return m_labels, run_names, m_cells, m_sep
    n_g = len(g_labels)
    sep_row = '─── Performance Metrics ───'
    row_labels = g_labels + [sep_row] + m_labels
    cell_text  = g_cells  + [[''] * len(run_names)] + m_cells
    sep_rows   = {n_g} | {n_g + 1 + i for i in m_sep}
    return row_labels, run_names, cell_text, sep_rows


# ── Style ─────────────────────────────────────────────────────────────────────
def apply_style():
    for s in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot'):
        try:
            plt.style.use(s)
            return
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main(paths):
    if len(paths) < 2:
        print(__doc__)
        sys.exit(1)

    apply_style()

    out_dir = Path(paths[0]).parent / 'gains_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f'  Saved → {p.name}')
        plt.close(fig)

    run_names   = []
    all_meta    = []
    all_metrics = []
    plot_data   = []   # (det_t, e_mu, e_ell) per run

    print(f'\nComparing {len(paths)} runs (window: 0 – {MAX_TIME_PLOT:.0f} s)\n')

    for path in paths:
        run = os.path.splitext(os.path.basename(path))[0]
        try:
            meta, data = _load_log(path)
        except Exception as e:
            print(f'  [error] {path}: {e}')
            continue

        det_t, det, t_end = _apply_window(data)
        if len(det_t) == 0:
            print(f'  [skip]  {run}: no detection frames in window')
            continue

        # Extract or compute errors
        e_mu  = det.get('e_mu',  np.full(len(det_t), np.nan))
        e_ell = det.get('e_ell', np.full(len(det_t), np.nan))

        if np.all(np.isnan(e_mu)):
            mu_d   = det.get('mu', np.full(len(det_t), np.nan))
            mu_des_arr = det.get('mu_des', np.array([]))
            mu_des = float(np.nanmean(mu_des_arr)) if len(mu_des_arr) and not np.all(np.isnan(mu_des_arr)) else 0.0
            e_mu   = mu_d - mu_des

        if np.all(np.isnan(e_ell)):
            ell_d   = det.get('ell', np.full(len(det_t), np.nan))
            ell_des_arr = det.get('ell_des', np.array([]))
            ell_des = float(np.nanmean(ell_des_arr)) if len(ell_des_arr) and not np.all(np.isnan(ell_des_arr)) else 0.8
            e_ell   = ell_d - ell_des

        # ── Pixel-space auto-detection ─────────────────────────────────────────
        _iw        = _to_float(_meta_any(meta, 'IMG_WIDTH'))
        _ih        = _to_float(_meta_any(meta, 'IMG_HEIGHT'))
        img_w      = int(_iw) if _iw else None
        img_h      = int(_ih) if _ih else None
        is_pixel   = img_w is not None and img_h is not None
        thresh_mu  = CONV_THRESH_MU  * img_w if is_pixel else CONV_THRESH_MU
        thresh_ell = CONV_THRESH_ELL * img_h if is_pixel else CONV_THRESH_ELL
        units_mu   = 'px' if is_pixel else '–'
        units_ell  = 'px' if is_pixel else 'norm'

        # Compute metrics
        rt_mu  = _rise_time(det_t, e_mu)
        rt_ell = _rise_time(det_t, e_ell)
        rv_mu  = _rise_val(det_t, e_mu)
        rv_ell = _rise_val(det_t, e_ell)
        st_mu  = _settling_time(det_t, e_mu,  thresh_mu)
        st_ell = _settling_time(det_t, e_ell, thresh_ell)
        bw_mu  = 0.35 / rt_mu  if not np.isnan(rt_mu)  and rt_mu > 0 else float('nan')
        bw_ell = 0.35 / rt_ell if not np.isnan(rt_ell) and rt_ell > 0 else float('nan')

        metrics = {
            'n_det':    int(np.sum(data.get('detected', np.ones(1)) == 1)),
            'duration': float(det_t[-1] - det_t[0]) if len(det_t) > 1 else float('nan'),
            't_end':    t_end,
            'rt_mu':    rt_mu,   'rt_ell':  rt_ell,
            'rv_mu':    rv_mu,   'rv_ell':  rv_ell,
            'st_mu':    st_mu,   'st_ell':  st_ell,
            'bw_mu':    bw_mu,   'bw_ell':  bw_ell,
            'mae_mu':    _mae(e_mu),    'mae_ell':    _mae(e_ell),
            'ss_mae_mu':  _ss_mae(det_t, e_mu,  st_mu),
            'ss_mae_ell': _ss_mae(det_t, e_ell, st_ell),
            'thresh_mu':  thresh_mu,
            'thresh_ell': thresh_ell,
            'units_mu':   units_mu,
            'units_ell':  units_ell,
        }

        g = extract_gains(meta)
        print(f'  {run}:  window {t_end:.1f} s | {len(det_t)} frames')
        print(f'    Kp: μ={_fmt(g["Kp_mu"])} ℓ={_fmt(g["Kp_ell"])}  '
              f'Ki: μ={_fmt(g["Ki_mu"])} ℓ={_fmt(g["Ki_ell"])}')
        print(f'    T_r: μ={_fmt(rt_mu)} s  ℓ={_fmt(rt_ell)} s | '
              f'T_s: μ={_fmt(st_mu)} s  ℓ={_fmt(st_ell)} s')
        print(f'    MAE: μ={_fmt(metrics["mae_mu"])}  ℓ={_fmt(metrics["mae_ell"])} | '
              f'SS-MAE: μ={_fmt(metrics["ss_mae_mu"])}  ℓ={_fmt(metrics["ss_mae_ell"])}')

        run_names.append(run)
        all_meta.append(meta)
        all_metrics.append(metrics)
        plot_data.append((det_t, e_mu, e_ell))

    if not run_names:
        print('[error] No valid data.')
        sys.exit(1)

    n = len(run_names)

    # ── Fig 1: Error convergence — both channels overlaid ────────────────────
    fig1, (ax_mu, ax_ell) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for i, (det_t, e_mu, e_ell) in enumerate(plot_data):
        c   = GAIN_COLORS[i % len(GAIN_COLORS)]
        lbl = run_names[i]
        ax_mu.plot(det_t,  e_mu,  color=c, linewidth=1.8, label=lbl, alpha=0.9)
        ax_ell.plot(det_t, e_ell, color=c, linewidth=1.8, label=lbl, alpha=0.9)

    _m0         = all_metrics[0] if all_metrics else {}
    _thresh_mu  = _m0.get('thresh_mu',  CONV_THRESH_MU)
    _thresh_ell = _m0.get('thresh_ell', CONV_THRESH_ELL)
    _units_mu   = _m0.get('units_mu',  '–')
    _units_ell  = _m0.get('units_ell', 'norm')
    for k, (ax, thresh, ylbl) in enumerate([
        (ax_mu,  _thresh_mu,  f'e_μ  [{_units_mu}]'),
        (ax_ell, _thresh_ell, f'e_ℓ  [{_units_ell}]'),
    ]):
        ax.axhline(0, color='k', linewidth=0.7)
        ax.axhspan(-thresh, thresh, alpha=0.10, color='green', zorder=0,
                   label=f'±{thresh:.3g} conv. band' if k == 0 else '_nolegend_')
        ax.set_xlabel('Time since first detection [s]')
        ax.set_ylabel(ylbl)
        ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = ax_mu.get_legend_handles_labels()
    fig1.legend(handles, labels, fontsize=8, loc='upper right',
                bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    plt.tight_layout()
    _save(fig1, 'fig1_error_convergence.png')

    # ── Fig 2: Combined parameters + metrics table ────────────────────────────
    row_c, col_c, cell_c, sep_c = _combined_table_data(run_names, all_meta, all_metrics)
    fig2_h = max(4.0, 0.38 * len(row_c) + 1.5)
    fig2 = plt.figure(figsize=(4 + 3.5 * n, fig2_h))
    _draw_table(fig2, row_c, col_c, cell_c,
                'Parameters & Performance Metrics', sep_rows=sep_c)
    plt.tight_layout()
    _save(fig2, 'fig2_combined_table.png')

    print(f'\n  All figures saved to: {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        print('\nExample:')
        print('  python3 experiment_gains.py run1.csv run2.csv run3.csv')
        sys.exit(1)
    main(sys.argv[1:])
