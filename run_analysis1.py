#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error
import os
import glob
import re as regex
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------------------------------
FONT_SIZE = 9
DPI = 600
FIG_WIDTH = 7.2
FIG_HEIGHT = 2.8

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.top": False,
    "ytick.right": False,
    "axes.grid": False,
    "grid.alpha": 0.12,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "grid.color": "#CCCCCC",
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---------------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------------
C_RED = "#D62728"
C_BLUE = "#1F77B4"
C_GREEN = "#2CA02C"
C_ORANGE = "#FF6D00"
C_PURPLE = "#7B2D8E"
C_BLACK = "#212121"
C_CYAN = "#17BECF"
C_GREY = "#757575"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPARISON_RE = [2500, 25000]
STREAMWISE_POSITIONS = [4.0, 5.0, 6.0, 7.0]
POSITION_TOLERANCE = 0.01
TIME_RANGE = (30, 99)
DT_SAVED = 0.5
PROBE_X, PROBE_Y = 5.0, 0.5

YLABEL_Y = r"Wall-normal position"

print("=" * 72)
print("   CONSOLIDATED MIXING LAYER ANALYSIS")
print("=" * 72)

# ---------------------------------------------------------------------------
# GLOBAL HELPER: force all 4 spines visible on any axes
# ---------------------------------------------------------------------------
def _ensure_boxed(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)

def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False, which='both')

def _light_grid(ax, axis="both"):
    ax.grid(True, alpha=0.12, linewidth=0.4, color="#CCCCCC", axis=axis)

def _add_centered_xlabel(fig, axes_row, label):
    fig.canvas.draw()
    if hasattr(axes_row, '__len__'):
        bbox_left = axes_row[0].get_position()
        bbox_right = axes_row[-1].get_position()
    else:
        bbox_left = axes_row.get_position()
        bbox_right = bbox_left
    x_center = 0.5 * (bbox_left.x0 + bbox_right.x1)
    y_below = bbox_left.y0 - 0.035
    fig.text(x_center, max(y_below, 0.01), label,
             ha="center", va="top", fontsize=FONT_SIZE - 1)

def _save(fig, filename):
    """Save figure to OUTPUT_DIR with high quality"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f" [OK] Saved: {filepath}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def discover_reynolds_numbers():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Re*.npz")))
    re_vals = []
    paths = {}
    for f in files:
        m = regex.search(r"Re(\d+)", os.path.basename(f))
        if m:
            rev = int(m.group(1))
            re_vals.append(rev)
            paths[rev] = f
    re_vals.sort()
    return re_vals, paths

ALL_RE, FILE_PATHS = discover_reynolds_numbers()
print(f"Found Reynolds numbers: {ALL_RE}")

DATA = {}

for rev in ALL_RE:
    d = np.load(FILE_PATHS[rev])
    DATA[rev] = {
        "coordinates":     d["coordinates"],
        "variables":       list(d["variables"]),
        "original":        d["original_data"],
        "reconstructed":   d["reconstructed_data"],
        "mean_field":      d["mean_field"] if "mean_field" in d else None,
        "temporal_coeffs": d["temporal_coefficients"] if "temporal_coefficients" in d else None,
        "time":            d["time"] if "time" in d else np.arange(d["original_data"].shape[0]) * DT_SAVED,
    }
    print(f"  Re={rev:>6d}  shape={DATA[rev]['original'].shape}  vars={DATA[rev]['variables']}")

# ---------------------------------------------------------------------------
# 2. METRIC COMPUTATION FUNCTIONS
# ---------------------------------------------------------------------------

def _var_idx(rev, name):
    return DATA[rev]["variables"].index(name)

def time_indices(rev):
    n = DATA[rev]["original"].shape[0]
    t0 = max(0, TIME_RANGE[0])
    t1 = min(TIME_RANGE[1], n - 1)
    return np.arange(t0, t1 + 1)

def pencil_indices(rev, x_target):
    coords = DATA[rev]["coordinates"]
    mask = np.abs(coords[:, 0] - x_target) < POSITION_TOLERANCE
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return idx[np.argsort(coords[idx, 1])]

def reference_velocity(rev):
    tidx = time_indices(rev)
    ui = _var_idx(rev, "U:0")
    u_all = DATA[rev]["original"][tidx, :, ui]
    return np.max(np.abs(np.mean(u_all, axis=0)))

def compute_turbulence(u_field, v_field):
    u_mean = np.mean(u_field, axis=0)
    v_mean = np.mean(v_field, axis=0)
    uf = u_field - u_mean[None, :]
    vf = v_field - v_mean[None, :]
    u_rms = np.sqrt(np.mean(uf**2, axis=0))
    v_rms = np.sqrt(np.mean(vf**2, axis=0))
    tke   = 0.5 * (u_rms**2 + v_rms**2)
    uv    = np.mean(uf * vf, axis=0)
    mag   = np.sqrt(u_mean**2 + v_mean**2 + 1e-12)
    ti    = np.sqrt(2.0 / 3.0 * tke) / mag
    return dict(u_mean=u_mean, v_mean=v_mean, u_rms=u_rms, v_rms=v_rms,
                tke=tke, reynolds_stress=uv, turbulence_intensity=ti)

def compute_vorticity(u_field, y_coords):
    omega = -np.gradient(u_field, y_coords, axis=1)
    omega_mean = np.mean(omega, axis=0)
    omega_rms  = np.sqrt(np.mean((omega - omega_mean[None, :])**2, axis=0))
    enstrophy  = 0.5 * np.mean(omega**2, axis=0)
    return dict(vorticity_mean=omega_mean, vorticity_rms=omega_rms,
                enstrophy=enstrophy, vorticity_field=omega)

def compute_intermittency(vorticity_field, y_coords, threshold_pct=85):
    threshold = np.percentile(np.abs(vorticity_field), threshold_pct)
    gamma = np.mean(np.abs(vorticity_field) > threshold, axis=0)
    return gamma

def compute_layer_scales(u_mean, s_mean, y):
    U_max, U_min = np.max(u_mean), np.min(u_mean)
    dU = U_max - U_min
    du_dy = np.gradient(u_mean, y)
    max_grad = np.max(np.abs(du_dy))
    delta_omega = dU / max_grad if max_grad > 0 else 0.0
    if dU > 0:
        integrand = (u_mean - U_min) * (U_max - u_mean) / dU**2
        theta = np.trapz(integrand, y)
    else:
        theta = 0.0
    s_norm = (s_mean - np.min(s_mean)) / (np.max(s_mean) - np.min(s_mean) + 1e-12)
    i99 = np.where(s_norm >= 0.99)[0]
    i01 = np.where(s_norm <= 0.01)[0]
    delta_s = np.abs(y[i99[0]] - y[i01[-1]]) if len(i99) > 0 and len(i01) > 0 else 0.0
    center_idx = np.argmax(np.abs(du_dy))
    return dict(delta_omega=delta_omega, theta=theta, delta_s=delta_s,
                dU=dU, center_y=y[center_idx], center_idx=center_idx)

def compute_scalar_mixing(s_field, y):
    s_mean = np.mean(s_field, axis=0)
    sf = s_field - s_mean[None, :]
    s_rms = np.sqrt(np.mean(sf**2, axis=0))
    ds_dy = np.gradient(s_field, y, axis=1)
    chi = 2.0 * np.mean(ds_dy**2, axis=0)
    s_n = (s_field - np.min(s_field)) / (np.max(s_field) - np.min(s_field) + 1e-12)
    mixedness = 4.0 * np.mean(s_n * (1.0 - s_n), axis=0)
    pdf_vals, bin_edges = np.histogram(s_n.flatten(), bins=50, range=(0, 1), density=True)
    pdf_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return dict(s_mean=s_mean, s_rms=s_rms, scalar_dissipation=chi,
                mixedness=mixedness, pdf_centers=pdf_centers, pdf_vals=pdf_vals)

def compute_tke_production(u_field, v_field, y):
    u_mean = np.mean(u_field, axis=0)
    v_mean = np.mean(v_field, axis=0)
    uf = u_field - u_mean[None, :]
    vf = v_field - v_mean[None, :]
    uv = np.mean(uf * vf, axis=0)
    du_dy = np.gradient(u_mean, y)
    P_tke = -uv * du_dy
    return dict(tke_production=P_tke, reynolds_stress=uv, velocity_gradient=du_dy)

def compute_spectrum(ts, dt):
    n = len(ts)
    fl = ts - np.mean(ts)
    freqs = fftfreq(n, dt)[:n // 2]
    psd = np.abs(fft(fl)[:n // 2])**2 / n
    pk = np.argmax(psd[1:]) + 1 if len(psd) > 1 else 0
    f_peak = freqs[pk] if pk < len(freqs) else 0.0
    acf = np.correlate(fl, fl, "full")[n - 1:]
    acf /= acf[0] + 1e-30
    zc = np.where(acf < 0)[0]
    T_int = np.trapz(acf[: zc[0]], dx=dt) if len(zc) > 0 else np.trapz(acf[: n // 4], dx=dt)
    return dict(freqs=freqs, psd=psd, f_peak=f_peak, T_int=T_int)

def compute_validation(orig, recon, y, dt):
    o_mean = np.mean(orig, axis=0)
    r_mean = np.mean(recon, axis=0)
    corr_profile = np.corrcoef(o_mean, r_mean)[0, 1] if len(o_mean) > 1 else 0.0
    rms_err = np.sqrt(np.mean((o_mean - r_mean)**2))
    rel_err = rms_err / (np.std(o_mean) + 1e-12) * 100
    ci = len(y) // 2
    corr_temporal = np.corrcoef(orig[:, ci], recon[:, ci])[0, 1] if orig.shape[0] > 1 else 0.0
    nperseg = min(256, max(8, len(orig) // 4))
    try:
        fc, coh = signal.coherence(orig[:, ci], recon[:, ci], fs=1 / dt, nperseg=nperseg)
    except Exception:
        fc = np.array([0.0])
        coh = np.array([1.0])
    return dict(profile_corr=corr_profile, relative_error=rel_err,
                temporal_corr=corr_temporal, mean_coherence=np.mean(coh[1:]) if len(coh) > 1 else 0.0,
                coh_freqs=fc, coherence=coh)

# ---------------------------------------------------------------------------
# 3. MASTER ANALYSIS
# ---------------------------------------------------------------------------
RESULTS = {}
GLOBAL  = {}

print("\n" + "-" * 72)
print("  Computing metrics for all Reynolds numbers ...")
print("-" * 72)

for rev in ALL_RE:
    d = DATA[rev]
    orig_all  = d["original"]
    recon_all = d["reconstructed"]
    coords    = d["coordinates"]
    vnames    = d["variables"]
    tidx      = time_indices(rev)
    dt        = DT_SAVED
    U_ref     = reference_velocity(rev)

    ui = _var_idx(rev, "U:0")
    vi = _var_idx(rev, "U:1")
    si = _var_idx(rev, "s")
    pi_idx = _var_idx(rev, "p")

    num_total = np.linalg.norm(orig_all - recon_all)
    den_total = np.linalg.norm(orig_all)
    total_rel = num_total / den_total * 100

    per_var_rmse = {}
    per_var_rel  = {}
    for k, vn in enumerate(vnames):
        per_var_rmse[vn] = np.sqrt(np.mean((orig_all[:, :, k] - recon_all[:, :, k])**2))
        den_v = np.linalg.norm(orig_all[:, :, k])
        per_var_rel[vn] = np.linalg.norm(orig_all[:, :, k] - recon_all[:, :, k]) / den_v * 100 if den_v > 0 else 0

    time_mse = np.mean((orig_all - recon_all)**2, axis=(1, 2))

    o_mag = np.sqrt(orig_all[:, :, ui]**2 + orig_all[:, :, vi]**2).flatten()
    r_mag = np.sqrt(recon_all[:, :, ui]**2 + recon_all[:, :, vi]**2).flatten()
    n_samp = min(3000, len(o_mag))
    samp_idx = np.random.RandomState(42).choice(len(o_mag), n_samp, replace=False)
    r2_val = r2_score(o_mag[samp_idx], r_mag[samp_idx])
    residuals_arr = o_mag[samp_idx] - r_mag[samp_idx]

    err_vel = np.linalg.norm(
        orig_all[:, :, [ui, vi]] - recon_all[:, :, [ui, vi]], axis=(1, 2)
    ) / (np.linalg.norm(orig_all[:, :, [ui, vi]], axis=(1, 2)) + 1e-30) * 100
    err_prs = np.linalg.norm(
        orig_all[:, :, pi_idx] - recon_all[:, :, pi_idx], axis=1
    ) / (np.linalg.norm(orig_all[:, :, pi_idx], axis=1) + 1e-30) * 100

    GLOBAL[rev] = dict(
        total_relative_error=total_rel, per_var_rmse=per_var_rmse,
        per_var_rel=per_var_rel, time_mse=time_mse, r2=r2_val,
        residuals=residuals_arr, samp_true=o_mag[samp_idx], samp_pred=r_mag[samp_idx],
        err_vel_time=err_vel, err_prs_time=err_prs,
        U_ref=U_ref,
    )

    RESULTS[rev] = {}
    for x_pos in STREAMWISE_POSITIONS:
        pidx = pencil_indices(rev, x_pos)
        if pidx is None or len(pidx) < 5:
            continue
        y = coords[pidx, 1]

        u_o = orig_all[tidx][:, pidx, ui]
        v_o = orig_all[tidx][:, pidx, vi]
        s_o = orig_all[tidx][:, pidx, si]
        u_r = recon_all[tidx][:, pidx, ui]
        v_r = recon_all[tidx][:, pidx, vi]
        s_r = recon_all[tidx][:, pidx, si]

        tb_o = compute_turbulence(u_o, v_o)
        tb_r = compute_turbulence(u_r, v_r)
        vt_o = compute_vorticity(u_o, y)
        vt_r = compute_vorticity(u_r, y)
        intm_o = compute_intermittency(vt_o["vorticity_field"], y)
        intm_r = compute_intermittency(vt_r["vorticity_field"], y)
        sc_o = compute_layer_scales(tb_o["u_mean"], np.mean(s_o, axis=0), y)
        sc_r = compute_layer_scales(tb_r["u_mean"], np.mean(s_r, axis=0), y)
        sm_o = compute_scalar_mixing(s_o, y)
        sm_r = compute_scalar_mixing(s_r, y)
        tp_o = compute_tke_production(u_o, v_o, y)
        tp_r = compute_tke_production(u_r, v_r, y)

        ci = sc_o["center_idx"]
        sp_o = compute_spectrum(u_o[:, ci], dt)
        sp_r = compute_spectrum(u_r[:, ci], dt)
        St = sp_o["f_peak"] * sc_o["delta_omega"] / sc_o["dU"] if sc_o["dU"] > 0 else 0

        val_u = compute_validation(u_o, u_r, y, dt)
        val_s = compute_validation(s_o, s_r, y, dt)

        RESULTS[rev][x_pos] = dict(
            y=y, U_ref=U_ref,
            t_u_mean=tb_o["u_mean"], t_v_mean=tb_o["v_mean"],
            t_u_rms=tb_o["u_rms"], t_v_rms=tb_o["v_rms"],
            t_tke=tb_o["tke"], t_uv=tb_o["reynolds_stress"],
            t_ti=tb_o["turbulence_intensity"],
            r_u_mean=tb_r["u_mean"], r_v_mean=tb_r["v_mean"],
            r_u_rms=tb_r["u_rms"], r_v_rms=tb_r["v_rms"],
            r_tke=tb_r["tke"], r_uv=tb_r["reynolds_stress"],
            r_ti=tb_r["turbulence_intensity"],
            t_omega=vt_o["vorticity_mean"], t_omega_rms=vt_o["vorticity_rms"],
            t_enstrophy=vt_o["enstrophy"],
            r_omega=vt_r["vorticity_mean"], r_omega_rms=vt_r["vorticity_rms"],
            r_enstrophy=vt_r["enstrophy"],
            t_intermittency=intm_o, r_intermittency=intm_r,
            t_scales=sc_o, r_scales=sc_r,
            t_s_mean=sm_o["s_mean"], t_s_rms=sm_o["s_rms"],
            t_chi=sm_o["scalar_dissipation"], t_mixedness=sm_o["mixedness"],
            t_pdf_c=sm_o["pdf_centers"], t_pdf_v=sm_o["pdf_vals"],
            r_s_mean=sm_r["s_mean"], r_s_rms=sm_r["s_rms"],
            r_chi=sm_r["scalar_dissipation"], r_mixedness=sm_r["mixedness"],
            r_pdf_c=sm_r["pdf_centers"], r_pdf_v=sm_r["pdf_vals"],
            t_P_tke=tp_o["tke_production"], r_P_tke=tp_r["tke_production"],
            t_du_dy=tp_o["velocity_gradient"], r_du_dy=tp_r["velocity_gradient"],
            t_spec=sp_o, r_spec=sp_r, strouhal=St,
            val_u=val_u, val_s=val_s,
        )

    print(f"  Re={rev:>6d}:  {len(RESULTS[rev])} positions,  "
          f"global error = {total_rel:.2f}%,  R2 = {r2_val:.4f}")

# ---------------------------------------------------------------------------
# CSV ENGINEERING REPORT
# ---------------------------------------------------------------------------
print("\n-- Generating CSV Report --")
rows = []
for rev in ALL_RE:
    for xp in STREAMWISE_POSITIONS:
        if rev not in RESULTS or xp not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][xp]
        rows.append(dict(
            Re=rev, x=xp,
            delta_omega_target=rr["t_scales"]["delta_omega"],
            delta_omega_predicted=rr["r_scales"]["delta_omega"],
            theta_target=rr["t_scales"]["theta"],
            theta_predicted=rr["r_scales"]["theta"],
            delta_c_target=rr["t_scales"]["delta_s"],
            delta_c_predicted=rr["r_scales"]["delta_s"],
            dU=rr["t_scales"]["dU"],
            max_TKE_target=np.max(rr["t_tke"]),
            max_TKE_predicted=np.max(rr["r_tke"]),
            max_uv_target=np.max(np.abs(rr["t_uv"])),
            max_uv_predicted=np.max(np.abs(rr["r_uv"])),
            max_mixedness_target=np.max(rr["t_mixedness"]),
            max_mixedness_predicted=np.max(rr["r_mixedness"]),
            max_enstrophy_target=np.max(rr["t_enstrophy"]),
            max_enstrophy_predicted=np.max(rr["r_enstrophy"]),
            Strouhal=rr["strouhal"],
            peak_freq=rr["t_spec"]["f_peak"],
            integral_time=rr["t_spec"]["T_int"],
            vel_profile_corr=rr["val_u"]["profile_corr"],
            vel_relative_error=rr["val_u"]["relative_error"],
            vel_temporal_corr=rr["val_u"]["temporal_corr"],
            vel_mean_coherence=rr["val_u"]["mean_coherence"],
            conc_profile_corr=rr["val_s"]["profile_corr"],
            conc_relative_error=rr["val_s"]["relative_error"],
            conc_temporal_corr=rr["val_s"]["temporal_corr"],
            conc_mean_coherence=rr["val_s"]["mean_coherence"],
        ))

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "engineering_report_all_Re.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"    -> Saved {csv_path}")

# ---------------------------------------------------------------------------
# FIGURE 1: Global Error vs Re + Benchmark Comparison
# ---------------------------------------------------------------------------
print("\n-- Figure 1: Global Error vs Re + Benchmark Comparison --")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.55), dpi=DPI)

# Left: Global Error - MATLAB colors
apply_style(ax1)
MATLAB_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#A2142F']
COLORS_LEFT = {"U:0": MATLAB_COLORS[0], "U:1": MATLAB_COLORS[1], "p": MATLAB_COLORS[2], "s": MATLAB_COLORS[3]}
VAR_LABELS = {"U:0": "Ux", "U:1": "Uy", "p": "p", "s": "c"}

for vn, color in COLORS_LEFT.items():
    y_vals = [GLOBAL[r]["per_var_rel"].get(vn, 0) for r in ALL_RE]
    ax1.plot(ALL_RE, y_vals, "o-", color=color, label=VAR_LABELS[vn], markersize=5, linewidth=1.4)

total = [GLOBAL[r]["total_relative_error"] for r in ALL_RE]
ax1.plot(ALL_RE, total, "s-", color=MATLAB_COLORS[5], lw=1.8, ms=5.5, label="Total")
mean_tot = np.mean(total)
ax1.axhline(mean_tot, color=C_GREY, ls=":", lw=0.8, alpha=0.6)
ax1.text(ALL_RE[-1] * 1.02, mean_tot, f"{mean_tot:.1f}%", fontsize=FONT_SIZE, color=C_GREY, va="center")
ax1.set_xlabel("Reynolds number", fontsize=FONT_SIZE)
ax1.set_ylabel("Relative error (%)", fontsize=FONT_SIZE)
ax1.set_ylim(bottom=0)
ax1.legend(ncol=2, loc="upper right", fontsize=FONT_SIZE - 1)
_light_grid(ax1)
ax1.set_title("(a) Global Reconstruction Error", fontsize=FONT_SIZE + 1, loc='left', pad=8)

# Right: Benchmark Comparison - Load from NPZ file only
apply_style(ax2)

# Load benchmark data from NPZ file (must exist)
benchmark_npz_path = os.path.join(DATA_DIR, "benchmark_data.npz")
if not os.path.exists(benchmark_npz_path):
    raise FileNotFoundError(f"Benchmark data file not found: {benchmark_npz_path}")

bench_data = np.load(benchmark_npz_path, allow_pickle=True)
RE_PLOT = bench_data['reynolds_numbers']

# Extract model data
BENCH_RMSE = {
    "SPECTRE": bench_data['spectre'],
    "POD-Galerkin": bench_data['pod_galerkin'],
    "DMD": bench_data['dmd'],
    "LSTM-ROM": bench_data['lstm_rom'],
    "FNO-3D": bench_data['fno_3d'],
    "PI-CNN": bench_data['pi_cnn'],
}

MODEL_COLORS_BENCH = {
    "SPECTRE": "#FF0000", "POD-Galerkin": "#8B008B", "DMD": "#0000FF",
    "LSTM-ROM": "#00FF00", "FNO-3D": "#FF8C00", "PI-CNN": "#00FFFF",
}
MODEL_MARKERS_BENCH = {
    "SPECTRE": "o", "POD-Galerkin": "s", "DMD": "^",
    "LSTM-ROM": "D", "FNO-3D": "v", "PI-CNN": "<",
}
MODEL_ORDER_BENCH = ["SPECTRE", "PI-CNN", "FNO-3D", "LSTM-ROM", "DMD", "POD-Galerkin"]

# Plot benchmark data - filtering out NaN values to connect remaining points
for model in MODEL_ORDER_BENCH:
    # Filter out NaN values
    re_vals = []
    rmse_vals = []
    for re, rmse in zip(RE_PLOT, BENCH_RMSE[model]):
        if not np.isnan(rmse):
            re_vals.append(re)
            rmse_vals.append(rmse)
    
    # Plot the filtered data with connecting lines
    ax2.plot(re_vals, rmse_vals, 
             marker=MODEL_MARKERS_BENCH[model],
             color=MODEL_COLORS_BENCH[model],
             ls="-" if model == "SPECTRE" else "--",
             lw=1.1 if model == "SPECTRE" else 1.0,
             ms=8 if model == "SPECTRE" else 5,
             markeredgecolor='white',
             markeredgewidth=0.5,
             label=model)

ax2.set_xlabel("Reynolds number", fontsize=FONT_SIZE)
ax2.set_ylabel("RMSE (%)", fontsize=FONT_SIZE)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([2000, 35000])
ax2.set_ylim([0, 65])
ax2.legend(ncol=2, loc="upper left", fontsize=FONT_SIZE - 1, frameon=False)
_light_grid(ax2)
ax2.set_title("(b) Model comparison", fontsize=FONT_SIZE + 1, loc='left', pad=8)
plt.subplots_adjust(wspace=0.33)
_save(fig, "figure01_global_error_benchmark.png")

# ---------------------------------------------------------------------------
# FIGURE 2: Diagnostic Dashboard
# ---------------------------------------------------------------------------
print("-- Figure 2: Diagnostic Dashboard --")

COLOR_PALETTE = {
    'scatter': '#D62728', 'diagonal': '#1F77B4', 'residuals': '#2CA02C',
    'error': '#7B2D8E', 'temporal': '#17BECF',
}

for rev in COMPARISON_RE:
    if rev not in GLOBAL:
        continue
    g = GLOBAL[rev]
    d = DATA[rev]
    coords_d = d["coordinates"]
    ui_d = _var_idx(rev, "U:0")
    vi_d = _var_idx(rev, "U:1")

    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    fig.suptitle(f"Diagnostic Dashboard — Re = {rev:,}", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.3, hspace=0.4)

    # Top-Left: Scatter
    ax = fig.add_subplot(gs[0, 0])
    apply_style(ax)
    ax.scatter(g["samp_true"], g["samp_pred"], s=8, alpha=0.15,
               color=COLOR_PALETTE['scatter'], edgecolors='none', rasterized=True)
    mn, mx = min(g["samp_true"].min(), g["samp_pred"].min()), max(g["samp_true"].max(), g["samp_pred"].max())
    ax.plot([mn, mx], [mn, mx], "-", color=COLOR_PALETTE['diagonal'], lw=1.0, alpha=0.8)
    ax.set_xlabel("Target velocity magnitude", fontsize=FONT_SIZE)
    ax.set_ylabel("Predicted velocity magnitude", fontsize=FONT_SIZE)
    ax.text(0.05, 0.92, f"R2 = {g['r2']:.4f}", transform=ax.transAxes, fontsize=FONT_SIZE - 1)
    _light_grid(ax)

    # Top-Middle: Residuals
    ax = fig.add_subplot(gs[0, 1])
    apply_style(ax)
    ax.hist(g["residuals"], bins=45, density=True, color=COLOR_PALETTE['residuals'],
            alpha=0.5, edgecolor='white', linewidth=0.3)
    mu, std = norm.fit(g["residuals"])
    xf = np.linspace(g["residuals"].min(), g["residuals"].max(), 100)
    ax.plot(xf, norm.pdf(xf, mu, std), color=COLOR_PALETTE['scatter'], lw=1.2)
    ax.axvline(0, color='black', ls=":", lw=0.6)
    ax.set_xlabel("Residual", fontsize=FONT_SIZE)
    ax.set_ylabel("Probability density", fontsize=FONT_SIZE)
    ax.text(0.05, 0.92, f"mu = {mu:.4f}", transform=ax.transAxes, fontsize=FONT_SIZE - 1)
    _light_grid(ax)

    # Top-Right: RMSE
    ax = fig.add_subplot(gs[0, 2])
    apply_style(ax)
    vn_list = [n for n in ["U:0", "U:1", "p", "s"] if n in d["variables"]]
    labels = ["Ux", "Uy", "p", "c"][:len(vn_list)]
    rmse_vals = [g["per_var_rmse"][n] for n in vn_list]
    bar_colors = ['#1F77B4', '#2CA02C', '#FF6D00', '#D62728']
    bars = ax.bar(labels, rmse_vals, color=bar_colors[:len(rmse_vals)], edgecolor='none', width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=FONT_SIZE - 1.5)
    ax.set_ylabel("RMSE", fontsize=FONT_SIZE)
    ax.set_ylim(0, max(rmse_vals) * 1.25)
    _light_grid(ax, axis="y")

    # Bottom-Left: Error vs magnitude
    ax = fig.add_subplot(gs[1, 0])
    apply_style(ax)
    ax.scatter(g["samp_true"], np.abs(g["residuals"]), s=8, alpha=0.15,
               color=COLOR_PALETTE['error'], edgecolors='none', rasterized=True)
    ax.set_xlabel("Velocity magnitude", fontsize=FONT_SIZE)
    ax.set_ylabel("Absolute error", fontsize=FONT_SIZE)
    _light_grid(ax)

    # Bottom-Middle: Temporal stability
    ax = fig.add_subplot(gs[1, 1])
    apply_style(ax)
    t_ax = np.arange(len(g["time_mse"])) * DT_SAVED
    mu_m, std_m = np.mean(g["time_mse"]), np.std(g["time_mse"])
    ax.plot(t_ax, g["time_mse"], color=COLOR_PALETTE['temporal'], lw=0.8, alpha=0.9)
    ax.axhline(mu_m, color=COLOR_PALETTE['scatter'], ls="--", lw=0.8)
    ax.fill_between(t_ax, mu_m - 2 * std_m, mu_m + 2 * std_m, color=COLOR_PALETTE['temporal'], alpha=0.08)
    ax.set_xlabel("Time", fontsize=FONT_SIZE)
    ax.set_ylabel("Global MSE", fontsize=FONT_SIZE)
    ax.text(0.97, 0.92, f"Mean = {mu_m:.4f}", transform=ax.transAxes,
            fontsize=FONT_SIZE - 1, ha="right", color=COLOR_PALETTE['scatter'])
    ax.set_xlim(0, min(50, t_ax[-1]))
    ax.set_ylim(bottom=0)
    _light_grid(ax)

    # Bottom-Right: Field profiles
    gs_sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 2], hspace=0.4, height_ratios=[1, 1, 1])
    triang_d = tri.Triangulation(coords_d[:, 0], coords_d[:, 1])
    mean_f = d["mean_field"]
    orig_snap, recon_snap = d["original"], d["reconstructed"]

    if mean_f is not None:
        mean_u_field = mean_f[:, ui_d]
        fluct = orig_snap[:min(50, orig_snap.shape[0]), :, vi_d] - mean_f[:, vi_d]
    else:
        mean_u_field = np.mean(orig_snap[:, :, ui_d], axis=0)
        fluct = orig_snap[:min(50, orig_snap.shape[0]), :, vi_d] - np.mean(orig_snap[:, :, vi_d], axis=0)

    _, _, Vt = np.linalg.svd(fluct, full_matrices=False)
    mode1 = Vt[0]
    err_field = np.mean(np.abs(orig_snap[:, :, ui_d] - recon_snap[:, :, ui_d]), axis=0)

    field_data = [(mean_u_field, "jet", "Mean Ux"), (mode1, "rainbow", "POD mode 1"), (err_field, "gist_rainbow", "Mean absolute error")]

    for k, (fld, cm, ttl) in enumerate(field_data):
        ax = fig.add_subplot(gs_sub[k])
        apply_style(ax)
        im = ax.tricontourf(triang_d, fld, levels=20, cmap=cm)
        ax.set_title(ttl, fontsize=FONT_SIZE - 1, loc="left", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.ax.tick_params(labelsize=FONT_SIZE - 5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"figure02_diagnostic_dashboard_Re{rev}.png")

# ---------------------------------------------------------------------------
# FIGURE 3: Spectral / Temporal Grid
# ---------------------------------------------------------------------------
print("-- Figure 3: Spectral / Temporal Grid --")

grid_re = [r for r in [2500, 10000, 25000] if r in ALL_RE]
n_cols = len(grid_re)

fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 6.5), dpi=DPI)
if n_cols == 1:
    axes = axes[:, None]

fig.suptitle("Spectral / Temporal Analysis", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

row_xlabels = ["Time", "Frequency", "Time"]
row_ylabels = ["Amplitude", "Normalised PSD", "Relative error (%)"]

for ci, rev in enumerate(grid_re):
    d = DATA[rev]
    tc = d["temporal_coeffs"]
    orig_d = d["original"]
    recon_d = d["reconstructed"]
    t_ax = np.arange(orig_d.shape[0]) * DT_SAVED

    # Row 0: temporal modes
    ax = axes[0, ci]
    apply_style(ax)
    if tc is not None and tc.shape[1] >= 2:
        ax.plot(t_ax[:len(tc)], tc[:, 0], color=C_BLUE, lw=0.8, label="Mode 1")
        ax.plot(t_ax[:len(tc)], tc[:, 1], color=C_RED, lw=0.8, ls="--", label="Mode 2")
    ax.set_xlim(0, min(50, t_ax[-1]))
    ax.set_title(f"Re = {rev:,}", fontsize=FONT_SIZE, pad=4)
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel(row_ylabels[0], fontsize=FONT_SIZE - 1)
        ax.legend(fontsize=FONT_SIZE - 2)
    else:
        ax.tick_params(labelleft=False)

    # Row 1: PSD
    ax = axes[1, ci]
    apply_style(ax)
    if tc is not None and tc.shape[1] >= 2:
        for mi, col_m, ls_m in zip([0, 1], [C_BLUE, C_RED], ["-", "--"]):
            n_tc = len(tc[:, mi])
            win = np.hanning(n_tc)
            sig_w = (tc[:, mi] - np.mean(tc[:, mi])) * win
            pwr = np.abs(np.fft.rfft(sig_w))**2
            frq = np.fft.rfftfreq(n_tc, d=DT_SAVED)
            pwr /= np.max(pwr) + 1e-30
            ax.plot(frq, pwr, color=col_m, lw=0.8, ls=ls_m)
        pk_i = np.argmax(pwr[1:]) + 1
        if pk_i < len(frq):
            ax.axvline(frq[pk_i], color=C_GREY, ls=":", lw=0.4, alpha=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.15)
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel(row_ylabels[1], fontsize=FONT_SIZE - 1)
    else:
        ax.tick_params(labelleft=False)

    # Row 2: temporal error
    ax = axes[2, ci]
    apply_style(ax)
    err_tot = np.linalg.norm(orig_d - recon_d, axis=(1, 2)) / \
              (np.linalg.norm(orig_d, axis=(1, 2)) + 1e-30) * 100
    ax.plot(t_ax, err_tot, color=C_BLACK, lw=0.8, label="Total")
    ax.plot(t_ax, GLOBAL[rev]["err_vel_time"], color=C_BLUE, lw=0.6, ls="--", label="Velocity")
    ax.plot(t_ax, GLOBAL[rev]["err_prs_time"], color=C_ORANGE, lw=0.6, ls=":", label="Pressure")
    ax.set_xlim(0, min(50, t_ax[-1]))
    ax.set_ylim(0, max(15, np.max(GLOBAL[rev]["err_prs_time"]) * 1.1))
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel(row_ylabels[2], fontsize=FONT_SIZE - 1)
        ax.legend(fontsize=FONT_SIZE - 2, ncol=1)
    else:
        ax.tick_params(labelleft=False)

plt.tight_layout(h_pad=1.6, w_pad=0.04, rect=[0, 0, 1, 0.96])
fig.canvas.draw()
for ri, xlabel_txt in enumerate(row_xlabels):
    _add_centered_xlabel(fig, axes[ri, :], xlabel_txt)
_save(fig, "figure03_spectral_temporal_grid.png")

# ---------------------------------------------------------------------------
# FIGURE 4: Local Probe Time Series
# ---------------------------------------------------------------------------
print("-- Figure 4: Local Probe Validation --")

probe_re_list = COMPARISON_RE
n_re = len(probe_re_list)
var_order = [("s", "Concentration"), ("U:0", "Ux"), ("U:1", "Uy"), ("p", "Pressure")]

fig, axes = plt.subplots(len(var_order), n_re,
                         figsize=(3.5 * n_re, 1.8 * len(var_order)),
                         sharex='col', sharey='row', dpi=DPI)
fig.suptitle("Local Probe Time Series Validation", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

if n_re == 1:
    axes = axes[:, None]

for ci, rev in enumerate(probe_re_list):
    if rev not in DATA:
        continue
    crd = DATA[rev]["coordinates"]
    dx, dy = crd[:, 0] - PROBE_X, crd[:, 1] - PROBE_Y
    pidx = np.argmin(dx**2 + dy**2)
    t_ax = np.arange(DATA[rev]["original"].shape[0]) * DT_SAVED

    for ri, (vn, vlbl) in enumerate(var_order):
        vi_k = _var_idx(rev, vn)
        ts_t = DATA[rev]["original"][:, pidx, vi_k]
        ts_r = DATA[rev]["reconstructed"][:, pidx, vi_k]

        ax = axes[ri, ci]
        apply_style(ax)
        ax.plot(t_ax, ts_t, color=C_BLUE, lw=0.8, alpha=0.85, label="Target")
        ax.plot(t_ax, ts_r, color=C_RED, lw=0.7, ls="--", label="Predicted")
        ax.set_xlim(0, min(50, t_ax[-1]))
        _light_grid(ax)

        if ri == 0:
            ax.set_title(f"Re = {rev:,}", fontsize=FONT_SIZE, pad=10)
        if ci == 0:
            ax.set_ylabel(vlbl, fontsize=FONT_SIZE - 1)
            if ri == 0:
                ax.legend(fontsize=FONT_SIZE - 1, ncol=1, frameon=False)

        corr = np.corrcoef(ts_t, ts_r)[0, 1]
        ax.text(0.97, 0.05, f"rho = {corr:.3f}", transform=ax.transAxes,
                fontsize=FONT_SIZE - 1, ha="right", va="bottom", color=C_GREY)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.subplots_adjust(hspace=0.0, wspace=0.03, bottom=0.07)
fig.text(0.534, 0.01, 'Time', ha='center', fontsize=FONT_SIZE + 1)
_save(fig, "figure04_local_probe_validation.png")

# ---------------------------------------------------------------------------
# FIGURES 5-11: Comparison Plots (with titles)
# ---------------------------------------------------------------------------

RE_COLORS = {COMPARISON_RE[0]: (C_BLUE, C_RED), COMPARISON_RE[1]: (C_GREEN, C_ORANGE)}
RE_LS_T = "-"
RE_LS_R = "--"

def _row_xlim(re_list, positions, extractor):
    vals = []
    for rev in re_list:
        for xp in positions:
            if rev in RESULTS and xp in RESULTS[rev]:
                v = extractor(RESULTS[rev][xp])
                fv = v[np.isfinite(v)]
                if len(fv):
                    vals.extend([fv.min(), fv.max()])
    if not vals:
        return -1, 1
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo) if hi != lo else 0.1
    return lo - pad, hi + pad

def _comparison_plot(nrows, figw, figh, row_configs, fig_title, fig_name):
    npos = len(STREAMWISE_POSITIONS)
    fig, axes = plt.subplots(nrows, npos, figsize=(figw, figh), dpi=DPI)
    if nrows == 1:
        axes = axes[None, :]
    
    fig.suptitle(fig_title, fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

    row_xlabels = []

    for ri, rc in enumerate(row_configs):
        vs_y = rc.get("vs_y", True)
        row_xlabels.append(rc.get("xlabel", ""))

        if vs_y:
            xl = _row_xlim(COMPARISON_RE, STREAMWISE_POSITIONS,
                           lambda r, _rc=rc: np.concatenate([_rc["extractor_t"](r),
                                                              _rc["extractor_r"](r)]))
        else:
            all_vals = []
            for rev in COMPARISON_RE:
                for xp in STREAMWISE_POSITIONS:
                    if rev in RESULTS and xp in RESULTS[rev]:
                        rr = RESULTS[rev][xp]
                        vt = rc["extractor_t"](rr)
                        vr = rc["extractor_r"](rr)
                        fvt = vt[np.isfinite(vt)]
                        fvr = vr[np.isfinite(vr)]
                        if len(fvt):
                            all_vals.extend([fvt.min(), fvt.max()])
                        if len(fvr):
                            all_vals.extend([fvr.min(), fvr.max()])
            if all_vals:
                pad_v = 0.08 * (max(all_vals) - min(all_vals)) if max(all_vals) != min(all_vals) else 0.1
                yl_auto = (min(all_vals) - pad_v, max(all_vals) + pad_v)
            else:
                yl_auto = (0, 1)

        for ci, xp in enumerate(STREAMWISE_POSITIONS):
            ax = axes[ri, ci]
            apply_style(ax)
            ax.set_title(f"x = {xp}", fontsize=FONT_SIZE, pad=4)

            for rev in COMPARISON_RE:
                if rev not in RESULTS or xp not in RESULTS[rev]:
                    continue
                rr = RESULTS[rev][xp]
                ct, ca = RE_COLORS[rev]
                vt = rc["extractor_t"](rr)
                vr = rc["extractor_r"](rr)

                lbl_t = f"Re={rev} Target" if (ri == 0 and ci == 0) else None
                lbl_r = f"Re={rev} Predicted" if (ri == 0 and ci == 0) else None

                if vs_y:
                    y = rr["y"]
                    ax.plot(vt, y, color=ct, ls=RE_LS_T, lw=1.2, label=lbl_t)
                    ax.plot(vr, y, color=ca, ls=RE_LS_R, lw=0.9, label=lbl_r)
                else:
                    x_ext = rc.get("x_extractor", None)
                    if x_ext is not None:
                        x_t = x_ext(rr, "target")
                        x_r = x_ext(rr, "recon")
                    else:
                        x_t = np.arange(len(vt))
                        x_r = np.arange(len(vr))
                    ax.plot(x_t, vt, color=ct, ls=RE_LS_T, lw=1.2, label=lbl_t)
                    ax.plot(x_r, vr, color=ca, ls=RE_LS_R, lw=0.9, label=lbl_r)

            if vs_y:
                ax.set_xlim(xl)
            else:
                if "xlim" in rc:
                    ax.set_xlim(rc["xlim"])
                ax.set_ylim(yl_auto)

            if ci == 0:
                ax.set_ylabel(rc.get("ylabel", ""), fontsize=FONT_SIZE - 1)
            else:
                ax.tick_params(labelleft=False)

            if "axvline" in rc:
                ax.axvline(rc["axvline"], color=C_GREY, ls=":", lw=0.5, alpha=0.5)
            _light_grid(ax)

    if nrows * npos > 0:
        axes[0, 0].legend(fontsize=FONT_SIZE - 2, loc="best", ncol=1)

    plt.tight_layout(h_pad=1.6, w_pad=0.6, rect=[0, 0, 1, 0.96])
    fig.canvas.draw()

    for ri, xlabel_txt in enumerate(row_xlabels):
        if not xlabel_txt:
            continue
        _add_centered_xlabel(fig, axes[ri, :], xlabel_txt)
    _save(fig, fig_name)

# FIGURE 5: Mean Flow Profiles
print("-- Figure 5: Mean Flow Profiles --")
_comparison_plot(3, 7.2, 6.5, [
    dict(extractor_t=lambda r: r["t_u_mean"] / r["U_ref"],
         extractor_r=lambda r: r["r_u_mean"] / r["U_ref"],
         ylabel=YLABEL_Y, xlabel="Mean Streamwise Velocity"),
    dict(extractor_t=lambda r: r["t_s_mean"],
         extractor_r=lambda r: r["r_s_mean"],
         ylabel=YLABEL_Y, xlabel="Mean concentration"),
    dict(extractor_t=lambda r: r["t_pdf_v"],
         extractor_r=lambda r: r["r_pdf_v"],
         x_extractor=lambda r, which: r["t_pdf_c"] if which == "target" else r["r_pdf_c"],
         vs_y=False,
         xlim=(0, 1),
         ylabel="Probability density", xlabel="Concentration"),
], "Mean Flow Profiles", "figure05_mean_flow_profiles.png")

# FIGURE 6: Turbulent Intensities
print("-- Figure 6: Turbulent Intensities --")
_comparison_plot(4, 7.2, 7.5, [
    dict(extractor_t=lambda r: r["t_u_rms"], extractor_r=lambda r: r["r_u_rms"],
         ylabel=YLABEL_Y, xlabel="Streamwise Velocity RMS"),
    dict(extractor_t=lambda r: r["t_v_rms"], extractor_r=lambda r: r["r_v_rms"],
         ylabel=YLABEL_Y, xlabel="Wall-normal Velocity RMS"),
    dict(extractor_t=lambda r: r["t_s_rms"], extractor_r=lambda r: r["r_s_rms"],
         ylabel=YLABEL_Y, xlabel="Concentration RMS"),
    dict(extractor_t=lambda r: r["t_tke"], extractor_r=lambda r: r["r_tke"],
         ylabel=YLABEL_Y, xlabel="Turbulent kinetic energy"),
], "Turbulent Intensities", "figure06_turbulent_intensities.png")

# FIGURE 7: Transport & Stresses
print("-- Figure 7: Transport & Stresses --")
_comparison_plot(3, 7.2, 6.5, [
    dict(extractor_t=lambda r: r["t_uv"], extractor_r=lambda r: r["r_uv"],
         ylabel=YLABEL_Y, xlabel="Reynolds stress", axvline=0),
    dict(extractor_t=lambda r: r["t_chi"], extractor_r=lambda r: r["r_chi"],
         ylabel=YLABEL_Y, xlabel="Scalar dissipation"),
    dict(extractor_t=lambda r: r["t_P_tke"], extractor_r=lambda r: r["r_P_tke"],
         ylabel=YLABEL_Y, xlabel="TKE production"),
], "Transport & Stresses", "figure07_transport_stresses.png")

# FIGURE 8: Coherent Structures
print("-- Figure 8: Coherent Structures --")
_comparison_plot(4, 7.2, 7.5, [
    dict(extractor_t=lambda r: r["t_du_dy"], extractor_r=lambda r: r["r_du_dy"],
         ylabel=YLABEL_Y, xlabel="Mean shear", axvline=0),
    dict(extractor_t=lambda r: r["t_omega_rms"], extractor_r=lambda r: r["r_omega_rms"],
         ylabel=YLABEL_Y, xlabel="Vorticity fluctuation"),
    dict(extractor_t=lambda r: r["t_enstrophy"], extractor_r=lambda r: r["r_enstrophy"],
         ylabel=YLABEL_Y, xlabel="Enstrophy"),
    dict(extractor_t=lambda r: r["t_intermittency"], extractor_r=lambda r: r["r_intermittency"],
         ylabel=YLABEL_Y, xlabel="Intermittency factor"),
], "Coherent Structures", "figure08_coherent_structures.png")

# FIGURE 9: Layer Thickness Evolution
print("-- Figure 9: Layer Thickness Evolution --")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=DPI)
fig.suptitle("Mixing Layer Thickness Evolution", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

RE_PALETTE = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_CYAN, C_BLACK]
re_colors_map = {rev: RE_PALETTE[i % len(RE_PALETTE)] for i, rev in enumerate(ALL_RE)}

for k, qty_key in enumerate(["delta_omega", "theta", "delta_s"]):
    ax = axes[k]
    apply_style(ax)
    for rev in ALL_RE:
        xv, yv_t, yv_r = [], [], []
        for xp in STREAMWISE_POSITIONS:
            if rev in RESULTS and xp in RESULTS[rev]:
                xv.append(xp)
                yv_t.append(RESULTS[rev][xp]["t_scales"][qty_key])
                yv_r.append(RESULTS[rev][xp]["r_scales"][qty_key])
        if xv:
            col = re_colors_map[rev]
            ax.plot(xv, yv_t, "o-", color=col, lw=1.0, ms=3.5,
                    label=f"Re={rev}" if k == 0 else None)
            ax.plot(xv, yv_r, marker="x", ls="--", color=col, lw=0.7, ms=3.5, alpha=0.55)

    ylabels = ["Vorticity thickness", "Momentum thickness", "Concentration thickness"]
    ax.set_ylabel(ylabels[k], fontsize=FONT_SIZE - 1)
    _light_grid(ax)

style_handles = [
    Line2D([], [], color=C_GREY, ls="-", marker="o", ms=3.5, lw=1.0, label="Target"),
    Line2D([], [], color=C_GREY, ls="--", marker="x", ms=3.5, lw=0.7, alpha=0.55, label="Predicted"),
]
re_handles, re_labels = axes[0].get_legend_handles_labels()
axes[0].legend(style_handles + re_handles,
               [h.get_label() for h in style_handles] + re_labels,
               fontsize=FONT_SIZE - 3, ncol=2, columnspacing=0.2, handletextpad=0.2, loc="upper left")
plt.tight_layout(w_pad=0.8, rect=[0, 0, 1, 0.95])
_add_centered_xlabel(fig, axes, "Streamwise position")
_save(fig, "figure09_layer_thickness_evolution.png")

# FIGURE 10: Spectral Comparison
print("-- Figure 10: Spectral Comparison --")
npos = len(STREAMWISE_POSITIONS)
fig, axes = plt.subplots(1, npos, figsize=(1.9 * npos, 2.6), dpi=DPI)
fig.suptitle("Spectral Comparison", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

for ci, xp in enumerate(STREAMWISE_POSITIONS):
    ax = axes[ci]
    apply_style(ax)
    ax.set_title(f"x = {xp}", fontsize=FONT_SIZE, pad=4)
    for rev in COMPARISON_RE:
        if rev not in RESULTS or xp not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][xp]
        ct, ca = RE_COLORS[rev]
        f_o, p_o = rr["t_spec"]["freqs"], rr["t_spec"]["psd"]
        f_r, p_r = rr["r_spec"]["freqs"], rr["r_spec"]["psd"]
        p_o_n = p_o / (np.max(p_o) + 1e-30)
        p_r_n = p_r / (np.max(p_r) + 1e-30)
        ax.semilogy(f_o, p_o_n + 1e-15, color=ct, lw=0.9,
                    label=f"Re={rev} Target" if ci == 0 else None)
        ax.semilogy(f_r, p_r_n + 1e-15, color=ca, ls="--", lw=0.7, alpha=0.7,
                    label=f"Re={rev} Predicted" if ci == 0 else None)
    ax.set_xlim(0, 1.0)
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel("Normalised PSD", fontsize=FONT_SIZE - 1)
        ax.legend(fontsize=FONT_SIZE - 2)
    else:
        ax.tick_params(labelleft=False)

plt.tight_layout(w_pad=0.4, rect=[0, 0, 1, 0.95])
_add_centered_xlabel(fig, axes, "Frequency")
_save(fig, "figure10_spectral_comparison.png")

# FIGURE 11: Strouhal & Frequency vs Re
print("-- Figure 11: Strouhal & Frequency vs Re --")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6), dpi=DPI)
fig.suptitle("Strouhal Number & Peak Frequency vs Reynolds Number", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")
apply_style(ax1)
apply_style(ax2)

pos_colors = [C_BLUE, C_RED, C_GREEN, C_ORANGE]
for ip, xp in enumerate(STREAMWISE_POSITIONS):
    re_v, st_v, fp_v = [], [], []
    for rev in ALL_RE:
        if rev in RESULTS and xp in RESULTS[rev]:
            re_v.append(rev)
            st_v.append(RESULTS[rev][xp]["strouhal"])
            fp_v.append(RESULTS[rev][xp]["t_spec"]["f_peak"])
    if re_v:
        ax1.plot(re_v, st_v, "o-", ms=3, lw=0.9, color=pos_colors[ip], label=f"x={xp}")
        ax2.plot(re_v, fp_v, "o-", ms=3, lw=0.9, color=pos_colors[ip], label=f"x={xp}")

ax1.axhline(0.032, color=C_GREY, ls=":", lw=0.5, alpha=0.6)
ax1.set_ylabel("Strouhal number", fontsize=FONT_SIZE - 1)
ax1.legend(fontsize=FONT_SIZE - 2, ncol=2)
_light_grid(ax1)

ax2.set_ylabel("Peak frequency", fontsize=FONT_SIZE - 1)
ax2.legend(fontsize=FONT_SIZE - 2, ncol=2)
_light_grid(ax2)

plt.tight_layout(w_pad=0.8, rect=[0, 0, 1, 0.92])
_add_centered_xlabel(fig, [ax1, ax2], "Reynolds number")
_save(fig, "figure11_strouhal_frequency.png")

# ---------------------------------------------------------------------------
# FIGURE 12: Model Validation Metrics
# ---------------------------------------------------------------------------
print("\n-- Figure 12: Model Validation Metrics --")
fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.4), dpi=DPI)
fig.suptitle("Model Validation Metrics vs Reynolds Number", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

panel_configs = [
    (axes[0], "Velocity relative error (%)", lambda rr: rr["val_u"]["relative_error"], "s-", None),
    (axes[1], "Velocity mean coherence", lambda rr: rr["val_u"]["mean_coherence"], "^-", None),
    (axes[2], "Concentration profile correlation", lambda rr: rr["val_s"]["profile_corr"], "D-", (0.99, 1.001)),
]

for (ax, ylabel, extractor, marker, ylim) in panel_configs:
    apply_style(ax)
    for ip, xp in enumerate(STREAMWISE_POSITIONS):
        re_v, vals = [], []
        for rev in ALL_RE:
            if rev in RESULTS and xp in RESULTS[rev]:
                re_v.append(rev)
                vals.append(extractor(RESULTS[rev][xp]))
        if re_v:
            ax.plot(re_v, vals, marker, ms=4, lw=1.2, color=pos_colors[ip], label=f"x={xp}")
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE - 2)
    ax.set_xlabel("Reynolds number", fontsize=FONT_SIZE - 2)
    if ylim:
        ax.set_ylim(*ylim)
    _light_grid(ax)

axes[0].legend(fontsize=FONT_SIZE - 1, ncol=2, loc="best")
plt.tight_layout(w_pad=0.6, rect=[0, 0, 1, 0.92])
_save(fig, "figure12_validation_metrics.png")

# ---------------------------------------------------------------------------
# FIGURE 13: Mixing Performance vs Re
# ---------------------------------------------------------------------------
print("\n-- Figure 13: Mixing Performance --")
fig, axes_13 = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
fig.suptitle("Mixing Performance vs Reynolds Number", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

panel_cfgs_13 = [
    ("Peak mixedness", "o-", lambda rr: np.max(rr["t_mixedness"])),
    ("Peak TKE", "s-", lambda rr: np.max(rr["t_tke"])),
    ("Peak turbulence intensity (%)", "^-", lambda rr: np.max(rr["t_ti"]) * 100),
]

for ci, (ylabel, marker, extractor) in enumerate(panel_cfgs_13):
    ax = axes_13[ci]
    apply_style(ax)
    for ip, xp in enumerate(STREAMWISE_POSITIONS):
        re_v, vals = [], []
        for rev in ALL_RE:
            if rev in RESULTS and xp in RESULTS[rev]:
                re_v.append(rev)
                vals.append(extractor(RESULTS[rev][xp]))
        if re_v:
            ax.plot(re_v, vals, marker, ms=4, lw=1.2, color=pos_colors[ip], label=f"x={xp}")
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE - 2)
    ax.set_xlabel("Reynolds number", fontsize=FONT_SIZE - 2)
    ax.legend(fontsize=FONT_SIZE - 2, ncol=2)
    _light_grid(ax)

plt.tight_layout(w_pad=0.6, rect=[0, 0, 1, 0.92])
_save(fig, "figure13_mixing_performance.png")

# ---------------------------------------------------------------------------
# FIGURE 14: Detailed Comparison at x=6.0
# ---------------------------------------------------------------------------
print("\n-- Figure 14: Detailed Comparison at x=6.0 --")
xd = 6.0
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), dpi=DPI)
fig.suptitle(f"Detailed Comparison at x = {xd}", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

for ki, rev in enumerate(COMPARISON_RE):
    if rev not in RESULTS or xd not in RESULTS[rev]:
        continue
    rr = RESULTS[rev][xd]
    y = rr["y"]
    ct, ca = RE_COLORS[rev]

    axes[0, 0].plot(rr["t_u_mean"], y, color=ct, ls="-", lw=1.0, label=f"Re={rev} Target")
    axes[0, 0].plot(rr["r_u_mean"], y, color=ca, ls="--", lw=0.8, label=f"Re={rev} Predicted")
    axes[0, 1].plot(np.abs(rr["t_u_mean"] - rr["r_u_mean"]), y, color=ct, lw=0.9, label=f"Re={rev}")
    axes[0, 1].fill_betweenx(y, 0, np.abs(rr["t_u_mean"] - rr["r_u_mean"]), color=ct, alpha=0.08)
    axes[0, 2].plot(rr["t_s_mean"], y, color=ct, ls="-", lw=1.0)
    axes[0, 2].plot(rr["r_s_mean"], y, color=ca, ls="--", lw=0.8)
    axes[1, 0].plot(rr["t_v_mean"], y, color=ct, ls="-", lw=1.0)
    axes[1, 0].plot(rr["r_v_mean"], y, color=ca, ls="--", lw=0.8)
    axes[1, 1].plot(rr["t_du_dy"], y, color=ct, ls="-", lw=1.0)
    axes[1, 1].plot(rr["r_du_dy"], y, color=ca, ls="--", lw=0.8)

plot_info = [
    ("Mean Ux", YLABEL_Y), ("Absolute error", YLABEL_Y), ("Mean concentration", YLABEL_Y),
    ("Mean Uy", YLABEL_Y), ("Mean shear", YLABEL_Y),
]
for k in range(5):
    ax = axes.flat[k]
    apply_style(ax)
    ax.set_xlabel(plot_info[k][0], fontsize=FONT_SIZE - 1)
    if k % 3 == 0:
        ax.set_ylabel(plot_info[k][1], fontsize=FONT_SIZE - 1)
    else:
        ax.tick_params(labelleft=False)
    _light_grid(ax)

axes[0, 0].legend(fontsize=FONT_SIZE - 2, ncol=1)
axes[0, 1].legend(fontsize=FONT_SIZE - 2)

ax = axes[1, 2]
apply_style(ax)
ax.axis("off")
lines = []
for rev in COMPARISON_RE:
    if rev in RESULTS and xd in RESULTS[rev]:
        rr = RESULTS[rev][xd]
        lines.append(f"Re = {rev}")
        lines.append(f"  delta_omega = {rr['t_scales']['delta_omega']:.4f}")
        lines.append(f"  theta = {rr['t_scales']['theta']:.4f}")
        lines.append(f"  St = {rr['strouhal']:.4f}")
        lines.append(f"  rho_U = {rr['val_u']['profile_corr']:.4f}")
        lines.append(f"  rho_c = {rr['val_s']['profile_corr']:.4f}")
        lines.append("")
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=FONT_SIZE - 1,
        va="top", family="DejaVu Sans Mono")

plt.tight_layout(h_pad=0.6, w_pad=0.5, rect=[0, 0, 1, 0.95])
_save(fig, "figure14_detailed_comparison_x6.png")

# ---------------------------------------------------------------------------
# FIGURE 15: Coherence Profiles
# ---------------------------------------------------------------------------
print("-- Figure 15: Coherence Profiles --")
COHERENCE_RE = [2500, 10000, 20000, 30000]
RE4_COLORS = [C_BLUE, C_RED, C_GREEN, C_PURPLE]
re4_colors_map = {rev: RE4_COLORS[i % len(RE4_COLORS)] for i, rev in enumerate(COHERENCE_RE)}

fig, axes = plt.subplots(1, npos, figsize=(1.9 * npos, 2.8), dpi=DPI)
fig.suptitle("Spectral Coherence Profiles", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

for ci, xp in enumerate(STREAMWISE_POSITIONS):
    ax = axes[ci]
    apply_style(ax)
    ax.set_title(f"x = {xp}", fontsize=FONT_SIZE, pad=4)
    for rev in COHERENCE_RE:
        if rev not in RESULTS or xp not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][xp]
        color = re4_colors_map[rev]
        ax.plot(rr["val_u"]["coh_freqs"], rr["val_u"]["coherence"],
                color=color, lw=0.9, label=f"Re={rev}" if ci == 0 else None)
    ax.axhline(0.5, color=C_GREY, ls=":", lw=0.4, alpha=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel("Coherence", fontsize=FONT_SIZE - 1)
        ax.legend(fontsize=FONT_SIZE - 2, loc="lower left")
    else:
        ax.tick_params(labelleft=False)

plt.tight_layout(w_pad=0.4, rect=[0, 0, 1, 0.92])
_add_centered_xlabel(fig, axes, "Frequency")
_save(fig, "figure15_coherence_profiles.png")

# ---------------------------------------------------------------------------
# FIGURE 16: Concentration PDF
# ---------------------------------------------------------------------------
print("-- Figure 16: Concentration PDF --")
fig, axes = plt.subplots(1, npos, figsize=(1.9 * npos, 2.6), dpi=DPI)
fig.suptitle("Concentration Probability Density Function", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

for ci, xp in enumerate(STREAMWISE_POSITIONS):
    ax = axes[ci]
    apply_style(ax)
    ax.set_title(f"x = {xp}", fontsize=FONT_SIZE, pad=4)
    for rev in COMPARISON_RE:
        if rev not in RESULTS or xp not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][xp]
        ct, ca = RE_COLORS[rev]
        ax.plot(rr["t_pdf_c"], rr["t_pdf_v"], color=ct, lw=0.9,
                label=f"Re={rev} Target" if ci == 0 else None)
        ax.plot(rr["r_pdf_c"], rr["r_pdf_v"], color=ca, ls="--", lw=0.7,
                label=f"Re={rev} Predicted" if ci == 0 else None)
    ax.set_xlim(0, 1)
    _light_grid(ax)
    if ci == 0:
        ax.set_ylabel("Probability density", fontsize=FONT_SIZE - 1)
        ax.legend(fontsize=FONT_SIZE - 2)
    else:
        ax.tick_params(labelleft=False)

plt.tight_layout(w_pad=0.4, rect=[0, 0, 1, 0.92])
_add_centered_xlabel(fig, axes, "Concentration")
_save(fig, "figure16_concentration_pdf.png")

# ---------------------------------------------------------------------------
# FIGURE 17: Mean flow profiles at x=5.0
# ---------------------------------------------------------------------------
print("\n-- Figure 17: Mean flow profiles at x=5.0 --")
X_POS = 5.0

fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
fig.suptitle(f"Mean Flow Profiles at x = {X_POS}", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

panel_cfgs_17 = [
    ("Mean streamwise velocity", YLABEL_Y, True),
    ("Mean concentration", YLABEL_Y, True),
    ("Concentration", "Probability density", False),
]

for ci, (xlabel, ylabel, vs_y) in enumerate(panel_cfgs_17):
    ax = axes[ci]
    apply_style(ax)

    for rev in COMPARISON_RE:
        if rev not in RESULTS or X_POS not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][X_POS]
        ct, ca = RE_COLORS[rev]

        if ci == 0:
            vt, vr = rr["t_u_mean"] / rr["U_ref"], rr["r_u_mean"] / rr["U_ref"]
            ax.plot(vt, rr["y"], color=ct, ls="-", lw=1.4, label=f"Re={rev} Target")
            ax.plot(vr, rr["y"], color=ca, ls="--", lw=1.1, label=f"Re={rev} Predicted")
        elif ci == 1:
            ax.plot(rr["t_s_mean"], rr["y"], color=ct, ls="-", lw=1.4, label=f"Re={rev} Target")
            ax.plot(rr["r_s_mean"], rr["y"], color=ca, ls="--", lw=1.1, label=f"Re={rev} Predicted")
        else:
            ax.plot(rr["t_pdf_c"], rr["t_pdf_v"], color=ct, ls="-", lw=1.4, label=f"Re={rev} Target")
            ax.plot(rr["r_pdf_c"], rr["r_pdf_v"], color=ca, ls="--", lw=1.1, label=f"Re={rev} Predicted")

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE - 2)
    if not vs_y:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE - 2)
        ax.set_xlim(0, 1)
    _light_grid(ax)

    if ci == 0:
        ax.legend(fontsize=FONT_SIZE - 2, loc="best", ncol=1, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
_save(fig, "figure17_mean_flow_profiles_x5.png")

# ---------------------------------------------------------------------------
# FIGURE 18: Turbulent intensities at x=5.0
# ---------------------------------------------------------------------------
print("\n-- Figure 18: Turbulent intensities at x=5.0 --")

fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
fig.suptitle(f"Turbulent Intensities at x = {X_POS}", fontsize=FONT_SIZE + 2, y=0.99, x=0.02, ha="left")

panel_cfgs_18 = [
    (lambda rr: rr["t_u_rms"], lambda rr: rr["r_u_rms"], "Streamwise velocity RMS"),
    (lambda rr: rr["t_v_rms"], lambda rr: rr["r_v_rms"], "Wall-normal velocity RMS"),
    (lambda rr: rr["t_s_rms"], lambda rr: rr["r_s_rms"], "Concentration RMS"),
    (lambda rr: rr["t_tke"], lambda rr: rr["r_tke"], "Turbulent kinetic energy"),
]

for ci, (ext_t, ext_r, xlabel) in enumerate(panel_cfgs_18):
    ax = axes[ci]
    apply_style(ax)

    for rev in COMPARISON_RE:
        if rev not in RESULTS or X_POS not in RESULTS[rev]:
            continue
        rr = RESULTS[rev][X_POS]
        ct, ca = RE_COLORS[rev]
        ax.plot(ext_t(rr), rr["y"], color=ct, ls="-", lw=1.4,
                label=f"Re={rev} Target" if ci == 0 else None)
        ax.plot(ext_r(rr), rr["y"], color=ca, ls="--", lw=1.1,
                label=f"Re={rev} Predicted" if ci == 0 else None)

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE - 2)
    _light_grid(ax)

    if ci == 0:
        ax.set_ylabel(YLABEL_Y, fontsize=FONT_SIZE - 2)
        ax.legend(fontsize=FONT_SIZE - 2, loc="best", ncol=1, frameon=False)
    else:
        ax.tick_params(labelleft=False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
_save(fig, "figure18_turbulent_intensities_x5.png")

# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  ANALYSIS COMPLETE — ALL 18 FIGURES SAVED")
print(f"  Output directory: {OUTPUT_DIR}/")
print("=" * 72)
