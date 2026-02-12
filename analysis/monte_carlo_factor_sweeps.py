"""
One factor at a time Monte Carlo sweeps with comparison plots.

Each figure varies a single uncertainty factor via Monte Carlo sampling
and compares trajectory/controller combinations:
  - S curve + Standard PD
  - Fourth order + Standard PD

Sweep factors:
  1. Inertia scaling
  2. Modal frequency scaling
  3. Modal damping scaling
  4. Disturbance frequency (sinusoidal torque)
  5. Sensor noise frequency (sinusoidal)

Outputs per factor: a PNG scatter + binned median plot and a CSV of
raw per sample metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure package paths are on sys.path
from pathlib import Path
_analysis_dir = Path(__file__).parent.resolve()
_basilisk_dir = _analysis_dir.parent
_src_dir = _basilisk_dir / "src"
_scripts_dir = _basilisk_dir / "scripts"
for path in (_src_dir, _scripts_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_mission as ms
from basilisk_sim.spacecraft_properties import HUB_INERTIA, compute_effective_inertia, compute_modal_gains


COMBOS = [
    ("s_curve", "standard_pd", "S-curve + Standard PD", "#ff7f0e"),
    ("fourth", "standard_pd", "Fourth-order + Standard PD", "#1f77b4"),
]

METRICS = [
    ("rms_vibration_mm", "RMS Vibration (mm)"),
    ("rms_pointing_error_deg", "RMS Pointing Error (deg)"),
    ("peak_torque_nm", "Peak Torque (N*m)"),
    ("rms_torque_nm", "RMS Torque (N*m)"),
]

FREQ_METRICS = [
    ("pointing_band_rms_deg", "Pointing RMS near f0 (deg)"),
    ("vibration_band_rms_mm", "Vibration RMS near f0 (mm)"),
    ("pointing_psd_at_f0_db", "Pointing PSD at f0 (dB)"),
    ("vibration_psd_at_f0_db", "Vibration PSD at f0 (dB)"),
]


def _combo_key(method: str, controller: str) -> str:
    """Build a unique key from method and controller names."""
    return f"{method}_{controller}"


def _format_eta(seconds: float) -> str:
    """Format remaining seconds as a human readable H:MM:SS string."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    secs = int(round(seconds))
    mins, sec = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {sec:02d}s"
    return f"{mins:02d}m {sec:02d}s"


def _update_progress(
    prefix: str,
    current: int,
    total: int,
    start_time: float,
    last_update: float,
    min_interval_s: float = 0.5,
    width: int = 32,
) -> float:
    """Print a single line progress bar with ETA. Returns the timestamp of last update."""
    now = time.time()
    if current < total and (now - last_update) < min_interval_s:
        return last_update
    total = max(total, 1)
    ratio = min(1.0, max(0.0, current / total))
    filled = int(round(width * ratio))
    bar = "=" * filled + "-" * (width - filled)
    elapsed = now - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else float("inf")
    eta = _format_eta(remaining)
    msg = f"\r{prefix} [{bar}] {current}/{total} ({ratio*100:5.1f}%) ETA {eta}"
    print(msg, end="", flush=True)
    if current >= total:
        print()
    return now


def _sample_uniform(rng: np.random.Generator, vmin: float, vmax: float, n: int) -> np.ndarray:
    """Draw *n* samples uniformly between *vmin* and *vmax*."""
    return rng.uniform(vmin, vmax, size=n)


def _sample_log_uniform(rng: np.random.Generator, vmin: float, vmax: float, n: int) -> np.ndarray:
    """Draw *n* samples from a log uniform distribution over [vmin, vmax]."""
    return np.exp(rng.uniform(np.log(vmin), np.log(vmax), size=n))


def _sample_uniform_stratified(
    rng: np.random.Generator,
    vmin: float,
    vmax: float,
    n_samples: int,
    n_bins: int,
) -> np.ndarray:
    """Sample uniformly while guaranteeing coverage across bins."""
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        return np.array([])
    n_bins = int(max(1, n_bins))
    edges = np.linspace(float(vmin), float(vmax), n_bins + 1)
    counts = np.full(n_bins, n_samples // n_bins, dtype=int)
    counts[: n_samples % n_bins] += 1
    parts = []
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        parts.append(rng.uniform(edges[i], edges[i + 1], size=count))
    values = np.concatenate(parts) if parts else np.array([])
    rng.shuffle(values)
    return values


def _sample_log_uniform_stratified(
    rng: np.random.Generator,
    vmin: float,
    vmax: float,
    n_samples: int,
    n_bins: int,
) -> np.ndarray:
    """Sample log uniformly while guaranteeing log space bin coverage."""
    if vmin <= 0 or vmax <= 0:
        raise ValueError("Log-uniform bounds must be positive")
    log_vals = _sample_uniform_stratified(
        rng,
        np.log(float(vmin)),
        np.log(float(vmax)),
        n_samples,
        n_bins,
    )
    return np.exp(log_vals)


def _filter_cutoff(cfg: ms.MissionConfig) -> float:
    """Return the configured low pass filter cutoff, defaulting to 8 Hz."""
    return float(cfg.control_filter_cutoff_hz) if cfg.control_filter_cutoff_hz is not None else 8.0


def _run_vizard_demo_batch(overrides: Dict[str, object], output_dir: str) -> None:
    """Launch run_vizard_demo.py for every method/controller combo with the given config overrides."""
    output_dir_abs = os.path.abspath(output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    cfg_path = os.path.join(output_dir_abs, "sweep_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f)

    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_vizard_demo.py")
    script_path = os.path.abspath(script_path)

    for method, controller, _, _ in COMBOS:
        cmd = [
            sys.executable,
            script_path,
            method,
            "--controller",
            controller,
            "--mode",
            "combined",
            "--config",
            cfg_path,
            "--output-dir",
            output_dir_abs,
        ]
        subprocess.run(cmd, cwd=output_dir_abs, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _estimate_total_runtime(
    sample_overrides: Dict[str, object],
    work_dir: str,
    total_samples: int,
) -> float:
    """Run a single sample batch and extrapolate total wall clock time."""
    start = time.time()
    _run_vizard_demo_batch(sample_overrides, work_dir)
    elapsed = time.time() - start
    return elapsed * total_samples


def _compute_post_slew_stats(
    time: np.ndarray, values: np.ndarray, slew_duration_s: float
) -> Tuple[float, float]:
    """Return (RMS, peak) of *values* in the post slew window."""
    if len(time) == 0 or len(values) == 0:
        return float("nan"), float("nan")
    mask = time >= slew_duration_s
    series = values[mask] if np.any(mask) else values
    rms = float(np.sqrt(np.mean(series**2))) if len(series) else float("nan")
    peak = float(np.max(np.abs(series))) if len(series) else float("nan")
    return rms, peak


def _compute_band_metrics(
    time: np.ndarray,
    series: np.ndarray,
    target_freq: float,
    band_half_hz: float,
) -> Tuple[float, float]:
    """Return band limited RMS and PSD (dB) at *target_freq*."""
    if len(time) == 0 or len(series) == 0 or not np.isfinite(target_freq):
        return float("nan"), float("nan")
    freq, psd = ms._compute_psd(time, series)
    if len(freq) == 0 or len(psd) == 0:
        return float("nan"), float("nan")
    mask = (freq >= target_freq - band_half_hz) & (freq <= target_freq + band_half_hz)
    if np.any(mask):
        df = np.gradient(freq[mask])
        band_rms = float(np.sqrt(np.sum(psd[mask] * df)))
    else:
        band_rms = float("nan")
    idx = int(np.argmin(np.abs(freq - target_freq)))
    psd_at = psd[idx] if idx >= 0 and idx < len(psd) else float("nan")
    psd_db = float(10.0 * np.log10(psd_at)) if np.isfinite(psd_at) and psd_at > 0 else float("nan")
    return band_rms, psd_db


def _build_config(
    base: ms.MissionConfig,
    inertia_scale: float = 1.0,
    freq_scale: float = 1.0,
    damping_scale: float = 1.0,
) -> ms.MissionConfig:
    """Create a perturbed MissionConfig by scaling inertia, modal frequency, and damping."""
    cfg = ms.MissionConfig(**asdict(base))
    cfg.inertia = compute_effective_inertia(hub_inertia=HUB_INERTIA.copy() * inertia_scale)
    cfg.modal_freqs_hz = [f * freq_scale for f in cfg.modal_freqs_hz]
    cfg.modal_damping = [max(0.001, d * damping_scale) for d in cfg.modal_damping]
    cfg.modal_gains = compute_modal_gains(cfg.inertia, cfg.rotation_axis)
    cfg.control_modal_gains = cfg.modal_gains
    return cfg


def _collect_metrics(
    config: ms.MissionConfig,
    data_dir: str,
    target_freq: float | None = None,
) -> Dict[str, Dict[str, float]]:
    """Load pointing and vibration data from *data_dir* and compute per combo performance metrics."""
    metrics: Dict[str, Dict[str, float]] = {}
    pointing_data = ms._load_all_pointing_data(data_dir, config=config, generate_if_missing=False)
    feedback_data = ms._collect_feedback_data(config, data_dir=data_dir, prefer_npz=True)

    for method, controller, _, _ in COMBOS:
        key = f"{method}_{controller}"
        combo_metrics: Dict[str, float] = {
            "rms_vibration_mm": float("nan"),
            "rms_pointing_error_deg": float("nan"),
            "peak_torque_nm": float("nan"),
            "rms_torque_nm": float("nan"),
            "pointing_band_rms_deg": float("nan"),
            "vibration_band_rms_mm": float("nan"),
            "pointing_psd_at_f0_db": float("nan"),
            "vibration_psd_at_f0_db": float("nan"),
        }

        # Pointing error
        data = pointing_data.get(method, {}).get(controller)
        if data:
            time = np.array(data.get("time", []), dtype=float)
            errors = ms._extract_pointing_error(data, config=config)
            time, aligned = ms._align_series(time, errors)
            errors = aligned[0]
            rms, _ = _compute_post_slew_stats(time, errors, config.slew_duration_s)
            combo_metrics["rms_pointing_error_deg"] = rms
            if target_freq is not None:
                band_half = min(1.0, max(0.05, 0.05 * target_freq))
                band_rms, psd_db = _compute_band_metrics(time, errors, target_freq, band_half)
                combo_metrics["pointing_band_rms_deg"] = band_rms
                combo_metrics["pointing_psd_at_f0_db"] = psd_db

        # Vibration + torque
        fb_key = f"{method}_{controller}"
        fb_data = feedback_data.get(fb_key)
        if fb_data:
            time = np.array(fb_data.get("time", []), dtype=float)
            # Prefer raw modal displacement to preserve sensitivity to parameter
            # variations; filtered data can flatten subtle trends.
            disp = np.array(
                fb_data.get("displacement_modal_raw", fb_data.get("displacement", [])),
                dtype=float,
            )
            torque = fb_data.get("torque_total", fb_data.get("torque", np.array([])))
            torque = np.array(torque, dtype=float)
            time, aligned = ms._align_series(time, disp, torque)
            disp = aligned[0]
            torque = aligned[1]
            rms_disp, _ = _compute_post_slew_stats(time, disp, config.slew_duration_s)
            combo_metrics["rms_vibration_mm"] = rms_disp * 1000.0 if np.isfinite(rms_disp) else float("nan")
            if len(torque):
                combo_metrics["peak_torque_nm"] = float(np.max(np.abs(torque)))
                combo_metrics["rms_torque_nm"] = float(np.sqrt(np.mean(torque**2)))
            if target_freq is not None:
                band_half = min(1.0, max(0.05, 0.05 * target_freq))
                band_rms, psd_db = _compute_band_metrics(time, disp, target_freq, band_half)
                combo_metrics["vibration_band_rms_mm"] = band_rms * 1000.0 if np.isfinite(band_rms) else float("nan")
                combo_metrics["vibration_psd_at_f0_db"] = psd_db

        metrics[key] = combo_metrics

    return metrics


def _plot_sweep(
    values: np.ndarray,
    results: Dict[str, Dict[str, List[float]]],
    title: str,
    x_label: str,
    out_path: str,
    log_x: bool = False,
    bins: int = 10,
    metrics: List[Tuple[str, str]] | None = None,
    log_y: bool = True,
) -> None:
    """Generate a 2x2 scatter + binned median figure for the given sweep results."""
    metric_list = metrics or METRICS
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    use_logx = log_x

    for idx, (metric_key, metric_label) in enumerate(metric_list):
        ax = axes[idx]
        use_logy = log_y and not metric_key.endswith("_db")
        for method, controller, combo_label, color in COMBOS:
            combo_key = _combo_key(method, controller)
            series = results.get(combo_key, {}).get(metric_key, [])
            if len(series) != len(values):
                continue
            x = np.array(values, dtype=float)
            y = np.array(series, dtype=float)
            if use_logy:
                y = np.where(y > 0, y, np.nan)
            if use_logx and use_logy:
                ax.loglog(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            elif use_logx:
                ax.semilogx(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            elif use_logy:
                ax.semilogy(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            else:
                ax.plot(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)

            # Bin stats
            if bins > 1:
                if use_logx:
                    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1)
                    centers = np.sqrt(edges[:-1] * edges[1:])
                else:
                    edges = np.linspace(x.min(), x.max(), bins + 1)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                medians = []
                p10 = []
                p90 = []
                for i in range(bins):
                    mask = (x >= edges[i]) & (x < edges[i + 1])
                    if not np.any(mask):
                        medians.append(np.nan)
                        p10.append(np.nan)
                        p90.append(np.nan)
                        continue
                    vals = y[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        medians.append(np.nan)
                        p10.append(np.nan)
                        p90.append(np.nan)
                        continue
                    medians.append(np.percentile(vals, 50))
                    p10.append(np.percentile(vals, 10))
                    p90.append(np.percentile(vals, 90))
                medians = np.array(medians)
                p10 = np.array(p10)
                p90 = np.array(p90)
                if use_logy:
                    medians = np.where(medians > 0, medians, np.nan)
                    p10 = np.where(p10 > 0, p10, np.nan)
                    p90 = np.where(p90 > 0, p90, np.nan)
                if use_logx and use_logy:
                    ax.loglog(centers, medians, color=color, linewidth=2, label=combo_label)
                elif use_logx:
                    ax.semilogx(centers, medians, color=color, linewidth=2, label=combo_label)
                elif use_logy:
                    ax.semilogy(centers, medians, color=color, linewidth=2, label=combo_label)
                else:
                    ax.plot(centers, medians, color=color, linewidth=2, label=combo_label)
                ax.fill_between(centers, p10, p90, color=color, alpha=0.15)
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=c, linewidth=3) for _, _, _, c in COMBOS]
    labels = [label for _, _, label, _ in COMBOS]
    fig.suptitle(title, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_raw_csv(
    path: str,
    factor_name: str,
    values: np.ndarray,
    results: Dict[str, Dict[str, List[float]]],
    metrics: List[Tuple[str, str]] | None = None,
) -> None:
    """Persist per sample sweep results to a CSV for later replotting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metric_list = metrics or METRICS
    headers = ["factor", "sample_index", "factor_value", "combo"] + [m for m, _ in metric_list]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for idx, val in enumerate(values):
            for combo_key in results:
                row = [
                    factor_name,
                    str(idx),
                    f"{float(val):.6g}",
                    combo_key,
                ]
                for metric_key, _ in metric_list:
                    series = results.get(combo_key, {}).get(metric_key, [])
                    metric_val = series[idx] if idx < len(series) else float("nan")
                    row.append(f"{metric_val:.6g}" if np.isfinite(metric_val) else "nan")
                f.write(",".join(row) + "\n")


def _read_raw_csv(
    path: str,
    metrics: List[Tuple[str, str]] | None = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, List[float]]]]:
    """Load one factor sweep CSV and reconstruct values/results for plotting."""
    metric_list = metrics or METRICS
    metric_keys = [k for k, _ in metric_list]
    combo_keys = [_combo_key(method, controller) for method, controller, _, _ in COMBOS]

    values_by_idx: Dict[int, float] = {}
    raw_by_combo: Dict[str, Dict[int, Dict[str, float]]] = {k: {} for k in combo_keys}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = row.get("combo", "")
            if combo not in raw_by_combo:
                continue
            try:
                sample_idx = int(row.get("sample_index", "-1"))
                factor_val = float(row.get("factor_value", "nan"))
            except (TypeError, ValueError):
                continue
            if sample_idx < 0 or not np.isfinite(factor_val):
                continue

            values_by_idx[sample_idx] = factor_val
            metric_map: Dict[str, float] = raw_by_combo[combo].setdefault(sample_idx, {})
            for metric_key in metric_keys:
                raw = row.get(metric_key, "nan")
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = float("nan")
                metric_map[metric_key] = val

    sorted_indices = sorted(values_by_idx.keys())
    values = np.array([values_by_idx[i] for i in sorted_indices], dtype=float)

    results: Dict[str, Dict[str, List[float]]] = {}
    for combo in combo_keys:
        results[combo] = {}
        for metric_key in metric_keys:
            series: List[float] = []
            combo_samples = raw_by_combo.get(combo, {})
            for sample_idx in sorted_indices:
                val = combo_samples.get(sample_idx, {}).get(metric_key, float("nan"))
                series.append(float(val))
            results[combo][metric_key] = series

    return values, results


def _replot_from_metrics_csvs(
    metrics_dir: str,
    plots_dir: str,
    bins: int,
    log_y: bool,
) -> None:
    """Regenerate all factor sweep plots from previously saved CSV metrics."""
    specs = [
        (
            "mc_sweep_inertia.csv",
            "mc_sweep_inertia.png",
            METRICS,
            "Sweep: Inertia Error (+/-20%)",
            "Inertia scale factor",
            False,
        ),
        (
            "mc_sweep_modal_frequency.csv",
            "mc_sweep_modal_frequency.png",
            METRICS,
            "Sweep: Modal Frequency Error",
            "Modal frequency scale factor",
            False,
        ),
        (
            "mc_sweep_modal_damping.csv",
            "mc_sweep_modal_damping.png",
            METRICS,
            "Sweep: Modal Damping Variation",
            "Modal damping scale factor",
            False,
        ),
        (
            "mc_sweep_disturbance_frequency.csv",
            "mc_sweep_disturbance_frequency.png",
            METRICS,
            "Sweep: Disturbance Frequency (sinusoidal torque)",
            "Disturbance frequency (Hz)",
            True,
        ),
        (
            "mc_sweep_disturbance_frequency.csv",
            "mc_sweep_disturbance_frequency_targeted.png",
            FREQ_METRICS,
            "Sweep: Disturbance Frequency (frequency-targeted metrics)",
            "Disturbance frequency (Hz)",
            True,
        ),
        (
            "mc_sweep_noise_level.csv",
            "mc_sweep_noise_level.png",
            METRICS,
            "Sweep: Sensor Noise Frequency (sinusoidal)",
            "Sensor noise frequency (Hz)",
            True,
        ),
        (
            "mc_sweep_noise_level.csv",
            "mc_sweep_noise_level_targeted.png",
            FREQ_METRICS,
            "Sweep: Sensor Noise Frequency (frequency-targeted metrics)",
            "Sensor noise frequency (Hz)",
            True,
        ),
    ]

    os.makedirs(plots_dir, exist_ok=True)
    for csv_name, plot_name, metric_list, title, x_label, log_x in specs:
        csv_path = os.path.join(metrics_dir, csv_name)
        if not os.path.isfile(csv_path):
            print(f"Missing CSV for replot: {csv_path}")
            continue
        values, results = _read_raw_csv(csv_path, metrics=metric_list)
        if len(values) == 0:
            print(f"No valid rows in CSV for replot: {csv_path}")
            continue
        out_path = os.path.join(plots_dir, plot_name)
        _plot_sweep(
            values=values,
            results=results,
            title=title,
            x_label=x_label,
            out_path=out_path,
            log_x=log_x,
            bins=bins,
            metrics=metric_list,
            log_y=log_y,
        )
        print(f"Replotted {out_path}")


def _run_sweep(
    values: np.ndarray,
    config_builder,
    overrides_builder,
    work_dir: str,
    progress_prefix: str,
    total_samples: int,
    start_time: float,
    last_update: float,
    progress_count: int,
    metric_keys: List[Tuple[str, str]],
    target_freq_func=None,
) -> Tuple[Dict[str, Dict[str, List[float]]], int, float]:
    """Iterate over *values*, run simulations, and collect metrics for each combo."""
    results: Dict[str, Dict[str, List[float]]] = {}
    for method, controller, _, _ in COMBOS:
        combo_key = _combo_key(method, controller)
        results[combo_key] = {metric: [] for metric, _ in metric_keys}

    for val in values:
        cfg = config_builder(val)
        overrides = overrides_builder(val, cfg)
        _run_vizard_demo_batch(overrides, work_dir)
        target_freq = target_freq_func(val) if target_freq_func is not None else None
        metrics = _collect_metrics(cfg, work_dir, target_freq=target_freq)
        for combo_key, combo_metrics in metrics.items():
            for metric_key, _ in metric_keys:
                results[combo_key][metric_key].append(combo_metrics.get(metric_key, float("nan")))
        progress_count += 1
        last_update = _update_progress(progress_prefix, progress_count, total_samples, start_time, last_update)

    return results, progress_count, last_update


def main() -> int:
    """CLI entry point: parse arguments, run sweeps or replot from CSVs."""
    parser = argparse.ArgumentParser(description="One-factor Monte Carlo sweep plots.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"), help="Output directory for plots")
    parser.add_argument("--work-dir", default=None, help="Working directory for temporary NPZs")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Metrics CSV directory for --plot-only (default: <out-dir>/metrics)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation and regenerate plots from existing sweep CSV files",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Run simulation sweeps without the final CSV replot pass",
    )
    parser.add_argument("--noise-baseline", type=float, default=0.0, help="Baseline sensor noise (rad/s)")
    parser.add_argument("--disturbance-bias", type=float, default=0.0, help="Baseline disturbance bias (N*m)")
    parser.add_argument("--disturbance-amplitude", type=float, default=1e-2,
                        help="Disturbance sine amplitude for frequency sweep (N*m)")
    parser.add_argument("--disturbance-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        help="Disturbance axis vector for sine disturbance")
    parser.add_argument("--samples", type=int, default=500, help="Monte Carlo samples per factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bins", type=int, default=10, help="Bins for percentile trend lines")
    parser.add_argument("--linear-y", action="store_true", help="Use linear y-scale instead of log")

    parser.add_argument("--inertia-min", type=float, default=0.7)
    parser.add_argument("--inertia-max", type=float, default=1.3)
    parser.add_argument("--inertia-points", type=int, default=9,
                        help="Stratification bins for inertia sampling")

    parser.add_argument("--freq-min", type=float, default=0.7)
    parser.add_argument("--freq-max", type=float, default=1.3)
    parser.add_argument("--freq-points", type=int, default=9,
                        help="Stratification bins for modal-frequency sampling")

    parser.add_argument("--damping-min", type=float, default=0.7)
    parser.add_argument("--damping-max", type=float, default=1.3)
    parser.add_argument("--damping-points", type=int, default=9,
                        help="Stratification bins for modal-damping sampling")

    parser.add_argument("--dist-freq-min", type=float, default=0.1)
    parser.add_argument("--dist-freq-max", type=float, default=50.0)
    parser.add_argument("--dist-freq-points", type=int, default=10,
                        help="Log-stratification bins for disturbance-frequency sampling")

    parser.add_argument("--noise-min", type=float, default=0.1,
                        help="Noise frequency sweep min (Hz)")
    parser.add_argument("--noise-max", type=float, default=50.0,
                        help="Noise frequency sweep max (Hz)")
    parser.add_argument("--noise-amplitude", type=float, default=3e-3,
                        help="Sensor noise amplitude for sine sweep (rad/s)")
    parser.add_argument("--noise-points", type=int, default=8,
                        help="Log-stratification bins for sensor-noise-frequency sampling")

    args = parser.parse_args()

    if args.plot_only and args.simulate_only:
        parser.error("--plot-only and --simulate-only cannot be used together")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Separate directories for plots and metrics
    plots_dir = os.path.join(out_dir, "plots")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    work_dir = os.path.abspath(args.work_dir or os.path.join(out_dir, "mc_sweep_work"))
    os.makedirs(work_dir, exist_ok=True)

    log_y = not args.linear_y

    if args.plot_only:
        metrics_src_dir = os.path.abspath(args.metrics_dir or metrics_dir)
        _replot_from_metrics_csvs(
            metrics_dir=metrics_src_dir,
            plots_dir=plots_dir,
            bins=args.bins,
            log_y=log_y,
        )
        print("Saved plots to:", plots_dir)
        print("Used metrics from:", metrics_src_dir)
        return 0

    rng = np.random.default_rng(args.seed)
    base_cfg = ms.default_config()

    total_samples = args.samples * 5
    start_time = time.time()
    last_update = start_time
    progress_count = 0
    _update_progress("Factor MC", 0, total_samples, start_time, last_update)

    # Estimate runtime based on one sample (all configured combos).
    estimate_overrides = {
        "inertia_scale": 1.0,
        "modal_freqs_hz": base_cfg.modal_freqs_hz,
        "modal_damping": base_cfg.modal_damping,
        "control_filter_cutoff_hz": _filter_cutoff(base_cfg),
        "sensor_noise_std_rad_s": args.noise_baseline,
        "disturbance_torque_nm": args.disturbance_bias,
    }
    est_total = _estimate_total_runtime(estimate_overrides, work_dir, total_samples)
    print(
        f"\nEstimated total runtime: {_format_eta(est_total)} "
        f"(based on 1 sample x {len(COMBOS)} combos)"
    )

    # 1) Inertia sweep (+/-20%)
    inertia_vals = _sample_uniform_stratified(
        rng,
        args.inertia_min,
        args.inertia_max,
        args.samples,
        args.inertia_points,
    )
    inertia_results, progress_count, last_update = _run_sweep(
        inertia_vals,
        lambda v: _build_config(base_cfg, inertia_scale=v),
        lambda v, cfg: {
            "inertia_scale": float(v),
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "control_filter_cutoff_hz": _filter_cutoff(cfg),
            "sensor_noise_std_rad_s": args.noise_baseline,
            "disturbance_torque_nm": args.disturbance_bias,
        },
        work_dir,
        "Factor MC",
        total_samples,
        start_time,
        last_update,
        progress_count,
        METRICS,
    )
    _plot_sweep(
        inertia_vals,
        inertia_results,
        "Sweep: Inertia Error (+/-20%)",
        "Inertia scale factor",
        os.path.join(plots_dir, "mc_sweep_inertia.png"),
        log_x=False,
        bins=args.bins,
        metrics=METRICS,
        log_y=log_y,
    )
    _write_raw_csv(
        os.path.join(metrics_dir, "mc_sweep_inertia.csv"),
        "inertia_scale",
        inertia_vals,
        inertia_results,
        metrics=METRICS,
    )

    # 2) Modal frequency error sweep
    freq_vals = _sample_uniform_stratified(
        rng,
        args.freq_min,
        args.freq_max,
        args.samples,
        args.freq_points,
    )
    freq_results, progress_count, last_update = _run_sweep(
        freq_vals,
        lambda v: _build_config(base_cfg, freq_scale=v),
        lambda v, cfg: {
            "inertia_scale": 1.0,
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "control_filter_cutoff_hz": _filter_cutoff(cfg),
            "sensor_noise_std_rad_s": args.noise_baseline,
            "disturbance_torque_nm": args.disturbance_bias,
        },
        work_dir,
        "Factor MC",
        total_samples,
        start_time,
        last_update,
        progress_count,
        METRICS,
    )
    _plot_sweep(
        freq_vals,
        freq_results,
        "Sweep: Modal Frequency Error",
        "Modal frequency scale factor",
        os.path.join(plots_dir, "mc_sweep_modal_frequency.png"),
        log_x=False,
        bins=args.bins,
        metrics=METRICS,
        log_y=log_y,
    )
    _write_raw_csv(
        os.path.join(metrics_dir, "mc_sweep_modal_frequency.csv"),
        "modal_frequency_scale",
        freq_vals,
        freq_results,
        metrics=METRICS,
    )

    # 3) Modal damping sweep
    damping_vals = _sample_uniform_stratified(
        rng,
        args.damping_min,
        args.damping_max,
        args.samples,
        args.damping_points,
    )
    damping_results, progress_count, last_update = _run_sweep(
        damping_vals,
        lambda v: _build_config(base_cfg, damping_scale=v),
        lambda v, cfg: {
            "inertia_scale": 1.0,
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "control_filter_cutoff_hz": _filter_cutoff(cfg),
            "sensor_noise_std_rad_s": args.noise_baseline,
            "disturbance_torque_nm": args.disturbance_bias,
        },
        work_dir,
        "Factor MC",
        total_samples,
        start_time,
        last_update,
        progress_count,
        METRICS,
    )
    _plot_sweep(
        damping_vals,
        damping_results,
        "Sweep: Modal Damping Variation",
        "Modal damping scale factor",
        os.path.join(plots_dir, "mc_sweep_modal_damping.png"),
        log_x=False,
        bins=args.bins,
        metrics=METRICS,
        log_y=log_y,
    )
    _write_raw_csv(
        os.path.join(metrics_dir, "mc_sweep_modal_damping.csv"),
        "modal_damping_scale",
        damping_vals,
        damping_results,
        metrics=METRICS,
    )

    # 4) Disturbance frequency sweep (sinusoid)
    dist_freq_vals = _sample_log_uniform_stratified(
        rng,
        args.dist_freq_min,
        args.dist_freq_max,
        args.samples,
        args.dist_freq_points,
    )
    dist_results, progress_count, last_update = _run_sweep(
        dist_freq_vals,
        lambda v: _build_config(base_cfg),
        lambda v, cfg: {
            "inertia_scale": 1.0,
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "control_filter_cutoff_hz": _filter_cutoff(cfg),
            "sensor_noise_std_rad_s": args.noise_baseline,
            "disturbance_torque_nm": 0.0,
            "disturbance_type": "sine",
            "disturbance_amplitude_nm": args.disturbance_amplitude,
            "disturbance_frequency_hz": float(v),
            "disturbance_axis": args.disturbance_axis,
        },
        work_dir,
        "Factor MC",
        total_samples,
        start_time,
        last_update,
        progress_count,
        METRICS + FREQ_METRICS,
        target_freq_func=lambda v: float(v),
    )
    _plot_sweep(
        dist_freq_vals,
        dist_results,
        "Sweep: Disturbance Frequency (sinusoidal torque)",
        "Disturbance frequency (Hz)",
        os.path.join(plots_dir, "mc_sweep_disturbance_frequency.png"),
        log_x=True,
        bins=args.bins,
        metrics=METRICS,
        log_y=log_y,
    )
    _plot_sweep(
        dist_freq_vals,
        dist_results,
        "Sweep: Disturbance Frequency (frequency-targeted metrics)",
        "Disturbance frequency (Hz)",
        os.path.join(plots_dir, "mc_sweep_disturbance_frequency_targeted.png"),
        log_x=True,
        bins=args.bins,
        metrics=FREQ_METRICS,
        log_y=log_y,
    )
    _write_raw_csv(
        os.path.join(metrics_dir, "mc_sweep_disturbance_frequency.csv"),
        "disturbance_frequency_hz",
        dist_freq_vals,
        dist_results,
        metrics=METRICS + FREQ_METRICS,
    )

    # 5) Noise frequency sweep (sensor noise as sinusoid)
    noise_vals = _sample_log_uniform_stratified(
        rng,
        args.noise_min,
        args.noise_max,
        args.samples,
        args.noise_points,
    )
    noise_results, progress_count, last_update = _run_sweep(
        noise_vals,
        lambda v: _build_config(base_cfg),
        lambda v, cfg: {
            "inertia_scale": 1.0,
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "control_filter_cutoff_hz": _filter_cutoff(cfg),
            "sensor_noise_std_rad_s": float(args.noise_amplitude),
            "sensor_noise_type": "sine",
            "sensor_noise_frequency_hz": float(v),
            "sensor_noise_axis": args.disturbance_axis,
            "disturbance_torque_nm": args.disturbance_bias,
        },
        work_dir,
        "Factor MC",
        total_samples,
        start_time,
        last_update,
        progress_count,
        METRICS + FREQ_METRICS,
        target_freq_func=lambda v: float(v),
    )
    _plot_sweep(
        noise_vals,
        noise_results,
        "Sweep: Sensor Noise Frequency (sinusoidal)",
        "Sensor noise frequency (Hz)",
        os.path.join(plots_dir, "mc_sweep_noise_level.png"),
        log_x=True,
        bins=args.bins,
        metrics=METRICS,
        log_y=log_y,
    )
    _plot_sweep(
        noise_vals,
        noise_results,
        "Sweep: Sensor Noise Frequency (frequency-targeted metrics)",
        "Sensor noise frequency (Hz)",
        os.path.join(plots_dir, "mc_sweep_noise_level_targeted.png"),
        log_x=True,
        bins=args.bins,
        metrics=FREQ_METRICS,
        log_y=log_y,
    )
    _write_raw_csv(
        os.path.join(metrics_dir, "mc_sweep_noise_level.csv"),
        "sensor_noise_frequency_hz",
        noise_vals,
        noise_results,
        metrics=METRICS + FREQ_METRICS,
    )

    if not args.simulate_only:
        _replot_from_metrics_csvs(
            metrics_dir=metrics_dir,
            plots_dir=plots_dir,
            bins=args.bins,
            log_y=log_y,
        )

    print("Saved plots to:", plots_dir)
    print("Saved metrics to:", metrics_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
