#!/usr/bin/env python3
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
Experiment Metrics Visualizer — Multi-panel (L2 + Collision) with single legend,
plus CSV & LaTeX export of aggregated numbers (mean ± std).

- Generates ONE figure directly (3 L2 only, or 3 L2 + 3 Collision) — no stitching.
- Single, compact legend centered on top across all subplots (horizontal) with tight spacing.
- No subplot titles; y-axis labels only (as requested).
- Collision metrics automatically converted to % (x100).
- Saves timestamped SVG (vector) and optionally PDF/PNG.
- Preserves styling and aesthetics from your implementation.
- NEW: Exports a timestamped CSV and a LaTeX tabular with mean ± std for each (Model/Epoch, Metric).

"""

from __future__ import annotations
import json
import math
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import cycle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
EPOCH_RE = re.compile(r"_e(\d+)(?:_|$)")


def find_epoch_number(name: str) -> Optional[int]:
    m = EPOCH_RE.search(name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON: {path}: {e}")
        return None


def safe_to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            if x.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
                return None
            return float(x)
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def find_metric_value(metrics: dict, metric_key: str) -> Optional[float]:
    # exact match
    if metric_key in metrics:
        return safe_to_float(metrics.get(metric_key))
    # one-level deep
    for _, v in metrics.items():
        if isinstance(v, dict) and metric_key in v:
            return safe_to_float(v.get(metric_key))
    # recursive
    for v in metrics.values():
        if isinstance(v, dict):
            found = find_metric_value(v, metric_key)
            if found is not None:
                return found
    return None


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
class RunSeries:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.epoch_to_val: Dict[int, float] = {}

    def add(self, epoch: int, value: float):
        self.epoch_to_val[epoch] = value

    def epochs_sorted(self) -> List[int]:
        return sorted(self.epoch_to_val.keys())


class ExperimentData:
    def __init__(self, exp_path: Path, label: str):
        self.exp_path = exp_path
        self.label = label
        self.runs: Dict[str, RunSeries] = {}

    def add_point(self, run_name: str, epoch: int, value: float):
        if run_name not in self.runs:
            self.runs[run_name] = RunSeries(run_name)
        self.runs[run_name].add(epoch, value)

    def all_epochs(self) -> List[int]:
        all_e = set()
        for r in self.runs.values():
            all_e.update(r.epoch_to_val.keys())
        return sorted(all_e)

    def aggregate(self, epochs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means: List[float] = []
        stds: List[float] = []
        for e in epochs:
            vals = [r.epoch_to_val[e] for r in self.runs.values() if e in r.epoch_to_val]
            if len(vals) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            elif len(vals) == 1:
                means.append(float(vals[0]))
                stds.append(0.0)
            else:
                arr = np.array(vals, dtype=float)
                means.append(float(np.nanmean(arr)))
                stds.append(float(np.nanstd(arr, ddof=0)))
        return np.array(epochs, dtype=int), np.array(means, dtype=float), np.array(stds, dtype=float)


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def is_run_folder(p: Path) -> bool:
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_dir() and find_epoch_number(child.name) is not None:
            fm = child / "formatted_metrics.json"
            if fm.exists():
                return True
    return False


def discover_runs(exp_path: Path) -> List[Path]:
    if is_run_folder(exp_path):
        return [exp_path]
    runs = [d for d in exp_path.iterdir() if is_run_folder(d)]
    runs.sort(key=lambda p: p.name)
    return runs


def load_experiment(
    exp_path: Path,
    metric_key: str,
    exclude_runs: Optional[List[str]] = None,
    label: Optional[str] = None,
    transform: Optional[callable] = None,
) -> ExperimentData:
    exclude_runs = exclude_runs or []
    label = label or exp_path.name
    exp = ExperimentData(exp_path, label)
    runs = discover_runs(exp_path)
    if not runs:
        print(f"[WARN] No runs discovered under: {exp_path}")
        return exp

    for run_path in runs:
        run_name = run_path.name
        if any(ex in run_name for ex in exclude_runs):
            print(f"[INFO] Excluding run: {run_name}")
            continue

        for epoch_dir in run_path.iterdir():
            if not epoch_dir.is_dir():
                continue
            epoch = find_epoch_number(epoch_dir.name)
            if epoch is None:
                continue
            fm_path = epoch_dir / "formatted_metrics.json"
            if not fm_path.exists():
                continue
            data = load_json(fm_path)
            if data is None:
                continue
            val = find_metric_value(data, metric_key)
            if val is None:
                continue
            if transform is not None:
                val = transform(val)
            exp.add_point(run_name, epoch, val)

    return exp


# -----------------------------------------------------------------------------
# Styling (kept consistent with your look)
# -----------------------------------------------------------------------------
def apply_pro_style(base_fontsize: int = 12):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 2,
            "axes.titleweight": "bold",
            "axes.labelsize": base_fontsize + 1,
            "axes.labelweight": "bold",
            "legend.fontsize": base_fontsize + 4,
            "xtick.labelsize": base_fontsize - 1,
            "ytick.labelsize": base_fontsize - 1,
            "axes.linewidth": 1.0,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def build_palette(labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
    cmap_names = ["tab10", "tab20"]
    colors: List[Tuple[float, float, float]] = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        try:
            colors.extend(list(cmap.colors))
        except AttributeError:
            colors.extend([cmap(i) for i in range(cmap.N)])
    unique_colors = []
    for c in colors:
        if c not in unique_colors:
            unique_colors.append(c)
    color_cycle = cycle(unique_colors)
    return {lbl: next(color_cycle) for lbl in labels}


def _nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / max(target_ticks, 1)
    exp = math.floor(math.log10(raw))
    frac = raw / (10 ** exp)
    for nice in (1, 2, 2.5, 5, 10):
        if frac <= nice:
            return nice * (10 ** exp)
    return 10 * (10 ** exp)


def _nice_bounds(ymin: float, ymax: float, step: float) -> Tuple[float, float]:
    lo = math.floor(ymin / step) * step
    hi = math.ceil(ymax / step) * step
    if math.isclose(lo, hi):
        hi = lo + step
    return lo, hi


def place_compact_top_legend(fig, handles, labels, max_cols: int = 6, extra_pad: float = 0.01):
    """
    Adds a single top legend, measures its bbox, then shrinks the top margin
    so the subplots sit as close as possible under the legend.
    extra_pad is in figure coordinates (0..1) for a tiny breathing room.
    """
    if not handles:
        fig.tight_layout()
        return

    # Deduplicate labels while preserving order
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    handles, labels = zip(*uniq)

    ncol = max(1, min(len(labels), max_cols))

    # Create the legend first
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        columnspacing=1.0,
        handlelength=2.2,
        borderaxespad=0.4,
        bbox_to_anchor=(0.5, 1.0),  # exactly at top edge
    )

    # Render once to get an accurate bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_fig = leg.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    # bbox_fig.y0 is the legend's bottom in figure coords; leave a hair of space
    top_rect = max(0.0, bbox_fig.y0 - extra_pad)

    # Now pack subplots tightly up to the computed top
    fig.tight_layout(rect=(0.02, 0.02, 0.98, top_rect))


# -------------------- Table helpers --------------------
def _metric_column_names(l2_keys: List[str], col_keys: Optional[List[str]]) -> List[str]:
    """Human-friendly column names in the order they are processed."""
    cols = []
    key2name = {
        "plan_L2_1s": "L2-1s",
        "plan_L2_2s": "L2-2s",
        "plan_L2_3s": "L2-3s",
        "plan_obj_box_col_1s": "Collision-1s",
        "plan_obj_box_col_2s": "Collision-2s",
        "plan_obj_box_col_3s": "Collision-3s",
    }
    for k in l2_keys:
        cols.append(key2name.get(k, k))
    if col_keys:
        for k in col_keys:
            cols.append(key2name.get(k, k))
    return cols


def _fmt_cell(mean: Optional[float], std: Optional[float], is_percent: bool, latex: bool = False) -> str:
    """Format 'mean ± std' (percent if requested). Empty if missing."""
    if mean is None or std is None or not np.isfinite(mean) or not np.isfinite(std):
        return ""
    if is_percent:
        return f"{mean:.2f}\\,\\% $\\pm$ {std:.2f}" if latex else f"{mean:.2f}% ± {std:.2f}"
    else:
        return f"{mean:.2f} $\\pm$ {std:.2f}" if latex else f"{mean:.2f} ± {std:.2f}"


def _latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for special chars used here."""
    return s.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_panel(
    exp_paths: List[str],
    legend_overrides: Dict[str, str],
    exclude_runs_map: Dict[str, List[str]],
    l2_keys: List[str],
    col_keys: Optional[List[str]],
    ylabels_l2: List[str],
    ylabels_col: Optional[List[str]],
    epochs_to_plot: Optional[List[int]] = None,
    base_fontsize: int = 12,
    save_dir: Path = Path("./plots_panel"),
    base_filename: str = "metrics_panel",
    save_svg: bool = True,
    save_pdf: bool = True,
    save_png: bool = True,
    png_dpi: int = 450,
):
    apply_pro_style(base_fontsize=base_fontsize)

    # Resolve labels for consistent palette
    def label_for_path(p: Path) -> str:
        sp = str(p.resolve())
        # exact match preferred
        for key, override in legend_overrides.items():
            if sp == str(Path(key).resolve()):
                return override
        # fallback: longest-key-first prefix match (optional)
        for key in sorted(legend_overrides.keys(), key=len, reverse=True):
            if sp.startswith(str(Path(key).resolve())):
                return legend_overrides[key]
        return p.name

    def excludes_for_path(p: Path) -> List[str]:
        sp = str(p.resolve())
        for key, ex_list in exclude_runs_map.items():
            if sp == str(Path(key).resolve()):
                return ex_list
        return []

    resolved_labels = []
    resolved_paths = []
    for p in exp_paths:
        pp = Path(p)
        if not pp.exists():
            print(f"[WARN] Path does not exist, skipping: {pp}")
            continue
        resolved_paths.append(pp)
        resolved_labels.append(label_for_path(pp))
    if not resolved_paths:
        raise RuntimeError("No valid experiment paths found.")

    color_map = build_palette(resolved_labels)

    # Which rows to plot
    have_collision = bool(col_keys)
    if have_collision and len(col_keys) != 3:
        print("[WARN] Collision keys provided but not length 3; plotting those found.")
    if len(l2_keys) != 3:
        print("[WARN] L2 keys should be 3; plotting those provided.")

    rows = 2 if have_collision else 1
    cols = 3

    # Figure size (kept per your version)
    fig_w = 10 * cols
    fig_h = 6 * rows + 0.5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300, squeeze=False, sharex=True)

    # ---- Store for table values: table[(model_label, epoch)][metric_name] = (mean, std, is_percent)
    table_store: Dict[Tuple[str, int], Dict[str, Tuple[float, float, bool]]] = {}

    def _store_cells(
        model_label: str, epochs: List[int], means: np.ndarray, stds: np.ndarray, col_name: str, is_percent: bool
    ):
        for e, m, s in zip(epochs, means, stds):
            if not np.isfinite(m) or not np.isfinite(s):
                continue
            key = (model_label, int(e))
            if key not in table_store:
                table_store[key] = {}
            table_store[key][col_name] = (float(m), float(s), is_percent)

    # Build a convenience to load and plot one metric on a specific axis
    def plot_metric_on_ax(ax, metric_key: str, y_label: str, percent: bool, col_name: str):
        # X axis config (major at 10, minor at 5); grid; remove top/right spines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        # Load experiments for this metric
        exps_for_metric: List[ExperimentData] = []
        transform = (lambda v: v * 100.0) if percent else None

        for p in resolved_paths:
            lbl = label_for_path(p)
            excludes = excludes_for_path(p)
            exp = load_experiment(p, metric_key=metric_key, exclude_runs=excludes, label=lbl, transform=transform)

            print(f"[INFO] Loaded experiment '{lbl}' with {len(exp.runs)} runs for metric '{metric_key}'")
            print(f"       Runs: {list(exp.runs.keys())}")

            exps_for_metric.append(exp)

        # Determine X limits
        all_epochs = []
        for exp in exps_for_metric:
            es = exp.all_epochs()
            if epochs_to_plot:
                s = set(epochs_to_plot)
                es = [e for e in es if e in s]
            all_epochs.extend(es)
        if all_epochs:
            ax.set_xlim(min(all_epochs) - 1, max(all_epochs) + 1)

        # Plot each experiment
        all_y_vals = []
        handles = []
        labels_ = []
        for exp in exps_for_metric:
            epochs = exp.all_epochs()
            if epochs_to_plot:
                s = set(epochs_to_plot)
                epochs = sorted([e for e in epochs if e in s])
            else:
                epochs = sorted(epochs)
            if not epochs:
                print(f"[WARN] No data to plot for {exp.label} on `{metric_key}`")
                continue

            xs, ys, ys_std = exp.aggregate(epochs)
            all_y_vals.append(ys)

            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                linestyle="-",
                linewidth=2.2,
                markersize=5.5,
                markeredgewidth=0.8,
                markeredgecolor="white",
                label=exp.label,
                color=color_map.get(exp.label),
                zorder=3,
            )
            if len(exp.runs) > 1:
                ax.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.18, color=line.get_color(), linewidth=0, zorder=2)

            handles.append(line)
            labels_.append(exp.label)

            # Register values to table (means/stds already percent-scaled if needed)
            _store_cells(exp.label, list(xs), ys, ys_std, col_name, percent)

        # Labels — no title; y-label only
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)

        # Y axis nice ticks & padding
        sf = ScalarFormatter(useOffset=False, useMathText=False)
        sf.set_scientific(False)
        ax.yaxis.set_major_formatter(sf)

        if all_y_vals:
            y_concat = np.concatenate([arr[np.isfinite(arr)] for arr in all_y_vals if arr.size > 0])
            if y_concat.size > 0:
                ymin, ymax = float(np.nanmin(y_concat)), float(np.nanmax(y_concat))
                if not np.isfinite(ymin) or not np.isfinite(ymax):
                    ymin, ymax = 0.0, 1.0
                span = ymax - ymin
                step = _nice_step(span if span > 0 else max(abs(ymax), 1.0), target_ticks=6)
                if span < 1e-9:
                    center = (ymin + ymax) / 2.0
                    half_span = 1.5 * step
                    lo, hi = center - half_span, center + half_span
                else:
                    lo, hi = _nice_bounds(ymin, ymax, step)
                    pad = max(0.1 * (hi - lo), 0.5 * step)
                    lo, hi = lo - pad, hi + pad
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(MultipleLocator(step))

        return handles, labels_

    # ---- Draw L2 row ----
    legend_handles = []
    legend_labels = []
    l2_colnames = ["L2-1s", "L2-2s", "L2-3s"]
    for j, (key, colname) in enumerate(zip(l2_keys[:3], l2_colnames)):
        ax = axes[0, j]
        ylab = ylabels_l2[j] if j < len(ylabels_l2) else "L2 (m)"
        h, l = plot_metric_on_ax(ax, key, ylab, percent=False, col_name=colname)
        if not legend_handles:
            legend_handles, legend_labels = h, l

    # ---- Draw Collision row (if provided) ----
    if have_collision:
        col_colnames = ["Collision-1s", "Collision-2s", "Collision-3s"]
        for j, (key, colname) in enumerate(zip(col_keys[:3], col_colnames)):
            ax = axes[1, j]
            ylab = ylabels_col[j] if (ylabels_col and j < len(ylabels_col)) else "Collision Rate (%)"
            plot_metric_on_ax(ax, key, ylab, percent=True, col_name=colname)

    # Share X axis labels only on bottom row
    if rows == 2:
        for j in range(cols):
            axes[0, j].set_xlabel("")

    # ---- Single top legend & compact spacing ----
    place_compact_top_legend(fig, legend_handles, legend_labels, max_cols=6, extra_pad=0.008)

    # ---- Save timestamped figure(s) ----
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = save_dir / f"{base_filename}_{ts}"

    if save_svg:
        fig.savefig(base.with_suffix(".svg"), facecolor="white")
        print(f"[INFO] Saved SVG to: {base.with_suffix('.svg')}")
    if save_pdf:
        fig.savefig(base.with_suffix(".pdf"), facecolor="white")
        print(f"[INFO] Saved PDF to: {base.with_suffix('.pdf')}")
    if save_png:
        fig.savefig(base.with_suffix(".png"), facecolor="white", dpi=png_dpi)
        print(f"[INFO] Saved PNG to: {base.with_suffix('.png')}")

    plt.close(fig)

    # -------------------- Write CSV and LaTeX tables --------------------
    # Final column order
    col_names = _metric_column_names(l2_keys, col_keys)

    # Row keys sorted by model label then epoch
    sorted_keys = sorted(table_store.keys(), key=lambda k: (k[0], k[1]))

    # Compose header and rows
    header = ["Model/Epoch"] + col_names
    csv_rows: List[List[str]] = [header]
    latex_rows: List[List[str]] = []

    for (model, epoch) in sorted_keys:
        row_id = f"{model}-e{epoch}"
        csv_row = [row_id]
        latex_row = [_latex_escape(row_id)]

        cells = table_store[(model, epoch)]
        for cn in col_names:
            if cn in cells:
                mean, std, is_pct = cells[cn]
                csv_row.append(_fmt_cell(mean, std, is_pct, latex=False))
                latex_row.append(_fmt_cell(mean, std, is_pct, latex=True))
            else:
                csv_row.append("")
                latex_row.append("")
        csv_rows.append(csv_row)
        latex_rows.append(latex_row)

    # Write CSV
    csv_path = base.with_name(base.name + "_table.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"[INFO] Saved CSV table to: {csv_path}")

    # Write LaTeX tabular (booktabs)
    tex_path = base.with_name(base.name + "_table.tex")
    with tex_path.open("w") as f:
        cols_spec = "l" + "c" * len(col_names)
        f.write("\\begin{tabular}{" + cols_spec + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(_latex_escape(h) for h in header) + " \\\\\n")
        f.write("\\midrule\n")
        for r in latex_rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"[INFO] Saved LaTeX table to: {tex_path}")


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
def main():
    # Choose which metrics to draw (always 3 L2; optional 3 Collision)
    l2_keys = [
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
    ]
    col_keys = [
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
    ]  # set to [] or None to disable collision row

    # y-axis labels (no titles on subplots)
    ylabels_l2 = [
        "L2 (m) — 1s",
        "L2 (m) — 2s",
        "L2 (m) — 3s",
    ]
    ylabels_col = [
        "Collision Rate (%) — 1s",
        "Collision Rate (%) — 2s",
        "Collision Rate (%) — 3s",
    ]

    # Experiments to visualize: path to exp folder that has formatted_metrics.json files inside
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1/4gpus_b6",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-VLP (SLP, Orig. Setup)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad": "VAD-VLP (SLP, Prob. Setup - 2 Samples)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6": "VAD-VLP (SLP, Prob. Setup - Scaled Up)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1/4gpus_b6": "VAD-VLP (SLP, Prob. Setup - 4 Samples)",
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "probabilistic_sampling-scaling"
    base_filename = "metrics_panel"
    base_fontsize = 16

    # If you only want L2 (3 plots), set col_keys = [] above.
    plot_panel(
        exp_paths=exp_paths,
        legend_overrides=legend_overrides,
        exclude_runs_map=exclude_runs_map,
        l2_keys=l2_keys,
        col_keys=col_keys,  # set to [] or None to plot only the 3 L2 panels
        ylabels_l2=ylabels_l2,
        ylabels_col=ylabels_col,
        epochs_to_plot=epochs_to_plot,
        base_fontsize=base_fontsize,
        save_dir=save_dir,
        base_filename=base_filename,
        save_svg=True,
        save_pdf=True,
        save_png=False,
        png_dpi=450,  # high DPI PNG in addition to vector files
    )


if __name__ == "__main__":
    main()
