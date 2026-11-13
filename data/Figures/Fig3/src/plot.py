#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined QC error and NEVPT2 error-reduction analysis for H2/Cu.
User-facing terminology uses QPU-QSCI and QC throughout the figures.

Outputs:
    qc_errors.png
    qpu_qsci_step5_step10_errors.png
    nevpt2_error_reduction.png
    qc_errors_with_nevpt2_reduction.png

Notes:
    No PDF or CSV files are written.
"""

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================================================
# User settings
# =========================================================

EXCEL_FILE = "data/QC_results_fr_adapt_vqe.xlsx"

OUT_QC_ERRORS_PNG = "output/Fig3_qc_errors.png"
OUT_NEVPT2_REDUCTION_PNG = "output/Fig3_nevpt2_error_reduction.png"
OUT_COMBINED_PNG = "output/Fig3_qc_errors_with_nevpt2_reduction.png"
OUT_QPU_QSCI_STEPS_PNG = "output/Fig3_qpu_qsci_step5_step10_errors.png"

DPI = 300

# Chemical accuracy
CHEMICAL_ACCURACY_HA = 0.0015936
HARTREE_TO_EV = 27.211386245988
CHEMICAL_ACCURACY_EV = CHEMICAL_ACCURACY_HA * HARTREE_TO_EV

# Coarse NEVPT2-reduction target
TARGET_QC_ERROR_EV = 0.20
TARGET_OVERLAP = 0.98
TARGET_NEVPT2_EV = CHEMICAL_ACCURACY_EV

# Map display states to the original scan directories. Source FS is
# dissociated (display IS); source IS is molecular (display FS).
NEVPT2_SCAN_DIRS = {
    "IS": [Path("data/nevpt2/FS")],
    "TS": [Path("data/nevpt2/TS")],
    "FS": [Path("data/nevpt2/IS")],
}

DISPLAY_STATE_ORDER = ["IS", "TS", "FS"]


# =========================================================
# Unified style
# =========================================================

TITLE_COLOR = "#222222"
LABEL_COLOR = "#222222"
TICK_COLOR  = "#333333"
SPINE_COLOR = "#333333"
ZERO_COLOR  = "#B8B8B8"
CHEM_ACC_COLOR = "#A99ACB"  # muted lavender

COLOR_IS = "#8FBBD9"   # soft blue
COLOR_TS = "#A6C59D"   # soft green
COLOR_FS = "#E7A299"   # soft salmon

COLOR_QC = "#7FAED0"             # soft blue: QC error
COLOR_QC_NEVPT2 = "#A99ACB"      # muted lavender: QC + NEVPT2 error
COLOR_ARROW = "#6F6F6F"     # neutral gray

STATE_COLORS = {
    "IS": COLOR_IS,
    "TS": COLOR_TS,
    "FS": COLOR_FS,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# General helpers
# =========================================================

def safe_float(x, default=math.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def style_axes(ax, right_spine=False):
    ax.spines["top"].set_visible(False)
    if not right_spine:
        ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    if right_spine:
        ax.spines["right"].set_color(SPINE_COLOR)
        ax.spines["right"].set_linewidth(1.0)

    ax.tick_params(axis="both", colors=TICK_COLOR)
    ax.grid(False)


def add_panel_label(ax, label, x=-0.10, y=1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        clip_on=False,
    )


def compute_ylim_positive(values_2d, frac=0.18, default=(0.0, 0.05), include_values=None):
    vals = np.asarray(values_2d, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]

    if include_values is not None:
        extra = np.asarray(include_values, dtype=float).ravel()
        extra = extra[np.isfinite(extra)]
        if len(extra) > 0:
            vals = np.concatenate([vals, extra]) if len(vals) > 0 else extra

    if len(vals) == 0:
        return default

    vmax = max(float(np.max(vals)), 0.0)
    margin = max(0.005, frac * max(vmax, 1e-8))
    return 0.0, vmax + margin


# =========================================================
# Part 1: QC energy error table from Excel
# =========================================================

def load_qc_table(excel_file):
    """
    Current xlsx layout:
      row 0-3 : basic info / metadata
      row 4   : header
      row 5+  : data
    """
    raw = pd.read_excel(excel_file, header=None)

    header_row_idx = 4
    data_start_idx = 5

    header = raw.iloc[header_row_idx].tolist()
    data = raw.iloc[data_start_idx:].copy()
    data.columns = header

    first_col = data.columns[0]
    data = data.rename(columns={first_col: "Method"})
    data = data.dropna(how="all").reset_index(drop=True)

    data["Method"] = data["Method"].astype(str).str.strip()

    for col in data.columns:
        if col != "Method":
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=7.5, offset_frac=0.018, color=TITLE_COLOR):
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    offset = yr * offset_frac

    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue

        x = b.get_x() + b.get_width() / 2.0
        ax.text(
            x, h + offset, fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color
        )


def prepare_qc_error_dataframe(df):
    """
    Prepare Figure A QC-error data.

    Required x-axis order:
        adapt-uccsd, adapt-uccgsd, QPU-QSCI

    Notes:
      - The display label is always "adapt-uccgsd" (not "adapt-guccsd").
      - Several aliases are accepted for the generalized UCCGSD and QPU-QSCI
        rows so that older Excel files can still be read.
      - Only step-10 is used for QPU-QSCI in Figure A.
    """
    adapt_uccsd_name = find_method_name(df, [
        "adapt-uccsd",
        "adapt_uccsd",
    ])
    adapt_uccgsd_name = find_method_name(df, [
        "adapt-uccgsd",
        "adapt_uccgsd",
        "adapt-guccsd",   # old/internal name; displayed as adapt-uccgsd
        "adapt_guccsd",
    ])
    qpu_qsci_name = find_method_name(df, [
        "tianyan_recover_step10",
        "qpu_recover_step10",
        "cpu_recover_step10",
        "tianyan_step10_recover",
        "step10_recover",
    ])

    wanted_methods = [
        adapt_uccsd_name,
        adapt_uccgsd_name,
        qpu_qsci_name,
    ]
    display_labels = {
        adapt_uccsd_name: "Adapt-UCCSD",
        adapt_uccgsd_name: "Adapt-UCCGSD",
        qpu_qsci_name: "QPU-QSCI",
    }

    # Preserve the display mapping used in the original script:
    # displayed IS <- original FS_error/eV
    # displayed TS <- original TS_error/eV
    # displayed FS <- original IS_error/eV
    source_cols = {
        "IS": "FS_error/eV",
        "TS": "TS_error/eV",
        "FS": "IS_error/eV",
    }

    dsub = df[df["Method"].isin(wanted_methods)].copy()
    dsub["Method"] = pd.Categorical(dsub["Method"], categories=wanted_methods, ordered=True)
    dsub = dsub.sort_values("Method").reset_index(drop=True)

    # Keep missing entries as NaN so the figure layout remains stable.
    dsub = dsub.set_index("Method").reindex(wanted_methods).reset_index()

    for src_col in source_cols.values():
        if src_col not in dsub.columns:
            dsub[src_col] = np.nan

    dsub["DisplayMethod"] = dsub["Method"].map(display_labels).astype(str)
    return dsub, source_cols



def _prepare_qc_panel_data(df):
    """
    Prepare QC-error data for Panel A.

    This thin wrapper keeps the plotting code readable and avoids depending on
    helper functions introduced in earlier experimental versions.
    """
    dsub, source_cols = prepare_qc_error_dataframe(df)
    state_to_col = {
        "IS": source_cols["IS"],
        "TS": source_cols["TS"],
        "FS": source_cols["FS"],
    }
    return dsub, state_to_col


def plot_state_errors_qc_only_on_ax(ax, df, show_title=True, show_legend=True):
    """
    Panel A: QC energy-error matrix with a companion chem.-acc.-status strip.

    The main matrix only lists the numerical errors, while a slim status strip
    on the right indicates whether each cell is within chemical accuracy.
    This keeps the data matrix clean while preserving an immediate visual cue.
    """
    dsub, state_to_col = _prepare_qc_panel_data(df)

    methods = dsub["DisplayMethod"].astype(str).tolist()
    states = DISPLAY_STATE_ORDER
    nrow = len(states)
    ncol = len(methods)

    # Geometry for main matrix and slim status strip.
    status_gap = 0.18
    status_w = 0.28
    main_x0 = -0.5
    status_x0 = ncol - 0.5 + status_gap
    total_xmax = status_x0 + status_w + 0.08

    # Main numeric matrix (uniform background).
    for i, st in enumerate(states):
        for j, method in enumerate(methods):
            val = safe_float(dsub.iloc[j][state_to_col[st]], math.nan)
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=MATRIX_CELL_COLOR,
                edgecolor=CELL_EDGE_COLOR,
                linewidth=2.0,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(
                j,
                i,
                _format_error_label(val),
                ha="center",
                va="center",
                fontsize=CELL_TEXT_SIZE,
                fontweight=TEXT_WEIGHT,
                color=TITLE_COLOR,
                zorder=3,
            )

    # Companion status strip.
    for i, st in enumerate(states):
        for j, method in enumerate(methods):
            val = safe_float(dsub.iloc[j][state_to_col[st]], math.nan)
            is_ok = np.isfinite(val) and abs(val) <= CHEMICAL_ACCURACY_EV
            if not np.isfinite(val):
                face = MISSING_CELL_COLOR
            else:
                face = WITHIN_CELL_COLOR if is_ok else ABOVE_CELL_COLOR
            rect = Rectangle(
                (status_x0 + j * status_w / ncol, i - 0.5),
                status_w / ncol,
                1.0,
                facecolor=face,
                edgecolor=STATUS_EDGE_COLOR,
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(rect)

    # Outline boxes.
    ax.add_patch(Rectangle((main_x0, -0.5), ncol, nrow,
                           facecolor="none", edgecolor="#D0D0D0",
                           linewidth=0.9, zorder=5))
    ax.add_patch(Rectangle((status_x0, -0.5), status_w, nrow,
                           facecolor="none", edgecolor="#D0D0D0",
                           linewidth=0.9, zorder=5))

    # Small status title.
    ax.text(
        status_x0 + status_w / 2.0,
        -0.73,
        "chem. acc. status",
        ha="center",
        va="top",
        fontsize=LEGEND_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TITLE_COLOR,
        zorder=6,
    )

    # Axes formatting.
    ax.set_xlim(-0.5, total_xmax)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(
        methods,
        fontsize=METHOD_TICK_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TICK_COLOR,
    )
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(
        states,
        fontsize=TICK_LABEL_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TICK_COLOR,
    )

    ax.set_xlabel("")
    # Fig. 3(a): keep IS/TS/FS tick labels but remove the redundant y-axis title "State".
    ax.set_ylabel("")

    if show_title:
        ax.set_title(
            "QC Energy Errors (eV)",
            color=TITLE_COLOR,
            pad=8,
            fontsize=MAIN_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.grid(False)

    if show_legend:
        handles = [
            Patch(facecolor=WITHIN_CELL_COLOR, edgecolor="none",
                  label="≤ chem. acc."),
            Patch(facecolor=ABOVE_CELL_COLOR, edgecolor="none",
                  label="above chem. acc."),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(0.985, -0.01),
            frameon=False,
            fontsize=LEGEND_SIZE_FINAL,
            handlelength=1.4,
            borderpad=0.10,
            labelspacing=0.28,
        )
        _bold_legend(leg)

    return dsub

def plot_state_errors_qc_only(df, outfile=OUT_QC_ERRORS_PNG):
    fig, ax = plt.subplots(figsize=(6.9, 4.8), constrained_layout=True)
    plot_state_errors_qc_only_on_ax(ax, df, show_title=True, show_legend=True)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def find_method_name(df, candidates):
    """Return the first candidate method name that exists in df['Method']."""
    method_map = {str(m).strip().lower(): str(m).strip() for m in df["Method"].dropna().unique()}
    for cand in candidates:
        key = cand.strip().lower()
        if key in method_map:
            return method_map[key]
    return candidates[0]


def prepare_qpu_qsci_steps_dataframe(df):
    """
    Prepare step-5 and step-10 QPU-QSCI errors.

    The Excel table in the current workflow uses tianyan_recover_stepXX for
    hardware-QPU results after configuration recovery. Several aliases are
    allowed so the script is less sensitive to future naming changes.
    """
    step5_name = find_method_name(df, [
        "tianyan_recover_step5",
        "qpu_recover_step5",
        "cpu_recover_step5",
        "tianyan_step5_recover",
        "step5_recover",
    ])
    step10_name = find_method_name(df, [
        "tianyan_recover_step10",
        "qpu_recover_step10",
        "cpu_recover_step10",
        "tianyan_step10_recover",
        "step10_recover",
    ])

    wanted_methods = [step5_name, step10_name]
    display_labels = {
        step5_name: "step-5\nQPU-QSCI",
        step10_name: "step-10\nQPU-QSCI",
    }

    # Keep exactly the same state-column mapping as qc_errors.png:
    # displayed IS <- original FS_error/eV
    # displayed TS <- original TS_error/eV
    # displayed FS <- original IS_error/eV
    source_cols = {
        "IS": "FS_error/eV",
        "TS": "TS_error/eV",
        "FS": "IS_error/eV",
    }

    dsub = df[df["Method"].isin(wanted_methods)].copy()
    dsub["Method"] = pd.Categorical(dsub["Method"], categories=wanted_methods, ordered=True)
    dsub = dsub.sort_values("Method").reset_index(drop=True)

    # Keep missing rows as NaN so missing step-5/step-10 data are obvious.
    dsub = dsub.set_index("Method").reindex(wanted_methods).reset_index()

    for src_col in source_cols.values():
        if src_col not in dsub.columns:
            dsub[src_col] = np.nan

    dsub["DisplayMethod"] = dsub["Method"].map(display_labels).astype(str)
    return dsub, source_cols


def plot_qpu_qsci_step5_step10_on_ax(ax, df, show_title=True, show_legend=True):
    dsub, source_cols = prepare_qpu_qsci_steps_dataframe(df)
    plot_cols = [source_cols["IS"], source_cols["TS"], source_cols["FS"]]

    ymin, ymax = compute_ylim_positive(
        dsub[plot_cols].to_numpy(dtype=float),
        frac=0.25,
        default=(0.0, 0.05),
        include_values=[CHEMICAL_ACCURACY_EV]
    )

    x = np.arange(len(dsub))
    width = 0.24

    bars_is = ax.bar(
        x - width, dsub[source_cols["IS"]].values,
        width=width, label="IS",
        color=COLOR_IS, edgecolor="none", zorder=3
    )
    bars_ts = ax.bar(
        x, dsub[source_cols["TS"]].values,
        width=width, label="TS",
        color=COLOR_TS, edgecolor="none", zorder=3
    )
    bars_fs = ax.bar(
        x + width, dsub[source_cols["FS"]].values,
        width=width, label="FS",
        color=COLOR_FS, edgecolor="none", zorder=3
    )

    chem_line = ax.axhline(
        CHEMICAL_ACCURACY_EV,
        color=CHEM_ACC_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.95,
        zorder=2,
        label="chem. acc."
    )
    ax.axhline(0.0, color=ZERO_COLOR, linewidth=1.0, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(dsub["DisplayMethod"].astype(str), rotation=0, ha="center", color=TICK_COLOR)
    ax.set_ylabel("Energy error (eV)", color=LABEL_COLOR)
    if show_title:
        ax.set_title("QPU-QSCI energy errors", color=TITLE_COLOR, pad=8, fontweight="normal")
    ax.set_ylim(ymin, ymax)

    style_axes(ax)
    ax.tick_params(axis="x", length=0)

    if show_legend:
        handles = [chem_line, bars_is, bars_ts, bars_fs]
        labels = [f"Chemical accuracy = {CHEMICAL_ACCURACY_EV:.3f} eV", "IS", "TS", "FS"]
        ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.98),
                  frameon=False, handlelength=2.3)

    add_bar_labels(ax, bars_is, fmt="{:.3f}", fontsize=7.2)
    add_bar_labels(ax, bars_ts, fmt="{:.3f}", fontsize=7.2)
    add_bar_labels(ax, bars_fs, fmt="{:.3f}", fontsize=7.2)

    return dsub


def plot_qpu_qsci_step5_step10(df, outfile_png=OUT_QPU_QSCI_STEPS_PNG):
    fig, ax = plt.subplots(figsize=(6.9, 4.8), constrained_layout=True)
    plot_qpu_qsci_step5_step10_on_ax(ax, df, show_title=True, show_legend=True)
    fig.savefig(outfile_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Part 2: NEVPT2 scan collection and reduction plot
# =========================================================

def read_runlog_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_selected_steps(path: Path):
    mp = {}
    if not path.exists():
        return mp

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                step = int(parts[0])
                err = float(parts[2])
                mp[step] = err
            except Exception:
                continue
    return mp


def parse_runlog_quantities(text: str):
    out = {}

    patterns = {
        "fcidump_ecore": r"FCIDUMP Ecore\s*=\s*([-\d\.Ee+]+)",
        "fcidump_etot": r"FCIDUMP Etot\s*=\s*([-\d\.Ee+]+)",
        "fci_ref_energy": r"FCI reference energy\s*=\s*([-\d\.Ee+]+)\s*Ha",
        # Keep this parser backward-compatible with old runlogs.
        # Some runlogs still print "QCI_subspace" even if the plotting labels
        # while internal dataframe columns use the shorter QC label.
        "QC_fci_overlap_abs": r"\|<\s*(?:QCW|QCI|QSCI|QC)_subspace\s*\|\s*FCI\s*>\|\s*=\s*([-\d\.Ee+]+)",
        "nevpt2_err_ev": r"NEVPT2 error\s*=\s*[-\d\.Ee+]+\s*Ha\s*=\s*([-\d\.Ee+]+)\s*eV",
        "nevpt2_err_ha": r"NEVPT2 error\s*=\s*([-\d\.Ee+]+)\s*Ha",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = safe_float(m.group(1))

    return out


def load_one_step(step_dir: Path, selected_step_err_map: dict):
    runlog_path = step_dir / "runlog"
    if not runlog_path.exists():
        raise FileNotFoundError(f"Missing runlog in {step_dir}")

    text = read_runlog_text(runlog_path)
    data = parse_runlog_quantities(text)

    m = re.match(r"step_(\d+)$", step_dir.name)
    if not m:
        raise RuntimeError(f"Cannot parse step index from directory name: {step_dir.name}")
    step = int(m.group(1))

    data["step"] = step
    data["step_dir"] = str(step_dir)
    data["runlog_err_ha"] = selected_step_err_map.get(step, math.nan)

    fcidump_ecore = safe_float(data.get("fcidump_ecore", math.nan))
    fcidump_etot = safe_float(data.get("fcidump_etot", math.nan))
    fci_ref_energy = safe_float(data.get("fci_ref_energy", math.nan))
    overlap_abs = safe_float(data.get("QC_fci_overlap_abs", math.nan))

    if not math.isnan(fcidump_ecore) and not math.isnan(fci_ref_energy):
        data["fci_total_from_fcidump_ha"] = fcidump_ecore + fci_ref_energy
    else:
        data["fci_total_from_fcidump_ha"] = math.nan

    if not math.isnan(fcidump_etot) and not math.isnan(data["fci_total_from_fcidump_ha"]):
        QC_fci_err_ha = abs(fcidump_etot - data["fci_total_from_fcidump_ha"])
        data["QC_fci_err_ha"] = QC_fci_err_ha
        data["QC_fci_err_ev"] = QC_fci_err_ha * HARTREE_TO_EV
    else:
        data["QC_fci_err_ha"] = math.nan
        data["QC_fci_err_ev"] = math.nan

    data["QC_fci_overlap_abs"] = overlap_abs

    return data


def resolve_scan_dir(candidates):
    return Path(candidates[0])


def collect_nevpt2_scan(scan_dir: Path):
    """Load a complete, traceable scan table reduced from the original logs.

    This publication copy does not require the historic runlogs or the
    incomplete selected_steps.txt files. All plotted values are preserved.
    """
    source_state = Path(scan_dir).name
    table = Path("data/nevpt2_scan.csv")
    if not table.is_file():
        raise FileNotFoundError(table)
    df = pd.read_csv(table)
    df = df[df["source_state"] == source_state].copy()
    if df.empty:
        raise RuntimeError("No scan rows for source state " + source_state)
    return df.sort_values("step").reset_index(drop=True)

def choose_representative_nevpt2_row(df):
    """
    Select the representative point for the NEVPT2 error-reduction diagram.

    The purpose of this panel is to show how much the NEVPT2 correction reduces
    the raw QC error once the NEVPT2-corrected total-energy error has reached
    chemical accuracy. Therefore the selected point is simply the earliest step
    with |NEVPT2 error| <= chemical accuracy.

    If no point reaches chemical accuracy, fall back to the step with the
    smallest |NEVPT2 error| so the figure still reports the best available
    reduction.
    """
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.sort_values("step").reset_index(drop=True)

    cand = d[d["nevpt2_err_ev_abs"] <= TARGET_NEVPT2_EV]
    if not cand.empty:
        row = cand.iloc[0].copy()
        row["selection_reason"] = "first nevpt2<=chem_acc"
        return row

    cand = d.dropna(subset=["nevpt2_err_ev_abs"])
    if not cand.empty:
        row = cand.sort_values(["nevpt2_err_ev_abs", "step"]).iloc[0].copy()
        row["selection_reason"] = "global minimum NEVPT2 error"
        return row

    row = d.iloc[0].copy()
    row["selection_reason"] = "fallback first available step"
    return row


def collect_all_nevpt2_reduction_data():
    selected_rows = []
    all_scan_tables = {}

    for state in DISPLAY_STATE_ORDER:
        scan_dir = resolve_scan_dir(NEVPT2_SCAN_DIRS[state])
        print(f"[INFO] Reading {state} NEVPT2 scan from: {scan_dir}")

        try:
            df_scan = collect_nevpt2_scan(scan_dir)
            df_scan.insert(0, "state", state)
            all_scan_tables[state] = df_scan

            row = choose_representative_nevpt2_row(df_scan)
            row = row.copy()
            row["state"] = state
            row["scan_dir"] = str(scan_dir)
            selected_rows.append(row)

        except Exception as e:
            print(f"[WARN] Failed to read {state} NEVPT2 scan: {e}")
            selected_rows.append(pd.Series({
                "state": state,
                "scan_dir": str(scan_dir),
                "step": math.nan,
                "QC_fci_err_ev_abs": math.nan,
                "QC_fci_overlap_abs": math.nan,
                "nevpt2_err_ev_abs": math.nan,
                "selection_reason": f"failed: {e}",
            }))

    summary = pd.DataFrame(selected_rows)
    summary = summary[[
        "state",
        "scan_dir",
        "step",
        "QC_fci_err_ev_abs",
        "QC_fci_overlap_abs",
        "nevpt2_err_ev_abs",
        "selection_reason",
    ]]
    return summary, all_scan_tables


def plot_nevpt2_error_reduction_on_ax(ax, summary):
    x = np.arange(len(DISPLAY_STATE_ORDER), dtype=float)
    half_width = 0.25

    before_vals = []
    after_vals = []

    for i, state in enumerate(DISPLAY_STATE_ORDER):
        row_df = summary[summary["state"] == state]
        if row_df.empty:
            before_vals.append(math.nan)
            after_vals.append(math.nan)
            continue

        row = row_df.iloc[0]
        before = safe_float(row.get("QC_fci_err_ev_abs", math.nan))
        after = safe_float(row.get("nevpt2_err_ev_abs", math.nan))
        overlap = safe_float(row.get("QC_fci_overlap_abs", math.nan))

        before_vals.append(before)
        after_vals.append(after)

        if not (np.isfinite(before) and np.isfinite(after)):
            ax.text(i, CHEMICAL_ACCURACY_EV * 1.25, "n/a",
                    ha="center", va="bottom", fontsize=9, color="#777777")
            continue

        ax.hlines(
            before, i - half_width, i + half_width,
            color=COLOR_QC, lw=4.0, zorder=3,
            label="QC error" if i == 0 else None,
        )
        ax.hlines(
            after, i - half_width, i + half_width,
            color=COLOR_QC_NEVPT2, lw=4.0, zorder=4,
            label="QC + NEVPT2 error" if i == 0 else None,
        )

        # Downward reduction arrow.
        if before >= after:
            ax.annotate(
                "",
                xy=(i, after),
                xytext=(i, before),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COLOR_ARROW,
                    lw=1.25,
                    mutation_scale=10,
                    shrinkA=2,
                    shrinkB=2,
                ),
                zorder=2,
            )
        else:
            ax.plot([i, i], [before, after], color=COLOR_ARROW, lw=1.25, zorder=2)

        # Label only the wavefunction overlap on the QC-error horizontal line.
        # Keep the plot clean: only show the overlap percentage here.
        # Define the meaning in the figure caption:
        # percentages above QC-error bars denote |<QC|FCI>|.
        if np.isfinite(overlap):
            overlap_label = f"{overlap * 100:.1f}%"
        else:
            overlap_label = "n/a"
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        label_offset = 0.025 * yr

        ax.text(
            i,
            before + label_offset,
            overlap_label,
            ha="center", va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color=COLOR_QC,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
    ax.axhline(
        CHEMICAL_ACCURACY_EV,
        color=CHEM_ACC_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.95,
        zorder=1,
        label="chem. acc.",
    )
    ax.axhline(0.0, color=ZERO_COLOR, linewidth=1.0, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_STATE_ORDER, rotation=0, ha="center")
    ax.set_ylabel("Energy error (eV)", color=LABEL_COLOR)
    ax.set_title("NEVPT2 error reduction", color=TITLE_COLOR, pad=8, fontweight="normal")

    vals = np.array(before_vals + after_vals + [CHEMICAL_ACCURACY_EV], dtype=float)
    vals = vals[np.isfinite(vals)]
    ymax = 0.25 if len(vals) == 0 else max(0.12, float(np.max(vals)) * 1.22)
    ax.set_ylim(0.0, ymax)

    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)


def plot_nevpt2_error_reduction(summary, outfile=OUT_NEVPT2_REDUCTION_PNG):
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    plot_nevpt2_error_reduction_on_ax(ax, summary)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Combined figure
# =========================================================

def plot_combined_figure(qc_df, nevpt2_summary):
    # Single final version: threshold-colored matrix + discrete chem.-acc.-status colorbar.
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14.2, 4.65),
        gridspec_kw={"width_ratios": [1.03, 1.00], "wspace": 0.36},
    )

    plot_state_errors_qc_only_on_ax(ax1, qc_df, show_title=True, show_legend=True)
    plot_nevpt2_error_reduction_on_ax(ax2, nevpt2_summary, show_title=True, show_legend=True)

    add_panel_label(ax1, "(a)", x=-0.13, y=1.08)
    add_panel_label(ax2, "(b)", x=-0.13, y=1.08)

    fig.subplots_adjust(left=0.070, right=0.972, bottom=0.20, top=0.84)
    fig.savefig(OUT_COMBINED_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)



# =========================================================
# Final checked plotting overrides
# =========================================================
# These definitions intentionally override earlier experimental plotting
# functions above.  They keep only the final design:
#   A: QC-error matrix with a discrete "chem. acc." colorbar.
#   B: linear horizontal NEVPT2-correction arrows.
# No PDF output is produced by these overrides.

from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable


# -------------------------
# Final typography and colors
# -------------------------
TEXT_WEIGHT = "bold"

MAIN_TITLE_SIZE = 15.2
AXIS_LABEL_SIZE_FINAL = 13.2
TICK_LABEL_SIZE_FINAL = 12.0
METHOD_TICK_SIZE_FINAL = 12.2
LEGEND_SIZE_FINAL = 12.0
PANEL_LABEL_SIZE_FINAL = 15.6
CELL_TEXT_SIZE = 10.9
VALUE_LABEL_SIZE_FINAL = 11.5
OVERLAP_LABEL_SIZE_FINAL = 11.4
# Vertical offset for overlap labels in Panel B, in y-axis state units.
# Smaller values place the label closer to the corresponding QC square.
OVERLAP_LABEL_Y_OFFSET = {
    "IS": 0.130,
    "TS": 0.135,
    "FS": 0.155,
}
CBAR_TICK_SIZE_FINAL = 10.0
CBAR_LABEL_SIZE_FINAL = 10.0

# Panel (a) chemical-accuracy status colors.
WITHIN_CELL_COLOR = "#E8F3EF"
ABOVE_CELL_COLOR = "#FBE7D8"
MISSING_CELL_COLOR = "#F4F4F4"
MATRIX_CELL_COLOR = "#FBFBFB"
CELL_EDGE_COLOR = "#FFFFFF"
STATUS_EDGE_COLOR = "#D8D8D8"

# Panel (b) uses exactly the same solid fill as the
# "within chemical accuracy" cells in Panel (a).
CHEM_ACC_BAND_COLOR = WITHIN_CELL_COLOR
CHEM_ACC_LINE_COLOR = "#5E9F8F"

COLOR_QC = "#7FAED0"             # soft blue: QC error
COLOR_QC_TEXT = "#4F7FA4"        # darker blue for QC labels
COLOR_QC_NEVPT2 = "#A99ACB"      # muted lavender: QC + NEVPT2 error
COLOR_QC_NEVPT2_TEXT = "#78699D" # darker lavender for labels
COLOR_ARROW = "#6F6F6F"

NEVPT2_XMAX_PAD = 1.15

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": TICK_LABEL_SIZE_FINAL,
    "font.weight": "normal",
    "axes.labelsize": AXIS_LABEL_SIZE_FINAL,
    "axes.titlesize": MAIN_TITLE_SIZE,
    "axes.labelweight": TEXT_WEIGHT,
    "axes.titleweight": TEXT_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE_FINAL,
    "ytick.labelsize": TICK_LABEL_SIZE_FINAL,
    "legend.fontsize": LEGEND_SIZE_FINAL,
    "figure.titlesize": MAIN_TITLE_SIZE,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _style_axes_final(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", colors=TICK_COLOR, labelsize=TICK_LABEL_SIZE_FINAL)
    ax.grid(False)

    ax.xaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.title.set_fontweight(TEXT_WEIGHT)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)


def add_panel_label(ax, label, x=-0.10, y=1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TITLE_COLOR,
        clip_on=False,
    )


def _bold_legend(leg):
    if leg is None:
        return
    for txt in leg.get_texts():
        txt.set_fontweight(TEXT_WEIGHT)
    title = leg.get_title()
    if title is not None:
        title.set_fontweight(TEXT_WEIGHT)


def _bold_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE_FINAL)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
    cbar.ax.yaxis.label.set_fontweight(TEXT_WEIGHT)


def _format_error_label(v):
    if not np.isfinite(v):
        return "n/a"
    av = abs(v)
    if av < 5.0e-8:
        return r"$\approx 0$"
    if av < 1.0e-4:
        return f"{av:.1e}"
    if av < 1.0e-3:
        return f"{av:.1e}"
    return f"{av:.3f}"


def _prepare_qc_panel_data(df):
    dsub, source_cols = prepare_qc_error_dataframe(df)
    state_to_col = {
        "IS": source_cols["IS"],
        "TS": source_cols["TS"],
        "FS": source_cols["FS"],
    }
    return dsub, state_to_col


def _get_qc_matrix(df):
    dsub, state_to_col = _prepare_qc_panel_data(df)
    methods = dsub["DisplayMethod"].astype(str).tolist()
    states = DISPLAY_STATE_ORDER

    mat = np.full((len(states), len(methods)), np.nan, dtype=float)
    for i, st in enumerate(states):
        for j, _method in enumerate(methods):
            mat[i, j] = safe_float(dsub.iloc[j][state_to_col[st]], math.nan)
    return mat, methods, states, dsub


def _add_discrete_chemacc_colorbar(ax):
    cmap = ListedColormap([WITHIN_CELL_COLOR, ABOVE_CELL_COLOR])
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cax = ax.inset_axes([1.030, 0.22, 0.032, 0.56])
    cbar = plt.colorbar(sm, cax=cax, boundaries=[0, 1, 2], ticks=[0.5, 1.5])
    cbar.ax.set_yticklabels(["≤ chem. acc.", "> chem. acc."])
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor("#999999")
    _bold_colorbar(cbar)
    return cbar


def plot_state_errors_qc_only_on_ax(ax, df, show_title=True, show_legend=True):
    """
    Final Panel A: threshold-colored QC energy-error matrix.

    The matrix shows numerical errors in eV.  The cell color encodes whether
    the error is within chemical accuracy, and the discrete colorbar explains
    this status.
    """
    mat, methods, states, dsub = _get_qc_matrix(df)

    nrow, ncol = mat.shape
    for i in range(nrow):
        for j in range(ncol):
            val = mat[i, j]
            if not np.isfinite(val):
                face = MISSING_CELL_COLOR
            elif abs(val) <= CHEMICAL_ACCURACY_EV:
                face = WITHIN_CELL_COLOR
            else:
                face = ABOVE_CELL_COLOR

            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=face,
                    edgecolor=CELL_EDGE_COLOR,
                    linewidth=2.0,
                    zorder=1,
                )
            )
            ax.text(
                j, i,
                _format_error_label(val),
                ha="center", va="center",
                fontsize=CELL_TEXT_SIZE,
                fontweight=TEXT_WEIGHT,
                color=TITLE_COLOR,
                zorder=3,
            )

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            ncol,
            nrow,
            facecolor="none",
            edgecolor="#D0D0D0",
            linewidth=0.9,
            zorder=5,
        )
    )

    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(
        methods,
        fontsize=METHOD_TICK_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TICK_COLOR,
    )
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(
        states,
        fontsize=TICK_LABEL_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TICK_COLOR,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    if show_title:
        ax.set_title(
            "QC Energy Errors (eV)",
            color=TITLE_COLOR,
            pad=8,
            fontsize=MAIN_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.grid(False)

    if show_legend:
        _add_discrete_chemacc_colorbar(ax)

    return dsub


def plot_state_errors_qc_only(df, outfile=OUT_QC_ERRORS_PNG):
    fig, ax = plt.subplots(figsize=(7.45, 4.35))
    plot_state_errors_qc_only_on_ax(ax, df, show_title=True, show_legend=True)
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.13, top=0.84)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _prepare_reduction_values(summary):
    rows = []
    for i, state in enumerate(DISPLAY_STATE_ORDER):
        row_df = summary[summary["state"] == state]
        if row_df.empty:
            rows.append({
                "state": state,
                "y": float(len(DISPLAY_STATE_ORDER) - 1 - i),
                "before": math.nan,
                "after": math.nan,
                "overlap": math.nan,
            })
            continue

        row = row_df.iloc[0]
        rows.append({
            "state": state,
            "y": float(len(DISPLAY_STATE_ORDER) - 1 - i),
            "before": safe_float(row.get("QC_fci_err_ev_abs", math.nan)),
            "after": safe_float(row.get("nevpt2_err_ev_abs", math.nan)),
            "overlap": safe_float(row.get("QC_fci_overlap_abs", math.nan)),
        })
    return rows


def _overlap_text(overlap):
    if not np.isfinite(overlap):
        return "overlap n/a"
    return f"{overlap * 100:.1f}% overlap"


def plot_nevpt2_error_reduction_on_ax(ax, summary, show_title=True, show_legend=True):
    """
    Final Panel B: linear horizontal correction-arrow diagram.

    QC errors are shown as soft-blue squares, while QC + NEVPT2 errors are
    shown as muted-lavender circles. Peach and green remain reserved for
    the chemical-accuracy status coding in Panel A.  Both values are shown parallel to the corresponding points.
    """
    rows = _prepare_reduction_values(summary)

    before_vals = np.asarray([r["before"] for r in rows if np.isfinite(r["before"])], dtype=float)
    after_vals = np.asarray([r["after"] for r in rows if np.isfinite(r["after"])], dtype=float)
    all_vals = np.concatenate([before_vals, after_vals, np.asarray([CHEMICAL_ACCURACY_EV])])
    xmax = max(0.34, float(np.nanmax(all_vals)) * NEVPT2_XMAX_PAD)

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(-0.55, len(DISPLAY_STATE_ORDER) - 0.45)

    ax.axvspan(
        0.0,
        CHEMICAL_ACCURACY_EV,
        color=CHEM_ACC_BAND_COLOR,
        alpha=1.0,
        lw=0,
        zorder=0,
    )
    ax.axvline(
        CHEMICAL_ACCURACY_EV,
        color=CHEM_ACC_LINE_COLOR,
        linestyle=(0, (4, 2.4)),
        linewidth=1.45,
        alpha=0.98,
        zorder=1,
    )

    for r in rows:
        y = r["y"]
        state = r["state"]
        before = r["before"]
        after = r["after"]
        overlap = r["overlap"]

        if not (np.isfinite(before) and np.isfinite(after)):
            continue

        ax.annotate(
            "",
            xy=(after, y),
            xytext=(before, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLOR_ARROW,
                lw=1.65,
                mutation_scale=15,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

        ax.scatter(
            [before],
            [y],
            s=118,
            marker="s",
            facecolor=COLOR_QC,
            edgecolor="white",
            linewidth=0.75,
            zorder=4,
            label="QC error" if state == DISPLAY_STATE_ORDER[0] else None,
        )
        ax.scatter(
            [after],
            [y],
            s=118,
            marker="o",
            facecolor=COLOR_QC_NEVPT2,
            edgecolor="white",
            linewidth=0.75,
            zorder=5,
            label="QC + NEVPT2 error" if state == DISPLAY_STATE_ORDER[0] else None,
        )

        ax.text(
            before + xmax * 0.020,
            y,
            f"{before:.3f}",
            ha="left",
            va="center",
            fontsize=VALUE_LABEL_SIZE_FINAL,
            fontweight=TEXT_WEIGHT,
            color=COLOR_QC_TEXT,
            clip_on=False,
        )
        ax.text(
            after - xmax * 0.022,
            y,
            f"{after:.3f}",
            ha="right",
            va="center",
            fontsize=VALUE_LABEL_SIZE_FINAL,
            fontweight=TEXT_WEIGHT,
            color=COLOR_QC_NEVPT2_TEXT,
            clip_on=False,
        )

        ax.text(
            before,
            y + OVERLAP_LABEL_Y_OFFSET.get(state, 0.14),
            _overlap_text(overlap),
            ha="center",
            va="bottom",
            fontsize=OVERLAP_LABEL_SIZE_FINAL,
            fontweight=TEXT_WEIGHT,
            color="#6F9EC6",
            bbox=dict(
                boxstyle="round,pad=0.14",
                facecolor="white",
                edgecolor="none",
                alpha=0.88,
            ),
            zorder=6,
        )

    y_ticks = [r["y"] for r in rows]
    y_labels = [r["state"] for r in rows]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        y_labels,
        fontsize=TICK_LABEL_SIZE_FINAL,
        fontweight=TEXT_WEIGHT,
        color=TICK_COLOR,
    )

    ax.set_xlabel("Energy error (eV)", fontsize=AXIS_LABEL_SIZE_FINAL, fontweight=TEXT_WEIGHT)
    ax.set_ylabel("State", fontsize=AXIS_LABEL_SIZE_FINAL, fontweight=TEXT_WEIGHT)

    if show_title:
        ax.set_title(
            "NEVPT2 error reduction",
            color=TITLE_COLOR,
            pad=8,
            fontsize=MAIN_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
        )

    _style_axes_final(ax)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=9.4,
                   markerfacecolor=COLOR_QC, markeredgecolor="white",
                   markeredgewidth=0.75, label="QC error"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=9.4,
                   markerfacecolor=COLOR_QC_NEVPT2, markeredgecolor="white",
                   markeredgewidth=0.75, label="QC + NEVPT2 error"),
            Patch(facecolor=CHEM_ACC_BAND_COLOR, edgecolor="none", alpha=1.0,
                  label="≤ chem. acc."),
        ]
        leg = ax.legend(
            handles=handles,
            frameon=False,
            loc="center right",
            bbox_to_anchor=(0.985, 0.50),
            handlelength=1.4,
            handletextpad=0.55,
            borderpad=0.10,
            labelspacing=0.42,
            fontsize=LEGEND_SIZE_FINAL,
        )
        _bold_legend(leg)


def plot_nevpt2_error_reduction(summary, outfile=OUT_NEVPT2_REDUCTION_PNG):
    fig, ax = plt.subplots(figsize=(7.35, 4.35))
    plot_nevpt2_error_reduction_on_ax(ax, summary, show_title=True, show_legend=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.20, top=0.84)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(qc_df, nevpt2_summary):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(13.8, 4.45),
        gridspec_kw={"width_ratios": [1.03, 1.00], "wspace": 0.36},
    )

    plot_state_errors_qc_only_on_ax(ax1, qc_df, show_title=True, show_legend=True)
    # Fig. 3(b): remove the title above the panel in the combined manuscript figure.
    plot_nevpt2_error_reduction_on_ax(ax2, nevpt2_summary, show_title=False, show_legend=True)

    add_panel_label(ax1, "(a)", x=-0.13, y=1.08)
    add_panel_label(ax2, "(b)", x=-0.13, y=1.08)

    fig.subplots_adjust(left=0.070, right=0.972, bottom=0.14, top=0.86)
    fig.savefig(OUT_COMBINED_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    os.makedirs("output", exist_ok=True)
    print(f"Chemical accuracy = {CHEMICAL_ACCURACY_EV:.6f} eV")
    print()

    qc_df = load_qc_table(EXCEL_FILE)
    print("Loaded QC data table:")
    print(qc_df)
    print()

    plot_state_errors_qc_only(qc_df, OUT_QC_ERRORS_PNG)
    print(f"Saved: {OUT_QC_ERRORS_PNG}")

    plot_qpu_qsci_step5_step10(qc_df, OUT_QPU_QSCI_STEPS_PNG)
    print(f"Saved: {OUT_QPU_QSCI_STEPS_PNG}")

    nevpt2_summary, _ = collect_all_nevpt2_reduction_data()
    print()
    print("Representative NEVPT2 reduction points:")
    print(nevpt2_summary.to_string(index=False))
    print()

    plot_nevpt2_error_reduction(nevpt2_summary, OUT_NEVPT2_REDUCTION_PNG)
    print(f"Saved: {OUT_NEVPT2_REDUCTION_PNG}")

    plot_combined_figure(qc_df, nevpt2_summary)
    print(f"Saved: {OUT_COMBINED_PNG}")
    print("\nDone.")


if __name__ == "__main__":
    main()
