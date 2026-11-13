#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the CAS(22e,20o)/40-qubit four-panel CO/Cu(100) figure.

Adapted copy: run from the standalone Fig6 directory as

    python src/plot.py

Default input
-------------
data/frontier_fig3d_mayer_bond_order_summary.csv

If the global CSV is unavailable because the Slurm-array post-processing has
not combined it, the script automatically merges the three adsorption-site
CSVs under
``data/site_analysis/{top,bridge,hollow}``.

Panels
------
(a) Adsorption-energy benchmark.
(b) Bridge/hollow stability relative to top.
(c) Top-site emb-DFT -> emb-QC Mayer-channel correction.
(d) Site-resolved emb-QC C--Cu total/sigma/pi comparison for top, bridge,
    and hollow, using the same heatmap style as the Cu(111) figure.

The ``total C--Cu`` quantity in panel (d) is explicitly the sum of the selected
projected channels, M_sigma + M_pi.  It is not relabelled as a full-AO atom-pair
Mayer index.

Outputs
-------
co_cu100_cas22e20o_four_panels_<variant>.png
co_cu100_cas22e20o_selected_values_<variant>.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator


# =============================================================================
# User-editable defaults
# =============================================================================

DEFAULT_INPUT = "data/frontier_fig3d_mayer_bond_order_summary.csv"
DEFAULT_OUTDIR = "output"
DEFAULT_SITES = ("top", "bridge", "hollow")
STATE_CSV_TEMPLATE = (
    "data/site_analysis/{state}/"
    "{state}_mayer_bond_order_summary.csv"
)

# Experimental adsorption-energy interval from TDS + chemical accommodation.
EXP_LOW = -0.588
EXP_HIGH = -0.472
EXP_CENTER = 0.5 * (EXP_LOW + EXP_HIGH)

# Site-resolved ZPE values used to correct the SCAN/RPA electronic site gaps.
SITE_ZPE = {
    "top": 1.680832,
    "abv": 1.628090,
    "bridge": 1.674575,
    "hollow": 1.662181,
}


def add_relative_zpe(electronic_gap: float, site: str) -> float:
    """Return electronic site gap + ZPE(site) - ZPE(top)."""
    return electronic_gap + SITE_ZPE[site] - SITE_ZPE["top"]


# SCAN/RPA paper site gaps are electronic values; add only the relative ZPE.
SCAN_BRIDGE_TOP = add_relative_zpe(-0.021, "bridge")
SCAN_HOLLOW_TOP = add_relative_zpe(+0.013, "hollow")
RPA_BRIDGE_TOP = add_relative_zpe(+0.114, "bridge")
RPA_HOLLOW_TOP = add_relative_zpe(+0.245, "hollow")

# Fig. 6(a,b) reference values (eV).  "PW91" denotes Zhang et al., Fuel
# Processing Technology 156, 253–264 (2017); the figure itself uses only
# the short method name.  "emb-MC-ftPBE" includes dispersion correction.
# Source-specific energy and ZPE conventions have been normalized to the
# manuscript's site-resolved values; panels (c,d) are unchanged.
# QC-DFET uses the four-site, ~1M singlet-CSF EN-SC-NEVPT2 results with
# the verified 5x5x4 ("554") PBE-without-dispersion reference energies
# and consistent site ZPE / adsorption thermodynamic corrections.
# Master: QSCI40_CO100_FOUR_SITES_1M_ENSC_QCDFET_554_FINAL_20261102.xlsx
METHODS = (
    "PBE-D3BJ",
    "SCAN",
    "PW91",
    "emb-MC-ftPBE",
    "RPA",
    "QC-DFET",
)

ADSORPTION_ENERGY = np.array(
    [-1.003958, -0.949000, -0.991612111695, -0.403908230000,
     -0.500000, -0.5878714459],
    dtype=float,
)
BRIDGE_TOP = np.array(
    [-0.082567, SCAN_BRIDGE_TOP, -0.033204,
     0.071798, RPA_BRIDGE_TOP, 0.3639687505],
    dtype=float,
)
HOLLOW_TOP = np.array(
    [-0.120781, SCAN_HOLLOW_TOP, -0.017615,
     0.319589, RPA_HOLLOW_TOP, 0.7609534447],
    dtype=float,
)


VARIANTS: Dict[str, Dict[str, object]] = {
    "p_allcu": {
        "description": "all C/O p shells; all embedded Cu atoms",
        "channels": {
            "co_sigma": "p_sigma_Cpz_Opz",
            "co_pi": "p_pi_Cpxy_Opxy",
            "cu_sigma": "p_sigma_Cpz_Cu_all",
            "cu_pi": "p_pi_Cpxy_Cu3d_all",
        },
    },
    "p2_allcu": {
        "description": "strict C/O 2p shells; all embedded Cu atoms",
        "channels": {
            "co_sigma": "p2_sigma_C2pz_O2pz",
            "co_pi": "p2_pi_C2pxy_O2pxy",
            "cu_sigma": "p2_sigma_C2pz_Cu_all",
            "cu_pi": "p2_pi_C2pxy_Cu3d_all",
        },
    },
}

CHANNEL_LABELS = {
    "co_sigma": "C–O σ",
    "co_pi": "C–O π",
    "cu_sigma": "C–Cu σ",
    "cu_pi": "C–Cu π",
}

# =============================================================================
# Figure style
# =============================================================================

METHOD_COLORS = {
    "PBE-D3BJ": "#8A94A6",
    "SCAN": "#D99A8D",
    "PW91": "#8A96A8",
    "emb-MC-ftPBE": "#8E78B7",
    "RPA": "#C9AD79",
    "QC-DFET": "#83BFAE",
}

SITE_COLORS = {
    "bridge-top": "#8FBBD8",
    "hollow-top": "#A9C7A1",
}

COLORS = {
    "emb_dft": "#B5E5FA",
    "emb_qc": "#F8CBC0",
    "connector": "#7F8B96",
    "arrow": "#707C88",
    "text": "#000000",
    "tick": "#000000",
    "spine": "#5F6B76",
    "grid": "#8D99A6",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "mayer_soft_peach",
    ["#FCFAF9", "#F8EEE9", "#F5E2DC", "#F8D8D0", "#F6D0C6"],
)

TEXT_WEIGHT = "bold"
TITLE_SIZE = 14.5
LABEL_SIZE = 12.5
TICK_SIZE = 11.0
LEGEND_SIZE = 10.5
PANEL_LABEL_SIZE = 16.0
PANEL_D_CELL_SIZE = 11.6
PANEL_D_CBAR_LABEL_SIZE = 12.8
PANEL_D_CBAR_TICK_SIZE = 11.0

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelweight": TEXT_WEIGHT,
        "axes.titleweight": TEXT_WEIGHT,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 1.05,
        "axes.edgecolor": COLORS["spine"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["tick"],
        "ytick.color": COLORS["tick"],
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =============================================================================
# Input and Mayer helpers
# =============================================================================

def resolve_input(path_text: str) -> Path:
    """Resolve relative input first from cwd, then from the script directory."""
    path = Path(path_text).expanduser()
    if path.is_absolute() and path.is_file():
        return path
    if path.is_file():
        return path.resolve()
    script_candidate = Path(__file__).resolve().parent / path
    if script_candidate.is_file():
        return script_candidate
    raise FileNotFoundError(
        f"Cannot find input CSV '{path_text}'. Run the script from the directory "
        "containing data/site_analysis, or provide "
        "--input /absolute/path/to/file.csv."
    )


def state_csv_paths(base_dir: Path) -> list[Path]:
    return [
        base_dir / STATE_CSV_TEMPLATE.format(state=state)
        for state in DEFAULT_SITES
    ]


def merge_state_csvs(base_dir: Path, output_path: Path) -> pd.DataFrame:
    """Merge the three adsorption-site Mayer CSVs with strict validation."""
    paths = state_csv_paths(base_dir)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The global Mayer CSV is unavailable and the following state-level "
            "CSVs are also missing:\n  " + "\n  ".join(str(path) for path in missing)
        )

    frames = []
    reference_columns: list[str] | None = None
    for expected_state, path in zip(DEFAULT_SITES, paths):
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"State-level Mayer CSV is empty: {path}")
        if "state" not in frame.columns:
            raise KeyError(f"State-level Mayer CSV lacks the 'state' column: {path}")
        states = frame["state"].astype(str).str.strip().str.lower().unique().tolist()
        if states != [expected_state]:
            raise ValueError(
                f"Expected only state='{expected_state}' in {path}, found {states}."
            )
        columns = frame.columns.tolist()
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                f"CSV schema differs for {path}; all three site outputs must use "
                "the same Mayer-analysis code."
            )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[INFO] Rebuilt global Mayer CSV from three site-level files: {output_path}")
    return merged


def load_mayer_input(path_text: str) -> tuple[pd.DataFrame, Path]:
    """Read the requested CSV, rebuilding the default global file if necessary."""
    try:
        input_path = resolve_input(path_text)
        try:
            dataframe = pd.read_csv(input_path)
        except pd.errors.EmptyDataError:
            if Path(path_text) != Path(DEFAULT_INPUT):
                raise
            dataframe = merge_state_csvs(Path.cwd(), Path.cwd() / DEFAULT_INPUT)
            input_path = (Path.cwd() / DEFAULT_INPUT).resolve()
        return normalize_dataframe(dataframe), input_path
    except FileNotFoundError:
        if Path(path_text) != Path(DEFAULT_INPUT):
            raise

    cwd_output = Path.cwd() / DEFAULT_INPUT
    try:
        dataframe = merge_state_csvs(Path.cwd(), cwd_output)
        return normalize_dataframe(dataframe), cwd_output.resolve()
    except FileNotFoundError as cwd_error:
        script_dir = Path(__file__).resolve().parent
        if script_dir == Path.cwd().resolve():
            raise cwd_error
        script_output = script_dir / DEFAULT_INPUT
        dataframe = merge_state_csvs(script_dir, script_output)
        return normalize_dataframe(dataframe), script_output.resolve()


def ensure_outdir(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in df.columns:
        raise KeyError("Input CSV must contain a 'state' column.")
    out = df.copy()
    out["state"] = out["state"].astype(str).str.strip().str.lower()
    if out["state"].duplicated().any():
        duplicates = sorted(out.loc[out["state"].duplicated(False), "state"].unique())
        raise ValueError(f"Duplicate state rows found: {duplicates}")
    for column in out.columns:
        if column != "state":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def column_for(channel_key: str, method: str) -> str:
    if method == "emb-DFT":
        return f"{channel_key}_embRKS"
    if method == "emb-QC":
        return f"{channel_key}_CASSCF"
    raise ValueError(f"Unknown method: {method}")


def row_for(df: pd.DataFrame, state: str) -> pd.Series:
    matched = df[df["state"] == state.lower()]
    if len(matched) != 1:
        raise KeyError(
            f"Expected one row for state='{state}', found {len(matched)}. "
            f"Available states: {df['state'].tolist()}"
        )
    return matched.iloc[0]


def mayer_value(
    df: pd.DataFrame, state: str, channel_key: str, method: str
) -> float:
    column = column_for(channel_key, method)
    if column not in df.columns:
        raise KeyError(f"Missing Mayer column: {column}")
    value = float(row_for(df, state)[column])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value for state={state}, column={column}")
    return value


def validate_mayer_input(
    df: pd.DataFrame, variant: str, sites: Sequence[str]
) -> None:
    missing_states = [state for state in sites if state.lower() not in set(df["state"])]
    if missing_states:
        raise KeyError(
            "Missing adsorption-site rows required for panel (d): "
            f"{missing_states}. Available states: {df['state'].tolist()}. "
            "Use --sites to change the requested site list if needed."
        )
    if "top" not in set(df["state"]):
        raise KeyError("A 'top' row is required for panel (c).")

    channels = VARIANTS[variant]["channels"]
    required_columns = []
    for key in channels.values():
        required_columns.extend(
            [column_for(str(key), "emb-DFT"), column_for(str(key), "emb-QC")]
        )
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "Missing Mayer columns for variant "
            f"'{variant}':\n  " + "\n  ".join(missing_columns)
        )


def fmt_2(value: float, _position=None) -> str:
    return "" if not np.isfinite(value) else f"{value:.2f}"


def fmt_3(value: float, _position=None) -> str:
    return "" if not np.isfinite(value) else f"{value:.3f}"


def style_open_axes(ax: plt.Axes, *, grid_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["spine"])
        ax.spines[side].set_linewidth(1.05)
    ax.tick_params(axis="both", colors=COLORS["tick"], pad=2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_color(COLORS["tick"])
    if grid_y:
        ax.grid(True, axis="y", alpha=0.11, linewidth=0.55, color=COLORS["grid"])
        ax.grid(False, axis="x")
    else:
        ax.grid(False)


PANEL_LABEL_X = -0.105  # relative to the subplot axes
PANEL_LABEL_Y = 1.025   # label bottom sits above the top axis

def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["text"],
        clip_on=False,
    )


def bold_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight(TEXT_WEIGHT)
        text.set_color(COLORS["text"])


# =============================================================================
# Panel (a): adsorption energy
# =============================================================================

def plot_adsorption_energy(ax: plt.Axes) -> None:
    x = np.arange(len(METHODS), dtype=float)

    ax.axhspan(EXP_LOW, EXP_HIGH, color="#DCEFE9", alpha=0.74, linewidth=0, zorder=0)
    ax.axhline(
        EXP_CENTER, color="#9AB7AD", linewidth=1.05, linestyle="--", zorder=1
    )

    for i, (method, energy) in enumerate(zip(METHODS, ADSORPTION_ENERGY)):
        is_qc = method == "QC-DFET"
        # Follow the Cu(111) CO figure: 76-pt diamonds for other methods
        # and a 76-pt filled hexagon for QC-DFET. The QC symbol retains
        # the original thin dark outline; white marker borders are removed.
        # All six points are single values (no reference-space error bars).
        ax.scatter(
            i,
            energy,
            s=76,
            marker="h" if is_qc else "D",
            facecolor=METHOD_COLORS[method],
            edgecolor="#2F2F2F" if is_qc else "none",
            linewidth=0.85 if is_qc else 0.0,
            alpha=0.98,
            zorder=5 if is_qc else 3,
        )

    #ax.text(
     #   len(METHODS) - 0.44,
     #   EXP_HIGH + 0.014,
     #   "Exp. TDS + chem. acc.",
     #   ha="right",
     #   va="bottom",
     #   fontsize=9.8,
      #  fontweight=TEXT_WEIGHT,
       # color="#5F7F76",
    #)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(-0.55, len(METHODS) - 0.45)
    ax.set_ylim(-1.085, -0.345)
    ax.set_yticks([-1.0, -0.8, -0.6, -0.4])
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    for tick, method in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(METHOD_COLORS[method])
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(7.7)
        tick.set_rotation(0)
        tick.set_ha("center")
    style_open_axes(ax)
    add_panel_label(ax, "(a)")


# =============================================================================
# Panel (b): site stability
# =============================================================================

def plot_site_stability(ax: plt.Axes) -> None:
    x = np.arange(len(METHODS), dtype=float)
    width = 0.27

    ax.bar(
        x - width / 2,
        BRIDGE_TOP,
        width=width,
        label="bridge–top",
        color=SITE_COLORS["bridge-top"],
        edgecolor="none",  # no white outline between adjacent bars
        linewidth=0.0,
        alpha=0.96,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        HOLLOW_TOP,
        width=width,
        label="hollow–top",
        color=SITE_COLORS["hollow-top"],
        edgecolor="none",  # same solid-bar styling as the Cu(111) reference
        linewidth=0.0,
        alpha=0.96,
        zorder=3,
    )
    ax.axhline(0.0, color=COLORS["spine"], linewidth=1.10, zorder=1)
    ax.set_ylabel("Relative energy (eV)")
    ax.set_xlim(-0.55, len(METHODS) - 0.45)
    ax.set_ylim(-0.20, 0.90)
    ax.set_yticks(np.arange(-0.2, 0.81, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    for tick, method in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(METHOD_COLORS[method])
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(7.7)
        tick.set_rotation(0)
        tick.set_ha("center")
    style_open_axes(ax, grid_y=True)
    legend = ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.45,
        handletextpad=0.55,
        borderaxespad=0.30,
        columnspacing=0.95,
    )
    bold_legend(legend)
    add_panel_label(ax, "(b)")


# =============================================================================
# Panel (c): top-site emb-DFT -> emb-QC correction
# =============================================================================

def plot_top_dft_to_qc(ax: plt.Axes, df: pd.DataFrame, variant: str) -> None:
    channels = VARIANTS[variant]["channels"]
    names = ("co_sigma", "co_pi", "cu_sigma", "cu_pi")
    labels = [CHANNEL_LABELS[name] for name in names]
    keys = [str(channels[name]) for name in names]

    dft = np.array(
        [mayer_value(df, "top", key, "emb-DFT") for key in keys], dtype=float
    )
    qc = np.array(
        [mayer_value(df, "top", key, "emb-QC") for key in keys], dtype=float
    )
    y = np.arange(len(names), dtype=float)

    for yi in y:
        ax.axhline(yi, color="#EFF2F5", lw=0.9, zorder=0)

    for yi, name, dft_value, qc_value in zip(y, names, dft, qc):
        if name.startswith("cu_"):
            direction = 1.0 if qc_value >= dft_value else -1.0
            gap = abs(qc_value - dft_value)
            shorten = min(0.025, 0.12 * gap)
            start = dft_value + direction * shorten
            end = qc_value - direction * shorten
            if (end - start) * direction > 0:
                ax.add_patch(
                    FancyArrowPatch(
                        (start, yi),
                        (end, yi),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        lw=1.75,
                        color=COLORS["arrow"],
                        zorder=2,
                    )
                )
            else:
                ax.plot(
                    [dft_value, qc_value],
                    [yi, yi],
                    color=COLORS["connector"],
                    lw=1.55,
                    zorder=2,
                )
        else:
            ax.plot(
                [dft_value, qc_value],
                [yi, yi],
                color=COLORS["connector"],
                lw=1.55,
                zorder=2,
            )

        ax.scatter(
            dft_value,
            yi,
            s=78,
            marker="D",
            color=COLORS["emb_dft"],
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
            label="emb-DFT" if yi == 0 else None,
        )
        ax.scatter(
            qc_value,
            yi,
            s=90,
            marker="h",
            color=COLORS["emb_qc"],
            edgecolor="#2F2F2F",
            linewidth=0.80,
            zorder=4,
            label="emb-QC" if yi == 0 else None,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mayer bond index, M")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_3))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    all_values = np.concatenate([dft, qc])
    xmin = float(np.min(all_values))
    xmax = float(np.max(all_values))
    value_range = max(xmax - xmin, 1e-8)
    ax.set_xlim(max(0.0, xmin - 0.18 * value_range), xmax + 0.20 * value_range)
    style_open_axes(ax)
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        handlelength=1.2,
        borderaxespad=0.0,
    )
    bold_legend(legend)
    add_panel_label(ax, "(c)")


# =============================================================================
# Panel (d): site-resolved emb-QC C--Cu coupling
# =============================================================================

def plot_site_qc_ccu(
    ax: plt.Axes, df: pd.DataFrame, variant: str, sites: Sequence[str]
) -> np.ndarray:
    channels = VARIANTS[variant]["channels"]
    sigma_key = str(channels["cu_sigma"])
    pi_key = str(channels["cu_pi"])

    sigma = np.array(
        [mayer_value(df, site, sigma_key, "emb-QC") for site in sites], dtype=float
    )
    pi = np.array(
        [mayer_value(df, site, pi_key, "emb-QC") for site in sites], dtype=float
    )
    total = sigma + pi
    matrix = np.vstack([total, sigma, pi])

    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmin = 0.0
        vmax = float(np.max(finite))
        if vmax <= 0.0:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=HEAT_CMAP,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    row_labels = ("total C--Cu", "C--Cu σ", "C--Cu π")
    ax.set_xticks(np.arange(len(sites)))
    ax.set_xticklabels(
        list(sites),
        fontsize=TICK_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["tick"],
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(
        row_labels,
        fontsize=TICK_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["tick"],
    )
    ax.set_xlabel("Adsorption site")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                label = "n/a"
                text_color = "#555555"
            else:
                label = f"{value:.3f}"
                normalized = (value - vmin) / max(vmax - vmin, 1e-12)
                text_color = "white" if normalized > 0.60 else COLORS["text"]
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=PANEL_D_CELL_SIZE,
                fontweight=TEXT_WEIGHT,
                color=text_color,
                zorder=3,
            )

    ax.set_xticks(np.arange(-0.5, len(sites), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_color(COLORS["tick"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.030)
    colorbar.set_label(
        "M emb-QC",
        fontsize=PANEL_D_CBAR_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        labelpad=8,
    )
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt_3))
    colorbar.ax.tick_params(
        labelsize=PANEL_D_CBAR_TICK_SIZE,
        width=0.8,
        length=3.0,
    )
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
    colorbar.outline.set_linewidth(0.50)
    colorbar.outline.set_edgecolor("#8A94A6")

    add_panel_label(ax, "(d)")
    return matrix


# =============================================================================
# Output table and figure
# =============================================================================

def export_selected_values(
    df: pd.DataFrame,
    variant: str,
    sites: Sequence[str],
    outdir: Path,
) -> Path:
    channels = VARIANTS[variant]["channels"]
    rows = []

    for name in ("co_sigma", "co_pi", "cu_sigma", "cu_pi"):
        key = str(channels[name])
        for method in ("emb-DFT", "emb-QC"):
            rows.append(
                {
                    "panel": "c_top_DFT_to_QC",
                    "state": "top",
                    "quantity": CHANNEL_LABELS[name],
                    "definition": key,
                    "method": method,
                    "mayer_bond_index": mayer_value(df, "top", key, method),
                }
            )

    sigma_key = str(channels["cu_sigma"])
    pi_key = str(channels["cu_pi"])
    for site in sites:
        sigma = mayer_value(df, site, sigma_key, "emb-QC")
        pi = mayer_value(df, site, pi_key, "emb-QC")
        definitions: Iterable[tuple[str, str, float]] = (
            ("total C--Cu", f"{sigma_key} + {pi_key}", sigma + pi),
            ("C--Cu σ", sigma_key, sigma),
            ("C--Cu π", pi_key, pi),
        )
        for quantity, definition, value in definitions:
            rows.append(
                {
                    "panel": "d_site_emb_QC",
                    "state": site,
                    "quantity": quantity,
                    "definition": definition,
                    "method": "emb-QC (40q)",
                    "mayer_bond_index": value,
                }
            )

    output = outdir / f"co_cu100_cas22e20o_selected_values_{variant}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def create_figure(
    df: pd.DataFrame,
    variant: str,
    sites: Sequence[str],
) -> plt.Figure:
    fig = plt.figure(figsize=(13.6, 8.75), dpi=300)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.00, 1.06),
        height_ratios=(1.00, 1.03),
        wspace=0.34,
        hspace=0.45,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_adsorption_energy(ax_a)
    plot_site_stability(ax_b)
    plot_top_dft_to_qc(ax_c, df, variant)
    plot_site_qc_ccu(ax_d, df, variant, sites)

    fig.subplots_adjust(left=0.080, right=0.975, bottom=0.095, top=0.945)
    return fig


def save_figure(fig: plt.Figure, variant: str, outdir: Path) -> Path:
    """Save a high-resolution PNG only (no PDF output)."""
    stem = f"co_cu100_cas22e20o_four_panels_{variant}"
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the four-panel CO/Cu(100) energetics and Mayer-bonding figure."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Mayer summary CSV; default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR, help=f"Output directory; default: {DEFAULT_OUTDIR}"
    )
    parser.add_argument(
        "--variant",
        default="p_allcu",
        choices=tuple(VARIANTS),
        help="Mayer projector definition; default: p_allcu",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        help="Ordered adsorption sites for panel (d); default: top bridge hollow",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    sites = tuple(site.strip().lower() for site in args.sites)

    dataframe, input_path = load_mayer_input(args.input)
    validate_mayer_input(dataframe, args.variant, sites)

    figure = create_figure(dataframe, args.variant, sites)
    png = save_figure(figure, args.variant, outdir)
    values = export_selected_values(dataframe, args.variant, sites, outdir)

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Variant: {args.variant} ({VARIANTS[args.variant]['description']})")
    print(f"[INFO] Sites: {', '.join(sites)}")
    print(f"[INFO] PNG: {png}")
    print(f"[INFO] Selected values: {values}")


if __name__ == "__main__":
    main()
