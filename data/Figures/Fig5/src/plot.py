#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process CO/Cu Mayer bond-order results into one mechanism figure.

Input from:
    wavefunction_analysis_frontier_full_p_2p_contact_allcu.py

Default input:
    analysis_outputs/frontier_fig3d_mayer_bond_order_summary.csv

Main output:
    ./co_mayer_bonding_mechanism_v19_bold_p_allcu.png

Panels:
    A. CO--Cu molecular-orbital interaction scheme
    B. abv CO internal C--O bonding: emb-DFT vs emb-QC
    C. top-site bonding channels: emb-DFT vs emb-QC
    D. site-dependent emb-QC Mayer indices for adsorption sites

No PDF or CSV output.
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.text import Text


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_INPUT = "data/frontier_fig3d_mayer_bond_order_summary.csv"
DEFAULT_OUTDIR = "output"

STATE_ORDER = ["abv", "top", "bridge", "fcc", "hcp"]
ADS_SITE_ORDER = ["top", "bridge", "fcc", "hcp"]

# The p_allcu definition is the most robust default for the current Cu14 cluster:
# all p-like C/O shells and all Cu14 atoms for C--Cu terms.
VARIANTS = {
    "p_allcu": {
        "title": "all-p / all Cu$_{14}$",
        "channels": {
            "co_sigma": "p_sigma_Cpz_Opz",
            "co_pi": "p_pi_Cpxy_Opxy",
            "cu_sigma": "p_sigma_Cpz_Cu_all",
            "cu_pi": "p_pi_Cpxy_Cu3d_all",
        },
        "labels": {
            "co_sigma": r"$\sigma$(C$_{p_z}$--O$_{p_z}$)",
            "co_pi": r"$\pi$(C$_{p_{xy}}$--O$_{p_{xy}}$)",
            "cu_sigma": r"$\sigma$(C$_{p_z}$--Cu)",
            "cu_pi": r"$\pi$(C$_{p_{xy}}$--Cu d)",
        },
        "short_labels": {
            "co_sigma": "C--O σ",
            "co_pi": "C--O π",
            "cu_sigma": "C--Cu σ",
            "cu_pi": "C--Cu π",
        },
    },
    "p2_allcu": {
        "title": "2p / all Cu$_{14}$",
        "channels": {
            "co_sigma": "p2_sigma_C2pz_O2pz",
            "co_pi": "p2_pi_C2pxy_O2pxy",
            "cu_sigma": "p2_sigma_C2pz_Cu_all",
            "cu_pi": "p2_pi_C2pxy_Cu3d_all",
        },
        "labels": {
            "co_sigma": r"$\sigma$(C$_{2p_z}$--O$_{2p_z}$)",
            "co_pi": r"$\pi$(C$_{2p_{xy}}$--O$_{2p_{xy}}$)",
            "cu_sigma": r"$\sigma$(C$_{2p_z}$--Cu)",
            "cu_pi": r"$\pi$(C$_{2p_{xy}}$--Cu d)",
        },
        "short_labels": {
            "co_sigma": "C--O σ",
            "co_pi": "C--O π",
            "cu_sigma": "C--Cu σ",
            "cu_pi": "C--Cu π",
        },
    },
}

# Soft palette aligned with the previous H2/CO figures.
COLORS = {
    "emb_dft": "#B5E5FA",     # soft blue
    "emb_qc": "#F8CBC0",      # QC-DFET-like mint
    "emb_qc_alt": "#E8B5AA",  # soft rose, used only for contrast when needed
    # Orbital-channel schematic colors:
    # sigma channels use one rose/pink family; pi channels use one blue family.
    "co_sigma": "#E58AAE",
    "co_pi": "#78BDEB",
    "cu_sigma": "#E58AAE",
    "cu_pi": "#78BDEB",
    "text": "#222222",
    "tick": "#333333",
    "spine": "#333333",
    "zero": "#9DA7B1",
    "light": "#EEF3F5",
    "panel_bg": "#FAFBFC",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "mayer_soft_peach",
    ["#FCFAF9", "#F8EEE9", "#F5E2DC", "#F8D8D0", "#F6D0C6"],
)

# Panel A orbital-scheme transform.
# MO_SCHEME_SCALE controls the size of the whole schematic.
# MO_SCHEME_SHIFT_X / MO_SCHEME_SHIFT_Y move the whole schematic in data coordinates.
# Positive X moves right; positive Y moves up.
MO_SCHEME_SCALE = 1.20
MO_SCHEME_CENTER = (0.500, 0.505)
MO_SCHEME_SHIFT_X = 0.015
MO_SCHEME_SHIFT_Y = 0.000

# Panel C arrow tuning for C--Cu sigma/pi rows.
# To make arrows longer: decrease TOP_ARROW_SHRINK_FRAC and/or TOP_ARROW_SHRINK_MAX.
# To make arrows shorter: increase TOP_ARROW_SHRINK_FRAC and/or TOP_ARROW_SHRINK_MAX.
TOP_ARROW_SHRINK_FRAC = 0.12
TOP_ARROW_SHRINK_MAX = 0.025
# Panel C DFT-to-QC correction arrows/connectors.
# Darkened and slightly thickened so the correction direction remains clear
# after all labels/ticks have been bolded.
TOP_CONNECTOR_COLOR = "#7F8B96"
TOP_ARROW_COLOR = "#707C88"
TOP_ARROW_MUTATION_SCALE = 14
TOP_ARROW_LW = 1.70
TOP_CONNECTOR_LW = 1.55

# Unified typography controls.  These are aligned with the recent H2/Cu12
# and CO adsorption figures: larger, bold labels/ticks/titles for readability.
TEXT_WEIGHT = "bold"
BASE_FONT_SIZE = 12.0
AXIS_LABEL_SIZE = 13.8
AXIS_TITLE_SIZE = 14.2
TICK_LABEL_SIZE = 12.0
LEGEND_FONT_SIZE = 11.8
PANEL_LABEL_SIZE = 16.2
SUPTITLE_SIZE = 16.2
ANNOTATION_SIZE = 11.2
VALUE_LABEL_SIZE = 11.6
CELL_VALUE_SIZE = 11.2

# Heatmap cell-value text color for Panel D.
# The low end of HEAT_CMAP is no longer white, so all values can stay white.
HEATMAP_VALUE_TEXT_COLOR = None

CBAR_LABEL_SIZE = 12.4
CBAR_TICK_SIZE = 10.8
ATOM_LABEL_SIZE = 10.8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": BASE_FONT_SIZE,
    "font.weight": "normal",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": AXIS_TITLE_SIZE,
    "axes.labelweight": TEXT_WEIGHT,
    "axes.titleweight": TEXT_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": 0.90,
    "xtick.major.width": 0.90,
    "ytick.major.width": 0.90,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_3(x, _pos=None):
    if not np.isfinite(x):
        return ""
    return f"{x:.3f}"


def _bold_ticklabels(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_color(COLORS["tick"])


def _bold_legend(leg):
    if leg is None:
        return
    for txt in leg.get_texts():
        txt.set_fontweight(TEXT_WEIGHT)
        txt.set_color(COLORS["text"])


def _bold_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, length=2.5, pad=1.2)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_color(COLORS["text"])
    # Important: colorbar labels do not always inherit rcParams / set_label
    # font properties, especially when mathtext is used.  Set them explicitly.
    cbar.ax.yaxis.label.set_size(CBAR_LABEL_SIZE)
    cbar.ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    cbar.ax.yaxis.label.set_color(COLORS["text"])

def _force_bold_all_text(fig):
    """Final safety pass: make every Text artist in the figure bold.

    This catches tick labels, colorbar labels, legend labels, panel labels,
    annotation text, and any text added by Matplotlib after axis formatting.
    For Greek channel labels we use unicode text instead of mathtext elsewhere,
    so fontweight is applied reliably.
    """
    for txt in fig.findobj(match=Text):
        txt.set_fontweight(TEXT_WEIGHT)


def style_axes(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COLORS["spine"])
        ax.spines[side].set_linewidth(0.90)
    ax.tick_params(axis="both", colors=COLORS["tick"], pad=2)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.title.set_color(COLORS["text"])
    ax.title.set_fontweight(TEXT_WEIGHT)
    _bold_ticklabels(ax)
    if grid:
        ax.grid(True, axis="y", alpha=0.11, linewidth=0.48, color="#8D99A6")
        ax.grid(False, axis="x")
    else:
        ax.grid(False)


def add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def safe_numeric(df):
    out = df.copy()
    for col in out.columns:
        if col != "state":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def standardize_state_order(df):
    out = df.copy()
    out["state"] = out["state"].astype(str).str.strip().str.lower()
    present = [s for s in STATE_ORDER if s in set(out["state"])]
    other = [s for s in out["state"].tolist() if s not in present]
    categories = present + other
    out["_state_order"] = pd.Categorical(out["state"], categories=categories, ordered=True)
    out = out.sort_values("_state_order").drop(columns=["_state_order"])
    return out.reset_index(drop=True)


def get_row(df, state):
    d = df[df["state"].astype(str).str.lower() == state.lower()]
    if d.empty:
        raise KeyError(f"State '{state}' not found in input CSV. Available: {df['state'].tolist()}")
    return d.iloc[0]


def col_abs(channel_key, method):
    if method == "emb-DFT":
        return f"{channel_key}_embRKS"
    if method == "emb-QC":
        return f"{channel_key}_CASSCF"
    raise ValueError(method)


def value(df, state, channel_key, method):
    row = get_row(df, state)
    col = col_abs(channel_key, method)
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    return float(row[col])


def values_for_sites(df, sites, channel_key, method):
    vals = []
    for st in sites:
        try:
            vals.append(value(df, st, channel_key, method))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def check_columns(df, variant):
    cfg = VARIANTS[variant]
    missing = []
    for key in cfg["channels"].values():
        for method in ["emb-DFT", "emb-QC"]:
            col = col_abs(key, method)
            if col not in df.columns:
                missing.append(col)
    if missing:
        print("[ERROR] Missing required columns:")
        for col in missing:
            print(f"  {col}")
        print("\nAvailable columns:")
        for col in df.columns:
            print(f"  {col}")
        raise KeyError("Required Mayer columns are missing.")
    return True


def autoscale_ylim(ax, vals, pad_frac=0.18, include_zero=True):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if include_zero:
        arr = np.concatenate([arr, [0.0]])
    if arr.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    ymin, ymax = float(np.min(arr)), float(np.max(arr))
    yr = max(ymax - ymin, 1e-8)
    ax.set_ylim(ymin - pad_frac * yr, ymax + pad_frac * yr)


# =============================================================================
# Panel A: horizontal unified orbital-shape MO scheme
# =============================================================================

def add_lobe(ax, xy, width, height, angle, color, alpha=0.58, ec="white", lw=0.65, zorder=2):
    """Draw one schematic orbital lobe as a soft ellipse."""
    e = Ellipse(
        xy=xy,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(e)
    return e


def add_teardrop_lobe(ax, xy, length, width, angle, color, alpha=0.58, ec="white", lw=0.65, zorder=2):
    """Draw a tapered orbital lobe, closer to textbook p/d orbital cartoons."""
    verts = [
        (0.0, 0.0),
        (-0.38, 0.16), (-0.44, 0.74), (0.0, 1.0),
        (0.44, 0.74), (0.38, 0.16), (0.0, 0.0),
        (0.0, 0.0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    path = MplPath(verts, codes)
    trans = (Affine2D()
             .scale(width, length)
             .rotate_deg(angle)
             .translate(*xy))
    patch = PathPatch(
        path,
        facecolor=color,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        transform=trans + ax.transData,
        zorder=zorder,
        joinstyle='round',
        capstyle='round',
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def add_p_orbital(ax, center, length, width, color, alpha=0.50, orientation="vertical", zorder=2):
    """Draw a two-lobe p orbital centered on an atom."""
    if orientation == "vertical":
        angles = (0, 180)
    elif orientation == "horizontal":
        angles = (90, 270)
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    for ang in angles:
        add_teardrop_lobe(
            ax, center, length, width, ang, color,
            alpha=alpha, ec="white", lw=0.60, zorder=zorder
        )


def add_d_orbital_four_lobes(ax, center, length, width, color, alpha=0.50, rotation=45, zorder=2):
    """Draw a four-lobe d-like orbital centered on an atom."""
    for ang in [rotation, rotation + 90, rotation + 180, rotation + 270]:
        add_teardrop_lobe(
            ax, center, length, width, ang, color,
            alpha=alpha, ec="white", lw=0.60, zorder=zorder
        )


def add_atom(ax, xy, symbol, radius, color, ec="#4A4F55", zorder=7):
    circ = Circle(
        xy,
        radius=radius,
        facecolor=color,
        edgecolor=ec,
        linewidth=0.75,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(circ)
    ax.text(
        xy[0], xy[1], symbol,
        ha="center", va="center",
        fontsize=ATOM_LABEL_SIZE,
        color="#222222",
        fontweight=TEXT_WEIGHT,
        zorder=zorder + 1,
        clip_on=False,
    )


def curved_arrow(ax, xy1, xy2, color, text, text_xy, rad=0.10, lw=1.35):
    arr = FancyArrowPatch(
        xy1,
        xy2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        alpha=0.95,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(arr)
    ax.text(
        text_xy[0], text_xy[1], text,
        ha="center", va="center",
        fontsize=ANNOTATION_SIZE,
        color=color,
        fontweight=TEXT_WEIGHT,
        zorder=11,
        clip_on=False,
    )


def plot_mo_scheme(ax):
    """
    Qualitative horizontal CO--Cu orbital-channel schematic.

    Same v13 layout, but the whole Panel-A schematic is enlarged by a scale
    factor of 1.20 around the visual center.  The title itself is not scaled.
    """
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # ------------------------------------------------------------------
    # Panel-A schematic transform.
    # Adjust MO_SCHEME_SHIFT_X / MO_SCHEME_SHIFT_Y near the top of this file
    # to move the whole orbital cartoon without changing the title position.
    # ------------------------------------------------------------------
    def sc_xy(x, y):
        cx, cy = MO_SCHEME_CENTER
        return (cx + MO_SCHEME_SCALE * (x - cx) + MO_SCHEME_SHIFT_X,
                cy + MO_SCHEME_SCALE * (y - cy) + MO_SCHEME_SHIFT_Y)

    def sc_len(v):
        return MO_SCHEME_SCALE * v

    # Horizontal O--C···Cu geometry.
    O = sc_xy(0.175, 0.505)
    C = sc_xy(0.340, 0.505)
    CU = sc_xy(0.740, 0.505)

    sigma_color = COLORS["co_sigma"]
    pi_color = COLORS["co_pi"]

    # ------------------------------------------------------------------
    # Sigma channels: axial lobes along O--C--Cu.
    # ------------------------------------------------------------------
    add_lobe(ax, sc_xy(0.425, 0.505), sc_len(0.230), sc_len(0.072), 0,
             sigma_color, alpha=0.56, zorder=2)
    add_lobe(ax, sc_xy(0.085, 0.505), sc_len(0.115), sc_len(0.050), 0,
             sigma_color, alpha=0.36, zorder=2)

    # Cu sigma-symmetry acceptor/coupling manifold.
    add_lobe(ax, sc_xy(0.615, 0.505), sc_len(0.205), sc_len(0.078), 0,
             sigma_color, alpha=0.42, zorder=2)

    # ------------------------------------------------------------------
    # Pi channels: C/O p lobes plus full Cu 3d four-lobe motif.
    # ------------------------------------------------------------------
    add_lobe(ax, sc_xy(0.175, 0.670), sc_len(0.062), sc_len(0.155), 0,
             pi_color, alpha=0.44, zorder=2)
    add_lobe(ax, sc_xy(0.175, 0.340), sc_len(0.062), sc_len(0.155), 0,
             pi_color, alpha=0.44, zorder=2)
    add_lobe(ax, sc_xy(0.340, 0.670), sc_len(0.068), sc_len(0.165), 0,
             pi_color, alpha=0.52, zorder=2)
    add_lobe(ax, sc_xy(0.340, 0.340), sc_len(0.068), sc_len(0.165), 0,
             pi_color, alpha=0.52, zorder=2)

    # Cu d-type donor manifold: schematic four-lobe motif centered around Cu.
    d_w, d_h = sc_len(0.076), sc_len(0.165)
    add_lobe(ax, sc_xy(0.740 - 0.078, 0.635), d_w, d_h,  38,
             pi_color, alpha=0.46, zorder=2)
    add_lobe(ax, sc_xy(0.740 + 0.078, 0.635), d_w, d_h, -38,
             pi_color, alpha=0.46, zorder=2)
    add_lobe(ax, sc_xy(0.740 - 0.078, 0.375), d_w, d_h, -38,
             pi_color, alpha=0.46, zorder=2)
    add_lobe(ax, sc_xy(0.740 + 0.078, 0.375), d_w, d_h,  38,
             pi_color, alpha=0.46, zorder=2)

    # Molecular axis and atoms.
    ax.plot([O[0] + sc_len(0.032), C[0] - sc_len(0.032)], [O[1], C[1]],
            color="#555E67", lw=1.10, zorder=7)
    ax.plot([C[0] + sc_len(0.038), CU[0] - sc_len(0.044)], [C[1], CU[1]],
            color="#9FA8B2", lw=1.00, ls=(0, (3.0, 2.3)), zorder=6)

    add_atom(ax, O, "O", sc_len(0.032), "#EAF0F7", ec="#AAB3BD", zorder=8)
    add_atom(ax, C, "C", sc_len(0.032), "#F6EFE7", ec="#AAB3BD", zorder=8)
    add_atom(ax, CU, "Cu", sc_len(0.036), "#DCEFE9", ec="#8CBFAF", zorder=8)

    # Interaction arrows show electron-donation directions in the
    # Blyholder-type picture:
    #   sigma donation: occupied CO 5sigma -> Cu sigma-symmetry manifold;
    #   pi back-donation: occupied Cu d-type states -> unoccupied CO 2pi*.
    # The Cu sigma channel is intentionally labelled as a symmetry manifold,
    # rather than as one specific atomic orbital, because the Mayer analysis
    # uses C p_z against all Cu AOs for this channel.
    # The pi channel is labelled "Cu d" (plain d, not d*): the numerical
    # projector contains every d-type AO on all selected Cu atoms and does not
    # perform a local d_pi/d_sigma decomposition.
    curved_arrow(
        ax,
        sc_xy(0.340 + 0.080, 0.420),
        sc_xy(0.740 - 0.120, 0.420),
        sigma_color,
        "CO 5σ → Cu",
        sc_xy(0.535, 0.355),
        rad=0.00,
        lw=1.35,
    )
    curved_arrow(
        ax,
        sc_xy(0.740 - 0.150, 0.700),
        sc_xy(0.340 + 0.070, 0.700),
        pi_color,
        "CO 2π* ← Cu d",
        sc_xy(0.505, 0.780),
        rad=-0.18,
        lw=1.35,
    )

    # Two-family legend, also scaled with the schematic.
    legend_items = [
        (0.170, 0.145, sigma_color, "σ channels"),
        (0.535, 0.145, pi_color, "π channels"),
    ]
    for x, y, col, lab in legend_items:
        sx, sy = sc_xy(x, y)
        add_lobe(ax, (sx, sy), sc_len(0.052), sc_len(0.027), 0,
                 col, alpha=0.70, zorder=4)
        ax.text(sx + sc_len(0.052), sy, lab,
                ha="left", va="center",
                fontsize=ANNOTATION_SIZE,
                fontweight=TEXT_WEIGHT,
                color="#4A4F55")



# =============================================================================
# Panel B: abv CO internal bonding
# =============================================================================

def plot_abv_internal(ax, df, variant):
    cfg = VARIANTS[variant]
    ch = cfg["channels"]
    labels = [cfg["short_labels"]["co_sigma"], cfg["short_labels"]["co_pi"]]
    keys = [ch["co_sigma"], ch["co_pi"]]

    y_dft = np.array([value(df, "abv", k, "emb-DFT") for k in keys], dtype=float)
    y_qc = np.array([value(df, "abv", k, "emb-QC") for k in keys], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.30

    ax.bar(x - width/2, y_dft, width=width, color=COLORS["emb_dft"],
           edgecolor="white", linewidth=0.65, label="emb-DFT", zorder=3)
    ax.bar(x + width/2, y_qc, width=width, color=COLORS["emb_qc"],
           edgecolor="white", linewidth=0.65, label="emb-QC", zorder=3)

    ymax_ref = np.nanmax(np.concatenate([y_dft, y_qc]))
    yoff = max(0.012, 0.030 * ymax_ref)
    for xi, v in zip(x - width/2, y_dft):
        if np.isfinite(v):
            ax.text(xi, v + yoff, f"{v:.3f}", ha="center", va="bottom", fontsize=VALUE_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["text"])
    for xi, v in zip(x + width/2, y_qc):
        if np.isfinite(v):
            ax.text(xi, v + yoff, f"{v:.3f}", ha="center", va="bottom", fontsize=VALUE_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["tick"])
    ax.set_ylabel("Mayer bond index M", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    vals = np.concatenate([y_dft, y_qc])
    vals = vals[np.isfinite(vals)]
    ymax = float(np.max(vals)) if vals.size else 1.0
    ax.set_ylim(0.0, ymax * 1.22)
    ax.set_xlim(-0.45, len(labels) - 0.55)

    style_axes(ax, grid=True)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_zorder(5)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", pad=3)
    leg = ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    _bold_legend(leg)


# =============================================================================
# Panel C: top-site emb-DFT vs emb-QC
# =============================================================================

def plot_top_dumbbell(ax, df, variant):
    cfg = VARIANTS[variant]
    ch = cfg["channels"]

    keys = [ch["co_sigma"], ch["co_pi"], ch["cu_sigma"], ch["cu_pi"]]
    labels = [
        cfg["short_labels"]["co_sigma"],
        cfg["short_labels"]["co_pi"],
        cfg["short_labels"]["cu_sigma"],
        cfg["short_labels"]["cu_pi"],
    ]

    y = np.arange(len(keys), dtype=float)
    x_dft = np.array([value(df, "top", k, "emb-DFT") for k in keys], dtype=float)
    x_qc = np.array([value(df, "top", k, "emb-QC") for k in keys], dtype=float)

    for yi in y:
        ax.axhline(yi, color="#EFF2F5", lw=0.9, zorder=0)

    cu_keys = {ch["cu_sigma"], ch["cu_pi"]}
    for yi, key, a, b in zip(y, keys, x_dft, x_qc):
        if np.isfinite(a) and np.isfinite(b):
            if key in cu_keys:
                # Parameterized arrow: tune TOP_ARROW_SHRINK_FRAC / TOP_ARROW_SHRINK_MAX
                # near the top of this script to adjust length.
                direction = 1.0 if b >= a else -1.0
                gap = abs(b - a)
                shorten = min(TOP_ARROW_SHRINK_MAX, TOP_ARROW_SHRINK_FRAC * gap)
                x_start = a + direction * shorten
                x_end = b - direction * shorten
                if (x_end - x_start) * direction > 0:
                    arr = FancyArrowPatch(
                        (x_start, yi), (x_end, yi),
                        arrowstyle='-|>',
                        mutation_scale=TOP_ARROW_MUTATION_SCALE,
                        lw=TOP_ARROW_LW,
                        color=TOP_ARROW_COLOR,
                        zorder=2,
                    )
                    ax.add_patch(arr)
                else:
                    ax.plot([a, b], [yi, yi], color=TOP_CONNECTOR_COLOR, lw=TOP_CONNECTOR_LW, zorder=2)
            else:
                ax.plot([a, b], [yi, yi], color=TOP_CONNECTOR_COLOR, lw=TOP_CONNECTOR_LW, zorder=2)
        if np.isfinite(a):
            ax.scatter(a, yi, s=72, marker="D", color=COLORS["emb_dft"],
                       edgecolor="white", linewidth=0.7, zorder=3, label="emb-DFT" if yi == 0 else None)
        if np.isfinite(b):
            ax.scatter(b, yi, s=82, marker="h", color=COLORS["emb_qc"],
                       edgecolor="#2F2F2F", linewidth=0.75, zorder=4, label="emb-QC" if yi == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["tick"])
    ax.invert_yaxis()
    ax.set_xlabel("Mayer bond index M", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_3))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    vals = np.concatenate([x_dft, x_qc])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        xmin, xmax = float(np.min(vals)), float(np.max(vals))
        xr = max(xmax - xmin, 1e-8)
        ax.set_xlim(max(0.0, xmin - 0.18*xr), xmax + 0.20*xr)

    style_axes(ax, grid=False)
    leg = ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    _bold_legend(leg)


# =============================================================================
# Panel D: emb-QC site comparison
# =============================================================================

def plot_site_ccu_coupling(ax, df, variant):
    """
    Heatmap of site-dependent C--Cu coupling at emb-QC level.

    Rows:
      total C--Cu = M(C--Cu sigma) + M(C--Cu pi)
      C--Cu sigma
      C--Cu pi
    """
    cfg = VARIANTS[variant]
    ch = cfg["channels"]

    ccu_sigma = values_for_sites(df, ADS_SITE_ORDER, ch["cu_sigma"], "emb-QC")
    ccu_pi = values_for_sites(df, ADS_SITE_ORDER, ch["cu_pi"], "emb-QC")
    ccu_total = ccu_sigma + ccu_pi

    mat = np.vstack([ccu_total, ccu_sigma, ccu_pi])
    row_labels = [
        r"total C--Cu",
        "C--Cu σ",
        "C--Cu π",
    ]

    finite = mat[np.isfinite(mat)]
    if finite.size:
        vmin, vmax = 0.0, float(np.max(finite))
        if vmax <= 0:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(mat, aspect="auto", cmap=HEAT_CMAP, vmin=vmin, vmax=vmax, zorder=1)

    ax.set_xticks(np.arange(len(ADS_SITE_ORDER)))
    ax.set_xticklabels(ADS_SITE_ORDER, fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["tick"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["tick"])
    ax.set_xlabel("Adsorption site", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    _bold_ticklabels(ax)

    # Cell values.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "n/a"
                color = "#555555"
            else:
                txt = f"{v:.3f}"
                norm = (v - vmin) / max(vmax - vmin, 1e-12)
                color = "white" if norm > 0.60 else COLORS["text"]
            ax.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=CELL_VALUE_SIZE,
                fontweight=TEXT_WEIGHT,
                color=color,
                zorder=3,
            )

    # White cell boundaries, as in the previous heatmap style.
    ax.set_xticks(np.arange(-0.5, len(ADS_SITE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.030)
    # Use bold mathtext and then enforce the font properties via _bold_colorbar().
    cbar.set_label("M emb-QC", fontsize=CBAR_LABEL_SIZE, fontweight=TEXT_WEIGHT, labelpad=8)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt_3))
    _bold_colorbar(cbar)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#8A94A6")


# Backward-compatible function names used by older calls.
def plot_site_bonding_balance(ax, df, variant):
    plot_site_ccu_coupling(ax, df, variant)


def plot_site_bonding_fingerprint(ax, df, variant):
    plot_site_ccu_coupling(ax, df, variant)


def plot_site_heatmap(ax, df, variant):
    plot_site_ccu_coupling(ax, df, variant)



# =============================================================================
# Additional Fig.3D-style postplots
# =============================================================================

# These two plots reproduce the key outputs from the older
# plot_fig3d_full_p_2p_contact_allcu_postprocess.py workflow, but are now
# generated directly from this main plotting script.
#
# Outputs:
#   fig3d_<variant>_delta_bars.png
#   fig3d_<variant>_abs_embRKS_embQC_all_sites.png
#
# For the default variant=p_allcu, these become:
#   fig3d_p_allcu_delta_bars.png
#   fig3d_p_allcu_abs_embRKS_embQC_all_sites.png

FIG3D_STATE_ORDER = ["abv", "top", "fcc", "hcp", "bridge"]
FIG3D_SITE_COLORS = {
    "abv": "#8A94A6",
    "top": "#C9A56E",
    "fcc": "#83BFAE",
    "hcp": "#D78C96",
    "bridge": "#8FBBD9",
}


def fig3d_reorder_states(df):
    out = df.copy()
    out["state"] = out["state"].astype(str).str.strip().str.lower()
    present = [s for s in FIG3D_STATE_ORDER if s in set(out["state"])]
    other = [s for s in out["state"].tolist() if s not in present]
    cats = present + other
    out["_fig3d_state_order"] = pd.Categorical(out["state"], categories=cats, ordered=True)
    out = out.sort_values("_fig3d_state_order").drop(columns=["_fig3d_state_order"])
    return out.reset_index(drop=True)


def fig3d_channel_items(variant):
    """Return channels in the Fig.3D order with main-figure-style short labels."""
    cfg = VARIANTS[variant]
    ch = cfg["channels"]
    lab = cfg["short_labels"]
    return [
        (ch["co_sigma"], lab["co_sigma"]),
        (ch["cu_sigma"], lab["cu_sigma"]),
        (ch["co_pi"], lab["co_pi"]),
        (ch["cu_pi"], lab["cu_pi"]),
    ]


def fig3d_delta_matrix(df, variant):
    items = fig3d_channel_items(variant)
    cols = [f"delta_{key}_CASSCF_minus_RKS" for key, _label in items]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Cannot make Fig.3D delta bars because these columns are missing:\n  "
            + "\n  ".join(missing)
        )
    labels = [label for _key, label in items]
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return labels, mat


def fig3d_abs_arrays(df, key):
    rks_col = f"{key}_embRKS"
    qc_col = f"{key}_CASSCF"
    missing = [c for c in [rks_col, qc_col] if c not in df.columns]
    if missing:
        raise KeyError(
            "Cannot make Fig.3D absolute plot because these columns are missing:\n  "
            + "\n  ".join(missing)
        )
    y_rks = pd.to_numeric(df[rks_col], errors="coerce").to_numpy(dtype=float)
    y_qc = pd.to_numeric(df[qc_col], errors="coerce").to_numpy(dtype=float)
    return y_rks, y_qc


def plot_fig3d_delta_bars(df, variant, outdir):
    """Grouped Delta M bars for all sites and the four Fig.3D channels."""
    df2 = fig3d_reorder_states(df)
    labels, mat = fig3d_delta_matrix(df2, variant)
    states = df2["state"].astype(str).tolist()

    x = np.arange(len(labels), dtype=float)
    n_state = len(states)
    width = 0.78 / max(n_state, 1)
    offsets = (np.arange(n_state) - (n_state - 1) / 2.0) * width

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        ymin, ymax = -0.01, 0.01
    else:
        ymin = min(float(np.min(finite)), 0.0)
        ymax = max(float(np.max(finite)), 0.0)
        yr = max(ymax - ymin, 1e-6)
        ymin -= 0.18 * yr
        ymax += 0.24 * yr

    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=300)
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.85, zorder=1)
    ax.axvline(1.5, color="#777777", linestyle="--", linewidth=0.80, alpha=0.55, zorder=1)

    for si, st in enumerate(states):
        ax.bar(
            x + offsets[si],
            mat[si, :],
            width=width * 0.88,
            color=FIG3D_SITE_COLORS.get(st, "#999999"),
            edgecolor="white",
            linewidth=0.60,
            label=st,
            zorder=3,
        )

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    ax.set_ylabel(r"$\Delta M$ = emb-QC $-$ emb-DFT", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axes(ax, grid=True)

    ax.text(
        0.25, 0.90, "C--O internal",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top",
        fontsize=ANNOTATION_SIZE,
        fontweight=TEXT_WEIGHT,
        color="#555555",
    )
    ax.text(
        2.55, 0.90, "CO--Cu coupling",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top",
        fontsize=ANNOTATION_SIZE,
        fontweight=TEXT_WEIGHT,
        color="#555555",
    )

    leg = ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.13),
        handlelength=1.5,
        ncol=max(1, len(states)),
        columnspacing=1.2,
        borderpad=0.10,
    )
    _bold_legend(leg)

    _force_bold_all_text(fig)
    fig.tight_layout(rect=[0.000, 0.000, 1.000, 0.920])
    out_png = Path(outdir) / f"fig3d_{variant}_delta_bars.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_fig3d_abs_all_sites(df, variant, outdir):
    """Absolute emb-DFT/emb-QC Mayer block indices for all adsorption sites.

    The method legend is moved above the subplot area so it no longer overlaps
    the bars, which was the main visual issue in the older postprocess plot.
    """
    df2 = fig3d_reorder_states(df)
    items = fig3d_channel_items(variant)
    states = df2["state"].astype(str).tolist()
    x = np.arange(len(states), dtype=float)
    width = 0.33

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.25), dpi=300, sharex=True)
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None

    for ax, (key, label) in zip(axes, items):
        y_rks, y_qc = fig3d_abs_arrays(df2, key)

        ax.bar(
            x - width / 2,
            y_rks,
            width=width,
            color=COLORS["emb_dft"],
            edgecolor="white",
            linewidth=0.60,
            alpha=0.96,
            label="emb-DFT",
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            y_qc,
            width=width,
            color=COLORS["emb_qc"],
            edgecolor="white",
            linewidth=0.60,
            alpha=0.96,
            label="emb-QC",
            zorder=3,
        )

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Put the channel label below the subplot instead of as a top title.
        # This keeps the Fig.3D absolute-bar panels consistent with the other
        # plots where the channel labels are below the data region.
        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=TICK_LABEL_SIZE, fontweight=TEXT_WEIGHT)
        ax.text(
            0.5,
            -0.17,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=AXIS_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
            clip_on=False,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_3))

        vals = np.concatenate([y_rks[np.isfinite(y_rks)], y_qc[np.isfinite(y_qc)]])
        if vals.size:
            ymax = float(np.max(vals))
            ax.set_ylim(0.0, ymax * 1.22 if ymax > 0 else 1.0)

        style_axes(ax, grid=True)

    if legend_handles is not None:
        leg = axes[0].legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.60, 1.17),
            ncol=2,
            handlelength=1.45,
            columnspacing=1.05,
            labelspacing=0.25,
            borderpad=0.08,
            fontsize=LEGEND_FONT_SIZE,
        )
        _bold_legend(leg)

    fig.supylabel(
        r"Mayer block index $M$",
        x=0.035,
        fontsize=AXIS_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["text"],
    )

    _force_bold_all_text(fig)
    # Compact row spacing; leave modest room for below-axis channel labels
    # and the first-subplot legend placed just above the axes.
    fig.tight_layout(rect=[0.040, 0.065, 1.000, 0.945], h_pad=1.25, w_pad=2.2)

    out_png = Path(outdir) / f"fig3d_{variant}_abs_embRKS_embQC_all_sites.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


# =============================================================================
# Output table
# =============================================================================

def export_selected_values(df, variant, outdir):
    cfg = VARIANTS[variant]
    rows = []

    panel_defs = [
        ("B_abv_internal", ["abv"], ["co_sigma", "co_pi"], ["emb-DFT", "emb-QC"]),
        ("C_top_method_compare", ["top"], ["co_sigma", "co_pi", "cu_sigma", "cu_pi"], ["emb-DFT", "emb-QC"]),
        ("D_site_embQC", ADS_SITE_ORDER, ["co_sigma", "co_pi", "cu_sigma", "cu_pi"], ["emb-QC"]),
    ]

    for panel, states, channel_names, methods in panel_defs:
        for st in states:
            for cname in channel_names:
                key = cfg["channels"][cname]
                for method in methods:
                    try:
                        v = value(df, st, key, method)
                    except Exception:
                        v = np.nan
                    rows.append({
                        "panel": panel,
                        "state": st,
                        "channel_label": cfg["short_labels"][cname],
                        "channel_key": key,
                        "method": method,
                        "mayer_bond_index": v,
                    })

    out_csv = Path(outdir) / f"co_mayer_selected_values_{variant}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


# =============================================================================
# Main figure
# =============================================================================

def plot_combined_figure(df, variant, outdir):
    fig = plt.figure(figsize=(13.2, 8.6), dpi=300)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.00, 1.12],
        height_ratios=[1.00, 1.08],
        wspace=0.38,
        hspace=0.40,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_mo_scheme(ax_a)
    plot_abv_internal(ax_b, df, variant)
    plot_top_dumbbell(ax_c, df, variant)
    plot_site_ccu_coupling(ax_d, df, variant)

    # With subplot titles removed, keep (a)-(d) close to the corresponding
    # axes top edge rather than at the former title level.
    # Figure-coordinate vertical offset: smaller values move labels down.
    panel_label_y_offset = 0.002
    # Horizontal offset relative to each axes width: larger moves left.
    panel_label_x_offset_frac = 0.120
    for ax, label in [
        (ax_a, "(a)"),
        (ax_b, "(b)"),
        (ax_c, "(c)"),
        (ax_d, "(d)"),
    ]:
        pos = ax.get_position()
        fig.text(
            pos.x0 - panel_label_x_offset_frac * pos.width,
            pos.y1 + panel_label_y_offset,
            label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
        )


    _force_bold_all_text(fig)

    out_png = Path(outdir) / f"co_mayer_bonding_mechanism_v22_darker_panelC_arrows_{variant}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot combined CO/Cu Mayer bond-order mechanism figure."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV. Default: {DEFAULT_INPUT}")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"Output directory. Default: {DEFAULT_OUTDIR}")
    parser.add_argument(
        "--variant",
        default="p_allcu",
        choices=list(VARIANTS.keys()),
        help="Mayer channel definition to use. Default: p_allcu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inpath = Path(args.input)
    outdir = ensure_dir(args.outdir)

    if not inpath.exists():
        raise FileNotFoundError(f"Cannot find input CSV: {inpath}")

    df = pd.read_csv(inpath)
    if "state" not in df.columns:
        raise KeyError("Input CSV must contain a 'state' column.")

    df = safe_numeric(df)
    df = standardize_state_order(df)
    check_columns(df, args.variant)

    print(f"[INFO] Loaded: {inpath}")
    print(f"[INFO] Variant: {args.variant} ({VARIANTS[args.variant]['title']})")
    print(f"[INFO] States: {df['state'].tolist()}")

    out_png = plot_combined_figure(df, args.variant, outdir)
    # [R2 harness adaptation] main-text panel only; auxiliary Fig3D-style outputs skipped.
    return
    out_delta = plot_fig3d_delta_bars(df, args.variant, outdir)
    out_abs = plot_fig3d_abs_all_sites(df, args.variant, outdir)

    print("[INFO] Saved combined figure:")
    print(f"  {out_png}")
    print("[INFO] Saved additional Fig.3D-style figures:")
    print(f"  {out_delta}")
    print(f"  {out_abs}")


if __name__ == "__main__":
    main()
