#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze H2/Cu QC-DFET outputs and generate all publication/checking plots.

Expected run location:
    project_root/
        analysis_outputs/IS/IS_analysis.json
        analysis_outputs/TS/TS_analysis.json
        analysis_outputs/FS/FS_analysis.json
        IS/*.xyz, TS/*.xyz, FS/*.xyz

Important reaction-coordinate convention:
    displayed IS <- raw FS   # fully dissociated 2H state
    displayed TS <- raw TS
    displayed FS <- raw IS   # molecular H2 state

Main changes in this version:
    1. Main Fig. 2b keeps the slab-to-H2 Mulliken charge-transfer bars but
       replaces the old TS atom-resolved Mulliken inset with a native,
       high-resolution emb-DFT/emb-QC promolecular deformation-density map.
    2. The IS/TS/FS top-view charge-map summary is retained for the SI:
           upper row = emb-QC
           lower row = emb-DFT
       A larger padding is used to avoid cutting atoms at map boundaries.
    3. figure_2_ci_compactness_entropy_dual_axis.png is redrawn in the
       cleaner style shown in your uploaded example.
    4. All previous PNG figures are still generated.
    5. The density colorbar uses symmetric +/- labels (three per side) at two decimals.
    6. No PDF files are generated.
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# =============================================================================
# Basic settings
# =============================================================================

ANALYSIS_ROOT = "data"
PLOT_DIR = "output"

RAW_STATES = ["IS", "TS", "FS"]
DISPLAY_STATE_ORDER = ["IS", "TS", "FS"]

DISPLAY_TO_SOURCE = {
    "IS": "FS",
    "TS": "TS",
    "FS": "IS",
}

# Energy benchmark data used for H2/Cu(111) barrier plot.
BENCHMARK_IS_TS = 0.77
BENCHMARK_FS_TS = 0.59
CHEMICAL_ACCURACY_EV = 0.0436
ENERGY_METHODS = {
    "PBE-D3BJ": {
        "is_ts": 0.49,
        "fs_ts": 0.29,
    },
    "ENEVPT2": {
        "is_ts": 0.57,
        "fs_ts": 0.62,
    },
    "FEMION": {
        "is_ts": 0.79,
        "fs_ts": None,  # reverse barrier was not reported/calculated
    },
    "XYG3:PBE-D3BJ": {
        "is_ts": 0.81,
        "fs_ts": 0.61,
    },
    "QC-DFET": {
        "is_ts": 0.75,
        "fs_ts": 0.57,
    },
}

# Increase this if atoms are still visually clipped in top-view maps.
# This is the key parameter for your current clipping issue.
TOPVIEW_PADDING_SCALE = 1.45
TOPVIEW_ALL_STATE_FIGSIZE = (11.2, 6.0)

# Inset controls for the main two-panel figure.
ENERGY_ERROR_INSET_BBOX = [0.405, 0.135, 0.245, 0.270]

# Legacy TS Mulliken-charge-map inset controls are retained because the script
# still generates the standalone atom-resolved charge maps used in the SI.
CHARGE_MAPS_BG_BBOX = [0.390, 0.575, 0.600, 0.310]
CHARGE_MAP_RKS_BBOX  = [0.025, 0.020, 0.385, 0.845]
CHARGE_MAP_QC_BBOX   = [0.425, 0.020, 0.385, 0.845]
CHARGE_MAP_CBAR_BBOX = [0.835, 0.135, 0.032, 0.620]

# Main Fig. 2b uses the TS promolecular deformation-density map as the inset.
# IMPORTANT: the inset is now drawn natively from the saved numerical grid,
# not pasted from a PNG.  This keeps text/contours crisp and allows the final
# combined figure to be exported at 900 dpi.
PROMOLECULAR_DENSITY_DATA_DIR = os.path.join("data", "inset")
PROMOLECULAR_DENSITY_PLANE_FILE = os.path.join(
    PROMOLECULAR_DENSITY_DATA_DIR,
    "promolecular_deformation_plane.npz",
)
PROMOLECULAR_DENSITY_SUMMARY_FILE = os.path.join(
    PROMOLECULAR_DENSITY_DATA_DIR,
    "promolecular_deformation_summary.json",
)

# [left, bottom, width, height] in Panel-B axes coordinates.
# Enlarged slightly toward the left while keeping the right edge fixed at x=0.960.
PROMOLECULAR_DENSITY_INSET_BBOX = [0.335, 0.615, 0.625, 0.295]
PROMOLECULAR_DENSITY_INSET_BORDER = False

# Publication-style typography for the density inset.  These sizes are
# intentionally close to the previous Fig. 2b inset, but every controlled
# element is bold to match the manuscript figures.
DENSITY_INSET_METHOD_SIZE = 7.6
DENSITY_INSET_AXIS_LABEL_SIZE = 6.6
DENSITY_INSET_TICK_SIZE = 6.0
DENSITY_INSET_CBAR_LABEL_SIZE = 6.0
DENSITY_INSET_CBAR_TICK_SIZE = 5.7

# Article-consistent diverging palette: muted blue for depletion and the
# emb-QC peach/salmon family for accumulation, with a near-white zero.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "article_density_diverging",
    [
        "#315D86",   # strong depletion
        "#78B4D3",
        "#B5E5FA",
        "#F7F7F7",  # zero
        "#F8CBC0",
        "#D9816E",
        "#A9473E",  # strong accumulation
    ],
    N=256,
)
DENSITY_POSITIVE_CONTOUR = "#9D4B42"
DENSITY_NEGATIVE_CONTOUR = "#315D86"
MAIN_FIGURE_DPI = 900

CHARGE_TRANSFER_NEGATIVE_ROOM = 0.035

COLORS = {
    "pbe": "#8A94A6",
    "nevpt2": "#B7AA9A",
    "afqmc_rpa": "#6F8FB7",
    "xyg3": "#C38755",
    "qc_dfet": "#83BFAE",
    "emb_dft": "#B5E5FA",
    "emb_qc": "#F8CBC0",
    "ci_entropy": "#6FAE9C",
    "top1": "#A99ACB",
    "adsorbate": "#C47A44",
    "slab": "#4F7C6A",
    "text": "#222222",
    "tick": "#333333",
    "spine": "#333333",
}

METHOD_COLORS = {
    "PBE-D3BJ": COLORS["pbe"],
    "ENEVPT2": COLORS["nevpt2"],
    "FEMION": COLORS["afqmc_rpa"],
    "XYG3:PBE-D3BJ": COLORS["xyg3"],
    "QC-DFET": COLORS["qc_dfet"],
}

BARRIER_METHOD_ORDER = [
    "PBE-D3BJ",
    "ENEVPT2",
    "FEMION",
    "XYG3:PBE-D3BJ",
    "QC-DFET",
]

# Marker style for the separated barrier panels.
# Use a distinct but comparable marker for QC-DFET so it stands out
# without relying on a much larger size.
BARRIER_MARKER_DEFAULT = "D"   # filled diamond for literature / reference methods
BARRIER_MARKER_QC = "h"        # filled hexagon for QC-DFET, aligned with the CO figure

SLAB_CMAP = LinearSegmentedColormap.from_list(
    "soft_diverging",
    ["#4C6EDB", "#DCE6F2", "#F4F4F4", "#F3D6C6", "#C73E3A"],
)

COV_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "Cu": 1.32,
}

# Unified typography controls.
# All controlled text is bold, but the sizes are arranged hierarchically so the
# figure does not look visually overloaded.
#
# Size hierarchy:
#   MAIN_TITLE_SIZE       : only the two main panel titles
#   SUBPANEL_TITLE_SIZE   : TS-IS / TS-FS and similar subplot titles
#   AXIS_LABEL_SIZE       : x/y axis labels
#   TICK_LABEL_SIZE       : ordinary tick labels
#   METHOD_TICK_LABEL_SIZE: method names on Panel A; slightly smaller to avoid overlap
#   LEGEND_FONT_SIZE      : legend text
#   INSET_*               : inset charge-map labels/colorbar ticks
MAIN_TITLE_SIZE = 12.8
SUBPANEL_TITLE_SIZE = 10.8
AXIS_LABEL_SIZE = 10.8
TICK_LABEL_SIZE = 9.6
METHOD_TICK_LABEL_SIZE = 8.5
LEGEND_FONT_SIZE = 9.6
CHARGE_LEGEND_FONT_SIZE = 8.2
PANEL_LABEL_SIZE = 13.8
INSET_LABEL_SIZE = 7.4
INSET_CBAR_TICK_SIZE = 6.8
MAP_TITLE_SIZE = 10.0
CBAR_LABEL_SIZE = 9.2
CBAR_TICK_SIZE = 8.6
ANNOTATION_SIZE = 9.2

BASE_FONT_SIZE = TICK_LABEL_SIZE
AXIS_TITLE_SIZE = SUBPANEL_TITLE_SIZE
SUPTITLE_SIZE = MAIN_TITLE_SIZE
COMMON_FONT_SIZE = TICK_LABEL_SIZE

# One consistent bold weight everywhere.  Use explicit fontweight control rather
# than relying on mixed default/bold settings.
TEXT_WEIGHT = "bold"
AXIS_LABEL_WEIGHT = TEXT_WEIGHT
TICK_LABEL_WEIGHT = TEXT_WEIGHT
METHOD_TICK_LABEL_WEIGHT = TEXT_WEIGHT
ALL_TEXT_WEIGHT = TEXT_WEIGHT

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": BASE_FONT_SIZE,
    "font.weight": "normal",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": AXIS_TITLE_SIZE,
    "axes.titleweight": TEXT_WEIGHT,
    "axes.labelweight": TEXT_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": 0.82,
    "xtick.major.width": 0.82,
    "ytick.major.width": 0.82,
    "xtick.major.size": 3.8,
    "ytick.major.size": 3.8,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# Logging / IO
# =============================================================================

def log_info(msg):
    print(f"[INFO] {msg}", flush=True)


def log_warn(msg):
    print(f"[WARN] {msg}", flush=True)


def log_error(msg):
    print(f"[ERROR] {msg}", flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def state_json_path(raw_state):
    return os.path.join(ANALYSIS_ROOT, raw_state, f"{raw_state}_analysis.json")


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def save_png(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {path}")


def apply_clean_axis_style(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", colors=COLORS["tick"], pad=2)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.title.set_color(COLORS["text"])
    ax.title.set_fontweight(TEXT_WEIGHT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
    if grid:
        ax.grid(True, axis="y", alpha=0.115, linewidth=0.48, color="#8D99A6")
        ax.grid(False, axis="x")
    else:
        ax.grid(False)


def add_panel_label(ax, label, x=-0.10, y=1.06):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["text"],
        clip_on=False,
    )


# =============================================================================
# Data loading and summaries
# =============================================================================

def load_remapped_state_data():
    raw_data = {}
    for raw_state in RAW_STATES:
        path = state_json_path(raw_state)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        raw_data[raw_state] = load_json(path)

    display_data = {}
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        display_data[display_state] = raw_data[src_state]
    return display_data


def top1_info(state_dict):
    dom = state_dict.get("dominant_ci", [])
    if dom:
        x = dom[0]
        sig = f"a:{x.get('alpha_occ_str')} | b:{x.get('beta_occ_str')}"
        return sig, safe_float(x.get("weight")), safe_float(x.get("coeff"))
    return None, np.nan, np.nan


def get_active_orbital_rows(state_dict):
    rows = state_dict.get("active_orbital_ao_composition", [])
    return rows if rows is not None else []


def get_active_noons(state_dict):
    arr = state_dict.get("active_noons", [])
    arr = arr if arr is not None else []
    return [safe_float(x) for x in arr]


def summarize_active_orbitals(state_dict, topn=12):
    rows = []
    for item in get_active_orbital_rows(state_dict)[:topn]:
        rows.append({
            "orbital_index": item.get("orbital_index"),
            "h_contribution": item.get("h_contribution"),
            "cu_contribution": item.get("cu_contribution"),
            "sigma_like_score": item.get("sigma_like_score"),
            "pi_like_score": item.get("pi_like_score"),
            "h_s_like_score": item.get("h_s_like_score"),
            "h_p_total_score": item.get("h_p_total_score"),
        })
    return pd.DataFrame(rows)


def make_energy_table(state_data):
    rows = []
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        d = state_data[display_state]
        top1_sig, top1_w, top1_c = top1_info(d)
        rows.append({
            "display_state": display_state,
            "source_state": src_state,
            "H-H distance (A)": d.get("h_h_distance_angstrom"),
            "E_UKS (Ha)": d.get("e_uks"),
            "E_RKS (Ha)": d.get("e_rks"),
            "E_embDFT (Ha)": d.get("e_emb_rks"),
            "E_embQC (Ha)": d.get("e_casscf"),
            "dE_NEVPT2 (Ha)": d.get("de_nevpt2"),
            "E_embQC+NEVPT2 (Ha)": d.get("e_casscf_nevpt2"),
            "N_eff": d.get("n_eff"),
            "Top1 weight": top1_w,
            "Top1 coeff": top1_c,
            "Top1 config": top1_sig,
        })
    return pd.DataFrame(rows)


def make_charge_table(state_data):
    rows = []
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        d = state_data[display_state]
        rows.append({
            "display_state": display_state,
            "source_state": src_state,
            "H2_embDFT_Mulliken": d.get("emb_rks_fragment_charge_h2_mulliken"),
            "H2_embDFT_Lowdin": d.get("emb_rks_fragment_charge_h2_lowdin"),
            "H2_embDFT_metaLowdin": d.get("emb_rks_fragment_charge_h2_meta_lowdin"),
            "H2_embQC_Mulliken": d.get("casscf_fragment_charge_h2_mulliken"),
            "H2_embQC_Lowdin": d.get("casscf_fragment_charge_h2_lowdin"),
            "H2_embQC_metaLowdin": d.get("casscf_fragment_charge_h2_meta_lowdin"),
            "Cu12_embDFT_Mulliken": d.get("emb_rks_fragment_charge_cu12_mulliken"),
            "Cu12_embQC_Mulliken": d.get("casscf_fragment_charge_cu12_mulliken"),
        })
    return pd.DataFrame(rows)


def make_ci_table(state_data, topn=20):
    rows = []
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        dom = state_data[display_state].get("dominant_ci", []) or []
        for item in dom[:topn]:
            rows.append({
                "display_state": display_state,
                "source_state": src_state,
                "Rank": item.get("rank"),
                "alpha_occ": item.get("alpha_occ_str"),
                "beta_occ": item.get("beta_occ_str"),
                "coeff": item.get("coeff"),
                "weight": item.get("weight"),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Frontier descriptor analysis
# =============================================================================

def build_hh_frontier_weights(
    state_dict,
    power_h=1.0,
    power_hs=1.0,
    power_sigma=1.0,
    include_pi=0.15,
    normalize=True,
):
    noons = get_active_noons(state_dict)
    rows = get_active_orbital_rows(state_dict)
    n_orb = min(len(noons), len(rows))
    if n_orb == 0:
        return np.array([], dtype=float)

    weights = []
    for i in range(n_orb):
        r = rows[i]
        h = max(safe_float(r.get("h_contribution"), 0.0), 0.0)
        hs = max(safe_float(r.get("h_s_like_score"), 0.0), 0.0)
        sig = max(safe_float(r.get("sigma_like_score"), 0.0), 0.0)
        pi = max(safe_float(r.get("pi_like_score"), 0.0), 0.0)
        wi = (h ** power_h) * (hs ** power_hs) * ((sig + include_pi * pi) ** power_sigma)
        weights.append(wi)

    weights = np.array(weights, dtype=float)
    if np.all(weights < 1e-15):
        weights = np.ones_like(weights)

    if normalize:
        s = np.sum(weights)
        if s > 1e-15:
            weights = weights / s
    return weights


def take_topk_by_weight(weights, k=4):
    weights = np.array(weights, dtype=float)
    if len(weights) == 0:
        return np.array([], dtype=int)
    k = min(int(k), len(weights))
    idx = np.argsort(weights)[::-1][:k]
    return np.sort(idx)


def get_frontier_detail_rows(state_dict, topk=4):
    rows = get_active_orbital_rows(state_dict)
    noons = get_active_noons(state_dict)
    weights = build_hh_frontier_weights(state_dict, normalize=True)
    idx = take_topk_by_weight(weights, k=topk)

    out = []
    for i in idx:
        r = rows[i]
        out.append({
            "orbital_index": i + 1,
            "weight": weights[i],
            "NOON": noons[i],
            "adsorbate_contribution": safe_float(r.get("h_contribution")),
            "slab_contribution": safe_float(r.get("cu_contribution")),
            "sigma_like_score": safe_float(r.get("sigma_like_score")),
            "pi_like_score": safe_float(r.get("pi_like_score")),
            "h_s_like_score": safe_float(r.get("h_s_like_score")),
            "h_p_total_score": safe_float(r.get("h_p_total_score")),
        })
    return out


def weighted_frontier_component_summary(state_dict, topk=4):
    detail = get_frontier_detail_rows(state_dict, topk=topk)
    if len(detail) == 0:
        return {
            "adsorbate_contribution": np.nan,
            "slab_contribution": np.nan,
            "sigma_like_score": np.nan,
            "pi_like_score": np.nan,
            "h_s_like_score": np.nan,
            "h_p_total_score": np.nan,
        }

    w = np.array([safe_float(x["weight"], 0.0) for x in detail], dtype=float)
    if np.sum(w) <= 1e-15:
        w = np.ones(len(detail), dtype=float) / len(detail)
    else:
        w = w / np.sum(w)

    out = {}
    for key in [
        "adsorbate_contribution",
        "slab_contribution",
        "sigma_like_score",
        "pi_like_score",
        "h_s_like_score",
        "h_p_total_score",
    ]:
        vals = np.array([safe_float(x[key], np.nan) for x in detail], dtype=float)
        out[key] = float(np.nansum(w * vals))
    return out


def make_frontier_component_summary_table(state_data, topk=4):
    rows = []
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        comp = weighted_frontier_component_summary(state_data[display_state], topk=topk)
        row = {"display_state": display_state, "source_state": src_state}
        row.update(comp)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# CI complexity
# =============================================================================

def get_ci_weights(state_dict, topn=20, renormalize=True):
    dom = state_dict.get("dominant_ci", [])
    dom = dom if dom is not None else []
    if topn is not None:
        dom = dom[:topn]

    weights = []
    for item in dom:
        w = safe_float(item.get("weight"), np.nan)
        if np.isfinite(w) and w >= 0.0:
            weights.append(w)
    weights = np.array(weights, dtype=float)

    if len(weights) == 0:
        return np.array([], dtype=float)

    s = np.sum(weights)
    if renormalize and s > 1e-15:
        weights = weights / s
    return weights


def shannon_entropy(weights, eps=1e-15):
    w = np.array(weights, dtype=float)
    if len(w) == 0:
        return np.nan
    w = w[w > eps]
    if len(w) == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def cumulative_weight(weights, k):
    w = np.array(weights, dtype=float)
    if len(w) == 0:
        return np.nan
    k = min(int(k), len(w))
    return float(np.sum(w[:k]))


def ci_complexity_metrics(state_dict, topn=20):
    weights = get_ci_weights(state_dict, topn=topn, renormalize=True)
    if len(weights) == 0:
        return {
            "CI_n_terms_used": 0,
            "CI_top1_weight": np.nan,
            "CI_top5_cum_weight": np.nan,
            "CI_top10_cum_weight": np.nan,
            "CI_top20_cum_weight": np.nan,
            "CI_entropy": np.nan,
        }
    return {
        "CI_n_terms_used": int(len(weights)),
        "CI_top1_weight": cumulative_weight(weights, 1),
        "CI_top5_cum_weight": cumulative_weight(weights, 5),
        "CI_top10_cum_weight": cumulative_weight(weights, 10),
        "CI_top20_cum_weight": cumulative_weight(weights, 20),
        "CI_entropy": shannon_entropy(weights),
    }


def make_ci_complexity_table(state_data, topn=20):
    rows = []
    for display_state in DISPLAY_STATE_ORDER:
        src_state = DISPLAY_TO_SOURCE[display_state]
        metrics = ci_complexity_metrics(state_data[display_state], topn=topn)
        row = {"display_state": display_state, "source_state": src_state}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def get_ci_top1_series(state_data, topn=20):
    return np.array([
        ci_complexity_metrics(state_data[s], topn=topn)["CI_top1_weight"]
        for s in DISPLAY_STATE_ORDER
    ], dtype=float)


def get_ci_entropy_series(state_data, topn=20):
    return np.array([
        ci_complexity_metrics(state_data[s], topn=topn)["CI_entropy"]
        for s in DISPLAY_STATE_ORDER
    ], dtype=float)


# =============================================================================
# XYZ and top-view map helpers
# =============================================================================

def parse_xyz_coords(path):
    if path is None or not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        lines = [x.rstrip("\n") for x in f]
    if len(lines) < 3:
        return None, None

    try:
        natm = int(lines[0].strip())
    except Exception:
        return None, None

    labels = []
    coords = []
    for i in range(2, min(2 + natm, len(lines))):
        toks = lines[i].split()
        if len(toks) < 4:
            continue
        labels.append(toks[0])
        coords.append([float(toks[1]), float(toks[2]), float(toks[3])])

    if len(coords) != natm:
        log_warn(f"Parsed atom count mismatch for xyz: {path} (expected {natm}, got {len(coords)})")
        return None, None
    return labels, np.array(coords, dtype=float)


def auto_find_xyz_for_state(state_dir):
    if not os.path.isdir(state_dir):
        log_warn(f"State directory not found for xyz search: {state_dir}")
        return None

    xyz_files = [f for f in os.listdir(state_dir) if f.lower().endswith(".xyz")]
    if len(xyz_files) == 0:
        log_warn(f"No xyz files found in {state_dir}")
        return None

    if len(xyz_files) == 1:
        return os.path.join(state_dir, xyz_files[0])

    preferred_keywords = ["coords", "coord", "final", "state", "opt", "geom"]
    scored = []
    for f in xyz_files:
        name = f.lower()
        score = sum(k in name for k in preferred_keywords)
        scored.append((score, f))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = os.path.join(state_dir, scored[0][1])
    log_info(f"Multiple xyz files found in {state_dir}; chose {chosen}")
    return chosen


def build_raw_xyz_map():
    return {st: auto_find_xyz_for_state(st) for st in RAW_STATES}


def prepare_display_xyz_cache(raw_xyz_map):
    xyz_cache = {}
    for display_state in DISPLAY_STATE_ORDER:
        source_state = DISPLAY_TO_SOURCE[display_state]
        xyz_path = raw_xyz_map.get(source_state)
        labels, coords = parse_xyz_coords(xyz_path) if xyz_path else (None, None)
        xyz_cache[display_state] = (labels, coords)
    return xyz_cache


def get_substrate_atom_idxs(labels):
    return [i for i, lab in enumerate(labels) if str(lab).strip().lower() == "cu"]


def get_adsorbate_atom_idxs(labels):
    return [i for i, lab in enumerate(labels) if str(lab).strip().lower() != "cu"]


def estimate_bonds(labels, coords, ads_idxs, include_ads_sub=True):
    bonds = []
    n = len(labels)
    ads_set = set(ads_idxs)
    sub_set = set(range(n)) - ads_set

    for i in range(n):
        for j in range(i + 1, n):
            in_ads = (i in ads_set) and (j in ads_set)
            ads_sub = ((i in ads_set and j in sub_set) or (j in ads_set and i in sub_set))
            if not in_ads and not (include_ads_sub and ads_sub):
                continue

            ri = COV_RADII.get(labels[i], 0.8)
            rj = COV_RADII.get(labels[j], 0.8)
            d = np.linalg.norm(coords[i] - coords[j])

            if in_ads:
                thresh = 1.25 * (ri + rj)
            else:
                thresh = max(2.35, 1.15 * (ri + rj))

            if d <= thresh:
                bonds.append((i, j))
    return bonds


def cluster_z_layers(zvals, tol=0.35):
    zvals = np.asarray(zvals, dtype=float)
    order = np.argsort(zvals)
    layer_id = np.empty(len(zvals), dtype=int)
    current_layer = 0
    ref_z = zvals[order[0]]
    layer_id[order[0]] = current_layer

    for idx in order[1:]:
        if abs(zvals[idx] - ref_z) <= tol:
            layer_id[idx] = current_layer
        else:
            current_layer += 1
            ref_z = zvals[idx]
            layer_id[idx] = current_layer
    return layer_id


def estimate_packed_radius_xy(coords_xy, idxs):
    if idxs is None or len(idxs) < 2:
        return 0.55

    pts = np.asarray(coords_xy[idxs], dtype=float)
    nn_dists = []
    for i in range(len(pts)):
        d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
        d = d[d > 1e-10]
        if len(d) > 0:
            nn_dists.append(np.min(d))

    if len(nn_dists) == 0:
        return 0.55
    return 0.48 * float(np.median(nn_dists))


def get_charge_norm_for_all_state_maps(state_data):
    vals = []
    for display_state in DISPLAY_STATE_ORDER:
        for key in ["emb_rks_mulliken_atomic_charges", "casscf_mulliken_atomic_charges"]:
            charges = state_data[display_state].get(key)
            if charges is None:
                continue
            arr = np.array(charges, dtype=float)
            if arr.size > 0:
                vals.extend(list(np.abs(arr)))
    if len(vals) == 0:
        return None
    vmax = max(float(np.nanmax(vals)), 1e-8)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def get_charge_norm_for_ts_maps(state_data):
    vals = []
    for key in ["emb_rks_mulliken_atomic_charges", "casscf_mulliken_atomic_charges"]:
        charges = state_data["TS"].get(key)
        if charges is None:
            continue
        vals.extend(list(np.abs(np.array(charges, dtype=float))))
    if len(vals) == 0:
        return None
    vmax = max(float(np.nanmax(vals)), 1e-8)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def draw_topview_charge_map_on_ax(
    ax,
    labels,
    coords,
    charges,
    norm,
    method_label="",
    minimal=True,
    padding_scale=TOPVIEW_PADDING_SCALE,
):
    ads_idxs = get_adsorbate_atom_idxs(labels)
    sub_idxs = get_substrate_atom_idxs(labels)
    if len(sub_idxs) == 0:
        raise RuntimeError("No Cu substrate atoms found in xyz file.")

    bonds = estimate_bonds(labels, coords, ads_idxs, include_ads_sub=True)

    xy = coords[:, :2]
    z = coords[:, 2]
    sub_layer_id = cluster_z_layers(z[sub_idxs], tol=0.35)
    unique_layers = sorted(np.unique(sub_layer_id))
    base_radius = estimate_packed_radius_xy(xy, sub_idxs)

    # substrate atoms: soft packed circles
    for lid in unique_layers:
        sub_mask = np.where(sub_layer_id == lid)[0]
        idxs_layer = [sub_idxs[i] for i in sub_mask]
        rank = unique_layers.index(lid)
        nlayer = len(unique_layers)

        if nlayer == 1:
            alpha_fill = 0.92
            r_scale = 1.00
        else:
            frac = rank / max(nlayer - 1, 1)
            alpha_fill = 0.28 + 0.58 * frac
            r_scale = 0.88 + 0.10 * frac

        for iatom in idxs_layer:
            circ = plt.Circle(
                (coords[iatom, 0], coords[iatom, 1]),
                radius=base_radius * r_scale,
                facecolor=SLAB_CMAP(norm(charges[iatom])),
                edgecolor="none",
                linewidth=0.0,
                alpha=alpha_fill,
                zorder=1 + rank,
            )
            ax.add_patch(circ)

    # bonds
    for i, j in bonds:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=(0.35, 0.35, 0.35, 0.62),
            lw=1.10,
            zorder=5,
        )

    # adsorbate atoms
    ads_radius = 0.53 * base_radius
    for iatom in ads_idxs:
        circ = plt.Circle(
            (coords[iatom, 0], coords[iatom, 1]),
            radius=ads_radius,
            facecolor=SLAB_CMAP(norm(charges[iatom])),
            edgecolor=(0.30, 0.30, 0.30, 0.42),
            linewidth=0.45,
            alpha=0.98,
            zorder=8,
        )
        ax.add_patch(circ)

        highlight = plt.Circle(
            (coords[iatom, 0] - 0.10 * ads_radius, coords[iatom, 1] + 0.10 * ads_radius),
            radius=0.23 * ads_radius,
            facecolor=(1, 1, 1, 0.22),
            edgecolor="none",
            zorder=9,
        )
        ax.add_patch(highlight)

    ax.set_aspect("equal")
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

    # This is the key fix: larger visible margin around atoms.
    pad = padding_scale * base_radius
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(method_label, pad=3.0, fontsize=MAP_TITLE_SIZE)

    if minimal:
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(False)
    else:
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_color("#BBBBBB")
            ax.spines[side].set_linewidth(0.65)
    ax.grid(False)


# =============================================================================
# Energy and charge plots
# =============================================================================

def build_energy_methods():
    methods = {name: vals.copy() for name, vals in ENERGY_METHODS.items()}
    for name in methods:
        is_ts = safe_float(methods[name].get("is_ts"), np.nan)
        fs_ts = safe_float(methods[name].get("fs_ts"), np.nan)
        methods[name]["is_ts"] = is_ts
        methods[name]["fs_ts"] = fs_ts
        methods[name]["E0"] = 0.0
        methods[name]["ETS"] = is_ts
        methods[name]["EFS"] = is_ts - fs_ts if np.isfinite(is_ts) and np.isfinite(fs_ts) else np.nan
        methods[name]["is_ts_err"] = is_ts - BENCHMARK_IS_TS if np.isfinite(is_ts) else np.nan
        methods[name]["fs_ts_err"] = fs_ts - BENCHMARK_FS_TS if np.isfinite(fs_ts) else np.nan
    return methods


def ordered_energy_method_names(methods, exclude_methods=None):
    """Return barrier-plot methods in publication order, with optional exclusions."""
    exclude_methods = set(exclude_methods or [])
    return [
        name for name in BARRIER_METHOD_ORDER
        if name in methods and name not in exclude_methods
    ]



def plot_one_barrier_by_method_on_ax(
    ax,
    barrier_key,
    benchmark_value,
    title,
    band_color,
    ref_color,
    show_xticklabels=True,
    show_xlabel=True,
    exclude_methods=None,
):
    """
    One separated method-wise barrier panel.

    Design choices in this refined version:
      - forward and reverse barriers are shown on separate axes;
      - all literature / reference methods use the same filled-diamond marker;
      - QC-DFET uses a filled hexagon so it is emphasized without simply making
        the symbol much bigger;
      - the missing FEMION reverse barrier is simply omitted from the plot.
    """
    methods = build_energy_methods()
    names = ordered_energy_method_names(methods, exclude_methods=exclude_methods)
    x = np.arange(len(names), dtype=float)
    vals = np.array([methods[name].get(barrier_key, np.nan) for name in names], dtype=float)

    finite_vals = vals[np.isfinite(vals)]
    ref_low = benchmark_value - CHEMICAL_ACCURACY_EV
    ref_high = benchmark_value + CHEMICAL_ACCURACY_EV
    all_vals = np.concatenate([finite_vals, np.array([ref_low, ref_high, benchmark_value])])
    ymin = max(0.0, float(np.nanmin(all_vals)) - 0.085)
    ymax = float(np.nanmax(all_vals)) + 0.065

    # Chemical-accuracy window and reference line.
    ax.axhspan(ref_low, ref_high, color=band_color, alpha=0.52, lw=0, zorder=0)
    ax.axhline(
        benchmark_value,
        color=ref_color,
        lw=1.05,
        ls=(0, (3.2, 2.3)),
        alpha=0.96,
        zorder=1,
    )

    # Plot one data series only, so the trend is visually unambiguous.
    for i, name in enumerate(names):
        val = vals[i]
        c = METHOD_COLORS[name]
        if np.isfinite(val):
            is_qc = (name == "QC-DFET")
            marker = BARRIER_MARKER_QC if is_qc else BARRIER_MARKER_DEFAULT
            ax.scatter(
                i,
                val,
                s=76,
                marker=marker,
                facecolor=c,
                edgecolor="#2F2F2F" if is_qc else "white",
                linewidth=1.00 if is_qc else 0.75,
                alpha=0.98,
                zorder=4 if is_qc else 3,
            )
        else:
            # Omit the missing FEMION reverse barrier from the main visual.
            pass

    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, loc="left", pad=4.5, fontsize=SUBPANEL_TITLE_SIZE, fontweight=TEXT_WEIGHT)
    ax.set_ylabel("Barrier (eV)", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)

    ax.set_xticks(x)
    if show_xticklabels:
        display_labels = list(names)
        ax.set_xticklabels(display_labels, rotation=0, ha="center")
        for tick, name in zip(ax.get_xticklabels(), names):
            tick.set_color(METHOD_COLORS[name])
            tick.set_fontweight(TEXT_WEIGHT)
            tick.set_fontsize(METHOD_TICK_LABEL_SIZE)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if show_xlabel:
        ax.set_xlabel("Method")
    else:
        ax.set_xlabel("")

    # Match the cleaner CO compact-panel style: no faint background grid.
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", colors=COLORS["tick"], pad=2)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.title.set_color(COLORS["text"])
    ax.title.set_fontweight(TEXT_WEIGHT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)


def create_energy_barrier_summary_axes(fig, subplot_spec, exclude_methods=None):
    """
    Create a clean two-row Figure A:
      top: forward barrier;
      bottom: reverse barrier.

    Separating the two barriers makes it immediately clear that QC-DFET is
    closest to the benchmark window for both directions.
    """
    inner = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=subplot_spec,
        height_ratios=[1.0, 1.0],
        hspace=0.24,
    )
    ax_forward = fig.add_subplot(inner[0, 0])
    ax_reverse = fig.add_subplot(inner[1, 0])

    plot_one_barrier_by_method_on_ax(
        ax_forward,
        barrier_key="is_ts",
        benchmark_value=BENCHMARK_IS_TS,
        title="TS-FS",
        band_color="#DCEFE9",
        ref_color="#5E9F8F",
        show_xticklabels=False,
        show_xlabel=False,
        exclude_methods=exclude_methods,
    )
    plot_one_barrier_by_method_on_ax(
        ax_reverse,
        barrier_key="fs_ts",
        benchmark_value=BENCHMARK_FS_TS,
        title="TS-IS",
        # Use the same green reference window as the forward barrier.
        band_color="#DCEFE9",
        ref_color="#5E9F8F",
        show_xticklabels=True,
        show_xlabel=False,
        exclude_methods=exclude_methods,
    )

    return ax_forward, ax_reverse



def plot_energy_profile_on_ax(ax, show_legend=True):
    methods = build_energy_methods()
    x_is, x_ts, x_fs = 0.0, 1.0, 2.0
    w = 0.17

    zorders = {
        "PBE-D3BJ": 2,
        "ENEVPT2": 3,
        "FEMION": 4,
        "XYG3:PBE-D3BJ": 5,
        "QC-DFET": 6,
    }
    plateau_lw = {
        "PBE-D3BJ": 3.00,
        "ENEVPT2": 3.00,
        "FEMION": 3.25,
        "XYG3:PBE-D3BJ": 3.25,
        "QC-DFET": 4.75,
    }
    connector_lw = {
        "PBE-D3BJ": 1.45,
        "ENEVPT2": 1.45,
        "FEMION": 1.60,
        "XYG3:PBE-D3BJ": 1.60,
        "QC-DFET": 2.05,
    }
    alphas = {
        "PBE-D3BJ": 0.66,
        "ENEVPT2": 0.68,
        "FEMION": 0.86,
        "XYG3:PBE-D3BJ": 0.86,
        "QC-DFET": 0.98,
    }

    for name, vals in methods.items():
        c = METHOD_COLORS[name]
        z = zorders.get(name, 3)
        a = alphas.get(name, 0.90)
        E0, ETS, EFS = vals["E0"], vals["ETS"], vals["EFS"]

        ax.plot([x_is - w, x_is + w], [E0, E0], lw=plateau_lw.get(name, 3.0), color=c,
                alpha=a, solid_capstyle="round", zorder=z)
        if np.isfinite(ETS):
            ax.plot([x_ts - w, x_ts + w], [ETS, ETS], lw=plateau_lw.get(name, 3.0), color=c,
                    alpha=a, solid_capstyle="round", label=name, zorder=z)
            ax.plot([x_is + w, x_ts - w], [E0, ETS], lw=connector_lw.get(name, 1.5), color=c,
                    ls=(0, (7, 5)), alpha=a * 0.82, zorder=z)

        # Some references, e.g. FEMION, only report the forward barrier.
        # In that case, draw the IS->TS part only and skip the FS plateau/error.
        if np.isfinite(EFS):
            ax.plot([x_fs - w, x_fs + w], [EFS, EFS], lw=plateau_lw.get(name, 3.0), color=c,
                    alpha=a, solid_capstyle="round", zorder=z)
            ax.plot([x_ts + w, x_fs - w], [ETS, EFS], lw=connector_lw.get(name, 1.5), color=c,
                    ls=(0, (7, 5)), alpha=a * 0.82, zorder=z)

    all_y = []
    for vals in methods.values():
        all_y.extend([vals["E0"], vals["ETS"], vals["EFS"]])
    all_y = [y for y in all_y if np.isfinite(y)]

    ax.set_xlim(-0.30, 2.30)
    ax.set_ylim(min(all_y) - 0.18, max(all_y) + 0.16)
    ax.set_xticks([x_is, x_ts, x_fs])
    ax.set_xticklabels(["IS", "TS", "FS"])
    ax.set_ylabel("Energy (eV)")
    ax.set_title(r"H$_2$ dissociation/desorption on Cu(111)", pad=7)
    apply_clean_axis_style(ax, grid=False)

    if show_legend:
        leg = ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            handlelength=2.65,
            borderpad=0.15,
            labelspacing=0.34,
            fontsize=LEGEND_FONT_SIZE,
        )
        for line in leg.get_lines():
            line.set_linewidth(3.65)
        for txt in leg.get_texts():
            txt.set_fontweight(TEXT_WEIGHT)


def plot_energy_error_bars_on_ax(ax, compact=False):
    methods = build_energy_methods()
    method_names = list(methods.keys())
    x = np.arange(2)

    # Keep grouped bars readable after adding FEMION and XYG3:PBE-D3BJ.
    n_methods = len(method_names)
    bar_width = 0.115 if compact else 0.135
    offsets = (np.arange(n_methods) - 0.5 * (n_methods - 1)) * (bar_width * 1.12)

    ax.axhspan(-CHEMICAL_ACCURACY_EV, CHEMICAL_ACCURACY_EV,
               color=COLORS["qc_dfet"], alpha=0.14, zorder=0, lw=0)
    ax.axhline(+CHEMICAL_ACCURACY_EV, color=COLORS["qc_dfet"], lw=0.55,
               ls=(0, (3, 2)), alpha=0.55, zorder=1)
    ax.axhline(-CHEMICAL_ACCURACY_EV, color=COLORS["qc_dfet"], lw=0.55,
               ls=(0, (3, 2)), alpha=0.55, zorder=1)

    finite_errs = []
    for i, name in enumerate(method_names):
        vals = [methods[name]["is_ts_err"], methods[name]["fs_ts_err"]]
        finite_errs.extend([v for v in vals if np.isfinite(v)])
        ax.bar(
            x + offsets[i], vals,
            width=bar_width,
            color=METHOD_COLORS[name],
            edgecolor="white" if compact else "none",
            linewidth=0.35 if compact else 0.0,
            alpha=0.96,
            label=name,
            zorder=3,
        )

    ax.axhline(0.0, color="#4A4A4A", lw=0.70, alpha=0.85, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["IS-TS", "FS-TS"], fontsize=INSET_CBAR_TICK_SIZE if compact else 9.5)

    if len(finite_errs) == 0:
        finite_errs = [0.0]
    ymin = min(min(finite_errs), -CHEMICAL_ACCURACY_EV)
    ymax = max(max(finite_errs), +CHEMICAL_ACCURACY_EV)
    pad = max(0.040, 0.20 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    if compact:
        ax.set_title("Barrier error", loc="left", fontsize=INSET_CBAR_TICK_SIZE, pad=1.8)
        ax.set_ylabel(r"$\Delta E$ (eV)", fontsize=INSET_CBAR_TICK_SIZE, labelpad=2.0)
        ax.tick_params(axis="both", pad=1.0, length=2.0, width=0.50)
        ax.grid(True, axis="y", alpha=0.10, linewidth=0.38, color="#8D99A6")
        for spine in ax.spines.values():
            spine.set_linewidth(0.48)
            spine.set_color("#C9D7DD")
        ax.set_facecolor((0.985, 0.995, 1.000, 0.94))
        ax.text(0.985, 0.965, "chem. acc.", transform=ax.transAxes,
                ha="right", va="top", fontsize=INSET_CBAR_TICK_SIZE, color=COLORS["qc_dfet"])
    else:
        ax.set_ylabel("Error (eV)")
        ax.set_title("Barrier errors", pad=3, fontsize=MAP_TITLE_SIZE)
        apply_clean_axis_style(ax, grid=True)


def add_energy_error_inset(ax):
    axins = ax.inset_axes(ENERGY_ERROR_INSET_BBOX)
    plot_energy_error_bars_on_ax(axins, compact=True)
    return axins


def plot_h2_barriers_standalone(outdir):
    fig = plt.figure(figsize=(8.7, 5.0))
    outer = GridSpec(1, 1, figure=fig)
    create_energy_barrier_summary_axes(fig, outer[0, 0])
    fig.suptitle(
        r"H$_2$ dissociation/desorption on Cu(111)",
        y=0.985,
        fontsize=MAIN_TITLE_SIZE,
        color=COLORS["text"],
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.135, top=0.88)
    save_png(fig, os.path.join(outdir, "h2-cu.png"))


def get_charge_series(state_data, key_suffix="mulliken"):
    y_dft = [state_data[s].get(f"emb_rks_fragment_charge_h2_{key_suffix}", np.nan)
             for s in DISPLAY_STATE_ORDER]
    y_qc = [state_data[s].get(f"casscf_fragment_charge_h2_{key_suffix}", np.nan)
            for s in DISPLAY_STATE_ORDER]
    return np.array(y_dft, dtype=float), np.array(y_qc, dtype=float)


def plot_charge_transfer_on_ax(
    ax, state_data, key_suffix="mulliken", as_bars=True, show_title=True
):
    x = np.arange(len(DISPLAY_STATE_ORDER), dtype=float)
    y_dft_raw, y_qc_raw = get_charge_series(state_data, key_suffix=key_suffix)

    # slab -> H2 convention
    y_dft = -np.array(y_dft_raw, dtype=float)
    y_qc = -np.array(y_qc_raw, dtype=float)

    if as_bars:
        bar_width = 0.30
        ax.bar(x - bar_width / 2, y_dft, width=bar_width,
               color=COLORS["emb_dft"], edgecolor="white", linewidth=0.65,
               alpha=0.96, label="emb-DFT", zorder=3)
        ax.bar(x + bar_width / 2, y_qc, width=bar_width,
               color=COLORS["emb_qc"], edgecolor="white", linewidth=0.65,
               alpha=0.96, label="emb-QC", zorder=3)
    else:
        ax.plot(x, y_dft, marker="o", linewidth=2.0, markersize=5.4,
                color=COLORS["emb_dft"], label="emb-DFT")
        ax.plot(x, y_qc, marker="o", linewidth=2.0, markersize=5.4,
                color=COLORS["emb_qc"], label="emb-QC")

    # No vertical background guides; keep the panel consistent with the
    # cleaner CO charge-transfer style.

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_STATE_ORDER, fontweight=TEXT_WEIGHT)
    for tick in ax.get_xticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(TICK_LABEL_SIZE)
    ax.set_ylabel(r"Slab$\rightarrow$H$_2$ transfer (e)", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    if show_title:
        ax.set_title(
            "Charge transfer along reaction coordinate",
            pad=10.0,
            fontsize=MAIN_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
        )
    leg = ax.legend(frameon=False, loc="upper left", handlelength=1.0,
                    borderpad=0.15, labelspacing=0.28, columnspacing=0.8,
                    fontsize=CHARGE_LEGEND_FONT_SIZE)
    for txt in leg.get_texts():
        txt.set_fontweight(TEXT_WEIGHT)
        txt.set_fontsize(CHARGE_LEGEND_FONT_SIZE)

    apply_clean_axis_style(ax, grid=False)
    ax.grid(False)

    # Keep the x-axis at the bottom of Panel B so that it aligns with
    # the lower x-axis of Panel A.  Draw a separate zero line instead of
    # moving the bottom spine to y=0.
    ax.axhline(0.0, color="#9A9A9A", lw=0.70, alpha=0.95, zorder=2)
    ax.spines["bottom"].set_position(("outward", 0))
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_linewidth(0.82)
    ax.tick_params(axis="x", pad=4.0, length=3.0, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    for tick in ax.get_xticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(TICK_LABEL_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(TICK_LABEL_SIZE)

    y_all = np.concatenate([y_dft[np.isfinite(y_dft)], y_qc[np.isfinite(y_qc)]])
    if len(y_all) > 0:
        ymin = float(np.nanmin(y_all))
        ymax = max(float(np.nanmax(y_all)), 0.10)
        lower = min(-CHARGE_TRANSFER_NEGATIVE_ROOM, ymin - 0.006)
        upper = ymax * 1.15
        ax.set_ylim(lower, upper)


def plot_charge_transfer_mulliken(state_data, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plot_charge_transfer_on_ax(ax, state_data, key_suffix="mulliken", as_bars=True)
    fig.tight_layout()
    save_png(fig, os.path.join(outdir, "figure_1_charge_transfer.png"))


def plot_charge_transfer_lowdin(state_data, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plot_charge_transfer_on_ax(ax, state_data, key_suffix="lowdin", as_bars=False)
    ax.set_ylabel(r"Slab$\rightarrow$H$_2$ transfer (Löwdin, e)")
    ax.set_title("Löwdin charge transfer along reaction coordinate", pad=6)
    fig.tight_layout()
    save_png(fig, os.path.join(outdir, "figure_1b_charge_transfer_lowdin.png"))


def plot_charge_transfer_meta_lowdin(state_data, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plot_charge_transfer_on_ax(ax, state_data, key_suffix="meta_lowdin", as_bars=False)
    ax.set_ylabel(r"Slab$\rightarrow$H$_2$ transfer (meta-Löwdin, e)")
    ax.set_title("meta-Löwdin charge transfer along reaction coordinate", pad=6)
    fig.tight_layout()
    save_png(fig, os.path.join(outdir, "figure_1c_charge_transfer_meta_lowdin.png"))


# =============================================================================
# CI and frontier plots
# =============================================================================

def _style_ci_frontier_axes(ax, grid=False, right_spine=False):
    """Unified style for the two auxiliary CI/frontier descriptor plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["left"].set_linewidth(0.90)
    ax.spines["bottom"].set_linewidth(0.90)

    if right_spine:
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color(COLORS["spine"])
        ax.spines["right"].set_linewidth(0.90)
    else:
        ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", colors=COLORS["tick"], pad=2)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.yaxis.label.set_fontweight(TEXT_WEIGHT)
    ax.title.set_color(COLORS["text"])
    ax.title.set_fontweight(TEXT_WEIGHT)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)

    if grid:
        ax.grid(True, axis="y", alpha=0.105, linewidth=0.48, color="#8D99A6")
        ax.grid(False, axis="x")
    else:
        ax.grid(False)


def plot_ci_dual_axis_on_ax(ax1, state_data, topn=20):
    """Unified CI compactness/entropy plot used by all three pathway scripts."""
    x = np.arange(len(DISPLAY_STATE_ORDER), dtype=float)
    y_top1 = get_ci_top1_series(state_data, topn=topn)
    y_entropy = get_ci_entropy_series(state_data, topn=topn)

    ax1.plot(
        x, y_top1,
        marker="o", linewidth=1.95, markersize=5.8,
        color=COLORS["top1"],
        markeredgecolor=COLORS["top1"],
        markeredgewidth=0.0,
        zorder=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(DISPLAY_STATE_ORDER)
    ax1.set_ylabel("Top1 CI weight", color=COLORS["top1"], fontweight=TEXT_WEIGHT)
    ax1.tick_params(axis="y", labelcolor=COLORS["top1"])
    ax1.tick_params(axis="x", colors=COLORS["tick"])
    ax1.set_xlim(-0.10, len(DISPLAY_STATE_ORDER) - 0.90)
    ax1.set_title("CI compactness and complexity", pad=6, fontweight=TEXT_WEIGHT)

    if np.any(np.isfinite(y_top1)):
        pad = max(0.0025, 0.12 * (np.nanmax(y_top1) - np.nanmin(y_top1) + 1e-8))
        ax1.set_ylim(np.nanmin(y_top1) - pad, np.nanmax(y_top1) + pad)

    ax2 = ax1.twinx()
    ax2.plot(
        x, y_entropy,
        marker="s", linewidth=1.95, markersize=5.6,
        color=COLORS["ci_entropy"],
        markeredgecolor=COLORS["ci_entropy"],
        markeredgewidth=0.0,
        zorder=3,
    )
    ax2.set_ylabel(r"CI Shannon entropy, $S_{\mathrm{CI}}$", color=COLORS["ci_entropy"], fontweight=TEXT_WEIGHT)
    ax2.tick_params(axis="y", labelcolor=COLORS["ci_entropy"])

    if np.any(np.isfinite(y_entropy)):
        pad2 = max(0.010, 0.12 * (np.nanmax(y_entropy) - np.nanmin(y_entropy) + 1e-8))
        ax2.set_ylim(np.nanmin(y_entropy) - pad2, np.nanmax(y_entropy) + pad2)

    _style_ci_frontier_axes(ax1, grid=False, right_spine=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["spine"])
    ax2.spines["right"].set_linewidth(0.90)
    ax2.tick_params(axis="y", labelcolor=COLORS["ci_entropy"], colors=COLORS["ci_entropy"], pad=2)
    ax2.yaxis.label.set_fontweight(TEXT_WEIGHT)
    for tick in ax2.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)

    return ax1, ax2


def plot_ci_dual_axis_standalone(state_data, outdir, topn=20):
    # Save the original numbered filename and the shorter filename requested
    # for direct manuscript/supplementary use.  Both use the same style.
    for name in [
        "figure_2_ci_compactness_entropy_dual_axis.png",
        "ci_compactness_entropy_dual_axis.png",
    ]:
        fig, ax = plt.subplots(figsize=(6.30, 4.20))
        plot_ci_dual_axis_on_ax(ax, state_data, topn=topn)
        fig.tight_layout()
        save_png(fig, os.path.join(outdir, name))


def plot_frontier_component_evolution(state_data, outdir, topk=4):
    """Unified frontier-descriptor evolution plot used by all three pathway scripts."""
    x = np.arange(len(DISPLAY_STATE_ORDER), dtype=float)
    y_ads, y_slab = [], []
    for s in DISPLAY_STATE_ORDER:
        comp = weighted_frontier_component_summary(state_data[s], topk=topk)
        y_ads.append(comp["adsorbate_contribution"])
        y_slab.append(comp["slab_contribution"])

    for name in [
        "figure_3_frontier_descriptor_evolution.png",
        "frontier_descriptor_evolution.png",
    ]:
        fig, ax = plt.subplots(figsize=(6.90, 4.70))
        ax.plot(
            x, y_ads,
            marker="o", linewidth=2.0, markersize=5.4,
            color=COLORS["adsorbate"],
            label="Adsorbate contribution",
            zorder=3,
        )
        ax.plot(
            x, y_slab,
            marker="o", linewidth=2.0, markersize=5.4,
            color=COLORS["slab"],
            label="Cu slab contribution",
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_STATE_ORDER)
        ax.set_ylabel("Weighted average over frontier top4", fontweight=TEXT_WEIGHT)
        ax.set_title("Evolution of frontier orbital descriptors", pad=6, fontweight=TEXT_WEIGHT)
        leg = ax.legend(frameon=False, loc="upper left", handlelength=1.7, borderpad=0.15, labelspacing=0.35)
        for txt in leg.get_texts():
            txt.set_fontweight(TEXT_WEIGHT)

        _style_ci_frontier_axes(ax, grid=True, right_spine=False)
        fig.tight_layout()
        save_png(fig, os.path.join(outdir, name))

# =============================================================================
# Top-view charge maps
# =============================================================================

def plot_ts_charge_maps_on_axes(ax_dft, ax_qc, cax, state_data, raw_xyz_map, minimal=True):
    xyz_cache = prepare_display_xyz_cache(raw_xyz_map)
    labels, coords = xyz_cache.get("TS", (None, None))
    if labels is None or coords is None:
        raise RuntimeError("Cannot draw TS charge maps because TS xyz could not be loaded.")

    norm = get_charge_norm_for_ts_maps(state_data)
    if norm is None:
        raise RuntimeError("Cannot draw TS charge maps because atomic charge arrays are unavailable.")

    charges_dft = state_data["TS"].get("emb_rks_mulliken_atomic_charges")
    charges_qc = state_data["TS"].get("casscf_mulliken_atomic_charges")
    if charges_dft is None or charges_qc is None:
        raise RuntimeError("Missing emb-DFT or emb-QC TS Mulliken atomic charges.")

    charges_dft = np.array(charges_dft, dtype=float)
    charges_qc = np.array(charges_qc, dtype=float)
    if len(charges_dft) != len(labels) or len(charges_qc) != len(labels):
        raise RuntimeError("Charge length mismatch for TS maps.")

    draw_topview_charge_map_on_ax(ax_dft, labels, coords, charges_dft, norm,
                                  method_label="emb-DFT", minimal=minimal,
                                  padding_scale=TOPVIEW_PADDING_SCALE)
    draw_topview_charge_map_on_ax(ax_qc, labels, coords, charges_qc, norm,
                                  method_label="emb-QC", minimal=minimal,
                                  padding_scale=TOPVIEW_PADDING_SCALE)

    sm = ScalarMappable(norm=norm, cmap=SLAB_CMAP)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=6.8, pad=1.2, length=2.0, width=0.5)
    cbar.outline.set_linewidth(0.50)
    cbar.outline.set_edgecolor("#666666")


def plot_ts_charge_maps_standalone(state_data, raw_xyz_map, outdir):
    fig = plt.figure(figsize=(7.6, 3.7))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.10, figure=fig)
    ax_dft = fig.add_subplot(gs[0, 0])
    ax_qc = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    plot_ts_charge_maps_on_axes(ax_dft, ax_qc, cax, state_data, raw_xyz_map, minimal=True)
    ax_dft.text(-0.10, 1.08, "TS charge maps", transform=ax_dft.transAxes,
                ha="left", va="top", fontsize=MAP_TITLE_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["text"], clip_on=False)
    save_png(fig, os.path.join(outdir, "figure_4_ts_charge_maps.png"))


def plot_all_state_topview_charge_maps(state_data, raw_xyz_map, outdir):
    xyz_cache = prepare_display_xyz_cache(raw_xyz_map)
    norm = get_charge_norm_for_all_state_maps(state_data)
    if norm is None:
        raise RuntimeError("Cannot draw all-state top-view charge maps because atomic charges are unavailable.")

    fig = plt.figure(figsize=TOPVIEW_ALL_STATE_FIGSIZE)
    gs = GridSpec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.045],
                  wspace=0.14, hspace=0.22, figure=fig)

    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
    cax = fig.add_subplot(gs[:, 3])

    row_meta = [
        ("emb-QC", "casscf_mulliken_atomic_charges"),
        ("emb-DFT", "emb_rks_mulliken_atomic_charges"),
    ]

    for j, display_state in enumerate(DISPLAY_STATE_ORDER):
        labels, coords = xyz_cache.get(display_state, (None, None))
        if labels is None or coords is None:
            raise RuntimeError(f"Cannot draw top-view charge maps: missing xyz for displayed {display_state}.")

        for i, (row_label, charge_key) in enumerate(row_meta):
            charges = state_data[display_state].get(charge_key)
            if charges is None:
                raise RuntimeError(f"Missing {row_label} charges for displayed {display_state}; key={charge_key}")
            charges = np.array(charges, dtype=float)
            if len(charges) != len(labels):
                raise RuntimeError(
                    f"Charge length mismatch for displayed {display_state}, {row_label}: "
                    f"Natom={len(labels)}, Ncharge={len(charges)}"
                )

            ax = axes[i][j]
            draw_topview_charge_map_on_ax(
                ax, labels, coords, charges, norm,
                method_label="",
                minimal=True,
                padding_scale=TOPVIEW_PADDING_SCALE,
            )
            if i == 0:
                ax.set_title(display_state, pad=5.0, fontsize=12.0, color=COLORS["text"])

    for i, (row_label, _) in enumerate(row_meta):
        axes[i][0].text(
            -0.16, 0.50, row_label,
            transform=axes[i][0].transAxes,
            ha="right", va="center",
            rotation=90,
            fontsize=SUPTITLE_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
            clip_on=False,
        )

    sm = ScalarMappable(norm=norm, cmap=SLAB_CMAP)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=7.5, pad=1.5, length=2.2, width=0.55)
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor("#666666")
    cbar.set_label("Mulliken charge (e)", fontsize=CBAR_LABEL_SIZE, labelpad=8)

    save_png(fig, os.path.join(outdir, "figure_4_topview_charge_maps_all_states.png"))


def add_ts_charge_maps_inset_to_charge_axis(ax_parent, state_data, raw_xyz_map):
    bg = ax_parent.inset_axes(CHARGE_MAPS_BG_BBOX, zorder=8)
    bg.set_facecolor((1.0, 1.0, 1.0, 0.82))
    for side in ["top", "right", "bottom", "left"]:
        bg.spines[side].set_color("#E5E5E5")
        bg.spines[side].set_linewidth(0.55)
    bg.set_xticks([])
    bg.set_yticks([])

    ax_dft = bg.inset_axes(CHARGE_MAP_RKS_BBOX, zorder=10)
    ax_qc = bg.inset_axes(CHARGE_MAP_QC_BBOX, zorder=10)
    cax = bg.inset_axes(CHARGE_MAP_CBAR_BBOX, zorder=10)

    try:
        plot_ts_charge_maps_on_axes(ax_dft, ax_qc, cax, state_data, raw_xyz_map, minimal=True)

        # Keep method labels inside the white inset box.  The map axes are
        # slightly lowered above, and labels are drawn on the background axes.
        ax_dft.set_title("")
        ax_qc.set_title("")
        bg.text(
            CHARGE_MAP_RKS_BBOX[0] + 0.5 * CHARGE_MAP_RKS_BBOX[2],
            0.955,
            "emb-DFT",
            transform=bg.transAxes,
            ha="center",
            va="top",
            fontsize=INSET_LABEL_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
            clip_on=True,
        )
        bg.text(
            CHARGE_MAP_QC_BBOX[0] + 0.5 * CHARGE_MAP_QC_BBOX[2],
            0.955,
            "emb-QC",
            transform=bg.transAxes,
            ha="center",
            va="top",
            fontsize=INSET_LABEL_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
            clip_on=True,
        )
        cax.set_ylabel("")
        cax.set_title("")
        cax.tick_params(labelsize=5.8, pad=0.6, length=1.6, width=0.45)
    except Exception as exc:
        for a in [ax_dft, ax_qc, cax]:
            a.set_visible(False)
        bg.text(0.50, 0.48, "charge maps unavailable", transform=bg.transAxes,
                ha="center", va="center", fontsize=INSET_LABEL_SIZE, fontweight=TEXT_WEIGHT, color="#666666")
        log_warn(f"Could not add TS charge-map inset: {exc}")
    return bg


def _density_signed_levels(vmax, linthresh, n_each_side=26):
    magnitudes = np.geomspace(linthresh, vmax, n_each_side)
    return np.concatenate((-magnitudes[::-1], [0.0], magnitudes))


def _load_promolecular_density_inset_data(
    plane_path=PROMOLECULAR_DENSITY_PLANE_FILE,
    summary_path=PROMOLECULAR_DENSITY_SUMMARY_FILE,
):
    """Load the already-computed TS density grid used by the inset.

    No PySCF density evaluation is performed here.  We only read the saved
    numerical plane and metadata from the previous deformation-density analysis.
    """
    plane_path = os.path.abspath(os.fspath(plane_path))
    summary_path = os.path.abspath(os.fspath(summary_path))
    if not os.path.isfile(plane_path):
        raise FileNotFoundError(plane_path)

    raw = np.load(plane_path)
    x = np.asarray(raw["x_angstrom"], dtype=float)
    z = np.asarray(raw["z_angstrom"], dtype=float)
    deformation = {
        "emb_DFT": np.asarray(raw["deformation_emb_DFT"], dtype=float),
        "emb_QC": np.asarray(raw["deformation_emb_QC"], dtype=float),
    }

    summary = {}
    if os.path.isfile(summary_path):
        summary = load_json(summary_path)

    plotting = summary.get("plotting", {}) if isinstance(summary, dict) else {}
    vmax = safe_float(plotting.get("common_vmax_e_bohr-3"), np.nan)
    linthresh = safe_float(plotting.get("symlog_linear_threshold_e_bohr-3"), np.nan)

    # Robust fallback identical in spirit to the original deformation-density
    # plotting script if the summary JSON is unavailable.
    if not np.isfinite(vmax) or vmax <= 0:
        absolute = np.concatenate([
            np.abs(deformation["emb_DFT"]).ravel(),
            np.abs(deformation["emb_QC"]).ravel(),
        ])
        positive = absolute[absolute > 1.0e-12]
        if positive.size == 0:
            raise RuntimeError("Promolecular deformation density is identically zero")
        vmax = float(np.percentile(positive, 99.35))
    if not np.isfinite(linthresh) or linthresh <= 0 or linthresh >= vmax:
        linthresh = max(0.015 * vmax, 1.0e-6)

    atoms = []
    cutoff = 1.45
    if isinstance(summary, dict):
        atoms = summary.get("projected_atoms", []) or []
        plane_info = summary.get("plane", {}) or {}
        cutoff = safe_float(
            summary.get("command_line_options", {}).get("atom_plane_cutoff"),
            cutoff,
        ) if isinstance(summary.get("command_line_options", {}), dict) else cutoff

    return x, z, deformation, atoms, float(cutoff), float(vmax), float(linthresh)


def _add_density_atoms_inset(ax, atoms, cutoff):
    """Draw the projected H/Cu markers with the same geometry as the source map."""
    atom_colors = {"H": "#F5F5F5", "CU": "#B87333"}
    atom_sizes = {"H": 18.0, "CU": 24.0}
    for atom in atoms:
        symbol = str(atom.get("symbol", "")).upper()
        if symbol not in atom_colors:
            continue
        out_of_plane = abs(safe_float(atom.get("out_of_plane_angstrom"), np.inf))
        if not np.isfinite(out_of_plane) or out_of_plane > cutoff:
            continue
        alpha = max(0.38, 1.0 - 0.50 * out_of_plane / max(cutoff, 1e-12))
        ax.scatter(
            safe_float(atom.get("x_angstrom")),
            safe_float(atom.get("z_angstrom")),
            s=atom_sizes[symbol],
            c=atom_colors[symbol],
            edgecolors="#252525",
            linewidths=0.45,
            alpha=alpha,
            zorder=10,
            clip_on=True,
        )


def add_promolecular_density_inset_to_charge_axis(ax_parent):
    """Draw the TS promolecular deformation-density inset natively.

    This deliberately removes the standalone figure title
    ("Density rearrangement ...") and the red/blue footer.  Only the two method
    labels, axes and the common density colorbar are retained.  All inset text
    is bold and uses the manuscript font hierarchy.
    """
    bg = ax_parent.inset_axes(PROMOLECULAR_DENSITY_INSET_BBOX, zorder=9)
    bg.set_facecolor("white")
    bg.set_xticks([])
    bg.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        bg.spines[side].set_visible(PROMOLECULAR_DENSITY_INSET_BORDER)
        if PROMOLECULAR_DENSITY_INSET_BORDER:
            bg.spines[side].set_color("#D9D9D9")
            bg.spines[side].set_linewidth(0.45)

    # Layout inside the white inset box.  The inset is kept anchored at the same
    # right edge but widened toward the left for better readability while remaining
    # high enough to avoid excessive overlap with the tall IS/TS bars.
    # Bottom room is still reserved for the two Distance labels.
    ax_dft = bg.inset_axes([0.010, 0.185, 0.400, 0.730], zorder=10)
    ax_qc  = bg.inset_axes([0.450, 0.185, 0.400, 0.730], zorder=10)
    cax    = bg.inset_axes([0.870, 0.220, 0.032, 0.665], zorder=10)

    try:
        x, z, deformation, atoms, atom_cutoff, vmax, linthresh = _load_promolecular_density_inset_data()
        norm = SymLogNorm(
            linthresh=linthresh,
            linscale=0.75,
            vmin=-vmax,
            vmax=vmax,
            base=10,
        )
        levels = _density_signed_levels(vmax, linthresh, n_each_side=26)
        contour_abs = np.geomspace(linthresh * 2.0, vmax * 0.65, 4)

        last = None
        for iax, key, method_label in (
            (ax_dft, "emb_DFT", "emb-DFT"),
            (ax_qc, "emb_QC", "emb-QC"),
        ):
            field = np.asarray(deformation[key], dtype=float)
            clipped = np.clip(field, -vmax, vmax)
            last = iax.contourf(
                x, z, clipped,
                levels=levels,
                cmap=DENSITY_CMAP,
                norm=norm,
                extend="both",
                antialiased=True,
            )
            iax.contour(
                x, z, field,
                levels=contour_abs,
                colors=DENSITY_POSITIVE_CONTOUR,
                linewidths=0.42,
                alpha=0.78,
            )
            iax.contour(
                x, z, field,
                levels=-contour_abs[::-1],
                colors=DENSITY_NEGATIVE_CONTOUR,
                linewidths=0.42,
                linestyles="dashed",
                alpha=0.78,
            )
            _add_density_atoms_inset(iax, atoms, atom_cutoff)

            iax.set_xlim(float(x[0]), float(x[-1]))
            iax.set_ylim(float(z[0]), float(z[-1]))
            iax.set_aspect("equal")
            iax.set_title(
                method_label,
                fontsize=DENSITY_INSET_METHOD_SIZE,
                fontweight=TEXT_WEIGHT,
                color=COLORS["text"],
                pad=1.8,
            )
            iax.set_xlabel(
                r"Distance ($\AA$)",
                fontsize=DENSITY_INSET_AXIS_LABEL_SIZE,
                fontweight=TEXT_WEIGHT,
                labelpad=1.2,
                color=COLORS["text"],
            )
            iax.tick_params(
                axis="both",
                labelsize=DENSITY_INSET_TICK_SIZE,
                width=0.58,
                length=2.3,
                pad=1.0,
                colors=COLORS["tick"],
            )
            for tick in iax.get_xticklabels() + iax.get_yticklabels():
                tick.set_fontweight(TEXT_WEIGHT)
            for side in ["top", "right", "bottom", "left"]:
                iax.spines[side].set_linewidth(0.62)
                iax.spines[side].set_color(COLORS["spine"])
            iax.grid(False)

        ax_dft.set_ylabel(
            r"Surface-normal ($\AA$)",
            fontsize=DENSITY_INSET_AXIS_LABEL_SIZE,
            fontweight=TEXT_WEIGHT,
            labelpad=1.6,
            color=COLORS["text"],
        )
        # Same geometry scale, but suppress repeated right-panel y labels.
        ax_qc.tick_params(axis="y", labelleft=False)

        cbar = plt.colorbar(last, cax=cax, extend="both")

        # Keep the colorbar deliberately sparse.  Seven labels were too dense
        # for the small Fig. 2b inset, especially under the SymLog mapping.
        # Use only two nonzero magnitudes on each side plus zero:
        #
        #     -0.06, -0.01, 0.00, +0.01, +0.06
        #
        # for the present TS range.  This preserves the sign information while
        # keeping the labels visually separated.
        outer_tick = np.floor(vmax * 100.0 + 1.0e-12) / 100.0
        if outer_tick <= 0.0:
            outer_tick = round(vmax, 2)

        inner_tick = round(outer_tick / 6.0, 2)

        # Safety fallback for an unexpectedly small range.
        if inner_tick <= 0.0:
            inner_tick = max(vmax * 0.15, 1.0e-4)
        if inner_tick >= outer_tick:
            inner_tick = 0.5 * outer_tick

        symmetric_ticks = np.array(
            [-outer_tick, -inner_tick, 0.0, inner_tick, outer_tick],
            dtype=float,
        )

        # Set the five symmetric tick positions explicitly.
        cbar.set_ticks(symmetric_ticks)
        cbar.ax.minorticks_off()

        # Keep tick marks on the RIGHT side of the colorbar, but suppress the
        # default Matplotlib tick labels.  Default labels are anchored too close
        # to the bar; forcing them to be right-aligned makes them extend back
        # over the color scale.  Instead, draw a separate right-aligned text
        # column to the right of the colorbar.
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.tick_params(
            axis="y",
            right=True,
            left=False,
            labelright=False,
            labelleft=False,
            length=1.8,
            width=0.45,
            colors=COLORS["tick"],
        )

        density_tick_labels = [
            f"-{outer_tick:.2f}",
            f"-{inner_tick:.2f}",
            "0.00",
            f"+{inner_tick:.2f}",
            f"+{outer_tick:.2f}",
        ]

        # x is in colorbar-axes coordinates and y is in colorbar data
        # coordinates.  All labels share one fixed RIGHT edge and remain fully
        # outside the colorbar.  clip_on=False allows the text column to occupy
        # the reserved white space on the right of cax.
        yaxis_transform = cbar.ax.get_yaxis_transform()
        for yval, label in zip(symmetric_ticks, density_tick_labels):
            cbar.ax.text(
                4.35, yval, label,
                transform=yaxis_transform,
                ha="right",
                va="center",
                fontsize=DENSITY_INSET_CBAR_TICK_SIZE,
                fontweight=TEXT_WEIGHT,
                color=COLORS["tick"],
                clip_on=False,
            )
        cbar.outline.set_linewidth(0.50)
        cbar.outline.set_edgecolor(COLORS["spine"])
        cbar.set_label("")

        log_info(
            "Using native promolecular deformation-density grid for Fig. 2b inset: "
            f"{os.path.abspath(PROMOLECULAR_DENSITY_PLANE_FILE)}"
        )
    except Exception as exc:
        for a in [ax_dft, ax_qc, cax]:
            a.set_visible(False)
        bg.text(
            0.50, 0.50,
            "promolecular density inset unavailable",
            transform=bg.transAxes,
            ha="center", va="center",
            fontsize=INSET_LABEL_SIZE,
            fontweight=TEXT_WEIGHT,
            color="#666666",
        )
        log_warn(f"Could not add native promolecular deformation-density inset: {exc}")

    return bg


# =============================================================================
# Main combined figure
# =============================================================================

def plot_main_figure(state_data, raw_xyz_map, outdir):
    fig = plt.figure(figsize=(13.1, 5.05))
    gs = GridSpec(
        1, 2,
        width_ratios=[1.12, 1.00],
        wspace=0.31,
        figure=fig,
    )

    # Main-text Fig. 2(a): omit XYG3:PBE-D3BJ from the barrier comparison.
    # Other diagnostic/standalone plots retain the full ENERGY_METHODS dataset.
    ax_forward, ax_reverse = create_energy_barrier_summary_axes(
        fig, gs[0, 0], exclude_methods={"XYG3:PBE-D3BJ"}
    )
    ax_charge = fig.add_subplot(gs[0, 1])

    # Keep panel labels, but remove the large panel titles requested for Fig. 2.
    add_panel_label(ax_forward, "(a)", x=-0.10, y=1.146)

    plot_charge_transfer_on_ax(
        ax_charge,
        state_data,
        key_suffix="mulliken",
        as_bars=True,
        show_title=False,
    )
    # Main-text Fig. 2b: retain the net slab-to-H2 Mulliken charge-transfer bars,
    # but replace the old atom-resolved Mulliken inset with the independent
    # real-space deformation-density comparison relative to the same neutral-atom
    # promolecule. The complete atom-resolved Mulliken maps are still generated
    # separately for Supplementary Fig. S17.
    add_promolecular_density_inset_to_charge_axis(ax_charge)
    add_panel_label(ax_charge, "(b)", x=-0.115, y=1.065)

    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.130, top=0.865)

    # Explicit new filename for the revised Fig. 2 with the promolecular-density inset.
    fig.savefig(os.path.join(outdir, "main_figure_h2_cu_promolecular_density_inset.png"), dpi=MAIN_FIGURE_DPI, bbox_inches="tight")

    # Keep the historical aliases so downstream manuscript assembly does not break.
    # These aliases now contain the revised promolecular-density inset as well.
    fig.savefig(os.path.join(outdir, "main_figure_h2_cu_soft_embdft_embqc_balanced_fonts_bold_states.png"), dpi=MAIN_FIGURE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "main_figure_h2_cu_soft_clean_inset_embdft_embqc_balanced_fonts_bold_states.png"), dpi=MAIN_FIGURE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "main_figure_h2_cu.png"), dpi=MAIN_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    log_info(f"Saved {os.path.join(outdir, 'main_figure_h2_cu_promolecular_density_inset.png')}")
    log_info(f"Saved {os.path.join(outdir, 'main_figure_h2_cu_soft_embdft_embqc_balanced_fonts_bold_states.png')}")
    log_info(f"Saved {os.path.join(outdir, 'main_figure_h2_cu_soft_clean_inset_embdft_embqc_balanced_fonts_bold_states.png')}")
    log_info(f"Saved {os.path.join(outdir, 'main_figure_h2_cu.png')}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(PLOT_DIR)
    # [R2 harness adaptation] main-text Fig. 2 only, from collected relative inputs.
    state_data = load_remapped_state_data()
    plot_main_figure(state_data, None, PLOT_DIR)
    return

    state_data = load_remapped_state_data()
    raw_xyz_map = build_raw_xyz_map()

    log_info("Detected raw xyz files:")
    for st in RAW_STATES:
        log_info(f"  {st}: {raw_xyz_map.get(st)}")

    df_energy = make_energy_table(state_data)
    df_charge = make_charge_table(state_data)
    df_ci = make_ci_table(state_data, topn=20)
    df_frontier = make_frontier_component_summary_table(state_data, topk=4)
    df_ci_complexity = make_ci_complexity_table(state_data, topn=20)

    print("\n" + "=" * 100)
    print("Energy / structure summary (display states remapped)")
    print("=" * 100)
    print(df_energy.to_string(index=False))

    print("\n" + "=" * 100)
    print("Fragment charge summary (display states remapped)")
    print("=" * 100)
    print(df_charge.to_string(index=False))

    print("\n" + "=" * 100)
    print("CI complexity summary (top20, display states remapped)")
    print("=" * 100)
    print(df_ci_complexity.to_string(index=False))

    for display_state in DISPLAY_STATE_ORDER:
        print("\n" + "=" * 100)
        print(f"Active orbital AO composition: displayed {display_state} (source {DISPLAY_TO_SOURCE[display_state]})")
        print("=" * 100)
        print(summarize_active_orbitals(state_data[display_state], topn=12).to_string(index=False))

    # Save summary tables.
    df_energy.to_csv("summary_energy.csv", index=False)
    df_charge.to_csv("summary_fragment_charges.csv", index=False)
    df_ci.to_csv("summary_top_ci_top20.csv", index=False)
    df_frontier.to_csv("summary_frontier_components_top4.csv", index=False)
    df_ci_complexity.to_csv("summary_ci_complexity_top20.csv", index=False)

    with pd.ExcelWriter("summary_tables.xlsx") as writer:
        df_energy.to_excel(writer, sheet_name="energy", index=False)
        df_charge.to_excel(writer, sheet_name="fragment_charge", index=False)
        df_ci.to_excel(writer, sheet_name="top_ci_top20", index=False)
        df_frontier.to_excel(writer, sheet_name="frontier_comp_top4", index=False)
        df_ci_complexity.to_excel(writer, sheet_name="ci_complexity_top20", index=False)
        for display_state in DISPLAY_STATE_ORDER:
            summarize_active_orbitals(state_data[display_state], topn=12).to_excel(
                writer,
                sheet_name=f"{display_state}_active_orb",
                index=False,
            )

    # All previous PNG plots + new top-view summary.
    plot_h2_barriers_standalone(PLOT_DIR)
    plot_charge_transfer_mulliken(state_data, PLOT_DIR)
    plot_charge_transfer_lowdin(state_data, PLOT_DIR)
    plot_charge_transfer_meta_lowdin(state_data, PLOT_DIR)
    plot_ci_dual_axis_standalone(state_data, PLOT_DIR, topn=20)
    plot_frontier_component_evolution(state_data, PLOT_DIR, topk=4)
    plot_ts_charge_maps_standalone(state_data, raw_xyz_map, PLOT_DIR)
    plot_all_state_topview_charge_maps(state_data, raw_xyz_map, PLOT_DIR)

    try:
        plot_main_figure(state_data, raw_xyz_map, PLOT_DIR)
    except Exception as exc:
        log_error("Failed to generate final combined main figure.")
        log_error(str(exc))
        traceback.print_exc()

    print("\nSaved files:")
    print("  summary_energy.csv")
    print("  summary_fragment_charges.csv")
    print("  summary_top_ci_top20.csv")
    print("  summary_frontier_components_top4.csv")
    print("  summary_ci_complexity_top20.csv")
    print("  summary_tables.xlsx")
    print("  plots/h2-cu.png")
    print("  plots/figure_1_charge_transfer.png")
    print("  plots/figure_1b_charge_transfer_lowdin.png")
    print("  plots/figure_1c_charge_transfer_meta_lowdin.png")
    print("  plots/figure_2_ci_compactness_entropy_dual_axis.png")
    print("  plots/ci_compactness_entropy_dual_axis.png")
    print("  plots/figure_3_frontier_descriptor_evolution.png")
    print("  plots/frontier_descriptor_evolution.png")
    print("  plots/figure_4_ts_charge_maps.png")
    print("  plots/figure_4_topview_charge_maps_all_states.png")
    print("  plots/main_figure_h2_cu_soft_embdft_embqc_balanced_fonts_bold_states.png")
    print("  plots/main_figure_h2_cu_soft_clean_inset_embdft_embqc_balanced_fonts_bold_states.png")
    print("  plots/main_figure_h2_cu.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_error("Script failed.")
        log_error(str(exc))
        traceback.print_exc()
        raise
