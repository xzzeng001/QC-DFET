#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch  # kept for backward-compatible optional inset helper
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

ANALYSIS_ROOT = "data"
STATE_ORDER = ["abv", "top", "fcc", "hcp", "bridge"]
PAIR_ORDER = [("abv", "top"), ("top", "fcc"), ("fcc", "hcp"), ("hcp", "bridge"), ("abv", "bridge")]
HARTREE_TO_EV = 27.211386245988

STATE_LABELS = {
    "abv": "abv",
    "top": "top",
    "fcc": "fcc",
    "hcp": "hcp",
    "bridge": "bridge",
}

COLORS = {
    # Keep CO and H2 figures visually consistent.
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

    # Backward-compatible keys used by the original CO script.
    "rks": "#8A94A6",
    "emb_rks": "#B5E5FA",
    "emb_casscf": "#F8CBC0",
}

SLAB_CMAP = LinearSegmentedColormap.from_list(
    "soft_diverging",
    ["#4C6EDB", "#DCE6F2", "#F4F4F4", "#F3D6C6", "#C73E3A"],
)

# Unified typography controls, copied from the H2/Cu12 plotting style.
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

TEXT_WEIGHT = "bold"
AXIS_LABEL_WEIGHT = TEXT_WEIGHT
TICK_LABEL_WEIGHT = TEXT_WEIGHT
METHOD_TICK_LABEL_WEIGHT = TEXT_WEIGHT
ALL_TEXT_WEIGHT = TEXT_WEIGHT

COV_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "Cu": 1.32,
}

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
# logging helpers
# =============================================================================

def log_info(msg):
    print(f"[INFO] {msg}", flush=True)


def log_warn(msg):
    print(f"[WARN] {msg}", flush=True)


def log_error(msg):
    print(f"[ERROR] {msg}", flush=True)


# =============================================================================
# IO
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def state_json_path(state):
    return os.path.join(ANALYSIS_ROOT, state, f"{state}_analysis.json")


def pair_json_path(a, b):
    return os.path.join(ANALYSIS_ROOT, "pairwise", f"{a}_to_{b}_pairwise.json")


def load_state_data():
    data = {}
    for st in STATE_ORDER:
        path = state_json_path(st)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        data[st] = load_json(path)
    return data


def load_pair_data():
    data = {}
    for a, b in PAIR_ORDER:
        path = pair_json_path(a, b)
        if os.path.exists(path):
            data[f"{a}-{b}"] = load_json(path)
        else:
            log_warn(f"Pairwise file not found: {path}")
    return data


# =============================================================================
# helpers
# =============================================================================

def rel_to_first(vals):
    arr = np.array(vals, dtype=float)
    return arr - arr[0]


def hartree_to_ev(vals):
    return np.array(vals, dtype=float) * HARTREE_TO_EV


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def top1_info(state_dict):
    dom = state_dict.get("dominant_ci", [])
    if dom:
        x = dom[0]
        sig = f"a:{x.get('alpha_occ_str')} | b:{x.get('beta_occ_str')}"
        return sig, safe_float(x.get("weight")), safe_float(x.get("coeff"))
    return None, np.nan, np.nan


def summarize_active_orbitals(state_dict, topn=12):
    arr = state_dict.get("active_orbital_ao_composition", [])
    rows = []
    for item in arr[:topn]:
        rows.append({
            "orbital_index": item.get("orbital_index"),
            "adsorbate_contribution": item.get("adsorbate_contribution"),
            "substrate_contribution": item.get("substrate_contribution"),
            "sigma_like_score": item.get("sigma_like_score"),
            "pi_like_score": item.get("pi_like_score"),
            "local_s_like_score": item.get("local_s_like_score"),
            "local_p_total_score": item.get("local_p_total_score"),
        })
    return pd.DataFrame(rows)


def parse_xyz_coords(path):
    if path is None:
        return None, None
    if not os.path.exists(path):
        log_warn(f"XYZ file does not exist: {path}")
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [x.rstrip("\n") for x in f]
    except Exception as e:
        log_error(f"Failed to read xyz file: {path}")
        log_error(str(e))
        return None, None

    if len(lines) < 3:
        log_warn(f"XYZ file too short: {path}")
        return None, None

    try:
        natm = int(lines[0].strip())
    except Exception:
        log_warn(f"Failed to parse atom count from xyz: {path}")
        return None, None

    coords = []
    labels = []
    for i in range(2, min(2 + natm, len(lines))):
        toks = lines[i].split()
        if len(toks) < 4:
            continue
        sym = toks[0]
        x, y, z = map(float, toks[1:4])
        labels.append(sym)
        coords.append([x, y, z])

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


def build_state_xyz_map():
    out = {}
    for st in STATE_ORDER:
        out[st] = auto_find_xyz_for_state(st)
    return out


def get_co_substrate_atom_idxs(labels):
    return [i for i, lab in enumerate(labels) if str(lab).strip().lower() == "cu"]


def get_co_adsorbate_atom_idxs(labels):
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
                thresh = max(2.45, 1.15 * (ri + rj))

            if d <= thresh:
                bonds.append((i, j))
    return bonds


def _cluster_z_layers(zvals, tol=0.35):
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


def _estimate_packed_radius_xy(coords_xy, idxs):
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

    dmed = float(np.median(nn_dists))
    return 0.48 * dmed


def safe_savefig(path):
    try:
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log_info(f"Saved {path}")
        return True
    except Exception as e:
        log_error(f"Failed to save figure: {path}")
        log_error(str(e))
        plt.close()
        return False


def plot_state_labels():
    return [STATE_LABELS.get(s, s) for s in STATE_ORDER]


def style_line_axes(ax, right_spine=False, grid=False):
    ax.spines["top"].set_visible(False)
    if not right_spine:
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["right"].set_color(COLORS["spine"])
        ax.spines["right"].set_linewidth(0.85)

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



# =============================================================================
# main figure: CO adsorption energetics + charge transfer
# =============================================================================

# CO adsorption-site energetics after ZPE correction.
# All plotted values are intentionally rounded to two decimal places.
# bridge/fcc/hcp are relative to the top site; abv is the CO adsorption energy.
# Missing values are stored as np.nan and are therefore not drawn.
CO_ADS_SITES = ["bridge", "fcc", "hcp", "abv"]
CO_ADS_QC_DFET = [0.19, 0.30, 0.56, -0.51]
CO_ADS_XYG3_PBE_D3BJ = [0.05, 0.11, 0.13, -0.59]
CO_ADS_ECASPT2 = [0.17, 0.19, 0.17, np.nan]
CO_ADS_AFQMC_RPA = [np.nan, 0.27, np.nan, -0.48]
CO_ADS_DMC = [0.45, 0.39, 0.39, -0.61]
CO_ADS_PBE_D3BJ = [-0.14, -0.18, -0.17, -0.89]

# Reference lines used in the all-site grouped-bar figure.
# The abv column now reports the negative CO adsorption energy.
CO_ADS_BENCHMARKS = {
    "abv": (-0.55, "Exp."),
}

# Compact two-panel summary.
# abv-top now denotes the negative CO adsorption energy.
# fcc-top = E_fcc - E_top.
CO_ADS_PAIR_LABELS = ["abv-top", "fcc-top"]
CO_ADS_PAIR_DATA = {
    # ZPE-corrected values, rounded to two decimal places.
    "PBE-D3BJ": {
        "abv-top": -0.89,
        "fcc-top": -0.18,
    },
    "ECASPT2": {
        "abv-top": np.nan,
        "fcc-top": 0.19,
    },
    "DMC": {
        "abv-top": -0.61,
        "fcc-top": 0.39,
    },
    "FEMION": {
        "abv-top": -0.48,
        "fcc-top": 0.27,
    },
    "XYG3:PBE-D3BJ": {
        "abv-top": -0.59,
        "fcc-top": 0.11,
    },
    "QC-DFET": {
        "abv-top": -0.51,
        "fcc-top": 0.30,
    },
}
CO_ADS_PAIR_METHOD_ORDER = [
    "PBE-D3BJ",
    "ECASPT2",
    "DMC",
    "FEMION",
    "XYG3:PBE-D3BJ",
    "QC-DFET",
]

# Experimental reference is used only for abv-top.
CO_ADS_PAIR_EXP_REFS = {
    "abv-top": -0.55,
}
# Use the same chemical-accuracy window convention as the H2/Cu(111) figure,
# centered at the experimental reference adsorption energy.
CO_CHEMICAL_ACCURACY_EV = 0.0436
CO_ADS_EXP_RANGE = {
    "abv-top": (-0.55 - CO_CHEMICAL_ACCURACY_EV, -0.55 + CO_CHEMICAL_ACCURACY_EV),
}

# Main-figure colors: matched exactly to the H2/Cu12 figure.
COLOR_QC = COLORS["qc_dfet"]
COLOR_ECAS = COLORS["nevpt2"]
COLOR_PBE = COLORS["pbe"]
COLOR_XYG3 = COLORS["xyg3"]
COLOR_DMC = "#5F6B7A"
COLOR_FEMION = COLORS["afqmc_rpa"]
COLOR_BENCH = "#7B8794"
COLOR_EMB_RKS = COLORS["emb_dft"]
COLOR_EMB_QC = COLORS["emb_qc"]
COLOR_ZERO = "#C8D1D8"

CO_METHOD_COLORS = {
    "PBE-D3BJ": COLOR_PBE,
    "ECASPT2": COLOR_ECAS,
    "DMC": COLOR_DMC,
    "FEMION": COLOR_FEMION,
    "XYG3:PBE-D3BJ": COLOR_XYG3,
    "QC-DFET": COLOR_QC,
}

# In the compact two-panel figure, all non-QC methods use the same symbol.
CO_MARKER_DEFAULT = "D"
CO_MARKER_QC = "h"
CO_MARKER_DMC = "D"

MAIN_FIG_PNG = "co_adsorption_energy_charge_main.png"
CO_ADS_STANDALONE_PNG = "co_adsorption_all_sites_barplot_benchmark_line.png"
CO_ADS_SPLIT_STANDALONE_PNG = "co_adsorption_abv_fcc_top_split_benchmark.png"

# Charge-transfer panel inset: top-site charge maps.
# BBox values are in Panel-B axis coordinates.
TOP_CHARGE_MAPS_BG_BBOX = [0.255, 0.135, 0.585, 0.365]

# Axes inside TOP_CHARGE_MAPS_BG_BBOX coordinates.
TOP_CHARGE_MAP_RKS_BBOX  = [0.030, 0.115, 0.385, 0.790]
TOP_CHARGE_MAP_QC_BBOX   = [0.455, 0.115, 0.385, 0.790]
TOP_CHARGE_MAP_CBAR_BBOX = [0.875, 0.210, 0.040, 0.560]

# Top-view map cropping.  Increase if boundary atoms are clipped.
TOP_CHARGE_MAP_PAD_SCALE = 1.45
TOPVIEW_ALL_SITE_FIGSIZE = (14.8, 5.8)
CHARGE_TRANSFER_NEGATIVE_ROOM = 0.035


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


def _finite_ylim_from_arrays(arrays, include_zero=True, pad_frac=0.16, min_pad=0.05):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()
        vals.extend(list(a[np.isfinite(a)]))
    if include_zero:
        vals.append(0.0)

    if len(vals) == 0:
        return -1.0, 1.0

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    yr = max(ymax - ymin, 1e-8)
    pad = max(min_pad, pad_frac * yr)
    return ymin - pad, ymax + pad



def plot_co_adsorption_energy_on_ax(ax, show_title=True, show_legend=True):
    """Panel A: all available CO adsorption-site energy differences relative to top."""
    x = np.arange(len(CO_ADS_SITES), dtype=float)

    method_specs = [
        ("QC-DFET", CO_ADS_QC_DFET, COLOR_QC),
        ("XYG3:PBE-D3BJ", CO_ADS_XYG3_PBE_D3BJ, COLOR_XYG3),
        ("ECASPT2", CO_ADS_ECASPT2, COLOR_ECAS),
        ("FEMION", CO_ADS_AFQMC_RPA, COLOR_FEMION),
        ("DMC", CO_ADS_DMC, COLOR_DMC),
        ("PBE-D3BJ", CO_ADS_PBE_D3BJ, COLOR_PBE),
    ]

    n_methods = len(method_specs)
    width = 0.115
    offsets = (np.arange(n_methods) - 0.5 * (n_methods - 1)) * width

    bar_handles = []
    for i, (label, values, color) in enumerate(method_specs):
        values = np.array(values, dtype=float)
        bars = ax.bar(
            x + offsets[i],
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.50,
            alpha=0.96,
            zorder=3,
        )
        bar_handles.append(bars)

    line_half_width = 0.35
    text_offset = 0.04
    dash_pattern = (0, (3.0, 2.2))

    for i, site in enumerate(CO_ADS_SITES):
        if site not in CO_ADS_BENCHMARKS:
            continue
        y, label_txt = CO_ADS_BENCHMARKS[site]

        ax.hlines(
            y,
            x[i] - line_half_width,
            x[i] + line_half_width,
            colors=COLOR_BENCH,
            linestyles=dash_pattern,
            linewidth=1.55,
            alpha=0.86,
            zorder=5,
        )
        ax.text(
            x[i] + line_half_width + text_offset,
            y,
            label_txt,
            va="center",
            ha="left",
            fontsize=9.3,
            color=COLOR_BENCH,
        )

    ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.90, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(CO_ADS_SITES)
    ax.set_ylabel("Energy (eV)", labelpad=8)
    if show_title:
        ax.set_title("CO adsorption on Cu(111)", pad=8, fontsize=MAIN_TITLE_SIZE, fontweight=TEXT_WEIGHT)

    arrays = [values for _, values, _ in method_specs]
    y0, y1 = _finite_ylim_from_arrays(arrays + [[v[0] for v in CO_ADS_BENCHMARKS.values()]],
                                      include_zero=True, pad_frac=0.15, min_pad=0.05)
    ax.set_ylim(y0, y1)
    ax.set_xlim(-0.55, len(CO_ADS_SITES) - 0.45)

    style_line_axes(ax)
    ax.grid(True, axis="y", alpha=0.10, linewidth=0.48, color="#8D99A6")
    ax.grid(False, axis="x")

    if show_legend:
        benchmark_handle = Line2D(
            [0], [0],
            color=COLOR_BENCH,
            linestyle=dash_pattern,
            linewidth=1.65,
            label="Exp. benchmark",
        )
        handles = [benchmark_handle] + bar_handles
        labels = ["Exp. benchmark"] + [m[0] for m in method_specs]
        leg = ax.legend(
            handles=handles,
            labels=labels,
            frameon=False,
            loc="upper left",
            handlelength=1.55,
            handletextpad=0.62,
            borderaxespad=0.35,
            labelspacing=0.30,
            fontsize=LEGEND_FONT_SIZE,
        )
        for t in leg.get_texts():
            t.set_color(COLORS["text"])
            t.set_fontweight(TEXT_WEIGHT)


def plot_co_adsorption_split_abv_fcc_on_ax(ax_abv, ax_fcc):
    """H2-style two-row plot: top = abv-top, bottom = fcc-top."""
    methods = CO_ADS_PAIR_METHOD_ORDER
    x = np.arange(len(methods), dtype=float)

    def draw_pair_panel(ax, pair, title, show_xticklabels, exp_ref=None):
        vals = np.array([CO_ADS_PAIR_DATA[m].get(pair, np.nan) for m in methods], dtype=float)
        finite_vals = vals[np.isfinite(vals)]

        # Determine y range. For abv-top, start from 0.4 eV to focus on
        # the adsorption-energy region near the experimental reference. For
        # fcc-top, keep zero in the range because its sign separates fcc- and
        # top-favored predictions.
        ref_vals = []
        exp_range = CO_ADS_EXP_RANGE.get(pair)
        if exp_range is not None:
            ref_vals = [exp_range[0], exp_range[1], exp_ref]
        elif exp_ref is not None:
            ref_vals = [exp_ref]

        all_vals = np.concatenate([finite_vals, np.array(ref_vals, dtype=float), np.array([0.0])])

        if pair == "abv-top":
            # Negative adsorption-energy convention.  Use -0.3 eV as the
            # upper display boundary so the experimental range
            # (-0.55 to -0.41 eV) has enough visual room above it.
            ymin = float(np.nanmin(np.concatenate([finite_vals, np.array(ref_vals, dtype=float)]))) - 0.075
            ymax = -0.30
        else:
            ymin = min(0.0, float(np.nanmin(all_vals))) - 0.075
            ymax = float(np.nanmax(all_vals)) + 0.075

        if pair == "fcc-top":
            # fcc-top = E_fcc - E_top. The zero line is a sign separator,
            # not an experimental reference, so use a solid line and remove
            # the positive/negative background bands.
            ax.axhline(0.0, color="#5F6B76", lw=1.10, ls="-", alpha=0.88, zorder=1)
        else:
            ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.85, zorder=1)

        if exp_ref is not None:
            exp_range = CO_ADS_EXP_RANGE.get(pair)
            if exp_range is not None:
                band_low, band_high = exp_range
            else:
                band_low, band_high = exp_ref, exp_ref

            # Experimental adsorption-energy range.  No dashed reference line
            # is drawn because the benchmark is now a range rather than a
            # single fixed value.
            ax.axhspan(
                band_low,
                band_high,
                color="#DCEFE9",
                alpha=0.68,
                lw=0,
                zorder=0,
            )

        for i, method in enumerate(methods):
            val = vals[i]
            if not np.isfinite(val):
                continue
            is_qc = (method == "QC-DFET")
            is_dmc = (method == "DMC")
            # Keep all method markers at the same size.  QC-DFET is
            # distinguished only by the star shape and black edge, not by a
            # larger marker size.
            marker = CO_MARKER_QC if is_qc else CO_MARKER_DEFAULT
            size = 76
            edgecolor = "#2F2F2F" if is_qc else "white"
            linewidth = 0.85 if is_qc else 0.70
            zorder = 5 if is_qc else 3

            ax.scatter(
                i,
                val,
                s=size,
                marker=marker,
                facecolor=CO_METHOD_COLORS[method],
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.98,
                zorder=zorder,
            )

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.55, len(methods) - 0.45)
        ax.set_title(title, loc="left", pad=4.5, fontsize=SUBPANEL_TITLE_SIZE, fontweight=TEXT_WEIGHT)
        ax.set_ylabel("Energy (eV)", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
        ax.set_xticks(x)

        if show_xticklabels:
            ax.set_xticklabels(methods, rotation=0, ha="center")
            for tick, method in zip(ax.get_xticklabels(), methods):
                tick.set_color(CO_METHOD_COLORS[method])
                tick.set_fontweight("bold")
                tick.set_fontsize(METHOD_TICK_LABEL_SIZE)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        style_line_axes(ax)
        # No grid lines in the compact energy panels.
        ax.grid(False)

    draw_pair_panel(
        ax_abv,
        pair="abv-top",
        title="Adsorption energy",
        show_xticklabels=False,
        exp_ref=CO_ADS_PAIR_EXP_REFS.get("abv-top"),
    )
    draw_pair_panel(
        ax_fcc,
        pair="fcc-top",
        title="fcc-top gap",
        show_xticklabels=True,
        exp_ref=None,
    )


def plot_co_adsorption_split_abv_fcc_standalone(outdir="."):
    fig = plt.figure(figsize=(8.8, 5.0), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.24)
    ax_abv = fig.add_subplot(gs[0, 0])
    ax_fcc = fig.add_subplot(gs[1, 0])
    plot_co_adsorption_split_abv_fcc_on_ax(ax_abv, ax_fcc)
    fig.suptitle("CO adsorption on Cu(111)", y=0.985, fontsize=MAIN_TITLE_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["text"])
    # Leave a little more headroom so the centered title does not collide with the
    # upper subplot label (Adsorption energy).
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.135, top=0.835)
    out = os.path.join(outdir, CO_ADS_SPLIT_STANDALONE_PNG)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out}")

def get_charge_transfer_arrays(state_data, charge_key="mulliken", direction="slab_to_co"):
    """Return emb-DFT and emb-QC charge-transfer arrays for the CO adsorbate.

    The analysis JSON stores adsorbate fragment charges.  By default, we plot
    charge transfer from the Cu slab to CO as -q_adsorbate, so electron donation
    to the CO fragment appears as a positive bar.
    """
    emb = np.array([
        state_data[s].get(f"emb_rks_adsorbate_fragment_charge_{charge_key}", np.nan)
        for s in STATE_ORDER
    ], dtype=float)

    qc = np.array([
        state_data[s].get(f"casscf_adsorbate_fragment_charge_{charge_key}", np.nan)
        for s in STATE_ORDER
    ], dtype=float)

    if direction == "slab_to_co":
        emb = -emb
        qc = -qc

    return emb, qc


def _draw_topview_charge_map_on_ax(
    ax,
    labels,
    coords,
    charges,
    norm,
    title=None,
    padding_scale=TOP_CHARGE_MAP_PAD_SCALE,
):
    """Draw a compact CO/Cu top-view Mulliken-charge map."""
    charges = np.array(charges, dtype=float)

    ads_idxs = get_co_adsorbate_atom_idxs(labels)
    sub_idxs = get_co_substrate_atom_idxs(labels)
    bonds = estimate_bonds(labels, coords, ads_idxs, include_ads_sub=True)

    xy = coords[:, :2]
    z = coords[:, 2]

    if len(sub_idxs) > 0:
        sub_layer_id = _cluster_z_layers(z[sub_idxs], tol=0.35)
        unique_layers = sorted(np.unique(sub_layer_id))
    else:
        sub_layer_id = np.array([], dtype=int)
        unique_layers = []

    base_radius = _estimate_packed_radius_xy(xy, sub_idxs)

    # Cu substrate atoms: soft packed circles, same layered alpha logic as H2/Cu12.
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

    # Bonds involving CO and nearest Cu atoms.
    for i, j in bonds:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=(0.35, 0.35, 0.35, 0.62),
            lw=1.10,
            zorder=5,
        )

    # CO atoms.
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
    pad = padding_scale * base_radius
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.grid(False)

    if title:
        ax.set_title(title, pad=3.0, fontsize=MAP_TITLE_SIZE, fontweight=TEXT_WEIGHT, color=COLORS["text"])


def add_top_charge_maps_inset(ax, state_data, state_xyz_map, state="top"):
    """Add top-site emb-DFT / emb-QC Mulliken charge maps as an inset in Panel B."""
    xyz_path = state_xyz_map.get(state)
    if xyz_path is None:
        log_warn(f"No xyz file found for {state}; skip top charge-map inset.")
        return

    labels, coords = parse_xyz_coords(xyz_path)
    if labels is None or coords is None:
        log_warn(f"Failed to parse xyz for {state}: {xyz_path}; skip top charge-map inset.")
        return

    charges_rks = state_data[state].get("emb_rks_mulliken_atomic_charges")
    charges_qc = state_data[state].get("casscf_mulliken_atomic_charges")
    if charges_rks is None or charges_qc is None:
        log_warn(f"Missing emb-DFT or emb-QC Mulliken atomic charges for {state}; skip inset.")
        return

    charges_rks = np.array(charges_rks, dtype=float)
    charges_qc = np.array(charges_qc, dtype=float)

    if len(charges_rks) != len(labels) or len(charges_qc) != len(labels):
        log_warn(f"Charge length mismatch for top-site inset; skip inset.")
        return

    vmax = max(float(np.nanmax(np.abs(np.concatenate([charges_rks, charges_qc])))), 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    bg_bbox = TOP_CHARGE_MAPS_BG_BBOX
    bg_patch = FancyBboxPatch(
        (bg_bbox[0], bg_bbox[1]),
        bg_bbox[2],
        bg_bbox[3],
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=(1.0, 1.0, 1.0, 0.82),
        edgecolor=(0.90, 0.90, 0.90, 0.90),
        linewidth=0.55,
        zorder=9,
        clip_on=False,
    )
    ax.add_patch(bg_patch)

    bg_ax = ax.inset_axes(bg_bbox, transform=ax.transAxes, zorder=10)
    bg_ax.set_axis_off()

    ax_rks = bg_ax.inset_axes(TOP_CHARGE_MAP_RKS_BBOX)
    ax_qc = bg_ax.inset_axes(TOP_CHARGE_MAP_QC_BBOX)
    cax = bg_ax.inset_axes(TOP_CHARGE_MAP_CBAR_BBOX)

    _draw_topview_charge_map_on_ax(ax_rks, labels, coords, charges_rks, norm, title="emb-DFT")
    _draw_topview_charge_map_on_ax(ax_qc, labels, coords, charges_qc, norm, title="emb-QC")

    sm = ScalarMappable(norm=norm, cmap=SLAB_CMAP)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=7.0, length=2.0, width=0.6, pad=1.5)
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("#8A94A6")


def plot_charge_transfer_line_on_ax(
    ax,
    state_data,
    charge_key="mulliken",
    show_title=True,
    show_legend=True,
    state_xyz_map=None,
    add_top_maps=False,
):
    """Panel B: H2-style grouped bar chart for CO charge transfer."""
    x = np.arange(len(STATE_ORDER), dtype=float)
    y_emb, y_qc = get_charge_transfer_arrays(state_data, charge_key=charge_key, direction="slab_to_co")

    bar_width = 0.30
    ax.bar(
        x - bar_width / 2,
        y_emb,
        width=bar_width,
        color=COLOR_EMB_RKS,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.96,
        label="emb-DFT",
        zorder=3,
    )
    ax.bar(
        x + bar_width / 2,
        y_qc,
        width=bar_width,
        color=COLOR_EMB_QC,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.96,
        label="emb-QC",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_state_labels(), rotation=0, ha="center")
    for tick in ax.get_xticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
        tick.set_fontsize(TICK_LABEL_SIZE)

    ax.set_ylabel("Charge transfer from Cu slab to CO (e)", fontsize=AXIS_LABEL_SIZE, fontweight=TEXT_WEIGHT)
    if show_title:
        ax.set_title(
            "Charge transfer along adsorption sites",
            pad=10.0,
            fontsize=MAIN_TITLE_SIZE,
            fontweight=TEXT_WEIGHT,
        )

    finite_vals = np.concatenate([
        np.asarray(y_emb, dtype=float)[np.isfinite(y_emb)],
        np.asarray(y_qc, dtype=float)[np.isfinite(y_qc)],
    ])
    if len(finite_vals) > 0:
        ymin = float(np.nanmin(finite_vals))
        ymax = max(float(np.nanmax(finite_vals)), 0.10)
        lower = min(0.0, ymin - 0.025)
        upper = ymax * 1.15
        ax.set_ylim(lower, upper)
    ax.set_xlim(-0.45, len(STATE_ORDER) - 0.55)

    style_line_axes(ax, grid=False)
    ax.axhline(0.0, color="#9A9A9A", lw=0.70, alpha=0.95, zorder=2)
    ax.spines["bottom"].set_position(("outward", 0))
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_linewidth(0.82)
    ax.tick_params(axis="x", pad=4.0, length=3.0, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)

    if show_legend:
        leg = ax.legend(
            frameon=False,
            loc="upper left",
            handlelength=1.0,
            borderpad=0.15,
            labelspacing=0.28,
            columnspacing=0.8,
            fontsize=CHARGE_LEGEND_FONT_SIZE,
        )
        for txt in leg.get_texts():
            txt.set_fontweight(TEXT_WEIGHT)
            txt.set_fontsize(CHARGE_LEGEND_FONT_SIZE)

    # The old top-site charge-map inset is intentionally disabled for the main Panel B.


# Backward-compatible name: previous code called this function from standalone plots.
def plot_charge_transfer_bar_on_ax(ax, state_data, charge_key="mulliken", show_title=True, show_legend=True):
    return plot_charge_transfer_line_on_ax(
        ax,
        state_data,
        charge_key=charge_key,
        show_title=show_title,
        show_legend=show_legend,
        state_xyz_map=None,
        add_top_maps=False,
    )


def plot_co_adsorption_standalone(outdir="."):
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    plot_co_adsorption_energy_on_ax(ax, show_title=True, show_legend=True)
    fig.tight_layout()
    out = os.path.join(outdir, CO_ADS_STANDALONE_PNG)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out}")


def plot_main_co_figure(state_data, state_xyz_map, outdir="."):
    """Final two-panel main figure.

    Panel A: compact two-row CO adsorption-energy summary.
    Panel B: H2-style charge-transfer grouped bars, without charge-map inset.
    """
    fig = plt.figure(figsize=(13.1, 5.05))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.18, 1.00],
        wspace=0.31,
    )

    gs_left = gs[0, 0].subgridspec(
        2, 1,
        height_ratios=[1.0, 1.0],
        hspace=0.24,
    )
    ax_abv = fig.add_subplot(gs_left[0, 0])
    ax_fcc = fig.add_subplot(gs_left[1, 0])
    ax_charge = fig.add_subplot(gs[0, 1])

    plot_co_adsorption_split_abv_fcc_on_ax(ax_abv, ax_fcc)

    plot_charge_transfer_line_on_ax(
        ax_charge,
        state_data,
        charge_key="mulliken",
        show_title=False,
        show_legend=True,
        state_xyz_map=None,
        add_top_maps=False,
    )

    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.130, top=0.835)

    # Use figure coordinates for both main panel titles and labels.  This avoids
    # the previous mismatch caused by using the same axes-fraction y value on
    # axes with different heights: ax_abv is only the upper half of Panel A,
    # whereas ax_charge spans the full height of Panel B.
    pos_abv = ax_abv.get_position()
    pos_charge = ax_charge.get_position()

    # Match the H2 / quantum-error figures: panel labels are placed in
    # figure coordinates at the same vertical level as the main panel titles,
    # rather than inside the axes.  This treats the title as part of the panel
    # and keeps (a)/(b) parallel and slightly title-level.
    main_title_y = max(pos_abv.y1, pos_charge.y1) + 0.070
    panel_label_y = max(pos_abv.y1, pos_charge.y1) + 0.025
    panel_label_x_a = pos_abv.x0 - 0.105 * pos_abv.width
    panel_label_x_b = pos_charge.x0 - 0.105 * pos_charge.width
    fig.text(
        panel_label_x_a,
        panel_label_y,
        "(a)",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["text"],
    )
    fig.text(
        panel_label_x_b,
        panel_label_y,
        "(b)",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight=TEXT_WEIGHT,
        color=COLORS["text"],
    )

    # Main panel headings removed; retain (a)/(b) and subpanel labels.

    png_path = os.path.join(outdir, MAIN_FIG_PNG)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    png_path_soft = os.path.join(outdir, "co_adsorption_energy_charge_main_embdft_embqc_soft.png")
    fig.savefig(png_path_soft, dpi=300, bbox_inches="tight")

    plt.close(fig)
    log_info(f"Saved {png_path}")
    log_info(f"Saved {png_path_soft}")


# =============================================================================
# frontier component analysis
# =============================================================================

def get_active_orbital_rows(state_dict):
    rows = state_dict.get("active_orbital_ao_composition", [])
    return rows if rows is not None else []


def get_active_noons(state_dict):
    arr = state_dict.get("active_noons", [])
    arr = arr if arr is not None else []
    return [safe_float(x) for x in arr]


def build_frontier_weights(
    state_dict,
    power_ads=1.0,
    power_local_s=1.0,
    power_sigma=1.0,
    include_pi=0.15,
    normalize=True,
):
    noons = get_active_noons(state_dict)
    rows = get_active_orbital_rows(state_dict)

    n_orb = min(len(noons), len(rows))
    if n_orb == 0:
        return np.array([], dtype=float)

    w = []
    for i in range(n_orb):
        r = rows[i]
        ads = max(safe_float(r.get("adsorbate_contribution"), 0.0), 0.0)
        local_s = max(safe_float(r.get("local_s_like_score"), 0.0), 0.0)
        sigma = max(safe_float(r.get("sigma_like_score"), 0.0), 0.0)
        pi = max(safe_float(r.get("pi_like_score"), 0.0), 0.0)

        wi = (ads ** power_ads) * (local_s ** power_local_s) * ((sigma + include_pi * pi) ** power_sigma)
        w.append(wi)

    w = np.array(w, dtype=float)

    if np.all(w < 1e-15):
        w = np.ones_like(w)

    if normalize:
        sw = np.sum(w)
        if sw > 1e-15:
            w = w / sw

    return w


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
    weights = build_frontier_weights(state_dict, normalize=True)
    idx = take_topk_by_weight(weights, k=topk)

    out = []
    for i in idx:
        r = rows[i]
        out.append({
            "orbital_index": i + 1,
            "weight": weights[i],
            "NOON": noons[i],
            "adsorbate_contribution": safe_float(r.get("adsorbate_contribution")),
            "substrate_contribution": safe_float(r.get("substrate_contribution")),
            "sigma_like_score": safe_float(r.get("sigma_like_score")),
            "pi_like_score": safe_float(r.get("pi_like_score")),
            "local_s_like_score": safe_float(r.get("local_s_like_score")),
            "local_p_total_score": safe_float(r.get("local_p_total_score")),
        })
    return out


def weighted_frontier_component_summary(state_dict, topk=4):
    detail = get_frontier_detail_rows(state_dict, topk=topk)
    if len(detail) == 0:
        return {
            "adsorbate_contribution": np.nan,
            "substrate_contribution": np.nan,
            "sigma_like_score": np.nan,
            "pi_like_score": np.nan,
            "local_s_like_score": np.nan,
            "local_p_total_score": np.nan,
        }

    w = np.array([safe_float(x["weight"], 0.0) for x in detail], dtype=float)
    if np.sum(w) <= 1e-15:
        w = np.ones(len(detail), dtype=float) / len(detail)
    else:
        w = w / np.sum(w)

    out = {}
    keys = [
        "adsorbate_contribution",
        "substrate_contribution",
        "sigma_like_score",
        "pi_like_score",
        "local_s_like_score",
        "local_p_total_score",
    ]
    for k in keys:
        vals = np.array([safe_float(x[k], np.nan) for x in detail], dtype=float)
        out[k] = float(np.nansum(w * vals))
    return out


# =============================================================================
# CI complexity
# =============================================================================

def get_ci_weights(state_dict, topn=20, renormalize=True):
    dom = state_dict.get("dominant_ci", [])
    dom = dom if dom is not None else []

    if topn is not None:
        dom = dom[:topn]

    w = []
    for item in dom:
        ww = safe_float(item.get("weight"), np.nan)
        if np.isfinite(ww) and ww >= 0.0:
            w.append(ww)

    w = np.array(w, dtype=float)

    if len(w) == 0:
        return np.array([], dtype=float)

    s = np.sum(w)
    if renormalize and s > 1e-15:
        w = w / s

    return w


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
    w = get_ci_weights(state_dict, topn=topn, renormalize=True)

    if len(w) == 0:
        return {
            "CI_n_terms_used": 0,
            "CI_top1_weight": np.nan,
            "CI_top5_cum_weight": np.nan,
            "CI_top10_cum_weight": np.nan,
            "CI_top20_cum_weight": np.nan,
            "CI_entropy": np.nan,
        }

    return {
        "CI_n_terms_used": int(len(w)),
        "CI_top1_weight": cumulative_weight(w, 1),
        "CI_top5_cum_weight": cumulative_weight(w, 5),
        "CI_top10_cum_weight": cumulative_weight(w, 10),
        "CI_top20_cum_weight": cumulative_weight(w, 20),
        "CI_entropy": shannon_entropy(w),
    }


# =============================================================================
# tables
# =============================================================================

def make_energy_table(state_data):
    rows = []
    for st in STATE_ORDER:
        d = state_data[st]
        top1_sig, top1_w, top1_c = top1_info(d)
        rows.append({
            "State": st,
            "CO bond distance (A)": d.get("co_bond_distance_angstrom"),
            "C to nearest Cu (A)": d.get("adsorbate_reference_to_nearest_substrate_distance_angstrom"),
            "Nearest Cu": d.get("nearest_substrate_atom_label"),
            "E_UKS (Ha)": d.get("e_uks"),
            "E_RKS (Ha)": d.get("e_rks"),
            "E_embRKS (Ha)": d.get("e_emb_rks"),
            "E_CASSCF (Ha)": d.get("e_casscf"),
            "dE_NEVPT2 (Ha)": d.get("de_nevpt2"),
            "E_CASSCF+NEVPT2 (Ha)": d.get("e_casscf_nevpt2"),
            "N_eff": d.get("n_eff"),
            "Top1 weight": top1_w,
            "Top1 coeff": top1_c,
            "Top1 config": top1_sig,
        })
    return pd.DataFrame(rows)


def make_charge_table(state_data):
    rows = []
    for st in STATE_ORDER:
        d = state_data[st]
        rows.append({
            "State": st,
            "Ads_embRKS_Mulliken": d.get("emb_rks_adsorbate_fragment_charge_mulliken"),
            "Ads_embRKS_Lowdin": d.get("emb_rks_adsorbate_fragment_charge_lowdin"),
            "Ads_embRKS_metaLowdin": d.get("emb_rks_adsorbate_fragment_charge_meta_lowdin"),
            "Ads_CASSCF_Mulliken": d.get("casscf_adsorbate_fragment_charge_mulliken"),
            "Ads_CASSCF_Lowdin": d.get("casscf_adsorbate_fragment_charge_lowdin"),
            "Ads_CASSCF_metaLowdin": d.get("casscf_adsorbate_fragment_charge_meta_lowdin"),
            "Sub_embRKS_Mulliken": d.get("emb_rks_substrate_fragment_charge_mulliken"),
            "Sub_embRKS_Lowdin": d.get("emb_rks_substrate_fragment_charge_lowdin"),
            "Sub_embRKS_metaLowdin": d.get("emb_rks_substrate_fragment_charge_meta_lowdin"),
            "Sub_CASSCF_Mulliken": d.get("casscf_substrate_fragment_charge_mulliken"),
            "Sub_CASSCF_Lowdin": d.get("casscf_substrate_fragment_charge_lowdin"),
            "Sub_CASSCF_metaLowdin": d.get("casscf_substrate_fragment_charge_meta_lowdin"),
            "Ads_RKS_Mulliken": d.get("rks_adsorbate_fragment_charge_mulliken"),
            "Sub_RKS_Mulliken": d.get("rks_substrate_fragment_charge_mulliken"),
        })
    return pd.DataFrame(rows)


def make_noon_table(state_data):
    rows = []
    for st in STATE_ORDER:
        d = state_data[st]
        noons = d.get("active_noons", [])
        row = {"State": st}
        for i, x in enumerate(noons, start=1):
            row[f"active_NOON_{i}"] = x
        rows.append(row)
    return pd.DataFrame(rows)


def make_ci_table(state_data, topn=20):
    rows = []
    for st in STATE_ORDER:
        dom = state_data[st].get("dominant_ci", [])
        for item in dom[:topn]:
            rows.append({
                "State": st,
                "Rank": item.get("rank"),
                "alpha_occ": item.get("alpha_occ_str"),
                "beta_occ": item.get("beta_occ_str"),
                "coeff": item.get("coeff"),
                "weight": item.get("weight"),
            })
    return pd.DataFrame(rows)


def make_pairwise_table(pair_data):
    rows = []
    for key in [f"{a}-{b}" for a, b in PAIR_ORDER]:
        d = pair_data.get(key, {})
        ci = d.get("ci_overlap", {})
        dom = d.get("dominant_configuration_comparison", {})
        dd_cas = d.get("difference_density_casscf", {})
        dd_emb = d.get("difference_density_emb_rks", {})
        rows.append({
            "Pair": key,
            "Approx_CI_overlap": ci.get("overlap"),
            "Top1_same": dom.get("top1_same"),
            "Top1_sig_a": dom.get("top1_sig_a"),
            "Top1_sig_b": dom.get("top1_sig_b"),
            "d(CO bond distance)": d.get("delta_co_bond_distance_angstrom"),
            "d(C to nearest Cu distance)": d.get("delta_adsorbate_reference_to_nearest_substrate_distance_angstrom"),
            "dN_eff": d.get("delta_n_eff"),
            "dCT_Ads_embRKS_Mulliken": d.get("delta_adsorbate_fragment_charge_emb_rks_mulliken"),
            "dCT_Ads_embRKS_Lowdin": d.get("delta_adsorbate_fragment_charge_emb_rks_lowdin"),
            "dCT_Ads_embRKS_metaLowdin": d.get("delta_adsorbate_fragment_charge_emb_rks_meta_lowdin"),
            "dCT_Ads_CASSCF_Mulliken": d.get("delta_adsorbate_fragment_charge_casscf_mulliken"),
            "dCT_Ads_CASSCF_Lowdin": d.get("delta_adsorbate_fragment_charge_casscf_lowdin"),
            "dCT_Ads_CASSCF_metaLowdin": d.get("delta_adsorbate_fragment_charge_casscf_meta_lowdin"),
            "dCT_Ads_RKS_Mulliken": d.get("delta_adsorbate_fragment_charge_rks_mulliken"),
            "diffdens_CASSCF_abs": dd_cas.get("integral_abs_drho"),
            "diffdens_embRKS_abs": dd_emb.get("integral_abs_drho"),
        })
    return pd.DataFrame(rows)


def make_atomic_charge_table(state_data, key_name):
    rows = []
    for st in STATE_ORDER:
        d = state_data[st]
        charges = d.get(key_name, [])
        row = {"State": st}
        labels = d.get("atom_labels", [f"A{i+1}" for i in range(len(charges))])
        for lab, chg in zip(labels, charges):
            row[lab] = chg
        rows.append(row)
    return pd.DataFrame(rows)


def make_frontier_component_summary_table(state_data, topk=4):
    rows = []
    for st in STATE_ORDER:
        comp = weighted_frontier_component_summary(state_data[st], topk=topk)
        row = {"State": st}
        row.update(comp)
        rows.append(row)
    return pd.DataFrame(rows)


def make_ci_complexity_table(state_data, topn=20):
    rows = []
    for st in STATE_ORDER:
        m = ci_complexity_metrics(state_data[st], topn=topn)
        row = {"State": st}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# plotting: match analyze_current_results (4).py
# =============================================================================

def plot_charge_transfer(state_data, outdir):
    """Standalone charge-transfer figure for Supporting Information.

    This uses the same bar-chart style as the main figure, without embedding any
    charge-map insets. Other diagnostic figures remain unchanged.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plot_charge_transfer_line_on_ax(
        ax,
        state_data,
        charge_key="mulliken",
        show_title=True,
        show_legend=True,
        state_xyz_map=None,
        add_top_maps=False,
    )
    fig.tight_layout()
    out = os.path.join(outdir, "figure_1_charge_transfer.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out}")


def plot_ci_dual_axis(state_data, outdir, topn=20):
    x = np.arange(len(STATE_ORDER))

    y_top1 = []
    y_entropy = []
    for s in STATE_ORDER:
        m = ci_complexity_metrics(state_data[s], topn=topn)
        y_top1.append(m["CI_top1_weight"])
        y_entropy.append(m["CI_entropy"])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))

    ax1.plot(
        x, y_top1,
        marker="o", linewidth=2.2, markersize=6,
        color=COLORS["top1"]
    )
    ax1.set_ylabel("Top1 CI weight", color=COLORS["top1"])
    ax1.tick_params(axis="y", labelcolor=COLORS["top1"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_state_labels())
    style_line_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        x, y_entropy,
        marker="s", linewidth=2.2, markersize=6,
        color=COLORS["ci_entropy"]
    )
    ax2.set_ylabel(r"CI Shannon entropy, $S_{\mathrm{CI}}$", color=COLORS["ci_entropy"])
    ax2.tick_params(axis="y", labelcolor=COLORS["ci_entropy"])
    style_line_axes(ax2, right_spine=True)

    plt.title("CI compactness and complexity")
    safe_savefig(os.path.join(outdir, "figure_2_ci_compactness_entropy.png"))


def plot_frontier_component_evolution(state_data, outdir, topk=4):
    x = np.arange(len(STATE_ORDER))

    y_ads = []
    y_slab = []
    for s in STATE_ORDER:
        comp = weighted_frontier_component_summary(state_data[s], topk=topk)
        y_ads.append(comp["adsorbate_contribution"])
        y_slab.append(comp["substrate_contribution"])

    plt.figure(figsize=(6.9, 4.7))
    plt.plot(
        x, y_ads,
        marker="o", linewidth=2.2, markersize=6,
        color=COLORS["adsorbate"], label="Adsorbate contribution"
    )
    plt.plot(
        x, y_slab,
        marker="o", linewidth=2.2, markersize=6,
        color=COLORS["slab"], label="Cu slab contribution"
    )

    plt.xticks(x, plot_state_labels())
    plt.ylabel("Weighted average over frontier top4")
    plt.title("Evolution of frontier orbital descriptors")
    plt.legend(frameon=False, handlelength=2.4)

    ax = plt.gca()
    style_line_axes(ax)

    safe_savefig(os.path.join(outdir, "figure_3_frontier_descriptor_evolution.png"))


# =============================================================================
# B-type top-view charge maps
# =============================================================================

def get_charge_norm_for_all_site_maps(state_data):
    vals = []
    for st in STATE_ORDER:
        for key in ["emb_rks_mulliken_atomic_charges", "casscf_mulliken_atomic_charges"]:
            charges = state_data[st].get(key)
            if charges is None:
                continue
            arr = np.array(charges, dtype=float)
            if arr.size > 0:
                vals.extend(list(np.abs(arr)))
    if len(vals) == 0:
        return None
    vmax = max(float(np.nanmax(vals)), 1e-8)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_all_site_topview_charge_maps(state_data, state_xyz_map, outdir):
    """All CO adsorption sites in one H2-style top-view summary figure.

    Row 1 = emb-QC; row 2 = emb-DFT.
    Columns = abv, top, fcc, hcp, bridge.
    """
    norm = get_charge_norm_for_all_site_maps(state_data)
    if norm is None:
        raise RuntimeError("Cannot draw all-site top-view charge maps because atomic charges are unavailable.")

    xyz_cache = {}
    for st in STATE_ORDER:
        xyz_path = state_xyz_map.get(st)
        labels, coords = parse_xyz_coords(xyz_path) if xyz_path else (None, None)
        if labels is None or coords is None:
            raise RuntimeError(f"Cannot draw top-view charge maps: missing xyz for {st}.")
        xyz_cache[st] = (labels, coords)

    fig = plt.figure(figsize=TOPVIEW_ALL_SITE_FIGSIZE)
    gs = fig.add_gridspec(
        2, len(STATE_ORDER) + 1,
        width_ratios=[1.0] * len(STATE_ORDER) + [0.045],
        wspace=0.12,
        hspace=0.20,
    )

    axes = [[fig.add_subplot(gs[i, j]) for j in range(len(STATE_ORDER))] for i in range(2)]
    cax = fig.add_subplot(gs[:, -1])

    row_meta = [
        ("emb-QC", "casscf_mulliken_atomic_charges"),
        ("emb-DFT", "emb_rks_mulliken_atomic_charges"),
    ]

    for j, st in enumerate(STATE_ORDER):
        labels, coords = xyz_cache[st]
        for i, (row_label, charge_key) in enumerate(row_meta):
            charges = state_data[st].get(charge_key)
            if charges is None:
                raise RuntimeError(f"Missing {row_label} charges for {st}; key={charge_key}")
            charges = np.array(charges, dtype=float)
            if len(charges) != len(labels):
                raise RuntimeError(
                    f"Charge length mismatch for {st}, {row_label}: "
                    f"Natom={len(labels)}, Ncharge={len(charges)}"
                )

            ax = axes[i][j]
            _draw_topview_charge_map_on_ax(
                ax,
                labels,
                coords,
                charges,
                norm,
                title=None,
                padding_scale=TOP_CHARGE_MAP_PAD_SCALE,
            )
            if i == 0:
                ax.set_title(
                    STATE_LABELS.get(st, st),
                    pad=5.0,
                    fontsize=12.0,
                    fontweight=TEXT_WEIGHT,
                    color=COLORS["text"],
                )

    for i, (row_label, _) in enumerate(row_meta):
        axes[i][0].text(
            -0.16,
            0.50,
            row_label,
            transform=axes[i][0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=SUPTITLE_SIZE,
            fontweight=TEXT_WEIGHT,
            color=COLORS["text"],
            clip_on=False,
        )

    sm = ScalarMappable(norm=norm, cmap=SLAB_CMAP)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, pad=1.5, length=2.2, width=0.55)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight(TEXT_WEIGHT)
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor("#666666")
    cbar.set_label("Mulliken charge (e)", fontsize=CBAR_LABEL_SIZE, fontweight=TEXT_WEIGHT, labelpad=8)

    out = os.path.join(outdir, "figure_4_topview_charge_maps_all_sites.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out}")


# Backward-compatible function name used by main().
def plot_topview_charge_bubble_maps(state_data, state_xyz_map, outdir):
    plot_all_site_topview_charge_maps(state_data, state_xyz_map, outdir)

# =============================================================================
# main
# =============================================================================

def main():
    ensure_dir("plots")
    # [R2 harness adaptation] main-text Fig. 4 only, from collected relative inputs.
    state_data = load_state_data()
    ensure_dir("output")
    plot_main_co_figure(state_data, None, "output")
    return

    state_data = load_state_data()
    pair_data = load_pair_data()
    state_xyz_map = build_state_xyz_map()

    log_info("Detected xyz files:")
    for st in STATE_ORDER:
        log_info(f"  {st}: {state_xyz_map[st]}")

    df_energy = make_energy_table(state_data)
    df_charge = make_charge_table(state_data)
    df_noon = make_noon_table(state_data)
    df_pair = make_pairwise_table(pair_data)
    df_ci = make_ci_table(state_data, topn=20)
    df_frontier = make_frontier_component_summary_table(state_data, topk=4)
    df_ci_complexity = make_ci_complexity_table(state_data, topn=20)

    df_emb_mull = make_atomic_charge_table(state_data, "emb_rks_mulliken_atomic_charges")
    df_emb_lowd = make_atomic_charge_table(state_data, "emb_rks_lowdin_atomic_charges")
    df_emb_meta = make_atomic_charge_table(state_data, "emb_rks_meta_lowdin_atomic_charges")

    df_cas_mull = make_atomic_charge_table(state_data, "casscf_mulliken_atomic_charges")
    df_cas_lowd = make_atomic_charge_table(state_data, "casscf_lowdin_atomic_charges")
    df_cas_meta = make_atomic_charge_table(state_data, "casscf_meta_lowdin_atomic_charges")

    print("\n" + "=" * 100)
    print("Energy / structure summary")
    print("=" * 100)
    print(df_energy.to_string(index=False))

    print("\n" + "=" * 100)
    print("Fragment charge summary")
    print("=" * 100)
    print(df_charge.to_string(index=False))

    print("\n" + "=" * 100)
    print("Active NOON summary")
    print("=" * 100)
    print(df_noon.to_string(index=False))

    print("\n" + "=" * 100)
    print("Pairwise summary")
    print("=" * 100)
    print(df_pair.to_string(index=False))

    print("\n" + "=" * 100)
    print("Top dominant CI summary (top20)")
    print("=" * 100)
    print(df_ci.to_string(index=False))

    print("\n" + "=" * 100)
    print("Frontier component summary (top4)")
    print("=" * 100)
    print(df_frontier.to_string(index=False))

    print("\n" + "=" * 100)
    print("CI complexity summary (top20)")
    print("=" * 100)
    print(df_ci_complexity.to_string(index=False))

    for st in STATE_ORDER:
        print("\n" + "=" * 100)
        print(f"Active orbital AO composition: {st}")
        print("=" * 100)
        print(summarize_active_orbitals(state_data[st], topn=12).to_string(index=False))

    # CSV and PDF files are intentionally not saved in this version.

    # save xlsx
    with pd.ExcelWriter("summary_tables.xlsx") as writer:
        df_energy.to_excel(writer, sheet_name="energy", index=False)
        df_charge.to_excel(writer, sheet_name="fragment_charge", index=False)
        df_noon.to_excel(writer, sheet_name="active_noons", index=False)
        df_pair.to_excel(writer, sheet_name="pairwise", index=False)
        df_ci.to_excel(writer, sheet_name="top_ci_top20", index=False)
        df_frontier.to_excel(writer, sheet_name="frontier_comp_top4", index=False)
        df_ci_complexity.to_excel(writer, sheet_name="ci_complexity_top20", index=False)

        df_emb_mull.to_excel(writer, sheet_name="embRKS_Mulliken_atoms", index=False)
        df_emb_lowd.to_excel(writer, sheet_name="embRKS_Lowdin_atoms", index=False)
        df_emb_meta.to_excel(writer, sheet_name="embRKS_metaLowdin_atoms", index=False)

        df_cas_mull.to_excel(writer, sheet_name="CASSCF_Mulliken_atoms", index=False)
        df_cas_lowd.to_excel(writer, sheet_name="CASSCF_Lowdin_atoms", index=False)
        df_cas_meta.to_excel(writer, sheet_name="CASSCF_metaLowdin_atoms", index=False)

        for st in STATE_ORDER:
            summarize_active_orbitals(state_data[st], topn=12).to_excel(
                writer,
                sheet_name=f"{st}_active_orb",
                index=False
            )

    # plots: main figure + supporting figures
    # Extra all-site grouped-bar figure requested for bridge/fcc/hcp/abv.
    plot_co_adsorption_standalone(".")
    plot_co_adsorption_split_abv_fcc_standalone(".")
    plot_charge_transfer(state_data, "plots")
    plot_ci_dual_axis(state_data, "plots", topn=20)
    plot_frontier_component_evolution(state_data, "plots", topk=4)
    plot_topview_charge_bubble_maps(state_data, state_xyz_map, "plots")
    plot_main_co_figure(state_data, state_xyz_map, ".")

    print("\nSaved files:")
    print("  summary_tables.xlsx")
    print("  co_adsorption_all_sites_barplot_benchmark_line.png")
    print("  co_adsorption_abv_fcc_top_split_benchmark.png")
    print("  co_adsorption_energy_charge_main.png  # B: grouped bar plot; no inset")
    print("  co_adsorption_energy_charge_main_embdft_embqc_soft.png")
    print("  plots/figure_1_charge_transfer.png")
    print("  plots/figure_2_ci_compactness_entropy.png")
    print("  plots/figure_3_frontier_descriptor_evolution.png")
    print("  plots/figure_4_topview_charge_maps_all_sites.png")



if __name__ == "__main__":
    main()

