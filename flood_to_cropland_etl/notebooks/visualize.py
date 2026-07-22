"""
Sen1Floods11 — Single Chip Visualizer
======================================
Visualizes one chip: S1 (VV, VH, VV/VH ratio), S2 RGB, NDWI, and the label mask.

Usage:
    python visualize_sen1floods11.py \
        --s1   /path/to/HandLabeled/S1Hand/Bolivia_103757_S1Hand.tif \
        --s2   /path/to/HandLabeled/S2Hand/Bolivia_103757_S2Hand.tif \
        --label /path/to/HandLabeled/LabelHand/Bolivia_103757_LabelHand.tif \
        --out   bolivia_103757_viz.png

Dependencies:
    pip install rasterio numpy matplotlib
"""

import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path


# ── Sen1Floods11 S2 band index mapping (0-indexed) ──────────────────────────
S2_BANDS = {
    "B1":  0,  "B2":  1,  "B3":  2,  "B4":  3,
    "B5":  4,  "B6":  5,  "B7":  6,  "B8":  7,
    "B8A": 8,  "B9":  9,  "B10": 10, "B11": 11, "B12": 12,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tif(path):
    with rasterio.open(path) as src:
        arr  = src.read().astype("float32")
        meta = src.meta.copy()
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, meta


def percentile_stretch(band, lo=2, hi=98):
    """Clip to percentile range then normalize to [0, 1]."""
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    vmin, vmax = np.percentile(valid, [lo, hi])
    stretched = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return stretched


def make_rgb(s2, r_idx, g_idx, b_idx):
    """Build an 8-bit RGB from three S2 band indices."""
    rgb = np.stack([
        percentile_stretch(s2[r_idx]),
        percentile_stretch(s2[g_idx]),
        percentile_stretch(s2[b_idx]),
    ], axis=-1)
    return rgb


def sar_to_db(arr):
    """Convert linear SAR backscatter to dB. Clamp negatives first."""
    arr = np.where(arr <= 0, 1e-10, arr)
    return 10 * np.log10(arr)


def compute_ndwi(s2):
    """NDWI = (Green - NIR) / (Green + NIR)  →  bands B3 and B8."""
    green = s2[S2_BANDS["B3"]].astype("float32")
    nir   = s2[S2_BANDS["B8"]].astype("float32")
    denom = green + nir
    ndwi  = np.divide(green - nir, denom,
                      out=np.zeros_like(green), where=np.abs(denom) > 1e-8)
    return ndwi


# ── Main viz ─────────────────────────────────────────────────────────────────

def visualize_chip(s1_path, s2_path, label_path, out_path=None, chip_id=None):

    s1,    _  = load_tif(s1_path)       # (2, H, W)  VV=0, VH=1
    s2,    _  = load_tif(s2_path)       # (13, H, W)
    label, _  = load_tif(label_path)    # (1, H, W)  values: -1, 0, 1
    label = label[0]

    vv_db   = sar_to_db(s1[0])
    vh_db   = sar_to_db(s1[1])
    ratio   = vv_db - vh_db             # VV/VH ratio in dB
    ndwi    = compute_ndwi(s2)

    # S2 composites
    rgb_true  = make_rgb(s2, r_idx=S2_BANDS["B4"],  g_idx=S2_BANDS["B3"],  b_idx=S2_BANDS["B2"])
    rgb_false = make_rgb(s2, r_idx=S2_BANDS["B8"],  g_idx=S2_BANDS["B4"],  b_idx=S2_BANDS["B3"])

    # Label colormap: -1=grey, 0=dark green, 1=blue
    label_cmap   = ListedColormap(["#888888", "#2d6a4f", "#1d6fa4"])
    label_display = label.copy()
    label_display[label == -1] = 0
    label_display[label ==  0] = 1
    label_display[label ==  1] = 2

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")

    title_id = chip_id or Path(s1_path).stem.replace("_S1Hand", "")
    fig.suptitle(
        f"Sen1Floods11  ·  {title_id}",
        color="white", fontsize=15, fontweight="bold", y=0.97,
        fontfamily="monospace"
    )

    axes_cfg = [
        (vv_db,        "SAR — VV (dB)",          "gray",    False),
        (vh_db,        "SAR — VH (dB)",          "gray",    False),
        (ratio,        "SAR — VV/VH Ratio (dB)", "RdBu",    False),
        (rgb_true,     "S2 — True Colour",        None,      True ),
        (rgb_false,    "S2 — False Colour (NIR)", None,      True ),
        (ndwi,         "S2 — NDWI",               "coolwarm",False),
        (label_display,"Label Mask",               label_cmap,False),
    ]

    n_cols = 4
    n_rows = 2
    positions = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]

    for ax_pos, (data, title, cmap, is_rgb) in zip(positions, axes_cfg):
        row, col = ax_pos
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax.set_facecolor("#0f1117")

        if is_rgb:
            ax.imshow(data)
        elif title == "Label Mask":
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            patches = [
                mpatches.Patch(color="#888888", label="Invalid / Cloud"),
                mpatches.Patch(color="#2d6a4f", label="No Flood"),
                mpatches.Patch(color="#1d6fa4", label="Flood"),
            ]
            ax.legend(handles=patches, loc="lower right",
                      fontsize=7, framealpha=0.6,
                      facecolor="#1c1f26", labelcolor="white")
        elif cmap == "RdBu" or cmap == "coolwarm":
            vabs = np.nanpercentile(np.abs(data[np.isfinite(data)]), 98)
            ax.imshow(data, cmap=cmap, vmin=-vabs, vmax=vabs)
        else:
            ax.imshow(percentile_stretch(data), cmap=cmap)

        ax.set_title(title, color="#cccccc", fontsize=9,
                     fontfamily="monospace", pad=4)
        ax.axis("off")

    # Stats panel — last cell
    ax_stats = fig.add_subplot(n_rows, n_cols, 8)
    ax_stats.set_facecolor("#1c1f26")
    ax_stats.axis("off")

    flood_px   = int(np.sum(label == 1))
    noflood_px = int(np.sum(label == 0))
    invalid_px = int(np.sum(label == -1))
    total_px   = label.size
    flood_pct  = 100 * flood_px / max(total_px - invalid_px, 1)

    stats_text = (
        f"CHIP STATS\n"
        f"{'─'*22}\n"
        f"Size        : {label.shape[0]}×{label.shape[1]}\n"
        f"Total px    : {total_px:,}\n\n"
        f"Flood       : {flood_px:,}\n"
        f"No Flood    : {noflood_px:,}\n"
        f"Invalid     : {invalid_px:,}\n\n"
        f"Flood frac  : {flood_pct:.1f}%\n\n"
        f"VV range    : [{np.nanmin(vv_db):.1f}, {np.nanmax(vv_db):.1f}] dB\n"
        f"VH range    : [{np.nanmin(vh_db):.1f}, {np.nanmax(vh_db):.1f}] dB\n"
        f"NDWI range  : [{np.nanmin(ndwi):.2f}, {np.nanmax(ndwi):.2f}]"
    )
    ax_stats.text(
        0.08, 0.92, stats_text,
        transform=ax_stats.transAxes,
        va="top", ha="left",
        color="#aad4f5",
        fontsize=8.5,
        fontfamily="monospace",
        linespacing=1.6
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
#USAGE: python3 4_visualize.py --s1 ../Sen1Floods11_data/v1.1/data/flood_events/HandLabeled/S1Hand/USA_181263_S1Hand.tif --s2  ../Sen1Floods11_data/v1.1/data/flood_events/HandLabeled/S2Hand/USA_181263_S2Hand.tif --label ../Sen1Floods11_data/v1.1/data/flood_events/HandLabeled/LabelHand/USA_181263_LabelHand.tif --out   USA_181263.png
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Sen1Floods11 chip")
    parser.add_argument("--s1",    required=True, help="Path to S1Hand .tif")
    parser.add_argument("--s2",    required=True, help="Path to S2Hand .tif")
    parser.add_argument("--label", required=True, help="Path to LabelHand .tif")
    parser.add_argument("--out",   default=None,  help="Output PNG path (optional)")
    parser.add_argument("--id",    default=None,  help="Chip ID for title (optional)")
    args = parser.parse_args()

    visualize_chip(
        s1_path    = args.s1,
        s2_path    = args.s2,
        label_path = args.label,
        out_path   = args.out,
        chip_id    = args.id,
    )


