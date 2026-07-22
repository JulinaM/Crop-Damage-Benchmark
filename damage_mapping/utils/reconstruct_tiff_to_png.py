"""
Poster-quality visualization for damage mapping segmentation results.
Produces 300 DPI TIFF figures suitable for print at conference poster scale.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import numpy as np
import pandas as pd
import rasterio as rio
from IPython.display import display

DISPLAY_NAMES = {
    "change.method": "Method",
    "curriculum.flood_epochs": "Flood Epoch",
    "curriculum.conflict_epochs": "Conflict Epoch",
}

# ---------------------------------------------------------------------------
# Semantics & colors
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: 'background', 1: 'no_damage', 2: 'damage'}

CLASS_COLORS = {
    0: np.array([42,  54,  66],  dtype=np.uint8),   # background  – cool dark slate
    1: np.array([52,  168, 130], dtype=np.uint8),   # no damage   – cool teal-green
    2: np.array([194, 68,  68],  dtype=np.uint8),   # damage      – cool desaturated red
}

# Difference panel: cool-toned diverging scheme
# 0 = background (masked)  → cool blue-gray
# 1 = correct prediction   → muted steel blue
# 2 = error / mismatch     → warm amber (contrast anchor)
DIFF_COLORS = {
    0: np.array([80,  96,  112], dtype=np.uint8),   # masked      – cool blue-gray
    1: np.array([74,  144, 196], dtype=np.uint8),   # agreement   – steel blue
    2: np.array([224, 138,  60], dtype=np.uint8),   # error       – muted amber
}

# Legend entries rendered on the figure
LEGEND_ENTRIES = {
    'Ground Truth / Prediction': [
        ('#1e1e1e', 'Background'),
        ('#3a7d44', 'No Damage'),
        ('#b4303a', 'Damage'),
    ],
    'Difference': [
        ('#f5f5f0', 'Agreement'),
        ('#0072b2', 'Error'),
        ('#1e1e1e', 'Masked (background)'),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_run(runs_dir: Path, eval_name: str | None = None) -> Path:
    candidates = [
        d for d in sorted(runs_dir.glob('terramind_*'))
        if _resolve_eval_paths(d, eval_name)["metrics_path"].exists()
        and _resolve_eval_paths(d, eval_name)["pred_dir"].exists()
    ]
    if not candidates:
        raise FileNotFoundError(f'No successful runs found in {runs_dir}')
    return candidates[-1]


def _resolve_eval_paths(run_dir: Path, eval_name: str | None = None) -> dict[str, Path]:
    if eval_name is None:
        if (run_dir / 'geotiffs').exists() or (run_dir / 'metrics.txt').exists():
            eval_dir = run_dir
        elif (run_dir / 'conflict').exists():
            eval_dir = run_dir / 'conflict'
        else:
            eval_dir = run_dir
    else:
        eval_dir = run_dir / eval_name

    return {
        "eval_dir": eval_dir,
        "pred_dir": eval_dir / 'geotiffs',
        "metrics_path": eval_dir / 'metrics.txt',
        "poster_dir": eval_dir / 'poster_figures',
    }


def _resolve_holdout_dirs(project_root: Path, eval_name: str | None = None) -> dict[str, Path]:
    if eval_name == "flood":
        input_root = project_root / 'data/input/flood_data/Test'
    else:
        input_root = project_root / 'data/input/Images_large/Test'

    return {
        "before": input_root / 'Before/S2L2A',
        "after": input_root / 'After/S2L2A',
        "labels": input_root / 'Labels',
    }


def read_single_band(path: Path) -> np.ndarray:
    with rio.open(path) as src:
        return src.read(1)


def colorize(label: np.ndarray, color_table: dict) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls, color in color_table.items():
        rgb[label == cls] = color
    return rgb


def build_diff_map(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Three-state difference: masked / agreement / error."""
    diff = np.zeros_like(truth, dtype=np.uint8)          # 0 = masked
    valid = truth != 0
    diff[valid & (truth == pred)] = 1                     # 1 = correct
    diff[valid & (truth != pred)] = 2                     # 2 = wrong
    return colorize(diff, DIFF_COLORS)


def parse_metrics(metrics_path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r'Image\s+(?P<image_id>\d+)\s+metrics:\s+'
        r'Accuracy:\s+(?P<Accuracy>[0-9.]+)\s+'
        r'Precision:\s+(?P<Precision>[0-9.]+)\s+'
        r'Recall:\s+(?P<Recall>[0-9.]+)\s+'
        r'F1:\s+(?P<F1>[0-9.]+)\s+'
        r'IoU:\s+(?P<IoU>[0-9.]+)',
        re.MULTILINE,
    )
    records = []
    for m in pattern.finditer(metrics_path.read_text()):
        row = m.groupdict()
        row['image_id'] = int(row['image_id'])
        for k in ('Accuracy', 'Precision', 'Recall', 'F1', 'IoU'):
            row[k] = float(row[k])
        records.append(row)
    return pd.DataFrame(records).sort_values('image_id').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Poster figure
# ---------------------------------------------------------------------------
POSTER_DPI = 300
PANEL_INCH = 4.5          # width of each image panel in inches
CBAR_PAD   = 0.18         # space between panels (fraction of panel width)


def make_poster_figure(
    tile_idx: int,
    label_rgb:      np.ndarray,
    prediction_rgb: np.ndarray,
    diff_rgb:       np.ndarray,
    metrics_row:    pd.Series,
    label_name:     str,
    pred_name:      str,
    output_path:    Path,
    config_info: dict,
    display_method: False
) -> None:
    """
    Render a single three-panel poster figure and save as 300 DPI TIFF.

    Parameters
    ----------
    tile_idx       : zero-based tile index (used only in the subtitle)
    label_rgb      : H×W×3 uint8 – colourised ground truth
    prediction_rgb : H×W×3 uint8 – colourised prediction
    diff_rgb       : H×W×3 uint8 – colourised difference map
    metrics_row    : pandas Series with IoU, F1, Precision, Recall, Accuracy
    label_name     : filename of the label GeoTIFF (subtitle text)
    pred_name      : filename of the prediction GeoTIFF (subtitle text)
    output_path    : destination .tiff path
    """
    # ------------------------------------------------------------------
    # Typography  (fall back to DejaVu Sans if Helvetica is unavailable)
    # ------------------------------------------------------------------
    available = {f.name for f in font_manager.fontManager.ttflist}
    sans = 'Helvetica' if 'Helvetica' in available else 'DejaVu Sans'

    TITLE_SIZE  = 22   # panel letter + label  (A) Ground Truth
    METRIC_SIZE = 16   # IoU / F1 annotation
    LEGEND_SIZE = 13
    SUPTITLE_SIZE = 14

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig_w  = PANEL_INCH * 3 + PANEL_INCH * CBAR_PAD * 2 + 1.6   # +1.6" legend column
    h, w   = label_rgb.shape[:2]
    aspect = h / w
    fig_h  = PANEL_INCH * aspect + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=POSTER_DPI, facecolor='white')

    # Reserve the rightmost ~1.6" for the legend; panels occupy the rest
    legend_frac = 1.6 / fig_w          # fraction of total width for legend column
    panels_right = 1.0 - legend_frac - 0.01

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        left=0.01, right=panels_right,
        top=0.82,  bottom=0.05,
        wspace=0.04,
    )

    ax_truth = fig.add_subplot(gs[0, 0])
    ax_pred  = fig.add_subplot(gs[0, 1])
    ax_diff  = fig.add_subplot(gs[0, 2])

    panels = [
        (ax_truth, label_rgb,      '(a)  Ground Truth'),
        (ax_pred,  prediction_rgb, '(b)  Predicted'),
        (ax_diff,  diff_rgb,       '(c)  Difference'),
    ]

    for ax, img, title in panels:
        ax.imshow(img, interpolation='nearest')
        ax.set_title(
            title,
            fontsize=TITLE_SIZE,
            fontfamily=sans,
            fontweight='bold',
            color='#111111',
            pad=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # (metrics rendered in the reserved right column — see metrics box below)

    # ------------------------------------------------------------------
    # Global title / suptitle (file info, small)
    # ------------------------------------------------------------------
    fig.text(
        0.5, 0.97,
        'Flood Damage Mapping (Holdout Set)',
        ha='center', va='top',
        fontsize=18, fontfamily=sans, fontweight='bold', color='#111111',
    )

    # fig.text(
    #     0.5, 0.93,
    #     # f'Tile {tile_idx + 1}   |   Label: {label_name}   |   Pred: {pred_name}',
    #     ha='center', va='top',
    #     fontsize=SUPTITLE_SIZE, fontfamily=sans, color='#555555',
    #     style='italic',
    # )

    # ------------------------------------------------------------------
    # Paper-style header (method + experiment info)
    # ------------------------------------------------------------------
    if display_method:
        method_parts = []
        for k, v in config_info.items():
            label = DISPLAY_NAMES.get(k, k.split('.')[-1])
            if v:
                method_parts.append(f"{label}={v}")
        method_text = ", ".join(method_parts)
        fig.text(
            0.5, 0.90,
            f"{method_text}",
            ha='center', va='top', fontsize=11, fontfamily=sans, color='#222222',
        )

    # ------------------------------------------------------------------
    # Legend — placed in the reserved right column, top-aligned, no overlap
    # ------------------------------------------------------------------
    def _patch(hex_col, label):
        return mpatches.Patch(facecolor=hex_col, edgecolor='#aabbcc',
                              linewidth=0.6, label=label)

    legend_handles = [
        _patch('#2a3642', 'Background (masked)'),
        _patch('#34a882', 'No Damage'),
        _patch('#c24444', 'Damage'),
        _patch('#4a90c4', 'Agreement'),
        _patch('#e08a3c', 'Error'),
    ]
    # bbox_to_anchor in figure coordinates: left edge of the legend column
    legend_x = panels_right + 0.02
    fig.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(legend_x, 0.84),   # aligns with top of plot area
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor='#ccd6e0',
        facecolor='#f7f9fb',
        fontsize=LEGEND_SIZE,
        prop={'family': sans, 'size': LEGEND_SIZE},
        handlelength=1.5,
        handleheight=1.1,
        borderpad=0.8,
        labelspacing=0.7,
        title='Legend',
        title_fontsize=LEGEND_SIZE,
    )

    # ------------------------------------------------------------------
    # Metrics box — reserved right column, bottom-aligned, no overlap
    # ------------------------------------------------------------------
    metrics = [
        ('IoU',       f'{metrics_row.IoU:.4f}'),
        ('F1',        f'{metrics_row.F1:.4f}'),
        ('Precision', f'{metrics_row.Precision:.4f}'),
        ('Recall',    f'{metrics_row.Recall:.4f}'),
        ('Accuracy',  f'{metrics_row.Accuracy:.4f}'),
    ]
    # Build a two-column text block: label (right-aligned) · value (left-aligned)
    metrics_lines = '\n'.join(f'{k:<10}{v}' for k, v in metrics)
    metrics_box_x = legend_x          # same left edge as the legend
    metrics_box_y = 0.10              # anchored near the figure bottom

    fig.text(
        metrics_box_x, metrics_box_y + 0.085,
        'Metrics',
        transform=fig.transFigure,
        ha='left', va='bottom',
        fontsize=LEGEND_SIZE,
        fontfamily=sans,
        fontweight='bold',
        color='#2a3642',
    )
    t = fig.text(
        metrics_box_x, metrics_box_y,
        metrics_lines,
        transform=fig.transFigure,
        ha='left', va='bottom',
        fontsize=LEGEND_SIZE - 1,
        fontfamily='monospace',
        color='#2a3642',
        linespacing=1.65,
        bbox=dict(
            boxstyle='round,pad=0.55',
            facecolor='#f7f9fb',
            edgecolor='#ccd6e0',
            linewidth=0.8,
        ),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=POSTER_DPI,
        format='png',
        bbox_inches='tight',
        pad_inches=0.2,          # no extra white border
    )
    # plt.show(fig)
    plt.close(fig)
    print(f'  ✓  Saved  →  {output_path}')

import yaml
def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_config_params(cfg: dict, keys: list) -> dict:
    """
    Extract nested keys like 'change.method' safely from config dict.
    """
    result = {}
    for key in keys:
        parts = key.split('.')
        val = cfg
        try:
            for p in parts:
                val = val[p]
        except (KeyError, TypeError):
            val = 'N/A'
        result[key] = val
    return result

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(run_dirr=None, display_method=False, config_keys=None, eval_name=None, dataset=None):
    PROJECT_ROOT = Path('/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2')
    EXP_DIR = PROJECT_ROOT / "data/experiments"
    if isinstance(display_method, (list, tuple)) and config_keys is None:
        config_keys = list(display_method)
        display_method = False
    eval_name = dataset or eval_name

    if run_dirr:
        run_dir = EXP_DIR / run_dirr
    else:
        run_dir = EXP_DIR / 'curriculum-learning/curriculum_terramind_2026-04-20_160535'
    # print("--->", run_dir)

    eval_paths = _resolve_eval_paths(run_dir, eval_name)
    holdout_dirs = _resolve_holdout_dirs(PROJECT_ROOT, eval_name)
    pred_dir = eval_paths["pred_dir"]
    holdout_before_dir = holdout_dirs["before"]
    holdout_after_dir = holdout_dirs["after"]
    holdout_label_dir = holdout_dirs["labels"]

    pred_files   = sorted(pred_dir.glob('predicted_map_*_colored.tif'))
    label_files  = sorted(holdout_label_dir.glob('*.tif'))
    before_files = sorted(holdout_before_dir.glob('*.tif'))
    after_files  = sorted(holdout_after_dir.glob('*.tif'))

    print(f'\nRun directory      : {run_dirr}')
    print(f'Evaluation set     : {eval_name or "default"}')
    print(f'Prediction dir     : {pred_dir}')
    print(f'Prediction files   : {len(pred_files)}')
    print(f'Holdout labels     : {len(label_files)}')
    assert len(pred_files) == len(label_files) == len(before_files) == len(after_files), 'Holdout file counts do not match.'
    metrics_df = parse_metrics(eval_paths["metrics_path"])
    metrics_df['label_file']      = [p.name for p in label_files]
    metrics_df['prediction_file'] = [p.name for p in pred_files]
    # display(metrics_df)

    poster_dir = eval_paths["poster_dir"]
    poster_dir.mkdir(exist_ok=True)

    cfg = load_config(run_dir / '.hydra/config.yaml')
    config_info = extract_config_params(cfg, config_keys or ["change.method","curriculum.flood_epochs", "curriculum.conflict_epochs",])

    print(f'Generating {len(pred_files)} poster-quality figure(s) → {poster_dir}')

    for tile_idx in range(len(pred_files)):
        label      = read_single_band(label_files[tile_idx]).astype(np.uint8)
        prediction = read_single_band(pred_files[tile_idx]).astype(np.uint8)

        label_rgb      = colorize(label, CLASS_COLORS)
        prediction_rgb = colorize(prediction, CLASS_COLORS)
        diff_rgb       = build_diff_map(label, prediction)

        row = metrics_df.iloc[tile_idx]
        out_path = poster_dir / f'poster_tile_{tile_idx + 1:02d}.png'

        make_poster_figure(
            tile_idx       = tile_idx,
            label_rgb      = label_rgb,
            prediction_rgb = prediction_rgb,
            diff_rgb       = diff_rgb,
            metrics_row    = row,
            label_name     = label_files[tile_idx].name,
            pred_name      = pred_files[tile_idx].name,
            output_path    = out_path,
            config_info    = config_info,
            display_method = display_method
        )

    # print('\nAll tiles complete.')


if __name__ == "__main__":
    main()
