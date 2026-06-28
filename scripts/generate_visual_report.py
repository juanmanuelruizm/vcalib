"""Generate a visual HTML report for the top-5 filters per illumination level.

For each of the top-5 filters (by test_mean) in level_1_vs_level_2 and
level_1_vs_level_3, show all 6 test scenes with:
  - Original image (level_1)
  - Filtered image (trained filter applied to level_1)
  - Target image  (level_2 or level_3)
  - Calibration map: |filtered - original|  (where the filter acted)
  - Improvement map: |orig - target| - |filtered - target|  (green=closer, red=farther)
  - Per-pair reduction metric bar

Usage::

    uv run python scripts/generate_visual_report.py
    uv run python scripts/generate_visual_report.py --out results/visual_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.filters import build_filter
from src.utils.activations import to_unit_rgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AUGMENTED_TEST = ROOT / "data" / "augmented" / "test"
RUNS_DIR = ROOT / "results" / "experiments" / "runs"
CONFIGS_DIR = ROOT / "configs" / "experiments"
RESULTS_CSV = ROOT / "results" / "experiments" / "experiment_results.csv"
DISPLAY_SIZE = 192  # pixels for HTML thumbnails
TOP_N = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor, size: Optional[int] = None) -> Image.Image:
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    if size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def _viridis_lut() -> np.ndarray:
    """64-entry viridis approximation as (256, 3) uint8 array."""
    pts = np.array([
        [68, 1, 84], [72, 20, 103], [68, 43, 121], [59, 65, 134],
        [49, 86, 140], [40, 107, 142], [34, 127, 141], [31, 148, 139],
        [36, 168, 131], [57, 188, 115], [93, 205, 90], [138, 219, 62],
        [186, 228, 35], [235, 234, 27], [253, 231, 37], [253, 231, 37],
    ], dtype=np.float32)
    x_in = np.linspace(0, 1, len(pts))
    x_out = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(x_out, x_in, pts[:, c]) for c in range(3)], axis=1)
    return lut.clip(0, 255).astype(np.uint8)


def _rdylgn_lut() -> np.ndarray:
    """256-entry RdYlGn approx: 0=dark-red, 128=yellow, 255=dark-green."""
    pts = np.array([
        [165, 0, 38], [215, 48, 39], [244, 109, 67], [253, 174, 97],
        [254, 224, 139], [255, 255, 191], [217, 239, 139], [166, 217, 106],
        [102, 189, 99], [26, 152, 80], [0, 104, 55],
    ], dtype=np.float32)
    x_in = np.linspace(0, 1, len(pts))
    x_out = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(x_out, x_in, pts[:, c]) for c in range(3)], axis=1)
    return lut.clip(0, 255).astype(np.uint8)


_VIRIDIS = _viridis_lut()
_RDYLGN = _rdylgn_lut()


def heatmap_to_pil(data: np.ndarray, cmap_name: str, size: Optional[int] = None) -> Image.Image:
    normed = (data - data.min()) / (data.max() - data.min() + 1e-8)
    idx = (normed * 255).clip(0, 255).astype(np.uint8)
    lut = _VIRIDIS if cmap_name == "viridis" else _RDYLGN
    rgb = lut[idx]
    img = Image.fromarray(rgb, "RGB")
    if size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def diverging_heatmap_to_pil(data: np.ndarray, size: Optional[int] = None) -> Image.Image:
    """RdYlGn: positive (improvement) = green, negative (regression) = red."""
    vmax = max(abs(data.min()), abs(data.max()), 1e-8)
    normed = (data / vmax + 1.0) / 2.0  # map [-vmax, vmax] -> [0, 1]
    idx = (normed * 255).clip(0, 255).astype(np.uint8)
    rgb = _RDYLGN[idx]
    img = Image.fromarray(rgb, "RGB")
    if size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def metric_bar_svg(value: float, max_val: float = 0.35, width: int = 160, height: int = 24) -> str:
    pct = min(value / max_val, 1.0) * 100
    color = "#4CAF50" if value >= 0.15 else "#FF9800" if value >= 0.08 else "#f44336"
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#e0e0e0" rx="4"/>'
        f'<rect width="{pct:.1f}%" height="{height}" fill="{color}" rx="4"/>'
        f'<text x="{width//2}" y="{height - 6}" text-anchor="middle" '
        f'font-size="11" font-family="monospace" fill="#222">{value:.4f}</text>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_top5() -> Dict[str, List[dict]]:
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    result: Dict[str, List[dict]] = {}
    for ds in ("level_1_vs_level_2", "level_1_vs_level_3"):
        sub = [r for r in rows if r["dataset"] == ds]
        top5 = sorted(sub, key=lambda r: float(r["test_mean"]), reverse=True)[:TOP_N]
        result[ds] = top5
    return result


def load_filter_from_run(config_name: str, filter_cfg: dict) -> torch.nn.Module:
    ckpt_path = RUNS_DIR / config_name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    filt = build_filter(filter_cfg)
    filt.load_state_dict(state)
    filt.eval()
    return filt


def get_test_scenes(b_level: int) -> List[Path]:
    suffix = f"_{b_level}"
    scenes = sorted(
        p for p in AUGMENTED_TEST.iterdir()
        if p.is_dir() and p.name.endswith(suffix)
    )
    return scenes


def apply_filter(filt: torch.nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return filt(img_tensor.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CAPTION_LABELS = [
    ("Original", "source"),
    ("Filtered", "filter applied"),
    ("Target", "reference"),
    ("Calibration map", "|filtered - original|"),
    ("Improvement map", "green=better, red=worse"),
]


def scene_row_html(
    scene_name: str,
    orig_pil: Image.Image,
    filt_pil: Image.Image,
    tgt_pil: Image.Image,
    calib_pil: Image.Image,
    improve_pil: Image.Image,
    metric: float,
) -> str:
    imgs = [orig_pil, filt_pil, tgt_pil, calib_pil, improve_pil]
    cells = "".join(
        f'<td style="text-align:center;padding:4px">'
        f'<img src="data:image/jpeg;base64,{pil_to_b64(im)}" '
        f'width="{DISPLAY_SIZE}" height="{DISPLAY_SIZE}" style="border-radius:4px;border:1px solid #ccc"/>'
        f'<br><small style="color:#555">{CAPTION_LABELS[i][0]}</small>'
        f'</td>'
        for i, im in enumerate(imgs)
    )
    bar = metric_bar_svg(metric)
    cells += (
        f'<td style="text-align:center;padding:4px;vertical-align:middle">'
        f'{bar}<br><small style="color:#555">per-pair reduction</small>'
        f'</td>'
    )
    return (
        f'<tr>'
        f'<td style="padding:4px 8px;font-family:monospace;font-size:12px;'
        f'vertical-align:middle;white-space:nowrap;color:#444">{scene_name}</td>'
        + cells +
        f'</tr>'
    )


def filter_section_html(row: dict, b_level: int, input_size: int) -> str:
    config_name = row["config"]
    filter_name = row["filter"]
    group = row["group"]
    test_mean = float(row["test_mean"])
    per_pair = [float(v) for v in row["per_pair_reductions"].split(";")]

    yaml_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    filt = load_filter_from_run(config_name, cfg["filter"])

    scenes = get_test_scenes(b_level)
    target_fname = f"level_{b_level}.jpg"

    scene_rows = []
    for idx, scene_dir in enumerate(scenes):
        a_path = scene_dir / "level_1.jpg"
        b_path = scene_dir / target_fname
        if not a_path.exists() or not b_path.exists():
            continue

        orig_t = to_unit_rgb(a_path, input_size)
        tgt_t = to_unit_rgb(b_path, input_size)
        filt_t = apply_filter(filt, orig_t)

        orig_pil = tensor_to_pil(orig_t, DISPLAY_SIZE)
        tgt_pil = tensor_to_pil(tgt_t, DISPLAY_SIZE)
        filt_pil = tensor_to_pil(filt_t, DISPLAY_SIZE)

        calib_arr = (filt_t - orig_t).abs().mean(0).cpu().numpy()
        calib_pil = heatmap_to_pil(calib_arr, "viridis", DISPLAY_SIZE)

        diff_orig = (orig_t - tgt_t).abs().mean(0).cpu().numpy()
        diff_filt = (filt_t - tgt_t).abs().mean(0).cpu().numpy()
        improve_arr = diff_orig - diff_filt
        improve_pil = diverging_heatmap_to_pil(improve_arr, DISPLAY_SIZE)

        metric = per_pair[idx] if idx < len(per_pair) else 0.0
        scene_rows.append(scene_row_html(
            scene_dir.name, orig_pil, filt_pil, tgt_pil, calib_pil, improve_pil, metric
        ))

    header = (
        f'<div style="margin:24px 0 8px;padding:10px 16px;background:#f5f5f5;border-left:4px solid #1565C0;border-radius:4px">'
        f'<strong style="font-size:15px;font-family:monospace">{filter_name}</strong>'
        f'&nbsp;&nbsp;<span style="color:#666;font-size:13px">group: <code>{group}</code></span>'
        f'&nbsp;&nbsp;<span style="background:#E8F5E9;color:#2E7D32;padding:2px 8px;border-radius:10px;font-size:13px">'
        f'test_mean = {test_mean:.4f}</span>'
        f'</div>'
    )
    table = (
        f'<div style="overflow-x:auto">'
        f'<table style="border-collapse:collapse;font-size:13px">'
        f'<thead><tr>'
        f'<th style="padding:4px 8px;text-align:left;border-bottom:1px solid #ddd">Scene</th>'
        + "".join(
            f'<th style="padding:4px 8px;text-align:center;border-bottom:1px solid #ddd">'
            f'{lbl}<br><small style="color:#888;font-weight:normal">{sub}</small></th>'
            for lbl, sub in CAPTION_LABELS
        )
        + f'<th style="padding:4px 8px;text-align:center;border-bottom:1px solid #ddd">Metric</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(scene_rows)}</tbody>'
        f'</table>'
        f'</div>'
    )
    return header + table


def build_html(top5_by_ds: Dict[str, List[dict]]) -> str:
    ds_sections = []
    for ds, top5 in top5_by_ds.items():
        b_level = int(re.search(r"level_(\d+)$", ds).group(1))
        input_size = int(yaml.safe_load(
            open(CONFIGS_DIR / f"{top5[0]['config']}.yaml")
        ).get("input_size", 384))

        filter_blocks = []
        for rank, row in enumerate(top5, 1):
            print(f"  [{ds}] rank {rank}/5: {row['config']} ...", flush=True)
            block = (
                f'<div style="margin-bottom:32px">'
                f'<div style="font-size:13px;color:#888;margin-bottom:4px">Rank {rank}</div>'
                + filter_section_html(row, b_level, input_size)
                + f'</div>'
            )
            filter_blocks.append(block)

        level_label = f"Level 1 &rarr; Level {b_level}"
        ds_sections.append(
            f'<section style="margin-bottom:48px">'
            f'<h2 style="font-family:sans-serif;border-bottom:2px solid #1565C0;padding-bottom:6px">'
            f'{level_label} &nbsp;<small style="color:#888;font-size:14px">({ds})</small></h2>'
            + "".join(filter_blocks)
            + f'</section>'
        )

    legend = (
        f'<div style="font-family:sans-serif;font-size:13px;color:#555;margin:16px 0;padding:10px 16px;'
        f'background:#fff9c4;border-radius:4px;border:1px solid #f9a825">'
        f'<strong>Legend:</strong> '
        f'<b>Calibration map</b> (viridis) shows where the filter changed pixel values. '
        f'<b>Improvement map</b> (RdYlGn) shows pixel-level change toward target: '
        f'<span style="color:#2E7D32">green = closer to target</span>, '
        f'<span style="color:#c62828">red = further from target</span>. '
        f'<b>Metric bar</b> = per-pair feature-space reduction computed by DINOv2 during training '
        f'(higher = better alignment).'
        f'</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VCalib - Top-5 Filter Visual Report</title>
<style>
  body {{ font-family: sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  table {{ background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 6px; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  img {{ display: block; }}
</style>
</head>
<body>
<h1>VCalib &mdash; Top-5 Filter Visual Report</h1>
<p style="color:#666;font-size:14px">Top 5 configs by <code>test_mean</code> (DINOv2 feature-space reduction on held-out test pairs) for each illumination shift level. All 6 test scenes shown per filter.</p>
{legend}
{"".join(ds_sections)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual HTML report")
    parser.add_argument("--out", default=str(ROOT / "results" / "visual_report.html"))
    args = parser.parse_args()

    print("Loading CSV ...", flush=True)
    top5_by_ds = load_csv_top5()

    for ds, rows in top5_by_ds.items():
        print(f"\n{ds}:")
        for r in rows:
            print(f"  {r['config']:45s} test_mean={r['test_mean']}")

    print("\nGenerating report ...", flush=True)
    html = build_html(top5_by_ds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {out}", flush=True)


if __name__ == "__main__":
    main()
