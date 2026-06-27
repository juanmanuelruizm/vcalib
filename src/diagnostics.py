"""Phase 1: Diagnostic sweep — which layers carry illumination signal?

Forwards A & B through frozen RF-DETR nano, computes per-layer L2(normalized) + cosine
distance per illumination level, aggregates mean±std over scenes, and writes the result
to ``results/phase1_diagnostics.json``. With ``--plot``, also generates a heatmap
(layer × level) and per-layer line plots with error bars.

Early bailout: if the distance is flat across all layers (model already invariant), the
script reports "no signal" and exits — the shift or dataset needs redesign.

Run after capturing the dev dataset:
    uv run python src/diagnostics.py --dataset-path data/raw/scenes_YYYYMMDD/ --output results/phase1_diagnostics.json --plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.activations import (
    DEFAULT_LAYERS,
    load_model,
)
from src.utils.data_pairs import Dataset, discover_pairs, load_pair_tensors

EPS = 1e-6


def _per_layer_distances(
    a_acts: Dict[str, torch.Tensor], b_acts: Dict[str, torch.Tensor], layers: List[str]
) -> Dict[str, Dict[str, float]]:
    """Return {layer: {"l2_rel": float, "cosine": float}} between A and B activations."""
    out: Dict[str, Dict[str, float]] = {}
    for name in layers:
        if name not in a_acts or name not in b_acts:
            continue
        a, b = a_acts[name], b_acts[name]
        l2_rel = float((a - b).flatten().norm() / a.flatten().norm().clamp(min=EPS))
        cos = float(
            torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
            )
        )
        out[name] = {"l2_rel": l2_rel, "cosine": cos}
    return out


def _extract_activations(
    model: Any, unit_image: torch.Tensor, layers: List[str]
) -> Dict[str, torch.Tensor]:
    """Forward a (3, H, W) [0,1] image through the frozen model; return per-layer activations."""
    from src.calibration import _grad_hooks, _make_nested_tensor, _normalize_batch, _model_device

    dev = _model_device(model)
    dtype = next(model.model.parameters()).dtype
    x = _normalize_batch(unit_image.unsqueeze(0).to(dev, dtype), dev, dtype)
    with _grad_hooks(model, layers) as acts:
        model.model(_make_nested_tensor(x))
    return {k: v.detach().clone() for k, v in acts.items()}


def run_diagnostics(
    dataset: Dataset,
    layers: List[str] = list(DEFAULT_LAYERS),
    size: str = "n",
    device: str = "auto",
) -> Dict[str, Any]:
    """Run the diagnostic sweep. Returns a serializable dict for JSON output."""
    print(f"[diagnostics] Loading model (size={size}, device={device})...")
    model = load_model(size=size, device=device)
    print(
        f"[diagnostics] {len(dataset)} pairs across {len(dataset.scene_ids)} scenes, levels={dataset.levels}"
    )

    # Collect per (layer, level) distances across scenes
    # distances[layer][level] = list of {l2_rel, cosine} over scenes
    distances: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for i, pair in enumerate(dataset.pairs):
        a_unit, b_unit = load_pair_tensors(pair, input_size=384)
        a_acts = _extract_activations(model, a_unit, layers)
        b_acts = _extract_activations(model, b_unit, layers)
        d = _per_layer_distances(a_acts, b_acts, layers)
        for layer_name, metrics in d.items():
            distances[layer_name][pair.level].append(metrics)
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset.pairs):
            print(f"[diagnostics] {i + 1}/{len(dataset.pairs)} pairs processed")

    # Aggregate mean ± std
    results: Dict[str, Any] = {
        "layers": layers,
        "levels": dataset.levels,
        "n_scenes": len(dataset.scene_ids),
        "n_pairs": len(dataset.pairs),
        "per_layer": {},
    }
    flat_any = False
    for layer_name in layers:
        per_level: Dict[str, Any] = {}
        l2_means, l2_stds = [], []
        for level in sorted(distances[layer_name].keys()):
            entries = distances[layer_name][level]
            l2_vals = [e["l2_rel"] for e in entries]
            cos_vals = [e["cosine"] for e in entries]
            l2_mean, l2_std = float(np.mean(l2_vals)), float(np.std(l2_vals))
            cos_mean = float(np.mean(cos_vals))
            per_level[str(level)] = {
                "l2_rel_mean": l2_mean,
                "l2_rel_std": l2_std,
                "cosine_mean": cos_mean,
                "n": len(entries),
            }
            l2_means.append(l2_mean)
            l2_stds.append(l2_std)
            if l2_mean > 0.01:
                flat_any = True
        # spread = max - min across levels (should scale with illumination shift)
        spread = max(l2_means) - min(l2_means) if l2_means else 0.0
        per_level["l2_rel_spread_across_levels"] = spread
        results["per_layer"][layer_name] = per_level

    results["signal_detected"] = flat_any
    if not flat_any:
        results["early_bailout"] = (
            "No signal: distance flat across all layers. Model may already be invariant to this shift. Redesign the shift or accept robustness."
        )
    else:
        # Rank layers by spread (descending)
        ranking = sorted(
            results["per_layer"].items(),
            key=lambda kv: kv[1].get("l2_rel_spread_across_levels", 0.0),
            reverse=True,
        )
        results["layer_ranking_by_spread"] = [name for name, _ in ranking]

    return results


def plot_diagnostics(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate heatmap (layer × level) + per-layer line plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[diagnostics] matplotlib not installed; skipping plots. Install with: uv sync --extra dev"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    layers = results["layers"]
    levels = [int(lv) for lv in results["levels"]]

    # Heatmap: rows=layers, cols=levels, color=l2_rel_mean
    matrix = np.zeros((len(layers), len(levels)))
    for i, layer in enumerate(layers):
        for j, level in enumerate(levels):
            pl = results["per_layer"].get(layer, {})
            entry = pl.get(str(level), {})
            matrix[i, j] = entry.get("l2_rel_mean", 0.0)

    fig, ax = plt.subplots(figsize=(8, max(6, len(layers) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"level {lv}" for lv in levels])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_title("Per-layer L2 distance (||A-B||/||A||) by illumination level")
    plt.colorbar(im, ax=ax, label="L2 (normalized)")
    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_l2.png", dpi=150)
    plt.close(fig)
    print(f"[diagnostics] heatmap -> {output_dir / 'heatmap_l2.png'}")

    # Line plots: one per layer, x=level, y=l2_rel_mean, error bars=std
    fig, ax = plt.subplots(figsize=(8, 6))
    for layer in layers:
        means, stds = [], []
        for level in levels:
            entry = results["per_layer"].get(layer, {}).get(str(level), {})
            means.append(entry.get("l2_rel_mean", 0.0))
            stds.append(entry.get("l2_rel_std", 0.0))
        ax.errorbar(levels, means, yerr=stds, label=layer, alpha=0.7, linewidth=1)
    ax.set_xlabel("Illumination level")
    ax.set_ylabel("L2 (normalized)")
    ax.set_title("Per-layer distance vs illumination level")
    ax.legend(fontsize=6, ncol=2, loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "lines_l2.png", dpi=150)
    plt.close(fig)
    print(f"[diagnostics] line plot -> {output_dir / 'lines_l2.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Diagnostic sweep")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to data/raw/ or data/raw/scenes_YYYYMMDD/",
    )
    parser.add_argument("--output", type=str, default="results/phase1_diagnostics.json")
    parser.add_argument("--plot", action="store_true", help="Generate heatmap + line plots")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=str, default="n")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    dataset = discover_pairs(args.dataset_path, val_split=args.val_split, seed=args.seed)
    print(f"[diagnostics] {dataset}")

    results = run_diagnostics(dataset, size=args.size, device=args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[diagnostics] results -> {output_path}")

    if results.get("early_bailout"):
        print(f"\n⚠ EARLY BAILOUT: {results['early_bailout']}")
    else:
        print(
            f"\n✓ Signal detected. Top layers by spread: {results.get('layer_ranking_by_spread', [])[:5]}"
        )

    if args.plot:
        plot_diagnostics(results, Path("results/phase1_plots"))


if __name__ == "__main__":
    main()
