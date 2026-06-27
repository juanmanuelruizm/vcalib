"""Phase 1: Diagnostic sweep to identify which layers carry illumination signal."""

import argparse
import json
from pathlib import Path

# TODO: Import necessary libraries
# - torch, transformers (RF-DETR)
# - numpy, PIL (image loading)
# - matplotlib (plotting)


def main():
    """
    Phase 1 diagnostic pipeline.

    Steps:
    1. Load RF-DETR nano model (frozen)
    2. Load image pairs (A, B) from dataset_path
    3. For each pair and illumination level:
       - Forward A and B through model with output_hidden_states=True
       - Extract activations from each layer
       - Compute L2 distance and cosine similarity
    4. Aggregate distances over scenes
    5. Output heatmap (layer × illumination level)
    """
    parser = argparse.ArgumentParser(description="Phase 1: Diagnostic sweep")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to data/raw/")
    parser.add_argument("--output", type=str, default="results/phase1_diagnostics.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # TODO: Implement Phase 1
    # 1. Load model
    # 2. Load image pairs
    # 3. Compute distances per layer
    # 4. Aggregate and save to JSON
    # 5. Generate heatmaps if --plot


if __name__ == "__main__":
    main()
