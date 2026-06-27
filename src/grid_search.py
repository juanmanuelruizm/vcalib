"""Phase 2: Grid search over filter configurations."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

# TODO: Import necessary libraries
# - yaml (config loading)
# - torch, transformers
# - Filter classes from src.filters


def main():
    """
    Phase 2 grid search pipeline.

    Steps:
    1. Load grid configuration from configs/grid.yaml
    2. For each combination (layer, filter_type, loss):
       - Train filter on 80% of scenes
       - Validate on 20% held-out scenes
       - Log convergence metrics
       - Save checkpoint
    3. Aggregate results in results/runs.csv
    4. Rank by validation distance reduction
    """
    parser = argparse.ArgumentParser(description="Phase 2: Grid search")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/grid.yaml")
    parser.add_argument("--dataset-path", type=str, default="data/raw/")
    parser.add_argument("--output-dir", type=str, default="results/")

    args = parser.parse_args()

    # TODO: Implement Phase 2
    # 1. Load config from YAML
    # 2. Create grid of all combinations
    # 3. For each combo:
    #    - Initialize filter
    #    - Train with early stopping
    #    - Log metrics
    #    - Save checkpoint
    # 4. Write results/runs.csv


if __name__ == "__main__":
    main()
