# vcalib — Calibration Filter for RF-DETR Under Illumination Shifts

Learnable differentiable preprocessing filter to recover RF-DETR detection performance under real-world illumination changes. See `AGENTS.md` for project rules and `docs/specs/` for design specs.

## Data Pipeline

**3-stage pipeline** for training illumination filters with geometry-augmented A/B pairs:

1. **Raw Data** → `data/raw/scenes_YYYYMMDD/scene_XXX/{level_1, level_2, level_3}.jpg`
   - 30 objects × 3 lighting conditions per object
   - level_1 = reference condition A, level_2/3 = shifted conditions B

2. **Augmentation** (`scripts/augment_dataset.py`)
   ```bash
   uv run python scripts/augment_dataset.py --raw data/raw --out data/augmented --n-aug 5 --seed 42
   ```
   - 80/20 scene split (train: 24, test: 6 scenes)
   - Geometric transforms (rotation ±5°, zoom 1.2×-1.35×, flip, crop offset)
   - **Identical transforms applied to both A and B** → preserves illumination relationship
   - Output: 288 train pairs (48 scenes × 6 variants) + 12 test pairs

3. **Dataset Split by Level** (`scripts/create_datasets.py`)
   ```bash
   uv run python scripts/create_datasets.py --augmented data/augmented --out data/datasets
   ```
   - Two independent experiments: `level_2/` and `level_3/`
   - Each: 144 train pairs + 6 test pairs (same scene split)
   - Uses symlinks to avoid data duplication
