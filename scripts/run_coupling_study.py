#!/usr/bin/env python3
"""Activation<->detection coupling study.

Trains the preprocessing filter on paired cooktop data and, every N epochs,
records TWO decoupled read-outs on the held-out (GT-backed) test scenes:

  * activation convergence  -> feature-gap closure at the layer group
                               (``evaluate_on_test`` == the classic ``test_mean``)
  * prediction quality      -> AP / AP50 of filter(B) vs GT (COCOeval)

Run for each loss mode {activation, detection, combined} with the SAME read-outs
so we can quantify, in both directions:
  - does closing the feature gap translate into better detection?  (activation mode)
  - when detection improves, do the activations converge too?      (detection mode)

The loss modes already exist in ``src/calibration.py`` (``loss_mode`` +
``detection_weight``); this script only adds the instrumentation.

Usage:
  uv run python scripts/run_coupling_study.py --device cuda
  uv run python scripts/run_coupling_study.py --device cuda --modes activation \
      --epochs 3 --eval-every 1 --tag smoke
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import contextlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pycocotools.coco import COCO  # noqa: E402

from src.calibration import CalibrationConfig, calibrate_epochs  # noqa: E402
from src.filters import build_filter  # noqa: E402
from src.utils.activations import load_model, to_unit_rgb  # noqa: E402
from src.utils.layer_groups import expand_layer_spec  # noqa: E402

# reuse the paired-pipeline glue and the detection eval helpers
from run_configs import discover_pairs, load_all_pairs, evaluate_on_test  # noqa: E402
from benchmark_detection import detect, evaluate, TEST_SCENES, DEFAULT_SPEC  # noqa: E402


def build_raw_test(gt: COCO, level: int, input_size: int
                   ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, Any]]]:
    """Load the raw held-out test scenes as (A,B) unit tensors + AP metadata.

    Uses the same scenes/GT that ``benchmark_detection`` reports on, so the
    activation read-out and the AP read-out sit on identical images.
    """
    name2id = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    dims = {im["id"]: (im["width"], im["height"]) for im in gt.dataset["images"]}
    raw = REPO / "data" / "raw"

    feat_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    meta: List[Dict[str, Any]] = []
    missing = []
    for sc in TEST_SCENES:
        a_name, b_name = f"IMG_{sc}_I1.jpg", f"IMG_{sc}_I{level}.jpg"
        if a_name not in name2id or b_name not in name2id:
            missing.append(sc)
            continue
        a_id, b_id = name2id[a_name], name2id[b_name]
        a_unit = to_unit_rgb(raw / a_name, input_size)
        b_unit = to_unit_rgb(raw / b_name, input_size)
        feat_pairs.append((a_unit, b_unit))
        meta.append({"a_id": a_id, "b_id": b_id, "aW": dims[a_id][0], "aH": dims[a_id][1],
                     "bW": dims[b_id][0], "bH": dims[b_id][1]})
    if missing:
        print(f"  [warn] test scenes without GT at level {level}: {missing}")
    return feat_pairs, meta


def ap_of(libre, filt, feat_pairs, meta, gt, max_det, device) -> Tuple[float, float]:
    """AP / AP50 of filter(B) vs GT over the raw test scenes."""
    preds, ids = [], []
    for (_, b_unit), m in zip(feat_pairs, meta):
        with torch.no_grad():
            fb = filt(b_unit.unsqueeze(0).to(device))[0].clamp(0, 1).cpu()
        preds += detect(libre, fb, m["bW"], m["bH"], m["b_id"], max_det, device)
        ids.append(m["b_id"])
    s = evaluate(gt, preds, ids)
    return (float(s[0]), float(s[1])) if s is not None else (0.0, 0.0)


def ap_raw(libre, feat_pairs, meta, gt, max_det, device, which: str) -> Tuple[float, float]:
    """AP / AP50 of the unfiltered A ('a') or B ('b') reference arms."""
    preds, ids = [], []
    for (a_unit, b_unit), m in zip(feat_pairs, meta):
        unit = a_unit if which == "a" else b_unit
        iid = m["a_id"] if which == "a" else m["b_id"]
        W, H = (m["aW"], m["aH"]) if which == "a" else (m["bW"], m["bH"])
        preds += detect(libre, unit, W, H, iid, max_det, device)
        ids.append(iid)
    s = evaluate(gt, preds, ids)
    return (float(s[0]), float(s[1])) if s is not None else (0.0, 0.0)


def run_mode(mode: str, args, model, train_pairs, es_test_pairs, layer_names,
             feat_pairs, meta, gt) -> List[Dict[str, Any]]:
    print(f"\n{'=' * 70}\nMODE: {mode}\n{'=' * 70}")
    filt = build_filter(dict(DEFAULT_SPEC))
    cfg = CalibrationConfig(
        max_epochs=args.epochs,
        early_stopping_patience=args.epochs + 1,   # disable early stop: full trajectory
        learning_rate=args.lr,
        reg_weight=args.reg_weight,
        seed=args.seed,
        loss_mode=mode,
        detection_weight=args.detection_weight,
        log_every=max(args.epochs, 1),
    )
    records: List[Dict[str, Any]] = []

    def cb(epoch: int, train_loss: float, val_loss, f: torch.nn.Module):
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            return
        f.eval()
        feat = evaluate_on_test(f, model, feat_pairs, layer_names)["mean"]
        ap, ap50 = ap_of(model, f, feat_pairs, meta, gt, args.max_det, args.device)
        f.train()
        rec = {"mode": mode, "epoch": epoch, "train_loss": train_loss,
               "val_loss": val_loss, "feat_closure": feat, "AP": ap, "AP50": ap50}
        records.append(rec)
        print(f"  ep {epoch:3d} | feat_closure {feat:+.4f} | AP {ap:.4f} | AP50 {ap50:.4f}"
              f" | train_loss {train_loss:.4f}")

    calibrate_epochs(filt, train_pairs, layer_names, model=model, cfg=cfg,
                     test_pairs=es_test_pairs, on_epoch_end=cb)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/datasets/level_1_vs_level_2")
    ap.add_argument("--level", type=int, default=2, choices=[2, 3])
    ap.add_argument("--gt", default="data/labels/instances_sam3.json")
    ap.add_argument("--layer-group", default="backbone.projector")
    ap.add_argument("--modes", default="activation,detection,combined")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg-weight", type=float, default=0.01)
    ap.add_argument("--detection-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/coupling")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    layer_names: List[str] = []
    for tok in args.layer_group.split(","):
        layer_names.extend(expand_layer_spec(tok.strip()))
    print(f"layer group {args.layer_group} -> {layer_names}")

    model = load_model(size="n", device=args.device, model_path=None)
    train_paths, test_paths = discover_pairs(Path(args.dataset))
    if not train_paths:
        raise RuntimeError(f"no train pairs under {args.dataset}")
    train_pairs = load_all_pairs(train_paths, args.input_size)
    es_test_pairs = load_all_pairs(test_paths, args.input_size) if test_paths else None
    print(f"train pairs: {len(train_pairs)} | early-stop pairs: "
          f"{len(es_test_pairs) if es_test_pairs else 0}")

    gt = COCO(args.gt)
    feat_pairs, meta = build_raw_test(gt, args.level, args.input_size)
    print(f"held-out test scenes for read-out: {len(feat_pairs)}")

    # reference AP lines (no filter): A ceiling and B floor
    a_ap, a_ap50 = ap_raw(model, feat_pairs, meta, gt, args.max_det, args.device, "a")
    b_ap, b_ap50 = ap_raw(model, feat_pairs, meta, gt, args.max_det, args.device, "b")
    print(f"reference | A(ceiling) AP {a_ap:.4f} | B(floor) AP {b_ap:.4f}")

    all_records: List[Dict[str, Any]] = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        t0 = time.time()
        recs = run_mode(mode, args, model, train_pairs, es_test_pairs, layer_names,
                        feat_pairs, meta, gt)
        with open(out / f"{mode}{suffix}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  [{mode}] {len(recs)} read-outs in {time.time() - t0:.0f}s "
              f"-> {out / (mode + suffix + '.jsonl')}")
        all_records += recs

    # combined CSV with the reference lines attached to every row
    csv_path = out / f"coupling{suffix}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "epoch", "feat_closure", "AP", "AP50",
                    "train_loss", "val_loss", "A_AP", "B_AP"])
        for r in all_records:
            w.writerow([r["mode"], r["epoch"], f"{r['feat_closure']:.4f}",
                        f"{r['AP']:.4f}", f"{r['AP50']:.4f}", f"{r['train_loss']:.4f}",
                        "" if r["val_loss"] is None else f"{r['val_loss']:.4f}",
                        f"{a_ap:.4f}", f"{b_ap:.4f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
