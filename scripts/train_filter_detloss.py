#!/usr/bin/env python3
"""Train a preprocessing filter WITHOUT pairs, supervised by detection GT.

The realistic deployment setting: the reference illumination is not reproducible,
so you cannot collect (reference, shifted) *pairs* — you can only label the new
shifted images. This trains the filter to make a **frozen** (optionally
domain-adapted) RF-DETR detect the labeled objects in shifted images, using the
detector's own DETR ``SetCriterion`` (Hungarian matcher + focal/L1/GIoU) against
the real GT boxes. No reference image A is used.

  filter(shifted) -> frozen RF-DETR -> SetCriterion(outputs, GT boxes) -> backprop to filter

Data is a COCO dataset produced by ``scripts/make_detection_coco.py`` (a shifted
level, e.g. ``--levels 2``). Evaluate later with ``scripts/benchmark_detection.py``.

Usage:
  uv run python scripts/train_filter_detloss.py \
      --data data/coco/cooktop_shift_lv2 \
      --model-checkpoint results/finetune/cooktop_ref_headonly/weights/best.pt \
      --out results/experiments/runs/pairfree_stc_lv2

  # ExDark (unpaired) off-the-shelf baseline arm -> remap labels to COCO-91:
  uv run python scripts/train_filter_detloss.py --data data/coco/exdark \
      --train-split dark_train --val-split dark_val --label-map exdark_coco \
      --out results/experiments/runs/exdark_pairfree_offtheshelf
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.filters import build_filter
from src.utils.activations import load_model, to_unit_rgb, normalize
from src.calibration import _make_nested_tensor, _grad_hooks

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ExDark contiguous class (by name) -> RF-DETR raw logit column (COCO 91-id space).
# Only for the off-the-shelf baseline arm, whose criterion is 91-class; the
# fine-tuned A' head is native 12-class, so it needs no remap. Derived from
# libreyolo's _COCO91_TO_COCO80 + the model's COCO-80 names (verified).
EXDARK_TO_COCO91 = {
    "Bicycle": 2, "Boat": 9, "Bottle": 44, "Bus": 6, "Car": 3, "Cat": 17,
    "Chair": 62, "Cup": 47, "Dog": 18, "Motorbike": 4, "People": 1, "Table": 67,
}


def load_split(data_dir: Path, split: str):
    """Return [(file_name, W, H, targets)] where targets = {labels, boxes cxcywh[0,1]}."""
    coco = json.load(open(data_dir / "annotations" / f"instances_{split}.json"))
    cat2idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    items = []
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        boxes, labels = [], []
        for a in anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
            labels.append(cat2idx[a["category_id"]])
        items.append((im["file_name"], W, H,
                      torch.tensor(boxes, dtype=torch.float32),
                      torch.tensor(labels, dtype=torch.long)))
    return items


def forward_step(model, crit, filt, img_dir, file_name, boxes, labels, size, dev, act_layer=None):
    """One forward of filter(image); return (det_GT_loss, projector_act or None).

    When ``act_layer`` is given the same forward also yields the grad-attached
    activation at that layer, so the CORAL term shares the detection forward.
    """
    unit = to_unit_rgb(img_dir / file_name, size).to(dev)
    fb = filt(unit.unsqueeze(0)).clamp(0, 1)
    normed = normalize(fb[0]).unsqueeze(0)
    if act_layer is not None:
        with _grad_hooks(model, [act_layer]) as acts:
            outputs = model.model(_make_nested_tensor(normed))
        act = acts[act_layer]
    else:
        outputs = model.model(_make_nested_tensor(normed))
        act = None
    targets = [{"labels": labels.to(dev), "boxes": boxes.to(dev)}]
    ld = crit(outputs, targets)
    wd = crit.weight_dict
    det = sum(ld[k] * wd[k] for k in ld if k in wd)
    return det, act


def _feat_moments(act):
    """(1,C,H,W) activation -> (mu[C], cov[C,C]) using spatial positions as samples.

    Population moments (divide by N) so per-image and reference stats are comparable.
    """
    c = act.shape[1]
    x = act.reshape(c, -1).transpose(0, 1)  # (N, C), N = H*W
    mu = x.mean(0)
    xc = x - mu
    cov = xc.transpose(0, 1) @ xc / x.shape[0]
    return mu, cov


def coral_loss(act, ref_mu, ref_cov):
    """Feature alignment (mean + covariance) between one image and the reference-domain
    statistics; returns the two raw squared terms separately.

    NB: we deliberately drop Deep-CORAL's classic 1/(4d^2) covariance scaling. At this
    layer (projector, d=256) that factor (=1/262144) crushes the covariance term to ~1e-6
    vs a mean term ~0.25 — i.e. it silently degenerates to mean-matching. Raw Frobenius
    keeps ||Sigma-Sigma_ref||_F^2 (~0.76) comparable to the mean term, so the covariance
    signal actually contributes (which is the whole point of choosing CORAL here)."""
    mu, cov = _feat_moments(act)
    mean_term = ((mu - ref_mu) ** 2).sum()
    cov_term = ((cov - ref_cov) ** 2).sum()
    return mean_term, cov_term


def compute_ref_moments(model, ref_dir, file_names, act_layer, size, dev, cache_path=None):
    """Streaming (mu, cov) of ``act_layer`` over the whole reference domain (unpaired).

    No GT needed — just the reference-condition images through the frozen model.
    Cached to ``cache_path`` (keyed by ref split + layer) so repeat runs skip it.
    """
    if cache_path is not None and cache_path.exists():
        d = torch.load(cache_path, map_location=dev)
        print(f"  ref moments: loaded cache {cache_path} (N={d['count']}, C={d['mu'].numel()})")
        return d["mu"].to(dev), d["cov"].to(dev)

    sum1 = sum2 = None
    count = 0
    n = len(file_names)
    print(f"  ref moments: forwarding {n} reference images through frozen model...")
    for i, fn in enumerate(file_names):
        unit = to_unit_rgb(ref_dir / fn, size).to(dev)
        normed = normalize(unit).unsqueeze(0)
        with _grad_hooks(model, [act_layer]) as acts:
            model.model(_make_nested_tensor(normed))
        x = acts[act_layer].detach().reshape(acts[act_layer].shape[1], -1).transpose(0, 1)  # (N,C)
        if sum1 is None:
            c = x.shape[1]
            sum1 = torch.zeros(c, device=dev, dtype=torch.float64)
            sum2 = torch.zeros(c, c, device=dev, dtype=torch.float64)
        xd = x.double()
        sum1 += xd.sum(0)
        sum2 += xd.transpose(0, 1) @ xd
        count += x.shape[0]
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"    ref moments: {i+1}/{n}")
    mu = (sum1 / count).float()
    cov = (sum2 / count - torch.outer(sum1 / count, sum1 / count)).float()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": mu.cpu(), "cov": cov.cpu(), "count": count}, cache_path)
        print(f"  ref moments: cached -> {cache_path}")
    return mu.to(dev), cov.to(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (make_detection_coco.py / exdark_to_coco.py)")
    ap.add_argument("--train-split", default="train", help="training split name (e.g. dark_train for ExDark)")
    ap.add_argument("--val-split", default="val", help="validation split name (e.g. dark_val for ExDark)")
    ap.add_argument("--label-map", choices=["none", "exdark_coco"], default="none",
                    help="'exdark_coco' remaps ExDark labels to RF-DETR's COCO-91 logit columns "
                         "(off-the-shelf baseline arm); 'none' keeps native dataset labels (A')")
    ap.add_argument("--model-checkpoint", default=None, help="fine-tuned detector; default off-the-shelf")
    ap.add_argument("--filter-type", default="spatial_tone_curve")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--out", required=True, help="filter checkpoint dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg-weight", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    # --- pair-free CORAL activation term (unpaired domain alignment) ---
    ap.add_argument("--act-weight", type=float, default=0.0,
                    help="lambda on the CORAL activation term; 0 = detection-only (default, unchanged)")
    ap.add_argument("--ref-split", default=None,
                    help="reference-domain split for CORAL stats (e.g. 'bright' for ExDark); "
                         "required when --act-weight > 0")
    ap.add_argument("--act-layer", default="backbone.projector",
                    help="layer whose activations are aligned to the reference domain")
    ap.add_argument("--ref-limit", type=int, default=0,
                    help="cap #reference images for the moment estimate (0 = all)")
    ap.add_argument("--train-limit", type=int, default=0, help="cap #train images (0 = all; for smoke/iteration)")
    ap.add_argument("--val-limit", type=int, default=0, help="cap #val images (0 = all)")
    args = ap.parse_args()

    if args.act_weight > 0.0 and not args.ref_split:
        ap.error("--ref-split is required when --act-weight > 0 (e.g. --ref-split bright)")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    data_dir = Path(args.data)
    train_dir = data_dir / "images" / args.train_split
    val_dir = data_dir / "images" / args.val_split

    model = load_model(size="n", device=dev, model_path=args.model_checkpoint)
    crit = model.model.build_criterion_and_postprocess()[0]
    crit.to(dev)

    spec = {"type": args.filter_type, "P": args.P, "grid_size": args.grid_size}
    filt = build_filter(spec).to(dev).train()
    opt = torch.optim.Adam(filt.parameters(), lr=args.lr)

    train = load_split(data_dir, args.train_split)
    val = load_split(data_dir, args.val_split)
    if args.train_limit:
        train = train[: args.train_limit]
    if args.val_limit:
        val = val[: args.val_limit]

    # optional label remap: the off-the-shelf baseline arm needs ExDark labels in
    # RF-DETR's COCO-91 logit space (A' is native 12-class -> remap stays None).
    remap = None
    if args.label_map == "exdark_coco":
        cats = sorted(
            json.load(open(data_dir / "annotations" / f"instances_{args.train_split}.json"))["categories"],
            key=lambda c: c["id"])
        remap = torch.tensor([EXDARK_TO_COCO91[c["name"]] for c in cats], dtype=torch.long)
        print(f"label-map exdark_coco: contiguous->COCO91 = {remap.tolist()}")

    # --- pair-free CORAL activation term: align filter(shifted) projector features
    # to the *reference domain* statistics (unpaired: no per-scene A image). ---
    use_coral = args.act_weight > 0.0
    ref_mu = ref_cov = None
    act_layer = args.act_layer if use_coral else None
    if use_coral:
        ref_dir = data_dir / "images" / args.ref_split
        ref_items = load_split(data_dir, args.ref_split)
        ref_files = [it[0] for it in ref_items]
        if args.ref_limit and args.ref_limit < len(ref_files):
            ref_files = ref_files[: args.ref_limit]
        # Activations are model-dependent, so the cache key MUST include the model
        # (else the off-the-shelf and A' arms would collide on the same file).
        if args.model_checkpoint:
            p = Path(args.model_checkpoint)
            mtag = p.parent.parent.name or p.parent.name or p.stem
        else:
            mtag = "offtheshelf"
        cache = data_dir / "ref_moments" / f"{args.ref_split}__{act_layer}__{mtag}__n{len(ref_files)}.pt"
        ref_mu, ref_cov = compute_ref_moments(
            model, ref_dir, ref_files, act_layer, args.input_size, dev, cache_path=cache)
        print(f"CORAL on: layer={act_layer} ref-split={args.ref_split} (N_imgs={len(ref_files)}) "
              f"lambda={args.act_weight}")

    print(f"filter {spec} | model={'off-the-shelf' if not args.model_checkpoint else args.model_checkpoint}")
    obj = "det-GT + lambda*CORAL (pair-free, unpaired)" if use_coral else "detector-GT loss (pair-free)"
    print(f"train={len(train)} val={len(val)} | splits={args.train_split}/{args.val_split} | {obj}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_f = open(out / "metrics.jsonl", "w")
    # Model selection tracks the real objective (detection), NOT the auxiliary CORAL
    # term — the coupling study showed selecting on the feature metric overshoots AP.
    best_val, best_epoch, since = float("inf"), -1, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        filt.train()
        random.shuffle(train)
        tr_det = tr_mean = tr_cov = 0.0
        for fn, W, H, boxes, labels in train:
            if len(boxes) == 0:
                continue
            opt.zero_grad()
            lbl = labels if remap is None else remap[labels]
            det, act = forward_step(model, crit, filt, train_dir, fn, boxes, lbl,
                                    args.input_size, dev, act_layer=act_layer)
            loss = det + args.reg_weight * filt.reg_loss()
            if use_coral:
                mean_term, cov_term = coral_loss(act, ref_mu, ref_cov)
                loss = loss + args.act_weight * (mean_term + cov_term)
                tr_mean += float(mean_term.detach())
                tr_cov += float(cov_term.detach())
            loss.backward()
            opt.step()
            tr_det += float(det.detach())
        nt = max(1, len(train))
        tr_det, tr_mean, tr_cov = tr_det / nt, tr_mean / nt, tr_cov / nt

        filt.eval()
        v_det = v_mean = v_cov = 0.0
        for fn, W, H, boxes, labels in val:
            if len(boxes) == 0:
                continue
            lbl = labels if remap is None else remap[labels]
            det, act = forward_step(model, crit, filt, val_dir, fn, boxes, lbl,
                                    args.input_size, dev, act_layer=act_layer)
            v_det += float(det.detach())
            if use_coral:
                mean_term, cov_term = coral_loss(act, ref_mu, ref_cov)
                v_mean += float(mean_term.detach())
                v_cov += float(cov_term.detach())
        nv = max(1, len(val))
        v_det, v_mean, v_cov = v_det / nv, v_mean / nv, v_cov / nv

        metrics_f.write(json.dumps({
            "epoch": epoch, "train_det": tr_det, "val_det": v_det,
            "train_act_mean": tr_mean, "train_act_cov": tr_cov,
            "val_act_mean": v_mean, "val_act_cov": v_cov,
        }) + "\n")
        metrics_f.flush()
        flag = ""
        if v_det < best_val:  # select on detection objective, not the CORAL term
            best_val, best_epoch, since = v_det, epoch, 0
            torch.save(filt.state_dict(), out / "best.pt")
            flag = "  *best"
        else:
            since += 1
        extra = f" act_mean(t/v)={tr_mean:.3f}/{v_mean:.3f} act_cov(t/v)={tr_cov:.3f}/{v_cov:.3f}" if use_coral else ""
        print(f"[{epoch:3}/{args.epochs}] train_det={tr_det:.4f} val_det={v_det:.4f}{extra}{flag}")
        if args.patience and since >= args.patience:
            print(f"early stop (no val_det improvement for {args.patience} epochs)")
            break

    metrics_f.close()
    print(f"\nDONE in {time.time()-t0:.0f}s | best val_det={best_val:.4f} @epoch {best_epoch}")
    print(f"  wrote {out/'best.pt'}")


if __name__ == "__main__":
    main()
