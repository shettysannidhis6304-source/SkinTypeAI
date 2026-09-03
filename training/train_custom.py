#!/usr/bin/env python3
"""
training/train_custom.py

Train the project's OWN multi-task skin model (models/custom_multi_task.py) on
the locally prepared ACNE04 crops + labels (scripts/prepare_dataset.py).

CPU-only, Dell Inspiron friendly:
  * torch.set_num_threads(4), num_workers=0
  * the backbone (mobilenetv3_small_100) is FROZEN -> we run it ONCE over every
    crop and cache the 576-d feature vectors to runs/feature_cache/. Training
    then optimises only the three classifier heads on those cached features,
    which takes seconds per epoch. Horizontal-flip features are cached too, so
    the one useful augmentation survives the caching step.
  * end-to-end target: well under 20 minutes (the feature pass is the only
    non-trivial cost; heads train almost instantly).

Outputs:
  weights/custom_skin_model.pt      final checkpoint (best mean val macro-F1)
  runs/training_metrics.json        machine-readable metrics
  runs/training_report.txt          human-readable viva report

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe training\\train_custom.py
    .venv-new\\Scripts\\python.exe training\\train_custom.py --epochs 8 --batch-size 16 --lr 1e-3
    .venv-new\\Scripts\\python.exe training\\train_custom.py --rebuild-cache
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.custom_multi_task import (  # noqa: E402
    HEAD_CLASSES,
    IMG_SIZE,
    MEAN,
    STD,
    CustomSkinModel,
    trainable_parameters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_custom")

PROCESSED = PROJECT_ROOT / "datasets" / "processed"
CACHE_DIR = PROJECT_ROOT / "runs" / "feature_cache"
WEIGHTS_OUT = PROJECT_ROOT / "weights" / "custom_skin_model.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

HEADS = list(HEAD_CLASSES)  # ["skin_type", "acne", "pigmentation"]
CLASS_TO_IDX = {h: {c: i for i, c in enumerate(HEAD_CLASSES[h])} for h in HEADS}
# CSV column name -> head name (prepare_dataset.py writes "acne_severity")
CSV_COL = {"skin_type": "skin_type", "acne": "acne_severity", "pigmentation": "pigmentation"}


# ============================================================ data
def read_labels_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    for ln in lines[1:]:
        parts = ln.split(",")
        rows.append(dict(zip(header, parts)))
    return rows


def _load_image_tensor(rel_path: str) -> torch.Tensor:
    from PIL import Image

    p = PROJECT_ROOT / rel_path
    with Image.open(p) as im:
        im = im.convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t - mean) / std


@torch.inference_mode()
def build_feature_cache(model: CustomSkinModel, rows: list[dict], split: str, flip_aug: bool) -> dict:
    """Run the frozen backbone once over every crop (and its h-flip for train)
    and cache features + integer labels. Returns {feats, labels{head:tensor}}."""
    cache_path = CACHE_DIR / f"{split}.pt"
    if cache_path.exists():
        logger.info("using existing feature cache %s", cache_path)
        return torch.load(cache_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    feats_all: list[torch.Tensor] = []
    labels_all: dict[str, list[int]] = {h: [] for h in HEADS}

    batch: list[torch.Tensor] = []
    batch_lbls: list[tuple[int, int, int]] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch)
        f = model.extract_features(x)
        feats_all.append(f.cpu())
        for lset in batch_lbls:
            for h, v in zip(HEADS, lset):
                labels_all[h].append(v)
        batch.clear()
        batch_lbls.clear()

    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            img = _load_image_tensor(r["image_path"])
        except Exception as e:
            logger.warning("skip %s (%s)", r.get("image_path"), e)
            continue
        lset = tuple(CLASS_TO_IDX[h][r[CSV_COL[h]]] for h in HEADS)
        batch.append(img)
        batch_lbls.append(lset)
        if flip_aug and split == "train":
            batch.append(torch.flip(img, dims=[2]))
            batch_lbls.append(lset)
        if len(batch) >= 32:
            flush()
        if i % 200 == 0:
            logger.info("  [%s] features %d / %d  (%.0fs)", split, i, len(rows), time.time() - t0)
    flush()

    feats = torch.cat(feats_all)
    labels = {h: torch.tensor(labels_all[h], dtype=torch.long) for h in HEADS}
    out = {"feats": feats, "labels": labels, "n": feats.shape[0], "feature_dim": feats.shape[1]}
    torch.save(out, cache_path)
    logger.info("cached %d %s feature rows -> %s (%.0fs)", feats.shape[0], split, cache_path, time.time() - t0)
    return out


def class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=n_classes).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


# ============================================================ metrics
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m.tolist()


# ============================================================ train / eval
def evaluate(model: CustomSkinModel, feats: torch.Tensor, labels: dict[str, torch.Tensor]) -> dict:
    model.eval()
    with torch.inference_mode():
        logits = model.heads_from_features(feats)
    out: dict = {}
    for h in HEADS:
        n_c = len(HEAD_CLASSES[h])
        pred = logits[h].argmax(1).numpy()
        true = labels[h].numpy()
        acc = float((pred == true).mean())
        out[h] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1(true, pred, n_c), 4),
            "confusion": confusion(true, pred, n_c),
            "classes": HEAD_CLASSES[h],
            "support": {c: int((true == i).sum()) for i, c in enumerate(HEAD_CLASSES[h])},
        }
    out["mean_macro_f1"] = round(float(np.mean([out[h]["macro_f1"] for h in HEADS])), 4)
    out["mean_accuracy"] = round(float(np.mean([out[h]["accuracy"] for h in HEADS])), 4)
    return out


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    train_rows = read_labels_csv(PROCESSED / "train_labels.csv")
    val_rows = read_labels_csv(PROCESSED / "val_labels.csv")
    logger.info("train rows %d | val rows %d", len(train_rows), len(val_rows))

    model = CustomSkinModel(backbone_name=args.backbone, pretrained_backbone=True)
    logger.info("model: %s backbone (frozen, %d-d) + heads %s",
                model.backbone_name, model.feature_dim,
                {h: len(HEAD_CLASSES[h]) for h in HEADS})

    if args.rebuild_cache:
        for f in CACHE_DIR.glob("*.pt"):
            f.unlink()

    tr = build_feature_cache(model, train_rows, "train", flip_aug=not args.no_flip)
    va = build_feature_cache(model, val_rows, "val", flip_aug=False)

    tr_feats, tr_labels = tr["feats"], tr["labels"]
    va_feats, va_labels = va["feats"], va["labels"]

    # per-head loss with inverse-frequency class weights (helps the imbalanced acne head)
    losses = {
        h: nn.CrossEntropyLoss(weight=class_weights(tr_labels[h], len(HEAD_CLASSES[h])))
        for h in HEADS
    }
    lw = dict(zip(HEADS, args.loss_weights))
    logger.info("loss weights: %s", lw)

    ds = TensorDataset(tr_feats, tr_labels["skin_type"], tr_labels["acne"], tr_labels["pigmentation"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(trainable_parameters(model), lr=args.lr, weight_decay=1e-4)

    history: list[dict] = []
    best_state = None
    best_score = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for fb, y_sk, y_ac, y_pg in dl:
            opt.zero_grad()
            logit = model.heads_from_features(fb)
            loss = (lw["skin_type"] * losses["skin_type"](logit["skin_type"], y_sk)
                    + lw["acne"] * losses["acne"](logit["acne"], y_ac)
                    + lw["pigmentation"] * losses["pigmentation"](logit["pigmentation"], y_pg))
            loss.backward()
            opt.step()
            running += float(loss) * fb.shape[0]
        train_loss = running / len(ds)

        val = evaluate(model, va_feats, va_labels)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_mean_macro_f1": val["mean_macro_f1"],
                        "val": {h: {"acc": val[h]["accuracy"], "f1": val[h]["macro_f1"]} for h in HEADS}})
        logger.info("epoch %d/%d  train_loss %.4f  val mean-F1 %.4f  (%s)",
                    epoch, args.epochs, train_loss, val["mean_macro_f1"],
                    {h: val[h]["accuracy"] for h in HEADS})

        if val["mean_macro_f1"] > best_score:
            best_score = val["mean_macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, va_feats, va_labels)
    final_train = evaluate(model, tr_feats, tr_labels)
    wall = time.time() - t0

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 1),
        "args": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "backbone": args.backbone, "loss_weights": lw, "flip_aug": not args.no_flip,
                 "seed": args.seed},
        "dataset": {"train": tr["n"], "val": va["n"], "feature_dim": tr["feature_dim"]},
        "history": history,
        "final_val": final_val,
        "final_train": final_train,
        "best_val_mean_macro_f1": best_score,
    }

    model.save(WEIGHTS_OUT, metrics=metrics,
               notes="Frozen mobilenetv3_small_100 trunk + 3 trained heads. "
                     "skin_type=distilled, acne=real ACNE04 grades, pigmentation=heuristic weak labels.")
    logger.info("saved checkpoint -> %s", WEIGHTS_OUT)
    return metrics


# ============================================================ report
def write_report(metrics: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prov = {}
    prov_path = PROCESSED / "label_provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())

    L: list[str] = []
    L.append("=" * 70)
    L.append("SkinTypeAI -- Custom Multi-Task Model -- Training Report")
    L.append("=" * 70)
    L.append(f"Trained at        : {metrics['trained_at']}")
    L.append(f"Wall time         : {metrics['wall_seconds']} s")
    a = metrics["args"]
    L.append(f"Backbone (frozen) : {a['backbone']}  ->  {metrics['dataset']['feature_dim']}-d features")
    L.append("Heads (trained)   : " + ", ".join(f"{h}({len(HEAD_CLASSES[h])})" for h in HEADS))
    L.append(f"Hyperparams       : epochs={a['epochs']} batch={a['batch_size']} lr={a['lr']} "
             f"loss_w={a['loss_weights']} flip_aug={a['flip_aug']}")
    L.append(f"Samples           : train={metrics['dataset']['train']}  val={metrics['dataset']['val']}")
    L.append("")
    L.append("LABEL PROVENANCE")
    L.append("-" * 70)
    for col, info in (prov.get("columns", {}) or {}).items():
        L.append(f"  {col:13s} : {info.get('source', '?')}")
        L.append(f"  {'':13s}   {info.get('method', '')}")
    L.append("")
    L.append("PER-EPOCH (validation mean macro-F1)")
    L.append("-" * 70)
    for h in metrics["history"]:
        per = " ".join(f"{k}:{v['acc']:.2f}/{v['f1']:.2f}" for k, v in h["val"].items())
        L.append(f"  epoch {h['epoch']:2d}  loss {h['train_loss']:.4f}  meanF1 {h['val_mean_macro_f1']:.4f}   [{per}]")
    L.append("")
    L.append("FINAL VALIDATION METRICS  (acc / macro-F1)")
    L.append("-" * 70)
    fv = metrics["final_val"]
    for h in HEADS:
        d = fv[h]
        L.append(f"  {h:13s}  acc {d['accuracy']:.4f}   macroF1 {d['macro_f1']:.4f}   support {d['support']}")
        L.append(f"  {'':13s}  classes  {d['classes']}")
        L.append(f"  {'':13s}  confusion (rows=true, cols=pred):")
        for cls, row in zip(d["classes"], d["confusion"]):
            L.append(f"  {'':15s}{cls:12s} {row}")
    L.append("")
    L.append(f"  MEAN over heads:  acc {fv['mean_accuracy']:.4f}   macroF1 {fv['mean_macro_f1']:.4f}")
    L.append("")
    L.append("FINAL TRAIN METRICS  (acc / macro-F1)  -- overfit check")
    L.append("-" * 70)
    ft = metrics["final_train"]
    for h in HEADS:
        L.append(f"  {h:13s}  acc {ft[h]['accuracy']:.4f}   macroF1 {ft[h]['macro_f1']:.4f}")
    L.append("")
    L.append("NOTES FOR THE VIVA")
    L.append("-" * 70)
    L.append("  * The trunk is ImageNet-pretrained and FROZEN (transfer learning); only the")
    L.append("    three heads were trained, on our own prepared dataset.")
    L.append("  * acne_severity labels are REAL ACNE04 Hayashi grades.")
    L.append("  * skin_type labels are DISTILLED once from a local ViT teacher and are")
    L.append("    domain-biased (teacher applied to acne-patient faces).")
    L.append("  * pigmentation labels are self-generated HEURISTIC weak labels (CIELAB L*")
    L.append("    unevenness + dark-spot fraction); this head is the weakest and is")
    L.append("    presented as indicative only.")
    L.append("  * At runtime the backend loads weights/custom_skin_model.pt with NO network")
    L.append("    / Hugging Face Hub dependency (backbone rebuilt with pretrained=False).")
    L.append("=" * 70)

    (RUNS_DIR / "training_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("wrote runs/training_report.txt and runs/training_metrics.json")
    print("\n".join(L))


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 1.0, 0.5],
                    metavar=("W_SKIN", "W_ACNE", "W_PIGMENT"))
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip feature augmentation")
    ap.add_argument("--rebuild-cache", action="store_true", help="delete and rebuild the feature cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (PROCESSED / "train_labels.csv").exists():
        logger.error("no datasets/processed/train_labels.csv -- run scripts/prepare_dataset.py first")
        return 2

    metrics = train(args)
    write_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
