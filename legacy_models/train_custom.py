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

Training modes:
  * --fine-tune-trunk  (DEFAULT, Phase 4)  End-to-end. Unfreezes blocks[-1] of the
      mobilenetv3_small_100 trunk (models/custom_multi_task.CustomSkinModel,
      fine_tune_top=True) and trains it + the 3 heads together with an ACTIVE
      image DataLoader (loads from datasets/processed/*_labels.csv on the fly,
      FFHQ crops swapped to the Phase-2 harmonized set). Differential AdamW:
      trunk lr 1e-4 / wd 1e-3, heads lr 1e-3 / wd 1e-4. The offline feature cache
      is bypassed. The new checkpoint overwrites weights/custom_skin_model.pt ONLY
      if val mean macro-F1 beats --save-if-beats (default 0.6654).
  * --no-fine-tune-trunk  Legacy. Frozen trunk; run it ONCE over every crop,
      cache the 1024-d feature vectors to runs/feature_cache/, and train only the
      3 heads on those cached features (seconds per epoch).

Acne head loss (--acne-loss):
  * ce       plain cross-entropy over the 5 ordered classes (legacy; for ablation).
  * ordinal  DEFAULT. Ordinal regression: the 5 logits are kept (so
             models/custom_multi_task.py and analysis_service.py are unchanged),
             loss is Smooth-L1 (Huber) on the softmax expected grade
             E[g] = sum_k softmax(logits)_k * k  against the 0..4 target, blended
             with CE (--acne-ordinal-beta, default 0.7). Penalises "off by 3
             grades" far more than "off by 1".
  Evaluation additionally reports Quadratic Weighted Kappa (QWK) and MAE for the
  ordinal heads (acne, pigmentation).

Outputs:
  weights/custom_skin_model.pt          checkpoint (only if it beats the gate)
  runs/ablation/trunk_finetuning.md     frozen-trunk vs fine-tuned comparison
  runs/training_metrics.json / .txt     viva report (written when the checkpoint is saved)

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe training\\train_custom.py --epochs 10
    .venv-new\\Scripts\\python.exe training\\train_custom.py --no-fine-tune-trunk --acne-loss ce
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:  # pragma: no cover - sklearn is a pinned dependency; fallback below is exact
    cohen_kappa_score = None

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
# Heads whose classes have a natural order -> report Quadratic Weighted Kappa + MAE,
# and (for acne) train with the ordinal Smooth-L1 loss. skin_type is nominal.
ORDINAL_HEADS = ("acne", "pigmentation")


# ============================================================ data
def read_labels_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    for ln in lines[1:]:
        parts = ln.split(",")
        rows.append(dict(zip(header, parts)))
    return rows


_MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
_STD_T = torch.tensor(STD).view(3, 1, 1)

# Phase 2 domain-harmonized the FFHQ clear-face crops into this dir; the label CSV
# still lists the original path, so swap when the harmonized file exists.
_CLEAR_SRC = "datasets/processed/clear_faces/"
_CLEAR_HARMONIZED = "datasets/processed/clear_faces_harmonized/"


def _resolve_image_path(rel_path: str) -> str:
    if _CLEAR_SRC in rel_path:
        cand = rel_path.replace(_CLEAR_SRC, _CLEAR_HARMONIZED)
        if (PROJECT_ROOT / cand).exists():
            return cand
    return rel_path


def _load_image_tensor(rel_path: str) -> torch.Tensor:
    from PIL import Image

    p = PROJECT_ROOT / rel_path
    with Image.open(p) as im:
        im = im.convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    return (t - _MEAN_T) / _STD_T


# ------------------------------------------------------------ active image loader
def _preload_images(rows: list[dict]) -> tuple[dict[str, np.ndarray], int]:
    """Decode every (harmonized-resolved) crop to a uint8 224x224 array once, in
    RAM. ~150 KB/image -> a few hundred MB for this dataset. Returns
    (cache_by_path, n_rows_remapped_to_harmonized)."""
    from PIL import Image

    cache: dict[str, np.ndarray] = {}
    remapped = 0
    for r in rows:
        rp = _resolve_image_path(r["image_path"])
        remapped += int(rp != r["image_path"])
        if rp in cache:
            continue
        with Image.open(PROJECT_ROOT / rp) as im:
            im = im.convert("RGB")
            if im.size != (IMG_SIZE, IMG_SIZE):
                im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            cache[rp] = np.asarray(im, dtype=np.uint8)
    return cache, remapped


class ImageLabelDataset(torch.utils.data.Dataset):
    """On-the-fly loader for end-to-end fine-tuning. Preprocessing matches
    models.custom_multi_task (MEAN/STD, 224px). Random horizontal flip when
    train + flip."""

    def __init__(self, rows: list[dict], cache: dict[str, np.ndarray], *, train: bool, flip: bool):
        self.rows = rows
        self.cache = cache
        self.train = train
        self.flip = flip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        arr = self.cache[_resolve_image_path(r["image_path"])]
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = (t - _MEAN_T) / _STD_T
        if self.train and self.flip and torch.rand(()).item() < 0.5:
            t = torch.flip(t, dims=[2])
        lab = tuple(CLASS_TO_IDX[h][r[CSV_COL[h]]] for h in HEADS)
        return t, lab[0], lab[1], lab[2]


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


def _qwk_from_confusion(cm: np.ndarray) -> float:
    """Quadratic Weighted Kappa computed directly from a confusion matrix.
    Exact match for sklearn.metrics.cohen_kappa_score(..., weights='quadratic').
    Used as a self-contained fallback when sklearn is unavailable."""
    n = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return 0.0
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    act = cm.sum(axis=1) / total          # true-label histogram
    pred = cm.sum(axis=0) / total         # predicted-label histogram
    expected = np.outer(act, pred)        # agreement expected by chance
    observed = cm / total
    denom = float((w * expected).sum())
    if denom == 0.0:
        return 0.0
    return float(1.0 - (w * observed).sum() / denom)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """QWK for an ordinal head. Prefers sklearn's implementation (as requested in
    the spec); falls back to an identical from-confusion computation."""
    if cohen_kappa_score is not None:
        try:
            k = cohen_kappa_score(
                y_true, y_pred, labels=list(range(n_classes)), weights="quadratic"
            )
            return float(k) if np.isfinite(k) else 0.0
        except Exception:  # pragma: no cover
            pass
    cm = np.asarray(confusion(y_true, y_pred, n_classes), dtype=np.float64)
    return _qwk_from_confusion(cm)


def grade_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in *grade steps* (predicted class index vs true index).
    For acne: 'off by one severity level' contributes 1.0."""
    return float(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64)).mean())


def acne_ordinal_loss(
    logits: torch.Tensor, target_idx: torch.Tensor, class_w: torch.Tensor, beta: float
) -> torch.Tensor:
    """Ordinal-aware loss for the acne head that KEEPS the existing 5-logit head
    (so nothing downstream changes shape).

        E[g]  = sum_k softmax(logits)_k * k        # differentiable expected grade
        L_reg = SmoothL1(E[g], target_grade)       # Huber, per-sample class-weighted
        L_ce  = CrossEntropy(logits, target_idx)   # keeps the per-class distribution sane
        L     = beta * L_reg + (1 - beta) * L_ce

    Smooth-L1 on the expected grade is what makes "predicted very_severe for a mild
    face" cost ~4x more than "predicted moderate for a mild face" -- plain CE treats
    both as equally wrong.
    """
    n_c = logits.shape[1]
    grades = torch.arange(n_c, dtype=logits.dtype, device=logits.device)
    e_grade = logits.softmax(dim=1) @ grades                 # (B,)
    tgt = target_idx.to(logits.dtype)
    sample_w = class_w.to(logits.device)[target_idx]         # inverse-freq weight per sample
    reg = F.smooth_l1_loss(e_grade, tgt, reduction="none")
    reg = (reg * sample_w).sum() / sample_w.sum().clamp_min(1e-8)
    ce = F.cross_entropy(logits, target_idx, weight=class_w.to(logits.device))
    return beta * reg + (1.0 - beta) * ce


# ============================================================ train / eval
def _eval_from_logits(logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict:
    out: dict = {}
    for h in HEADS:
        n_c = len(HEAD_CLASSES[h])
        pred = logits[h].argmax(1).cpu().numpy()
        true = labels[h].cpu().numpy()
        acc = float((pred == true).mean())
        entry = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1(true, pred, n_c), 4),
            "confusion": confusion(true, pred, n_c),
            "classes": HEAD_CLASSES[h],
            "support": {c: int((true == i).sum()) for i, c in enumerate(HEAD_CLASSES[h])},
        }
        if h in ORDINAL_HEADS:
            entry["qwk"] = round(quadratic_weighted_kappa(true, pred, n_c), 4)
            entry["mae"] = round(grade_mae(true, pred), 4)
        out[h] = entry
    out["mean_macro_f1"] = round(float(np.mean([out[h]["macro_f1"] for h in HEADS])), 4)
    out["mean_accuracy"] = round(float(np.mean([out[h]["accuracy"] for h in HEADS])), 4)
    out["acne_qwk"] = out["acne"]["qwk"]
    out["acne_mae"] = out["acne"]["mae"]
    return out


def evaluate(model: CustomSkinModel, feats: torch.Tensor, labels: dict[str, torch.Tensor]) -> dict:
    """Cached-feature eval (frozen-trunk path): heads only."""
    model.eval()
    with torch.inference_mode():
        logits = model.heads_from_features(feats)
    return _eval_from_logits(logits, labels)


@torch.inference_mode()
def evaluate_full(model: CustomSkinModel, loader: DataLoader) -> dict:
    """Full-model eval (image -> backbone -> heads) for the fine-tune path."""
    model.eval()
    acc_logits: dict[str, list] = {h: [] for h in HEADS}
    acc_labels: dict[str, list] = {h: [] for h in HEADS}
    for imgs, y_sk, y_ac, y_pg in loader:
        out = model(imgs)
        for h, y in zip(HEADS, (y_sk, y_ac, y_pg)):
            acc_logits[h].append(out[h])
            acc_labels[h].append(y)
    logits = {h: torch.cat(acc_logits[h]) for h in HEADS}
    labels = {h: torch.cat(acc_labels[h]) for h in HEADS}
    return _eval_from_logits(logits, labels)


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

    # per-head inverse-frequency class weights (helps the imbalanced acne head)
    class_w = {h: class_weights(tr_labels[h], len(HEAD_CLASSES[h])) for h in HEADS}
    losses = {h: nn.CrossEntropyLoss(weight=class_w[h]) for h in HEADS}
    lw = dict(zip(HEADS, args.loss_weights))
    logger.info("loss weights: %s", lw)
    logger.info("acne head loss: %s%s", args.acne_loss,
                f" (smooth-l1 beta={args.acne_ordinal_beta})" if args.acne_loss == "ordinal" else "")

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
            if args.acne_loss == "ordinal":
                l_acne = acne_ordinal_loss(logit["acne"], y_ac, class_w["acne"], args.acne_ordinal_beta)
            else:
                l_acne = losses["acne"](logit["acne"], y_ac)
            loss = (lw["skin_type"] * losses["skin_type"](logit["skin_type"], y_sk)
                    + lw["acne"] * l_acne
                    + lw["pigmentation"] * losses["pigmentation"](logit["pigmentation"], y_pg))
            loss.backward()
            opt.step()
            running += float(loss) * fb.shape[0]
        train_loss = running / len(ds)

        val = evaluate(model, va_feats, va_labels)

        def _vrow(h: str) -> dict:
            row = {"acc": val[h]["accuracy"], "f1": val[h]["macro_f1"]}
            if h in ORDINAL_HEADS:
                row["qwk"] = val[h]["qwk"]
                row["mae"] = val[h]["mae"]
            return row

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_mean_macro_f1": val["mean_macro_f1"],
                        "val_acne_qwk": val["acne_qwk"], "val_acne_mae": val["acne_mae"],
                        "val": {h: _vrow(h) for h in HEADS}})
        logger.info("epoch %d/%d  loss %.4f  val mean-F1 %.4f  acne F1/QWK/MAE %.3f/%.3f/%.3f  (acc %s)",
                    epoch, args.epochs, train_loss, val["mean_macro_f1"],
                    val["acne"]["macro_f1"], val["acne_qwk"], val["acne_mae"],
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
                 "seed": args.seed, "acne_loss": args.acne_loss,
                 "acne_ordinal_beta": args.acne_ordinal_beta},
        "dataset": {"train": tr["n"], "val": va["n"], "feature_dim": tr["feature_dim"]},
        "history": history,
        "final_val": final_val,
        "final_train": final_train,
        "best_val_mean_macro_f1": best_score,
        "final_val_acne_qwk": final_val["acne_qwk"],
        "final_val_acne_mae": final_val["acne_mae"],
    }

    _acne_desc = ("real ACNE04 grades, ordinal Smooth-L1 loss "
                  f"(beta={args.acne_ordinal_beta})" if args.acne_loss == "ordinal"
                  else "real ACNE04 grades, plain cross-entropy")
    model.save(WEIGHTS_OUT, metrics=metrics,
               notes="Frozen mobilenetv3_small_100 trunk + 3 trained heads. "
                     f"skin_type=distilled, acne={_acne_desc}, pigmentation=heuristic weak labels.")
    logger.info("saved checkpoint -> %s", WEIGHTS_OUT)
    return metrics


# ==================================================== end-to-end trunk fine-tuning
ABLATION_DIR = RUNS_DIR / "ablation"
FROZEN_BACKUP = ABLATION_DIR / "custom_skin_model.frozen_trunk.pt"


def _frozen_reference() -> dict | None:
    """Frozen-trunk (heads-only) val metrics for the before/after table -- read
    from the current checkpoint (or the frozen-trunk backup) BEFORE we overwrite."""
    for src in (FROZEN_BACKUP, WEIGHTS_OUT):
        if not src.exists():
            continue
        try:
            m = torch.load(src, map_location="cpu").get("metrics", {})
            fv = m.get("final_val", {})
            if not fv:
                continue
            return {
                "mean_macro_f1": fv.get("mean_macro_f1"),
                "acne_qwk": fv.get("acne_qwk") or fv.get("acne", {}).get("qwk"),
                "acne_mae": fv.get("acne_mae") or fv.get("acne", {}).get("mae"),
                "acne_f1": fv.get("acne", {}).get("macro_f1"),
                "skin_type_f1": fv.get("skin_type", {}).get("macro_f1"),
                "pigmentation_f1": fv.get("pigmentation", {}).get("macro_f1"),
                "wall_seconds": m.get("wall_seconds"),
                "trainable_total": (m.get("trainable_params") or {}).get("total", 12300),
                "acne_loss": m.get("args", {}).get("acne_loss"),
                "source": src.name,
            }
        except Exception:  # pragma: no cover
            continue
    return None


def _backup_frozen_checkpoint() -> None:
    if WEIGHTS_OUT.exists() and not FROZEN_BACKUP.exists():
        import shutil

        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEIGHTS_OUT, FROZEN_BACKUP)
        logger.info("backed up frozen-trunk checkpoint -> %s", FROZEN_BACKUP)


def train_finetune(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    ref = _frozen_reference()  # capture BEFORE any overwrite

    train_rows = read_labels_csv(PROCESSED / "train_labels.csv")
    val_rows = read_labels_csv(PROCESSED / "val_labels.csv")
    logger.info("train rows %d | val rows %d", len(train_rows), len(val_rows))

    model = CustomSkinModel(backbone_name=args.backbone, pretrained_backbone=True, fine_tune_top=True)

    warm = False
    if args.warm_start_heads and WEIGHTS_OUT.exists():
        try:
            prev_sd = torch.load(WEIGHTS_OUT, map_location="cpu").get("state_dict", {})
            missing, unexpected = model.load_state_dict(prev_sd, strict=False)
            warm = True
            logger.info("warm-started heads+trunk from %s (%d missing / %d unexpected keys)",
                        WEIGHTS_OUT.name, len(missing), len(unexpected))
        except Exception:
            logger.exception("warm-start failed -> training from ImageNet trunk + fresh heads")
    model._configure_trunk_trainability()   # re-assert: blocks[-1] trainable, rest frozen

    trunk_p = [p for p in model.backbone.parameters() if p.requires_grad]
    head_p = list(model.heads.parameters())
    if not trunk_p:
        raise RuntimeError("fine_tune_top=True produced no trainable trunk params")
    n_trunk = sum(p.numel() for p in trunk_p)
    n_head = sum(p.numel() for p in head_p)
    logger.info("fine-tune-trunk: trunk tensors %s (%s params) + heads (%s params)",
                model._unfrozen_trunk_names, f"{n_trunk:,}", f"{n_head:,}")

    tr_cache, tr_remap = _preload_images(train_rows)
    va_cache, va_remap = _preload_images(val_rows)
    logger.info("preloaded %d train / %d val unique images  (FFHQ->harmonized remap: %d train / %d val rows)",
                len(tr_cache), len(va_cache), tr_remap, va_remap)

    flip = not args.no_flip
    train_loader = DataLoader(ImageLabelDataset(train_rows, tr_cache, train=True, flip=flip),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ImageLabelDataset(val_rows, va_cache, train=False, flip=False),
                            batch_size=64, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(ImageLabelDataset(train_rows, tr_cache, train=False, flip=False),
                                   batch_size=64, shuffle=False, num_workers=0)

    tr_label_idx = {h: torch.tensor([CLASS_TO_IDX[h][r[CSV_COL[h]]] for r in train_rows], dtype=torch.long)
                    for h in HEADS}
    class_w = {h: class_weights(tr_label_idx[h], len(HEAD_CLASSES[h])) for h in HEADS}
    losses = {h: nn.CrossEntropyLoss(weight=class_w[h]) for h in HEADS}
    lw = dict(zip(HEADS, args.loss_weights))
    logger.info("loss weights %s | acne loss: %s%s", lw, args.acne_loss,
                f" (smooth-l1 beta={args.acne_ordinal_beta})" if args.acne_loss == "ordinal" else "")

    # ---- differential AdamW: slow LR + strong decay on the trunk, fast on the heads
    opt = torch.optim.AdamW([
        {"params": trunk_p, "lr": args.trunk_lr, "weight_decay": args.trunk_wd},
        {"params": head_p, "lr": args.head_lr, "weight_decay": args.head_wd},
    ])
    logger.info("AdamW groups: trunk lr=%g wd=%g | heads lr=%g wd=%g",
                args.trunk_lr, args.trunk_wd, args.head_lr, args.head_wd)

    history: list[dict] = []
    best_state = None
    best_score = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()                       # note: trunk BN stays in eval (CustomSkinModel.train)
        running, n_seen = 0.0, 0
        for imgs, y_sk, y_ac, y_pg in train_loader:
            opt.zero_grad()
            out = model(imgs)               # full forward -> grad to blocks[-1] + heads
            if args.acne_loss == "ordinal":
                l_acne = acne_ordinal_loss(out["acne"], y_ac, class_w["acne"], args.acne_ordinal_beta)
            else:
                l_acne = losses["acne"](out["acne"], y_ac)
            loss = (lw["skin_type"] * losses["skin_type"](out["skin_type"], y_sk)
                    + lw["acne"] * l_acne
                    + lw["pigmentation"] * losses["pigmentation"](out["pigmentation"], y_pg))
            loss.backward()
            opt.step()
            running += float(loss) * imgs.shape[0]
            n_seen += imgs.shape[0]
        train_loss = running / max(n_seen, 1)

        val = evaluate_full(model, val_loader)

        def _vrow(h: str) -> dict:
            row = {"acc": val[h]["accuracy"], "f1": val[h]["macro_f1"]}
            if h in ORDINAL_HEADS:
                row["qwk"] = val[h]["qwk"]
                row["mae"] = val[h]["mae"]
            return row

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_mean_macro_f1": val["mean_macro_f1"],
                        "val_acne_qwk": val["acne_qwk"], "val_acne_mae": val["acne_mae"],
                        "val": {h: _vrow(h) for h in HEADS}})
        logger.info("epoch %d/%d  loss %.4f  val mean-F1 %.4f  acne F1/QWK/MAE %.3f/%.3f/%.3f  (acc %s)",
                    epoch, args.epochs, train_loss, val["mean_macro_f1"],
                    val["acne"]["macro_f1"], val["acne_qwk"], val["acne_mae"],
                    {h: val[h]["accuracy"] for h in HEADS})

        if val["mean_macro_f1"] > best_score:
            best_score = val["mean_macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_full(model, val_loader)
    final_train = evaluate_full(model, train_eval_loader)
    wall = time.time() - t0

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 1),
        "mode": "fine_tune_trunk",
        "args": {"epochs": args.epochs, "batch_size": args.batch_size,
                 "lr": args.head_lr, "trunk_lr": args.trunk_lr, "head_lr": args.head_lr,
                 "trunk_wd": args.trunk_wd, "head_wd": args.head_wd, "optimizer": "AdamW (differential)",
                 "backbone": args.backbone, "loss_weights": lw, "flip_aug": flip,
                 "seed": args.seed, "acne_loss": args.acne_loss,
                 "acne_ordinal_beta": args.acne_ordinal_beta, "fine_tune_trunk": True,
                 "warm_start_heads": warm},
        "dataset": {"train": len(train_rows), "val": len(val_rows), "feature_dim": model.feature_dim,
                    "harmonized_remapped_train": tr_remap, "harmonized_remapped_val": va_remap},
        "trainable_params": {"trunk": n_trunk, "heads": n_head, "total": n_trunk + n_head,
                             "trunk_tensors": list(model._unfrozen_trunk_names)},
        "history": history,
        "final_val": final_val,
        "final_train": final_train,
        "best_val_mean_macro_f1": best_score,
        "final_val_acne_qwk": final_val["acne_qwk"],
        "final_val_acne_mae": final_val["acne_mae"],
    }

    # ---- save gate ----
    gate = args.save_if_beats
    saved = best_score > gate
    metrics["save_gate"] = gate
    metrics["saved_to_checkpoint"] = saved
    if saved:
        _backup_frozen_checkpoint()
        model.save(WEIGHTS_OUT, metrics=metrics,
                   notes=f"mobilenetv3_small_100 with blocks[-1] FINE-TUNED end-to-end on the "
                         f"harmonized dataset (AdamW differential LR: trunk {args.trunk_lr:g}, "
                         f"heads {args.head_lr:g}) + 3 trained heads. acne=ordinal "
                         f"Smooth-L1(beta={args.acne_ordinal_beta})+CE.")
        logger.info("val mean macro-F1 %.4f > %.4f gate -> SAVED fine-tuned checkpoint to %s",
                    best_score, gate, WEIGHTS_OUT)
    else:
        logger.warning("val mean macro-F1 %.4f <= %.4f gate -> checkpoint UNCHANGED (fine-tune not saved)",
                       best_score, gate)

    _write_trunk_ablation(metrics, ref)
    return metrics


def _write_trunk_ablation(metrics: dict, ref: dict | None) -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    fv = metrics["final_val"]
    a = metrics["args"]

    def row(name: str, before, after, fmt: str = ".4f") -> str:
        if before is None or after is None:
            return f"| {name} | {before} | {after} | - |"
        d = after - before
        return f"| {name} | {before:{fmt}} | {after:{fmt}} | {'+' if d >= 0 else ''}{d:{fmt}} |"

    L: list[str] = []
    L.append("# Trunk Fine-Tuning Ablation -- Frozen Trunk vs. `blocks[-1]` Fine-Tuned")
    L.append("")
    L.append(f"_Generated {metrics['trained_at']}_")
    L.append("")
    L.append(f"- **Dataset:** {metrics['dataset']['train']} train / {metrics['dataset']['val']} val  "
             f"({metrics['dataset']['harmonized_remapped_train']} train + "
             f"{metrics['dataset']['harmonized_remapped_val']} val FFHQ rows remapped to the "
             f"Phase-2 harmonized crops)")
    L.append(f"- **Fine-tuned:** `blocks[-1]` of {a['backbone']} unfrozen "
             f"({metrics['trainable_params']['trunk']:,} params) + 3 heads "
             f"({metrics['trainable_params']['heads']:,}); active image DataLoader (no feature cache)")
    L.append(f"- **Optimizer:** AdamW differential -- trunk lr {a['trunk_lr']:g} / wd {a['trunk_wd']:g}, "
             f"heads lr {a['head_lr']:g} / wd {a['head_wd']:g}; {a['epochs']} epochs; "
             f"acne loss = {a['acne_loss']} (beta {a['acne_ordinal_beta']}); "
             f"warm-start heads = {a.get('warm_start_heads')}")
    L.append(f"- **Frozen-trunk reference:** {ref['source'] if ref else 'unavailable'}")
    L.append(f"- **Save gate:** val mean macro-F1 > {metrics['save_gate']}  ->  "
             + ("**SAVED to `weights/custom_skin_model.pt`**" if metrics["saved_to_checkpoint"]
                else "**NOT saved (checkpoint unchanged)**"))
    L.append("")
    L.append("| Metric (validation) | Frozen trunk (heads only) | Fine-tuned trunk (`blocks[-1]` + heads) | Delta |")
    L.append("|---|---|---|---|")
    if ref:
        L.append(row("mean macro-F1", ref["mean_macro_f1"], fv["mean_macro_f1"]))
        L.append(row("acne QWK (quadratic)", ref["acne_qwk"], fv["acne_qwk"]))
        L.append(row("acne MAE (grades)", ref["acne_mae"], fv["acne_mae"]))
        L.append(row("acne macro-F1", ref["acne_f1"], fv["acne"]["macro_f1"]))
        L.append(row("skin_type macro-F1", ref["skin_type_f1"], fv["skin_type"]["macro_f1"]))
        L.append(row("pigmentation macro-F1", ref["pigmentation_f1"], fv["pigmentation"]["macro_f1"]))
    else:
        L.append("| _(frozen-trunk reference metrics unavailable)_ | - | - | - |")
    ref_tp = ref["trainable_total"] if ref else 12300
    L.append(f"| trainable params | {ref_tp:,} | {metrics['trainable_params']['total']:,} "
             f"| +{metrics['trainable_params']['trunk']:,} (trunk) |")
    L.append(f"| wall time | {ref['wall_seconds'] if ref else 'n/a'} s (cached features) "
             f"| {metrics['wall_seconds']} s (on-the-fly, {a['epochs']} ep) | - |")
    L.append("")
    L.append("## Per-epoch (fine-tuned run)")
    L.append("")
    L.append("| epoch | train loss | val mean macro-F1 | acne QWK | acne MAE |")
    L.append("|---|---|---|---|---|")
    for h in metrics["history"]:
        L.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_mean_macro_f1']:.4f} "
                 f"| {h['val_acne_qwk']:.4f} | {h['val_acne_mae']:.3f} |")
    L.append("")
    if ref:
        better = fv["mean_macro_f1"] > ref["mean_macro_f1"]
        L.append(f"**Verdict:** fine-tuning `blocks[-1]` "
                 + ("**improved**" if better else "**did not improve**")
                 + f" the validation mean macro-F1 "
                 + f"({ref['mean_macro_f1']:.4f} -> {fv['mean_macro_f1']:.4f}, "
                 + f"{'+' if fv['mean_macro_f1'] - ref['mean_macro_f1'] >= 0 else ''}"
                 + f"{fv['mean_macro_f1'] - ref['mean_macro_f1']:.4f}).")
    (ABLATION_DIR / "trunk_finetuning.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ABLATION_DIR / "trunk_finetuning_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("wrote runs/ablation/trunk_finetuning.md")


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
    _ft = a.get("fine_tune_trunk", False)
    if _ft:
        L.append(f"Backbone          : {a['backbone']}  ->  blocks[-1] FINE-TUNED end-to-end "
                 f"({metrics.get('trainable_params', {}).get('trunk', '?')} params) + heads")
        L.append(f"Optimizer         : AdamW differential  trunk lr={a.get('trunk_lr')}/wd={a.get('trunk_wd')}  "
                 f"heads lr={a.get('head_lr')}/wd={a.get('head_wd')}")
    else:
        L.append(f"Backbone (frozen) : {a['backbone']}  ->  {metrics['dataset']['feature_dim']}-d features")
    L.append("Heads (trained)   : " + ", ".join(f"{h}({len(HEAD_CLASSES[h])})" for h in HEADS))
    L.append(f"Hyperparams       : epochs={a['epochs']} batch={a['batch_size']} lr={a.get('lr', a.get('head_lr'))} "
             f"loss_w={a['loss_weights']} flip_aug={a['flip_aug']}")
    _al = a.get("acne_loss", "ce")
    L.append(f"Acne head loss    : {_al}"
             + (f"  (Smooth-L1 on expected grade, beta={a.get('acne_ordinal_beta')}, + CE)"
                if _al == "ordinal" else "  (plain cross-entropy)"))
    L.append(f"Samples           : train={metrics['dataset']['train']}  val={metrics['dataset']['val']}")
    L.append("")
    L.append("LABEL PROVENANCE")
    L.append("-" * 70)
    for col, info in (prov.get("columns", {}) or {}).items():
        L.append(f"  {col:13s} : {info.get('source', '?')}")
        L.append(f"  {'':13s}   {info.get('method', '')}")
    L.append("")
    L.append("PER-EPOCH  (validation:  per-head acc/macroF1  |  acne QWK / MAE)")
    L.append("-" * 70)
    for h in metrics["history"]:
        per = " ".join(f"{k}:{v['acc']:.2f}/{v['f1']:.2f}" for k, v in h["val"].items())
        aq = h.get("val_acne_qwk")
        am = h.get("val_acne_mae")
        tail = f"   acne QWK {aq:.4f} MAE {am:.3f}" if aq is not None else ""
        L.append(f"  epoch {h['epoch']:2d}  loss {h['train_loss']:.4f}  meanF1 {h['val_mean_macro_f1']:.4f}   [{per}]{tail}")
    L.append("")
    L.append("FINAL VALIDATION METRICS  (acc / macro-F1)")
    L.append("-" * 70)
    fv = metrics["final_val"]
    for h in HEADS:
        d = fv[h]
        L.append(f"  {h:13s}  acc {d['accuracy']:.4f}   macroF1 {d['macro_f1']:.4f}   support {d['support']}")
        if "qwk" in d:
            L.append(f"  {'':13s}  QWK {d['qwk']:.4f}   MAE {d['mae']:.4f}  (ordinal head)")
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
        extra = (f"   QWK {ft[h]['qwk']:.4f}  MAE {ft[h]['mae']:.4f}" if "qwk" in ft[h] else "")
        L.append(f"  {h:13s}  acc {ft[h]['accuracy']:.4f}   macroF1 {ft[h]['macro_f1']:.4f}{extra}")
    _tr_f1_gap = ft["acne"]["macro_f1"] - fv["acne"]["macro_f1"]
    _tr_qwk_gap = ft["acne"].get("qwk", 0.0) - fv["acne"].get("qwk", 0.0)
    L.append(f"  acne train/val gap:  macroF1 {_tr_f1_gap:+.3f}   QWK {_tr_qwk_gap:+.3f}")
    L.append("")
    L.append("NOTES FOR THE VIVA")
    L.append("-" * 70)
    if _ft:
        L.append("  * The trunk is ImageNet-pretrained; its FINAL inverted-residual block (blocks[-1])")
        L.append("    was fine-tuned end-to-end with an active image DataLoader, the rest stays frozen.")
        L.append("    Differential AdamW (low LR + strong weight-decay on the trunk). BatchNorm running")
        L.append("    stats stay frozen. FFHQ crops use the Phase-2 domain-harmonized images.")
    else:
        L.append("  * The trunk is ImageNet-pretrained and FROZEN (transfer learning); only the")
        L.append("    three heads were trained, on our own prepared dataset.")
    L.append("  * acne_severity labels are REAL ACNE04 Hayashi grades.")
    if _al == "ordinal":
        L.append("  * The acne head is trained as ORDINAL regression: Smooth-L1 (Huber) on the")
        L.append("    softmax expected grade E[g]=sum_k p_k*k against the 0..4 target, blended")
        L.append(f"    with CE (beta={a.get('acne_ordinal_beta')}). The 5-logit head is unchanged, so")
        L.append("    models/custom_multi_task.py and analysis_service.py are untouched.")
        L.append("  * Report Quadratic Weighted Kappa (penalises far-off grades) + grade MAE,")
        L.append("    not just accuracy/F1, for acne severity.")
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="heads LR for the frozen-trunk (cached) path")
    ap.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--fine-tune-trunk", action=argparse.BooleanOptionalAction, default=True,
                    help="Phase 4: end-to-end train blocks[-1] + heads with an active image loader "
                         "(default). --no-fine-tune-trunk = legacy frozen-trunk cached-feature path.")
    ap.add_argument("--trunk-lr", type=float, default=1e-4, help="AdamW LR for the unfrozen trunk block")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="AdamW LR for the classifier heads")
    ap.add_argument("--trunk-wd", type=float, default=1e-3, help="AdamW weight decay for the trunk block")
    ap.add_argument("--head-wd", type=float, default=1e-4, help="AdamW weight decay for the heads")
    ap.add_argument("--save-if-beats", type=float, default=0.6654,
                    help="overwrite weights/custom_skin_model.pt only if val mean macro-F1 exceeds this")
    ap.add_argument("--warm-start-heads", action=argparse.BooleanOptionalAction, default=True,
                    help="fine-tune from the existing checkpoint's head weights (default) rather than "
                         "re-learning the heads from scratch in 10 epochs")
    ap.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 1.0, 0.5],
                    metavar=("W_SKIN", "W_ACNE", "W_PIGMENT"))
    ap.add_argument("--acne-loss", choices=["ce", "ordinal"], default="ordinal",
                    help="acne head loss: 'ordinal' (default; Smooth-L1 on the softmax "
                         "expected grade + CE blend) or 'ce' (legacy plain cross-entropy, "
                         "kept for the ablation table)")
    ap.add_argument("--acne-ordinal-beta", type=float, default=0.7,
                    help="weight on the Smooth-L1 term when --acne-loss=ordinal (the rest is CE)")
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip feature augmentation")
    ap.add_argument("--rebuild-cache", action="store_true", help="delete and rebuild the feature cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (PROCESSED / "train_labels.csv").exists():
        logger.error("no datasets/processed/train_labels.csv -- run scripts/prepare_dataset.py first")
        return 2

    if args.fine_tune_trunk:
        metrics = train_finetune(args)
        if metrics.get("saved_to_checkpoint"):
            write_report(metrics)
        else:
            logger.warning(
                "fine-tuned model did not beat the %.4f gate -> weights/custom_skin_model.pt and "
                "runs/training_report.txt are UNCHANGED. Full comparison: runs/ablation/trunk_finetuning.md",
                metrics["save_gate"])
    else:
        metrics = train(args)
        write_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
