#!/usr/bin/env python3
"""
training/train_skin_analysis.py  --  STAGE 3 of the from-scratch rebuild

Trains models.skin_analysis_model.SkinAnalysisModel (< 450k params, 100% random
Kaiming init -- no pretrained weights) on the Stage-1 expanded set
(datasets/processed_scratch/) for the 5 model-backed characteristics:

    acne (4 grades) | wrinkles (3) | oiliness (3) | dryness (3) | pigmentation (3)

CPU-safe: torch.set_num_threads(4), batch size 32, num_workers=0, images
preloaded to RAM once.

Optimiser  : AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(T_max=epochs)

Loss (all of the below stack):
  * acne head           -> multi-class FocalLoss(gamma=2.0)  (hard-example focus,
                           the acne grade histogram is skewed).
  * oiliness / dryness  -> CrossEntropy with label smoothing (eps=0.1) -- their
                           Stage-1 weak labels are noisy tier boundaries.
  * wrinkles / pigment  -> plain CrossEntropy.
  * task balancing      -> dynamic homoscedastic uncertainty weighting
                           (Kendall, Gal & Cipolla 2018): 5 learnable log-variance
                           parameters optimised alongside the weights in AdamW.
                           `--loss-balance fixed` falls back to static weights.
  * augmentation        -> intra-grade MixUp (alpha=0.2) / CutMix on a random
                           fraction of batches. "Intra-grade" = each sample is only
                           mixed with another sample of the SAME acne grade, so the
                           acne label stays exact; the other 4 heads get the
                           standard lam-blended loss.

Outputs:
    weights/skin_analysis_model.pt     best-val checkpoint ({state_dict, config, metrics, ...})
    runs/skin_analysis_report.txt      human-readable report + per-head confusion matrices
    runs/skin_analysis_metrics.json    machine-readable metrics + loss history

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe training\\train_skin_analysis.py
    .venv-new\\Scripts\\python.exe training\\train_skin_analysis.py --epochs 15 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:  # pragma: no cover - sklearn is pinned; exact fallback below
    cohen_kappa_score = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skin_analysis_model import (  # noqa: E402
    HEAD_CLASSES,
    MEAN,
    STD,
    SkinAnalysisModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_skin_analysis")

SCRATCH = PROJECT_ROOT / "datasets" / "processed_scratch"
WEIGHTS_OUT = PROJECT_ROOT / "weights" / "skin_analysis_model.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

HEADS = list(HEAD_CLASSES)                       # acne, wrinkles, oiliness, dryness, pigmentation
CLASS_TO_IDX = {h: {c: i for i, c in enumerate(HEAD_CLASSES[h])} for h in HEADS}
MANIFEST_COL = {"acne": "acne_severity", "wrinkles": "wrinkles", "oiliness": "oiliness",
                "dryness": "dryness", "pigmentation": "pigmentation"}
# Static fallback weights (used only when --loss-balance fixed).
LOSS_WEIGHTS = {"acne": 1.0, "wrinkles": 1.0, "oiliness": 0.8, "dryness": 0.8, "pigmentation": 0.8}
ORDINAL_HEADS = ("acne",)                        # acne grade -> also report QWK + MAE

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1                            # oiliness + dryness heads
LABEL_SMOOTHED_HEADS = ("oiliness", "dryness")
MIXUP_ALPHA = 0.2
MIX_PROB = 0.5                                   # fraction of batches that get MixUp/CutMix

_MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
_STD_T = torch.tensor(STD).view(3, 1, 1)


# ============================================================ losses
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017): FL = (1 - p_t)^gamma * CE.

    `weight` (optional) is a per-class tensor, exactly like nn.CrossEntropyLoss.
    Reduction is always 'mean' over the batch -> a scalar.
    """

    def __init__(self, gamma: float = FOCAL_GAMMA, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)                                   # p_t of the true class
        return ((1.0 - pt) ** self.gamma * ce).mean()


class MultiTaskUncertaintyLoss(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, 2018).

    Holds one learnable log-variance s_h = log(sigma_h^2) per head. Given the raw
    per-head losses it returns

        L = sum_h [ exp(-s_h) * L_h + 0.5 * s_h ]

    The 0.5*s_h term stops the variances collapsing to 0. `s_h` is optimised by
    the SAME AdamW as the network weights (in its own no-weight-decay group).
    """

    def __init__(self, heads: list[str]) -> None:
        super().__init__()
        self.heads = list(heads)
        self.log_var = nn.Parameter(torch.zeros(len(self.heads)))

    def forward(self, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        total: torch.Tensor | None = None
        detail: dict[str, dict] = {}
        for i, h in enumerate(self.heads):
            s = self.log_var[i]
            precision = torch.exp(-s)
            term = precision * raw[h] + 0.5 * s
            total = term if total is None else total + term
            detail[h] = {
                "raw": round(float(raw[h].detach()), 4),
                "sigma": round(float(torch.exp(0.5 * s).detach()), 4),
                "precision": round(float(precision.detach()), 4),
            }
        return total, detail

    def sigmas(self) -> dict[str, float]:
        return {h: round(float(torch.exp(0.5 * self.log_var[i]).detach()), 4)
                for i, h in enumerate(self.heads)}


def build_head_criteria() -> dict[str, nn.Module]:
    """acne -> Focal(gamma=2); oiliness/dryness -> CE(label_smoothing=0.1);
    wrinkles/pigmentation -> plain CE."""
    crit: dict[str, nn.Module] = {}
    for h in HEADS:
        if h == "acne":
            crit[h] = FocalLoss(gamma=FOCAL_GAMMA)
        elif h in LABEL_SMOOTHED_HEADS:
            crit[h] = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        else:
            crit[h] = nn.CrossEntropyLoss()
    return crit


# ============================================================ mixup / cutmix
def mix_batch(
    imgs: torch.Tensor,
    labels: tuple[torch.Tensor, ...],
    *,
    alpha: float = MIXUP_ALPHA,
    p: float = MIX_PROB,
    mode: str = "both",
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], float, str]:
    """Intra-grade MixUp / CutMix.

    `labels` is (y_acne, y_wrinkles, y_oiliness, y_dryness, y_pigmentation) -- index
    0 MUST be the acne grade. Each sample is paired only with another sample that
    shares its acne grade, so the acne label is unambiguous after mixing.

    Returns (mixed_imgs, labels_a, labels_b, lam, applied) where the per-head loss
    is  lam * L(., labels_a) + (1 - lam) * L(., labels_b).  `applied` is one of
    "none" / "mixup" / "cutmix".
    """
    b = imgs.size(0)
    if b < 2 or float(torch.rand(())) >= p:
        return imgs, labels, labels, 1.0, "none"

    use_cutmix = mode == "cutmix" or (mode == "both" and float(torch.rand(())) < 0.5)
    lam = float(np.random.beta(alpha, alpha))

    # intra-grade permutation: shuffle indices only within each acne-grade bucket
    y_acne = labels[0]
    perm = torch.arange(b)
    for g in torch.unique(y_acne):
        idx = (y_acne == g).nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            perm[idx] = idx[torch.randperm(idx.numel())]

    if use_cutmix:
        h, w = imgs.shape[-2:]
        r = math.sqrt(max(0.0, 1.0 - lam))
        cw, ch = int(w * r), int(h * r)
        cx, cy = int(np.random.randint(w)), int(np.random.randint(h))
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        mixed = imgs.clone()
        mixed[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(h * w))     # true pasted-area lambda
        applied = "cutmix"
    else:
        mixed = lam * imgs + (1.0 - lam) * imgs[perm]
        applied = "mixup"

    labels_b = tuple(y[perm] for y in labels)
    return mixed, labels, labels_b, lam, applied


# ============================================================ data
def read_manifest(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    hdr = lines[0].split(",")
    return [dict(zip(hdr, ln.split(","))) for ln in lines[1:]]


class ManifestDataset(Dataset):
    """Reads a Stage-1 manifest. Images (128x128) preloaded to RAM as uint8;
    __getitem__ does float/normalise ([-1,1] via MEAN/STD)."""

    def __init__(self, rows: list[dict]) -> None:
        from PIL import Image

        self.rows = rows
        self._imgs: list[np.ndarray] = []
        for r in rows:
            with Image.open(PROJECT_ROOT / r["image_path"]) as im:
                self._imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        self._labels = [tuple(CLASS_TO_IDX[h][r[MANIFEST_COL[h]]] for h in HEADS) for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        t = torch.from_numpy(self._imgs[i].astype(np.float32) / 255.0).permute(2, 0, 1)
        t = (t - _MEAN_T) / _STD_T
        return (t, *self._labels[i])


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


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    if cohen_kappa_score is not None:
        try:
            k = cohen_kappa_score(y_true, y_pred, labels=list(range(n_classes)), weights="quadratic")
            return float(k) if np.isfinite(k) else 0.0
        except Exception:  # pragma: no cover
            pass
    cm = np.asarray(confusion(y_true, y_pred, n_classes), dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    w = np.array([[((i - j) ** 2) / ((n_classes - 1) ** 2) for j in range(n_classes)]
                  for i in range(n_classes)], dtype=np.float64)
    act, pred = cm.sum(1) / total, cm.sum(0) / total
    expected = np.outer(act, pred)
    denom = float((w * expected).sum())
    return float(1.0 - (w * (cm / total)).sum() / denom) if denom else 0.0


@torch.inference_mode()
def evaluate(model: SkinAnalysisModel, loader: DataLoader) -> dict:
    model.eval()
    preds: dict[str, list] = {h: [] for h in HEADS}
    trues: dict[str, list] = {h: [] for h in HEADS}
    for batch in loader:
        imgs, ys = batch[0], batch[1:]
        out = model(imgs)
        for j, h in enumerate(HEADS):
            preds[h].append(out[h].argmax(1))
            trues[h].append(ys[j])
    res: dict = {}
    for h in HEADS:
        pred = torch.cat(preds[h]).cpu().numpy()
        true = torch.cat(trues[h]).cpu().numpy()
        n_c = len(HEAD_CLASSES[h])
        entry = {
            "accuracy": round(float((pred == true).mean()), 4),
            "macro_f1": round(macro_f1(true, pred, n_c), 4),
            "confusion": confusion(true, pred, n_c),
            "classes": HEAD_CLASSES[h],
            "support": {c: int((true == i).sum()) for i, c in enumerate(HEAD_CLASSES[h])},
        }
        if h in ORDINAL_HEADS:
            entry["qwk"] = round(quadratic_weighted_kappa(true, pred, n_c), 4)
            entry["mae"] = round(float(np.abs(true.astype(float) - pred.astype(float)).mean()), 4)
        res[h] = entry
    res["mean_macro_f1"] = round(float(np.mean([res[h]["macro_f1"] for h in HEADS])), 4)
    res["mean_accuracy"] = round(float(np.mean([res[h]["accuracy"] for h in HEADS])), 4)
    return res


# ============================================================ train
def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    tr_rows = read_manifest(SCRATCH / "train_manifest.csv")
    va_rows = read_manifest(SCRATCH / "val_manifest.csv")
    logger.info("train %d | val %d images", len(tr_rows), len(va_rows))

    tr_ds, va_ds = ManifestDataset(tr_rows), ManifestDataset(va_rows)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    tr_eval_dl = DataLoader(tr_ds, batch_size=64, shuffle=False, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = SkinAnalysisModel()
    n_params = model.num_parameters()
    logger.info("SkinAnalysisModel: %s params (from scratch, Kaiming init, SE after %s)",
                f"{n_params:,}", model.se_after_widths)
    if n_params >= 450_000:
        raise RuntimeError(f"param budget exceeded: {n_params:,} >= 450,000")

    head_crit = build_head_criteria()
    mtl = MultiTaskUncertaintyLoss(HEADS) if args.loss_balance == "uncertainty" else None

    param_groups: list[dict] = [{"params": list(model.parameters()), "weight_decay": args.weight_decay}]
    if mtl is not None:
        param_groups.append({"params": list(mtl.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    logger.info("AdamW lr=%g wd=%g | CosineAnnealingLR T_max=%d | loss-balance=%s",
                args.lr, args.weight_decay, args.epochs, args.loss_balance)
    logger.info("acne=FocalLoss(gamma=%.1f) | %s=CE(label_smoothing=%.2f) | mixup(alpha=%.2f, p=%.2f, %s)",
                FOCAL_GAMMA, "+".join(LABEL_SMOOTHED_HEADS), LABEL_SMOOTHING,
                args.mixup_alpha, args.mix_prob, args.mix_mode)

    history: list[dict] = []
    best_score, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if mtl is not None:
            mtl.train()
        running, n_seen = 0.0, 0
        mix_counts = {"none": 0, "mixup": 0, "cutmix": 0}

        for batch in tr_dl:
            imgs, ys = batch[0], tuple(batch[1:])
            mixed, la, lb, lam, applied = mix_batch(
                imgs, ys, alpha=args.mixup_alpha, p=args.mix_prob, mode=args.mix_mode
            )
            mix_counts[applied] += 1

            opt.zero_grad()
            out = model(mixed)
            raw: dict[str, torch.Tensor] = {}
            for j, h in enumerate(HEADS):
                crit = head_crit[h]
                if lam < 1.0:
                    raw[h] = lam * crit(out[h], la[j]) + (1.0 - lam) * crit(out[h], lb[j])
                else:
                    raw[h] = crit(out[h], la[j])

            if mtl is not None:
                loss, _ = mtl(raw)
            else:
                loss = sum(LOSS_WEIGHTS[h] * raw[h] for h in HEADS)

            loss.backward()
            opt.step()
            running += float(loss) * imgs.shape[0]
            n_seen += imgs.shape[0]
        sched.step()
        train_loss = running / max(n_seen, 1)

        val = evaluate(model, va_dl)
        row = {
            "epoch": epoch, "lr": round(float(sched.get_last_lr()[0]), 7),
            "train_loss": round(train_loss, 4),
            "val_mean_macro_f1": val["mean_macro_f1"],
            "val_acne_qwk": val["acne"]["qwk"], "val_acne_mae": val["acne"]["mae"],
            "val": {h: {"acc": val[h]["accuracy"], "f1": val[h]["macro_f1"]} for h in HEADS},
            "mix_counts": mix_counts,
        }
        if mtl is not None:
            row["sigmas"] = mtl.sigmas()
        history.append(row)
        logger.info("epoch %2d/%d  lr %.5f  loss %.4f  | val meanF1 %.4f  acne F1/QWK/MAE %.3f/%.3f/%.3f  "
                    "(F1 %s)%s", epoch, args.epochs, sched.get_last_lr()[0], train_loss,
                    val["mean_macro_f1"], val["acne"]["macro_f1"], val["acne"]["qwk"], val["acne"]["mae"],
                    {h: val[h]["macro_f1"] for h in HEADS},
                    f"  sigmas {row['sigmas']}" if mtl is not None else "")

        if val["mean_macro_f1"] > best_score:
            best_score = val["mean_macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, va_dl)
    final_train = evaluate(model, tr_eval_dl)
    wall = time.time() - t0

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 1),
        "model": {"name": "SkinAnalysisModel", "params": n_params,
                  "from_scratch": True, "img_size": 128, "widths": list(model.widths),
                  "se_after_widths": list(model.se_after_widths), "se_reduction": model.se_reduction},
        "args": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "weight_decay": args.weight_decay, "scheduler": "CosineAnnealingLR",
                 "loss_balance": args.loss_balance, "loss_weights": LOSS_WEIGHTS,
                 "focal_gamma": FOCAL_GAMMA, "label_smoothing": LABEL_SMOOTHING,
                 "label_smoothed_heads": list(LABEL_SMOOTHED_HEADS),
                 "mixup_alpha": args.mixup_alpha, "mix_prob": args.mix_prob, "mix_mode": args.mix_mode,
                 "seed": args.seed, "num_workers": 0},
        "dataset": {"train": len(tr_rows), "val": len(va_rows),
                    "source": "datasets/processed_scratch (Stage 1)"},
        "best_epoch": best_epoch,
        "best_val_mean_macro_f1": best_score,
        "learned_sigmas": mtl.sigmas() if mtl is not None else None,
        "history": history,
        "final_val": final_val,
        "final_train": final_train,
    }

    model.save(WEIGHTS_OUT, metrics=metrics,
               notes=f"From-scratch SkinAnalysisModel ({n_params:,} params, Kaiming init, no pretrained; "
                     f"SE blocks after {list(model.se_after_widths)} ch). 5 heads: acne (real ACNE04 "
                     f"grades, FocalLoss gamma={FOCAL_GAMMA}) + wrinkles/oiliness/dryness/pigmentation "
                     f"(Stage-1 OpenCV weak labels; oiliness+dryness label-smoothed). Task balancing: "
                     f"{args.loss_balance}. MixUp/CutMix alpha={args.mixup_alpha}. 128x128 processed_scratch.")
    logger.info("saved best checkpoint (epoch %d, val meanF1 %.4f) -> %s", best_epoch, best_score, WEIGHTS_OUT)
    return metrics


# ============================================================ report
def write_report(metrics: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "skin_analysis_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("=" * 72)
    L.append("SkinTypeAI -- SkinAnalysisModel (from scratch) -- Stage 3 Training Report")
    L.append("=" * 72)
    m = metrics["model"]
    a = metrics["args"]
    L.append(f"Trained at   : {metrics['trained_at']}")
    L.append(f"Wall time    : {metrics['wall_seconds']} s")
    L.append(f"Model        : {m['name']}  {m['params']:,} params  (from scratch, Kaiming init, "
             f"no pretrained/timm/ImageNet)")
    L.append(f"Input        : 3x{m['img_size']}x{m['img_size']}   conv widths {m['widths']}   "
             f"SE after {m.get('se_after_widths')} ch (reduction {m.get('se_reduction')})")
    L.append(f"Optimiser    : AdamW lr={a['lr']} wd={a['weight_decay']}  +  {a['scheduler']}(T_max={a['epochs']})")
    L.append(f"Loss         : acne=Focal(gamma={a['focal_gamma']}) | {'+'.join(a['label_smoothed_heads'])}="
             f"CE(eps={a['label_smoothing']}) | wrinkles/pigmentation=CE")
    L.append(f"Task balance : {a['loss_balance']}"
             + (f"   learned sigmas {metrics.get('learned_sigmas')}" if a['loss_balance'] == 'uncertainty'
                else f"   fixed weights {a['loss_weights']}"))
    L.append(f"Augmentation : intra-grade MixUp/CutMix  alpha={a['mixup_alpha']}  p={a['mix_prob']}  mode={a['mix_mode']}")
    L.append(f"Data         : train {metrics['dataset']['train']} / val {metrics['dataset']['val']}  "
             f"(128x128, {metrics['dataset']['source']})")
    L.append(f"Best epoch   : {metrics['best_epoch']}  (val mean macro-F1 {metrics['best_val_mean_macro_f1']:.4f})")
    L.append("")
    L.append("PER-EPOCH  (val: mean macro-F1 | per-head F1 | acne QWK/MAE)")
    L.append("-" * 72)
    for h in metrics["history"]:
        pf = " ".join(f"{k[:4]}:{v['f1']:.2f}" for k, v in h["val"].items())
        L.append(f"  ep {h['epoch']:2d}  lr {h['lr']:.5f}  loss {h['train_loss']:.4f}  "
                 f"meanF1 {h['val_mean_macro_f1']:.4f}  [{pf}]  acne QWK {h['val_acne_qwk']:.3f} "
                 f"MAE {h['val_acne_mae']:.3f}")
    L.append("")
    L.append("FINAL VALIDATION METRICS")
    L.append("-" * 72)
    fv = metrics["final_val"]
    for h in HEADS:
        d = fv[h]
        extra = f"   QWK {d['qwk']:.4f}  MAE {d['mae']:.4f}" if "qwk" in d else ""
        L.append(f"  {h:13s}  acc {d['accuracy']:.4f}   macroF1 {d['macro_f1']:.4f}{extra}")
        L.append(f"  {'':13s}  classes  {d['classes']}   support {d['support']}")
        L.append(f"  {'':13s}  confusion (rows=true, cols=pred):")
        for cls, row in zip(d["classes"], d["confusion"]):
            L.append(f"  {'':15s}{cls:11s} {row}")
    L.append("")
    L.append(f"  MEAN over heads:  acc {fv['mean_accuracy']:.4f}   macroF1 {fv['mean_macro_f1']:.4f}")
    L.append("")
    L.append("FINAL TRAIN METRICS  (overfit check)")
    L.append("-" * 72)
    ft = metrics["final_train"]
    for h in HEADS:
        extra = f"   QWK {ft[h]['qwk']:.4f}  MAE {ft[h]['mae']:.4f}" if "qwk" in ft[h] else ""
        L.append(f"  {h:13s}  acc {ft[h]['accuracy']:.4f}   macroF1 {ft[h]['macro_f1']:.4f}{extra}")
    L.append(f"  mean macroF1  train {ft['mean_macro_f1']:.4f}  vs  val {fv['mean_macro_f1']:.4f}  "
             f"(gap {ft['mean_macro_f1'] - fv['mean_macro_f1']:+.4f})")
    L.append("")
    L.append("NOTES")
    L.append("-" * 72)
    L.append("  * 100% from scratch: every weight is Kaiming-random-initialised. No timm, no")
    L.append("    ImageNet, no pretrained backbone. SE blocks are trained from scratch too.")
    L.append("  * acne_severity is REAL (ACNE04 Hayashi grades), trained with FocalLoss + intra-grade")
    L.append("    MixUp/CutMix. wrinkles/oiliness/dryness/pigmentation are Stage-1 OpenCV weak labels --")
    L.append("    the CNN learns to imitate the heuristic; those heads do not add new clinical info.")
    L.append("  * Task weights are balanced by learnable homoscedastic uncertainty (Kendall et al.).")
    L.append("  * skin_type is NOT a head -- Stage 4 derives it from predicted oiliness + dryness.")
    L.append("=" * 72)

    (RUNS_DIR / "skin_analysis_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("wrote runs/skin_analysis_report.txt and runs/skin_analysis_metrics.json")
    print("\n".join(L))


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss-balance", choices=["uncertainty", "fixed"], default="uncertainty",
                    help="'uncertainty' = learnable homoscedastic task weights (Kendall et al., default); "
                         "'fixed' = the static LOSS_WEIGHTS dict")
    ap.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA)
    ap.add_argument("--mix-prob", type=float, default=MIX_PROB,
                    help="fraction of training batches that receive MixUp/CutMix")
    ap.add_argument("--mix-mode", choices=["both", "mixup", "cutmix", "none"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mix_mode == "none":
        args.mix_prob = 0.0

    if not (SCRATCH / "train_manifest.csv").exists():
        logger.error("no datasets/processed_scratch/train_manifest.csv -- run "
                     "scripts/expand_and_prepare_dataset.py first")
        return 2

    metrics = train(args)
    write_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
