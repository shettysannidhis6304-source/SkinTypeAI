#!/usr/bin/env python3
"""
dry_run_test.py  --  PHASE 1 validation (no training, no data, no network)

Feeds a dummy (2, 3, 128, 128) tensor through:
  1. the SE-augmented SkinAnalysisModel                -> head shapes, param budget
  2. the Grad-CAM hook (models/skin_analysis/gradcam)  -> heatmap shape + range
  3. the multi-task training loss stack               -> focal + label smoothing +
     homoscedastic uncertainty weighting + intra-grade MixUp/CutMix, then
     loss.backward() and a gradient-presence check (weights AND the 5 log-vars)

Exits 0 only if every shape / gradient computes with zero runtime errors.

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe dry_run_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skin_analysis_model import HEAD_CLASSES, SEBlock, SkinAnalysisModel  # noqa: E402
from models.skin_analysis.gradcam import GradCAM  # noqa: E402
from training.train_skin_analysis import (  # noqa: E402
    FocalLoss,
    MultiTaskUncertaintyLoss,
    build_head_criteria,
    mix_batch,
)

HEADS = list(HEAD_CLASSES)
PARAM_BUDGET = 450_000


def _section(name: str) -> None:
    print("\n" + "-" * 68 + f"\n{name}\n" + "-" * 68)


def test_model_forward() -> tuple[SkinAnalysisModel, torch.Tensor]:
    _section("1. SE-augmented SkinAnalysisModel -- forward + param budget")
    torch.manual_seed(0)
    model = SkinAnalysisModel()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert set(out) == set(HEADS), f"heads {set(out)} != {set(HEADS)}"
    for h in HEADS:
        exp = (2, len(HEAD_CLASSES[h]))
        assert tuple(out[h].shape) == exp, f"{h}: {tuple(out[h].shape)} != {exp}"
        print(f"  head {h:13s} -> logits {tuple(out[h].shape)}")

    se = [m for m in model.modules() if isinstance(m, SEBlock)]
    assert len(se) == 2, f"expected 2 SE blocks (after 128 & 256 ch), found {len(se)}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SE blocks       : {len(se)}  (after channels {model.se_after_widths})")
    print(f"  total params    : {n_params:,}  (budget < {PARAM_BUDGET:,})")
    assert n_params < PARAM_BUDGET, f"param budget exceeded: {n_params:,} >= {PARAM_BUDGET:,}"
    last_conv = model.last_conv_layer()
    print(f"  Grad-CAM target : {type(last_conv).__name__}({last_conv.in_channels}->{last_conv.out_channels})")
    return model, x


def test_gradcam(model: SkinAnalysisModel, x: torch.Tensor) -> None:
    _section("2. Grad-CAM hook -- heatmap shape + [0,1] range")
    cam = GradCAM(model)
    try:
        for head in ("acne", "wrinkles", "pigmentation"):
            heat = cam.generate(x[:1], head=head)
            assert heat.shape == (128, 128), f"{head}: heatmap shape {heat.shape} != (128, 128)"
            lo, hi = float(heat.min()), float(heat.max())
            assert 0.0 <= lo <= hi <= 1.0 + 1e-5, f"{head}: heatmap range [{lo}, {hi}] not in [0,1]"
            assert heat.std() > 0, f"{head}: heatmap is flat (no gradient signal)"
            print(f"  {head:13s} -> heatmap {heat.shape}  range [{lo:.3f}, {hi:.3f}]")
        batch = cam.generate(x, head="acne")
        assert batch.shape == (2, 128, 128), f"batched heatmap shape {batch.shape} != (2, 128, 128)"
        print(f"  batched (B=2)  -> heatmap {batch.shape}")
    finally:
        cam.remove()


def test_multitask_loss(model: SkinAnalysisModel, x: torch.Tensor) -> None:
    _section("3. Multi-task loss -- focal + label smoothing + uncertainty + MixUp/CutMix")
    torch.manual_seed(1)
    # dummy labels: acne grade 0..3, the other 4 heads 0..2
    labels = (
        torch.tensor([0, 2]),                       # acne
        torch.tensor([1, 2]),                       # wrinkles
        torch.tensor([0, 1]),                       # oiliness
        torch.tensor([2, 0]),                       # dryness
        torch.tensor([1, 1]),                       # pigmentation
    )

    head_crit = build_head_criteria()
    assert isinstance(head_crit["acne"], FocalLoss)
    assert getattr(head_crit["oiliness"], "label_smoothing", 0.0) == 0.1
    assert getattr(head_crit["dryness"], "label_smoothing", 0.0) == 0.1
    print("  criteria       : acne=FocalLoss(gamma=2.0), oiliness/dryness=CE(eps=0.1), rest=CE")

    # intra-grade MixUp/CutMix (force it on with p=1.0)
    mixed, la, lb, lam, applied = mix_batch(x, labels, alpha=0.2, p=1.0, mode="both")
    assert mixed.shape == x.shape, f"mixed shape {mixed.shape} != {x.shape}"
    assert 0.0 <= lam <= 1.0, f"lambda {lam} not in [0,1]"
    assert torch.equal(la[0], lb[0]), "intra-grade mix must preserve the acne label"
    print(f"  mix_batch      : applied={applied}  lambda={lam:.3f}  acne label preserved")

    mtl = MultiTaskUncertaintyLoss(HEADS)
    model.train()
    mtl.train()
    out = model(mixed)
    raw = {}
    for j, h in enumerate(HEADS):
        crit = head_crit[h]
        raw[h] = lam * crit(out[h], la[j]) + (1.0 - lam) * crit(out[h], lb[j])
    loss, detail = mtl(raw)
    assert loss.dim() == 0 and torch.isfinite(loss), f"loss not a finite scalar: {loss}"
    print(f"  joint loss     : {float(loss):.4f}   per-head sigmas "
          f"{ {h: detail[h]['sigma'] for h in HEADS} }")

    model.zero_grad(set_to_none=True)
    mtl.zero_grad(set_to_none=True)
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_with_grad > 0, "no model parameter received a finite gradient"
    assert mtl.log_var.grad is not None and torch.isfinite(mtl.log_var.grad).all(), \
        "the 5 uncertainty log-variance params did not receive a finite gradient"
    print(f"  gradients      : {n_with_grad} model tensors + 5 log-var params have finite grads")


def main() -> int:
    try:
        model, x = test_model_forward()
        test_gradcam(model, x)
        test_multitask_loss(model, x)
    except Exception:
        print("\n" + "=" * 68 + "\nDRY RUN FAILED\n" + "=" * 68)
        traceback.print_exc()
        return 1
    print("\n" + "=" * 68)
    print("DRY RUN PASSED -- SE model, Grad-CAM, and the multi-task loss all compute "
          "shapes + gradients with zero runtime errors.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
