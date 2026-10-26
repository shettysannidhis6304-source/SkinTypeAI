"""
models/skin_analysis/gradcam.py

Standalone Grad-CAM (Selvaraju et al., 2017) for
`models.skin_analysis_model.SkinAnalysisModel`.

It hooks the model's FINAL convolutional layer (Stage 4 conv, 256 channels,
8x8 spatially for a 128px input) and produces a normalised [0, 1] class-
activation heatmap for any of the 5 heads:

    acne | wrinkles | oiliness | dryness | pigmentation

Nothing here changes inference. It is a debugging / explainability helper --
`analysis_service.run_analysis()` does NOT call it. Use it from a notebook, a
test, or a future "show me why" endpoint.

Usage
-----
    from models.skin_analysis_model import load_skin_analysis_model
    from models.skin_analysis.gradcam import GradCAM

    model, ckpt = load_skin_analysis_model("weights/skin_analysis_model.pt")
    cam = GradCAM(model)                       # hooks the last conv
    heat = cam.generate(image_tensor, head="acne")   # (H, W) float32 in [0, 1]
    cam.remove()                              # detach the hooks

`image_tensor` is a normalised (B, 3, 128, 128) tensor -- e.g. from
`models.skin_analysis_model.preprocess(...)`. With B > 1 a stack of maps is
returned, shape (B, H, W).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HEADS = ("acne", "wrinkles", "oiliness", "dryness", "pigmentation")


class GradCAM:
    """Grad-CAM on the final conv layer of a SkinAnalysisModel."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        if target_layer is None:
            target_layer = (
                model.last_conv_layer() if hasattr(model, "last_conv_layer")
                else self._last_conv(model)
            )
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _last_conv(model: torch.nn.Module) -> torch.nn.Module:
        conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv = m
        if conv is None:
            raise RuntimeError("no Conv2d layer found to attach Grad-CAM")
        return conv

    def _save_activation(self, _module, _inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------ api
    def generate(
        self,
        image_tensor: torch.Tensor,
        head: str = "acne",
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Return a Grad-CAM heatmap for `head`.

        image_tensor : (B, 3, H, W) normalised tensor.
        head         : one of the 5 head names.
        class_idx    : target class for the CAM; defaults to the arg-max class
                       (per-sample) of that head.

        Returns a float32 array in [0, 1]: (H, W) if B == 1, else (B, H, W).
        """
        if head not in _HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {_HEADS}")
        if image_tensor.dim() != 4:
            raise ValueError(f"expected a (B,3,H,W) tensor, got shape {tuple(image_tensor.shape)}")

        x = image_tensor.clone().requires_grad_(True)
        h, w = x.shape[-2:]

        with torch.enable_grad():
            logits = self.model(x)[head]                       # (B, n_classes)
            if class_idx is None:
                targets = logits.argmax(dim=1)                 # (B,)
            else:
                targets = torch.full((x.shape[0],), int(class_idx), dtype=torch.long)
            score = logits.gather(1, targets.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()

        if self._activations is None or self._gradients is None:  # pragma: no cover
            raise RuntimeError("Grad-CAM hooks did not fire -- check the target layer")

        acts = self._activations                                # (B, C, h', w')
        grads = self._gradients                                 # (B, C, h', w')
        weights = grads.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (B, 1, h', w')
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                                    # (B, H, W)

        # per-sample min-max normalise to [0, 1]
        flat = cam.flatten(1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo + 1e-8)).reshape_as(cam)

        arr = cam.detach().cpu().numpy().astype(np.float32)
        return arr[0] if arr.shape[0] == 1 else arr

    def remove(self) -> None:
        """Detach the forward / backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()


def generate_gradcam(model: torch.nn.Module, image, target_head: str = "acne",
                     config: dict | None = None) -> np.ndarray:
    """One-shot Grad-CAM for a raw image (path / PIL / BGR-or-RGB ndarray).

    Preprocesses `image` the same way inference does, runs Grad-CAM on
    `target_head`, and returns an (img_size, img_size) float32 heatmap in [0, 1].
    Hooks are attached and removed within this call.
    """
    from models.skin_analysis_model import IMG_SIZE, preprocess

    cfg = config or (model.config_dict() if hasattr(model, "config_dict") else {})
    x = preprocess(image, cfg.get("img_size", IMG_SIZE), cfg.get("mean"), cfg.get("std"))
    cam = GradCAM(model)
    try:
        return cam.generate(x, head=target_head)
    finally:
        cam.remove()
