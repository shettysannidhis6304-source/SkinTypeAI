"""
models/custom_multi_task.py

The project's OWN multi-task skin classifier -- the model we train ourselves
(scripts/prepare_dataset.py -> training/train_custom.py) to replace the runtime
Hugging Face checkpoints.

Design (CPU-only, Dell Inspiron, <20 min end to end):
  * Backbone : timm `mobilenetv3_small_100`, ImageNet-pretrained, used purely as
               a FROZEN feature extractor (num_classes=0 -> 576-d pooled vector).
               Every backbone parameter has requires_grad = False. This is
               transfer learning, the standard and defensible approach: we do
               NOT fine-tune the trunk.
  * Heads    : three small, independent classifier heads on the pooled vector:
                 head_skin_type    -> 4  (dry, normal, oily, combination)
                 head_acne         -> 4  (mild, moderate, severe, very_severe)
                 head_pigmentation -> 3  (normal, mild, marked)
               Only these heads are trained.
  * Loss     : L_total = w1*CE(skin_type) + w2*CE(acne) + w3*CE(pigmentation)
               (computed in training/train_custom.py; the module just returns
               per-head logits).

Checkpoint format (`weights/custom_skin_model.pt`), saved by train_custom.py::

    {
      "state_dict":   <full module state_dict>,
      "config": {
        "backbone": "mobilenetv3_small_100",
        "img_size": 224,
        "mean": [...], "std": [...],
        "heads": {"skin_type": [...4 names], "acne": [...4], "pigmentation": [...3]},
      },
      "metrics": {...},          # per-head val accuracy / macro-F1
      "trained_at": "<iso8601>",
      "notes": "...",
    }

`load_custom_model()` rebuilds the module from that dict with NO network access
(timm backbone is created with pretrained=False; all weights come from the .pt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Canonical class orders -- the single source of truth for both training and
# the backend adapter. Do not reorder without retraining.
HEAD_CLASSES: dict[str, list[str]] = {
    "skin_type": ["dry", "normal", "oily", "combination"],
    # 5-class: "clear" comes from the FFHQ clean-face crops (ACNE04 has no
    # acne-free faces); "mild".."very_severe" are ACNE04 Hayashi grades 0..3.
    "acne": ["clear", "mild", "moderate", "severe", "very_severe"],
    "pigmentation": ["normal", "mild", "marked"],
}

DEFAULT_BACKBONE = "mobilenetv3_small_100"
IMG_SIZE = 224
# ImageNet normalisation (timm default for mobilenetv3)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _make_head(in_features: int, n_classes: int, p_drop: float = 0.2) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_features, n_classes))


class CustomSkinModel(nn.Module):
    """Frozen backbone + 3 trainable classification heads."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        head_classes: dict[str, list[str]] | None = None,
        pretrained_backbone: bool = True,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        self.head_classes = head_classes or {k: list(v) for k, v in HEAD_CLASSES.items()}

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained_backbone, num_classes=0)
        self.freeze_backbone()
        # timm reports `num_features` (e.g. 576 for mobilenetv3_small) but some
        # backbones keep a conv-head that expands the pooled vector (mobilenetv3
        # actually emits 1024). Probe once so the heads are sized correctly.
        with torch.no_grad():
            self.feature_dim = int(self.backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1])

        self.heads = nn.ModuleDict(
            {name: _make_head(self.feature_dim, len(classes), p_drop) for name, classes in self.head_classes.items()}
        )

    # ------------------------------------------------------------------ backbone
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):  # keep the frozen trunk in eval mode always
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features (B, feature_dim). No grad -- trunk is frozen.
        Used both in forward() and by the offline feature cache in train_custom.py."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.extract_features(images)
        return {name: head(feats) for name, head in self.heads.items()}

    def heads_from_features(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run only the trainable heads on a precomputed feature batch."""
        return {name: head(feats) for name, head in self.heads.items()}

    # ------------------------------------------------------------------ persistence
    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "img_size": IMG_SIZE,
            "mean": MEAN,
            "std": STD,
            "heads": {k: list(v) for k, v in self.head_classes.items()},
            "feature_dim": self.feature_dim,
        }

    def save(self, path: str | Path, *, metrics: dict | None = None, notes: str = "") -> None:
        from datetime import datetime, timezone

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config_dict(),
                "metrics": metrics or {},
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "notes": notes,
            },
            path,
        )


def load_custom_model(path: str | Path, device: str = "cpu") -> tuple[CustomSkinModel, dict]:
    """Rebuild CustomSkinModel from a checkpoint with NO network access.

    Returns (model.eval(), checkpoint_dict). Raises FileNotFoundError if the
    checkpoint is missing -- the caller is expected to fall back to the HF path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"custom model checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = CustomSkinModel(
        backbone_name=cfg.get("backbone", DEFAULT_BACKBONE),
        head_classes=cfg.get("heads", HEAD_CLASSES),
        pretrained_backbone=False,  # weights come entirely from the checkpoint
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def trainable_parameters(model: CustomSkinModel):
    """Just the head parameters -- what the optimiser should see."""
    return (p for p in model.parameters() if p.requires_grad)


# ============================================================ inference helpers
def preprocess(image, img_size: int = IMG_SIZE, mean=None, std=None) -> torch.Tensor:
    """Accepts a BGR np.ndarray (OpenCV), an RGB np.ndarray, a PIL.Image, or a
    path. Returns a normalised 1x3ximg_sizeximg_size float tensor.

    A BGR array is assumed when the input is an ndarray of dtype uint8 with 3
    channels (that is what heuristics.face_region / cv2.imread produce)."""
    import numpy as np
    from PIL import Image as PILImage

    mean = mean or MEAN
    std = std or STD

    if isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    elif isinstance(image, PILImage.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        a = image
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        # OpenCV crops are BGR -> flip to RGB
        arr = a[:, :, ::-1].copy() if a.shape[-1] == 3 else a
    else:
        raise TypeError(f"unsupported image type: {type(image)}")

    t = torch.from_numpy(arr).permute(2, 0, 1)
    if t.shape[-2:] != (img_size, img_size):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
        ).squeeze(0)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return ((t - m) / s).unsqueeze(0)


@torch.inference_mode()
def predict(model: CustomSkinModel, image, config: dict | None = None) -> dict[str, dict]:
    """Run all three heads on one image/crop. Returns, per head:

        {"label": str, "confidence": float, "distribution": {class: prob}, "index": int}
    """
    cfg = config or model.config_dict()
    x = preprocess(image, cfg.get("img_size", IMG_SIZE), cfg.get("mean"), cfg.get("std"))
    logits = model(x)
    out: dict[str, dict] = {}
    for head, classes in model.head_classes.items():
        probs = logits[head].softmax(-1)[0].tolist()
        idx = int(max(range(len(probs)), key=probs.__getitem__))
        out[head] = {
            "label": classes[idx],
            "index": idx,
            "confidence": round(float(probs[idx]), 4),
            "distribution": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        }
    return out
