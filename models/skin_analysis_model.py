"""
models/skin_analysis_model.py

`SkinAnalysisModel` -- the project's OWN multi-task CNN, trained 100% FROM SCRATCH
(zero pretrained weights, zero timm, zero ImageNet backbones -- every parameter is
Kaiming-random-initialised). It predicts the 5 model-backed characteristics:

    acne (4 Hayashi grades) | wrinkles (3) | oiliness (3) | dryness (3) | pigmentation (3)

`skin_type` is NOT a head -- Stage 4 derives it deterministically from the
predicted oiliness + dryness tiers. The other 4 characteristics (redness, dark
spots, pores, blackheads) come from models.skin_analysis.opencv_engine.

Architecture (input 3x128x128, < 450k params):
    features : 4 x [Conv2d(3x3, bias=False) -> BatchNorm2d -> LeakyReLU(0.1) -> MaxPool2d(2)]
               channels 3 -> 32 -> 64 -> 128 -> 256   (128 -> 64 -> 32 -> 16 -> 8 spatially)
               + a Squeeze-and-Excitation block at the output of Conv Stage 3 (128 ch)
                 and Conv Stage 4 (256 ch) -- lightweight channel re-weighting.
    pool     : AdaptiveAvgPool2d(1) -> 256-d vector
    dropout  : 0.3
    heads    : 5 x nn.Linear(256, n_classes)

Checkpoint format (weights/skin_analysis_model.pt), written by
training/train_skin_analysis.py::

    {"state_dict", "config": {"img_size", "mean", "std", "widths", "heads",
                              "se_after_widths", "se_reduction"},
     "metrics", "trained_at", "notes"}

`load_skin_analysis_model()` rebuilds the module from that dict with NO network
access (there is nothing pretrained to fetch -- it is a from-scratch model).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical class orders -- the single source of truth for training (Stage 3) and
# the backend adapter (Stage 4). Do not reorder without retraining.
HEAD_CLASSES: dict[str, list[str]] = {
    "acne": ["0", "1", "2", "3"],                    # ACNE04 Hayashi grade
    "wrinkles": ["minimal", "moderate", "prominent"],
    "oiliness": ["low", "balanced", "high"],
    "dryness": ["low", "balanced", "high"],
    "pigmentation": ["normal", "mild", "marked"],
}

IMG_SIZE = 128
WIDTHS = (32, 64, 128, 256)
# Squeeze-and-Excitation blocks are inserted after any conv stage whose output
# channel count is in SE_AFTER_WIDTHS -> for the default WIDTHS that is Stage 3
# (128 ch) and Stage 4 (256 ch), exactly as specified.
SE_AFTER_WIDTHS = (128, 256)
SE_REDUCTION = 16
# From-scratch model -> no ImageNet stats. Plain [-1, 1] scaling.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
DROPOUT = 0.3


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

        squeeze : AdaptiveAvgPool2d(1)            -> (B, C)
        excite  : Linear(C, C // reduction) -> ReLU -> Linear(.., C) -> Sigmoid
        scale   : x * excite  (broadcast over H, W)

    ~2.2k params for C=128, ~8.5k for C=256 (both far below the 450k budget).
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.pool(x).reshape(b, c)
        s = self.fc(s).reshape(b, c, 1, 1)
        return x * s


class _ConvStage(nn.Module):
    """Conv(3x3, no bias) -> BN -> LeakyReLU(0.1) -> MaxPool(2)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SkinAnalysisModel(nn.Module):
    """From-scratch multi-task CNN. `forward` returns a dict of per-head logits."""

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = WIDTHS,
        head_classes: dict[str, list[str]] | None = None,
        p_drop: float = DROPOUT,
        se_after_widths: tuple[int, ...] = SE_AFTER_WIDTHS,
        se_reduction: int = SE_REDUCTION,
    ) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.head_classes = head_classes or {k: list(v) for k, v in HEAD_CLASSES.items()}
        self.feature_dim = self.widths[-1]
        self.se_after_widths = tuple(se_after_widths or ())
        self.se_reduction = int(se_reduction)

        chs = [in_channels, *self.widths]
        layers: list[nn.Module] = []
        for i in range(len(self.widths)):
            c_out = chs[i + 1]
            layers.append(_ConvStage(chs[i], c_out))
            if c_out in self.se_after_widths:
                layers.append(SEBlock(c_out, reduction=self.se_reduction))
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p_drop)

        # 5 explicit linear heads (named attributes, as specified)
        self.head_acne = nn.Linear(self.feature_dim, len(self.head_classes["acne"]))
        self.head_wrinkles = nn.Linear(self.feature_dim, len(self.head_classes["wrinkles"]))
        self.head_oiliness = nn.Linear(self.feature_dim, len(self.head_classes["oiliness"]))
        self.head_dryness = nn.Linear(self.feature_dim, len(self.head_classes["dryness"]))
        self.head_pigmentation = nn.Linear(self.feature_dim, len(self.head_classes["pigmentation"]))

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        """Explicit 100% random Kaiming-normal initialisation. Nothing pretrained."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.features(x)
        z = self.gap(z).flatten(1)              # (B, 256)
        z = self.dropout(z)
        return {
            "acne": self.head_acne(z),
            "wrinkles": self.head_wrinkles(z),
            "oiliness": self.head_oiliness(z),
            "dryness": self.head_dryness(z),
            "pigmentation": self.head_pigmentation(z),
        }

    def last_conv_layer(self) -> nn.Conv2d:
        """The final convolutional layer -- the Grad-CAM target (Stage 4 conv, 256 ch)."""
        conv: nn.Conv2d | None = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv = m
        if conv is None:  # pragma: no cover - the model always has conv layers
            raise RuntimeError("SkinAnalysisModel has no Conv2d layer")
        return conv

    # ------------------------------------------------------------------ persistence
    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def config_dict(self) -> dict[str, Any]:
        return {
            "img_size": IMG_SIZE,
            "mean": MEAN,
            "std": STD,
            "widths": list(self.widths),
            "dropout": float(self.dropout.p),
            "heads": {k: list(v) for k, v in self.head_classes.items()},
            "feature_dim": self.feature_dim,
            "se_after_widths": list(self.se_after_widths),
            "se_reduction": self.se_reduction,
            "from_scratch": True,
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


def load_skin_analysis_model(path: str | Path, device: str = "cpu") -> tuple[SkinAnalysisModel, dict]:
    """Rebuild SkinAnalysisModel from a checkpoint. No network access -- there is
    nothing pretrained to fetch. Returns (model.eval(), checkpoint_dict).

    `strict=False` so a checkpoint written before the SE blocks were added still
    loads (missing SE weights stay at their random init) -- retrain to refresh it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"skin-analysis model checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = SkinAnalysisModel(
        widths=tuple(cfg.get("widths", WIDTHS)),
        head_classes=cfg.get("heads", HEAD_CLASSES),
        p_drop=float(cfg.get("dropout", DROPOUT)),
        se_after_widths=tuple(cfg.get("se_after_widths", SE_AFTER_WIDTHS)),
        se_reduction=int(cfg.get("se_reduction", SE_REDUCTION)),
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        logger.warning("load_skin_analysis_model: %d missing / %d unexpected keys "
                       "(retrain the checkpoint if the architecture changed): missing=%s unexpected=%s",
                       len(missing), len(unexpected), list(missing)[:6], list(unexpected)[:6])
    model.to(device).eval()
    return model, ckpt


def trainable_parameters(model: SkinAnalysisModel):
    return (p for p in model.parameters() if p.requires_grad)


# ============================================================ inference helpers
def preprocess(image, img_size: int = IMG_SIZE, mean=None, std=None) -> torch.Tensor:
    """Accepts a path, a PIL.Image, a BGR np.ndarray (OpenCV) or an RGB np.ndarray.
    Returns a normalised 1x3ximg_sizeximg_size float tensor."""
    import numpy as np
    from PIL import Image as PILImage

    mean = mean or MEAN
    std = std or STD

    if isinstance(image, (str, Path)):
        arr = np.asarray(PILImage.open(image).convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, PILImage.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        a = image.astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        arr = a[:, :, ::-1].copy() if a.ndim == 3 and a.shape[-1] == 3 else a  # BGR -> RGB
    else:
        raise TypeError(f"unsupported image type: {type(image)}")

    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    if t.shape[-2:] != (img_size, img_size):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
        ).squeeze(0)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return ((t - m) / s).unsqueeze(0)


@torch.inference_mode()
def predict(model: SkinAnalysisModel, image, config: dict | None = None) -> dict[str, dict]:
    """Run all 5 heads on one image/crop. Returns, per head:
        {"label": str, "index": int, "confidence": float, "distribution": {class: prob}}
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
