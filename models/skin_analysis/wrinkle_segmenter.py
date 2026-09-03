"""
models/skin_analysis/wrinkle_segmenter.py

Wrapper around the pretrained FFHQ-Wrinkle U-Net (ResNet50 encoder).

  weights: weights/hf_models/wrinkle_unet/wrinkle_model.safetensors
  source : https://huggingface.co/rmsandu/ffhq-wrinkle-unet
  licence: model code MIT (vendored under wrinkle_unet/);
           WEIGHTS are CC BY-NC-SA 4.0 (FFHQ-Wrinkle dataset) - academic use only.

Output: a wrinkle probability mask. We reduce it to a single
`wrinkle_area_fraction` (fraction of face pixels flagged as wrinkle) which is
the score surfaced to the user. Positive-case accuracy is NOT independently
validated by this project - see the registry entry.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WrinkleSegmenter:
    def __init__(self, weights_path: str | Path, *, device: str = "cpu", imgsz: int = 512, num_threads: int = 4):
        import torch
        from safetensors.torch import load_file
        from torchvision import transforms

        from models.skin_analysis.wrinkle_unet import UNet

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"No wrinkle weights at {self.weights_path}.")
        torch.set_num_threads(num_threads)
        self._torch = torch
        self.device = device
        self.imgsz = imgsz

        model = UNet(n_channels=3, n_classes=1, pretrained=False)
        state = load_file(str(self.weights_path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("WrinkleSegmenter load: %d missing / %d unexpected keys", len(missing), len(unexpected))
        self.model = model.to(device).eval()
        self.tf = transforms.Compose([transforms.Resize((imgsz, imgsz)), transforms.ToTensor()])
        logger.info("WrinkleSegmenter loaded %s", self.weights_path.name)

    def predict(self, image, *, threshold: float = 0.5, skin_mask=None) -> dict[str, Any]:
        """image: PIL.Image (ideally a face crop). skin_mask: optional bool ndarray
        (same HxW as `image`) restricting the area considered 'face'.
        Returns {wrinkle_area_fraction, max_prob, threshold, imgsz}."""
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        x = self.tf(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            prob = self._torch.sigmoid(self.model(x))[0, 0].cpu().numpy()

        mask = prob >= threshold
        if skin_mask is not None:
            import cv2

            sm = cv2.resize(skin_mask.astype(np.uint8), (prob.shape[1], prob.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            denom = max(int(sm.sum()), 1)
            frac = float((mask & sm).sum()) / denom
        else:
            frac = float(mask.mean())

        return {
            "wrinkle_area_fraction": round(frac, 5),
            "max_prob": round(float(prob.max()), 4),
            "threshold": threshold,
            "imgsz": self.imgsz,
        }


_lock = threading.Lock()
_instance: WrinkleSegmenter | None = None


def get_wrinkle_segmenter(weights_path: str | Path) -> WrinkleSegmenter:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = WrinkleSegmenter(weights_path)
    return _instance
