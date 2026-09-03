"""
models/skin_analysis/classifier.py

Thin wrapper around a Hugging Face `ViTForImageClassification` checkpoint,
used for both the skin-type model (dima806) and the acne-severity model
(skintelligent-acne).

Load once, reuse. CPU only on this machine. Returns plain dicts.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HFImageClassifier:
    def __init__(self, model_dir: str | Path, *, device: str = "cpu", num_threads: int = 4):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No HF model at {self.model_dir}. Download it first "
                f"(see weights/skin_analysis_models.metadata.json)."
            )
        torch.set_num_threads(num_threads)
        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
        self.model = AutoModelForImageClassification.from_pretrained(str(self.model_dir)).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        logger.info("HFImageClassifier loaded %s (labels=%s)", self.model_dir.name, self.id2label)

    def predict(self, image) -> dict[str, Any]:
        """image: PIL.Image. Returns {label, confidence, distribution, index}."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = logits.softmax(-1)[0]
        idx = int(probs.argmax())
        return {
            "index": idx,
            "label": self.id2label[idx],
            "confidence": round(float(probs[idx]), 4),
            "distribution": {self.id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
        }


# ---- module-level singletons ------------------------------------------------
_lock = threading.Lock()
_instances: dict[str, HFImageClassifier] = {}


def get_classifier(model_dir: str | Path, cache_key: str) -> HFImageClassifier:
    if cache_key not in _instances:
        with _lock:
            if cache_key not in _instances:
                _instances[cache_key] = HFImageClassifier(model_dir)
    return _instances[cache_key]
