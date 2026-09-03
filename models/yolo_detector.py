"""
models/yolo_detector.py

Thin, framework-agnostic wrapper around an Ultralytics YOLO detection model,
used by the FastAPI backend (backend/app/services/analysis_service.py, for the
acne lesion boxes) and the CLI scripts under training/yolo/.

The active SkinTypeAI acne detector is a single-class YOLOv8s model:

    0 -> acne   (see weights/skin_analysis_models.metadata.json)

Design notes
------------
* One `YoloDetector` instance owns one loaded `ultralytics.YOLO`. Construct it
  once and reuse it (the backend keeps a module-level singleton).
* `detect()` returns plain dicts / lists only -- no torch tensors, no numpy --
  so callers never need Ultralytics types.
* Bounding boxes are returned in **original source-image pixel coordinates**
  `[x1, y1, x2, y2]` (top-left, bottom-right), matching what the frontend needs
  to draw an overlay on the uploaded image.
* CPU by default. This project's dev machine has no NVIDIA GPU; passing
  device="cuda" is allowed but will raise inside Ultralytics if unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Kept in sync with datasets/skin_detect/data.yaml (`names:`).
DEFAULT_CLASS_NAMES: dict[int, str] = {0: "acne"}


@dataclass(frozen=True)
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in source-image pixels

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
        }


class YoloDetector:
    """Loads an Ultralytics YOLO detector once and runs structured inference."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights_path}. "
                "Download yolo11n.pt or train weights/skin_yolo11n.pt "
                "(see training/yolo/ and colab_train_yolo.ipynb)."
            )
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        # Imported here so importing this module never hard-requires ultralytics
        # (e.g. for type checking or docs) until a model is actually loaded.
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights_path))
        model_names = getattr(self._model, "names", None) or {}
        self.class_names: dict[int, str] = class_names or (
            {int(k): str(v) for k, v in model_names.items()} if model_names else dict(DEFAULT_CLASS_NAMES)
        )
        logger.info(
            "YoloDetector loaded %s (classes=%s, device=%s, conf=%.2f, iou=%.2f, imgsz=%d)",
            self.weights_path.name, self.class_names, self.device, self.conf, self.iou, self.imgsz,
        )

    @property
    def is_pretrained_coco(self) -> bool:
        """True if this looks like a stock COCO model (80 classes, no 'acne')
        rather than a fine-tuned skin detector. Callers surface this so a
        temporary COCO model is never mistaken for the real acne model."""
        return "acne" not in set(self.class_names.values())

    @property
    def model_version(self) -> str:
        """Honest identifier for the loaded weights, for API `model_version`.

        Priority:
          1. a "<weights>.version" sidecar file written by training/yolo/train_yolo.py
             (e.g. "yolo11n-acne04-proof" for an under-trained CPU proof run, or
             "yolo11n-acne04" for a completed GPU training run);
          2. "yolo11n-coco-pretrained-UNTRAINED" if the weights are stock COCO;
          3. "yolo11n-acne04-unknown" otherwise.
        """
        sidecar = self.weights_path.with_suffix(self.weights_path.suffix + ".version")
        if sidecar.is_file():
            tag = sidecar.read_text(encoding="utf-8").strip()
            if tag:
                return tag
        if self.is_pretrained_coco:
            return "yolo11n-coco-pretrained-UNTRAINED"
        return "yolo11n-acne04-unknown"

    def detect(
        self,
        image: str | Path,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> dict[str, Any]:
        """Run detection on one image.

        Returns:
            {
              "detections": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...],
              "counts": {class_name: int, ...},          # only classes with >=1 detection
              "image_size": [width, height],
            }
        """
        results = self._model.predict(
            source=str(image),
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        h, w = result.orig_shape  # (height, width)

        detections: list[Detection] = []
        counts: dict[str, int] = {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clsids = boxes.cls.tolist()
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clsids):
                name = self.class_names.get(int(k), str(int(k)))
                detections.append(Detection(name, float(c), (float(x1), float(y1), float(x2), float(y2))))
                counts[name] = counts.get(name, 0) + 1

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return {
            "detections": [d.to_dict() for d in detections],
            "counts": counts,
            "image_size": [int(w), int(h)],
        }
