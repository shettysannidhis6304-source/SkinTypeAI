"""
models/skin_analysis/wrinkle_segmenter.py

Wrapper around the pretrained FFHQ-Wrinkle U-Net (ResNet50 encoder).

  weights (PyTorch) : weights/hf_models/wrinkle_unet/wrinkle_model.safetensors  (~276 MB fp32)
  weights (OpenVINO): weights/openvino/wrinkle_unet.xml + .bin                   (~138 MB fp16)
  source  : https://huggingface.co/rmsandu/ffhq-wrinkle-unet
  licence : model code MIT (vendored under wrinkle_unet/);
            WEIGHTS are CC BY-NC-SA 4.0 (FFHQ-Wrinkle dataset) - academic use only.

Backend selection (transparent to callers):
  * If the OpenVINO IR exists AND `openvino` imports AND it compiles on the
    requested device -> use OpenVINO (fp16, ~2x faster, ~2x less RAM). The heavy
    276 MB PyTorch model is never allocated.
  * Otherwise -> fall back to the PyTorch .safetensors model.
Both paths use IDENTICAL preprocessing (PIL bilinear resize to imgsz, /255, CHW,
no mean/std) and return the SAME dict:
    {wrinkle_area_fraction, max_prob, threshold, imgsz, backend}

Output: a wrinkle probability mask reduced to a single `wrinkle_area_fraction`
(fraction of face pixels flagged as wrinkle). Positive-case accuracy is NOT
independently validated by this project - see the registry entry.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OPENVINO_IR = _PROJECT_ROOT / "weights" / "openvino" / "wrinkle_unet.xml"

# torch device string -> OpenVINO device name
_OV_DEVICE_MAP = {"cpu": "CPU", "gpu": "GPU", "igpu": "GPU", "cuda": "GPU", "npu": "NPU", "auto": "AUTO"}


class WrinkleSegmenter:
    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 512,
        num_threads: int = 4,
        openvino_ir: str | Path | None = "auto",
        prefer_openvino: bool = True,
    ):
        import numpy as np

        self.weights_path = Path(weights_path)
        self.device = device
        self.imgsz = imgsz
        self._np = np
        self.backend: str | None = None
        self.backend_detail: str = ""

        ir_path = (
            _DEFAULT_OPENVINO_IR if openvino_ir in ("auto", None)
            else Path(openvino_ir)
        )
        self.openvino_ir = ir_path

        if prefer_openvino and ir_path.exists():
            try:
                self._init_openvino(ir_path, device)
                self.backend = "openvino"
            except Exception as e:  # pragma: no cover - env dependent
                logger.warning("OpenVINO wrinkle backend unavailable (%s) -> falling back to PyTorch", e)

        if self.backend is None:
            self._init_pytorch(num_threads)
            self.backend = "pytorch"

        logger.info("WrinkleSegmenter backend=%s (%s)", self.backend, self.backend_detail)

    # ------------------------------------------------------------------ backends
    def _init_openvino(self, ir_path: Path, device: str) -> None:
        import openvino as ov

        ov_device = _OV_DEVICE_MAP.get(str(device).lower(), str(device).upper())
        core = ov.Core()
        if ov_device not in core.available_devices and ov_device != "AUTO":
            raise RuntimeError(f"OpenVINO device {ov_device!r} not in {core.available_devices}")
        self._ov_core = core
        self._ov_compiled = core.compile_model(str(ir_path), ov_device)
        self._ov_out = self._ov_compiled.output(0)
        self._ov_device = ov_device
        self.backend_detail = f"IR={ir_path.name}, device={ov_device}"

    def _init_pytorch(self, num_threads: int) -> None:
        import torch
        from safetensors.torch import load_file
        from torchvision import transforms

        from models.skin_analysis.wrinkle_unet import UNet

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No wrinkle weights: OpenVINO IR missing/unusable and no PyTorch "
                f"checkpoint at {self.weights_path}."
            )
        torch.set_num_threads(num_threads)
        self._torch = torch

        # OpenVINO device strings ("gpu", "auto", ...) are not valid torch devices
        d = str(self.device).lower()
        self._torch_device = d if (d == "cpu" or d.startswith("cuda")) else "cpu"

        model = UNet(n_channels=3, n_classes=1, pretrained=False)
        state = load_file(str(self.weights_path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("WrinkleSegmenter load: %d missing / %d unexpected keys",
                           len(missing), len(unexpected))
        self.model = model.to(self._torch_device).eval()
        self._tf = transforms.Compose([transforms.Resize((self.imgsz, self.imgsz)), transforms.ToTensor()])
        self.backend_detail = f"weights={self.weights_path.name}, device={self._torch_device}"

    # ------------------------------------------------------------------ inference
    def _prob_map(self, pil_rgb) -> "Any":
        """Wrinkle probability map (imgsz x imgsz, float32 in [0,1]). Identical
        preprocessing for both backends: bilinear resize -> [0,1] -> CHW."""
        np = self._np
        if self.backend == "openvino":
            from PIL import Image as _Image

            arr = np.asarray(
                pil_rgb.resize((self.imgsz, self.imgsz), _Image.BILINEAR), dtype=np.float32
            ) / 255.0
            x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])   # (1,3,H,W)
            logits = self._ov_compiled(x)[self._ov_out]                   # (1,1,H,W)
            return (1.0 / (1.0 + np.exp(-logits[0, 0]))).astype(np.float32)

        x = self._tf(pil_rgb).unsqueeze(0).to(self._torch_device)
        with self._torch.no_grad():
            return self._torch.sigmoid(self.model(x))[0, 0].cpu().numpy()

    def predict(self, image, *, threshold: float = 0.5, skin_mask=None) -> dict[str, Any]:
        """image: PIL.Image (ideally a face crop). skin_mask: optional bool ndarray
        (same HxW as `image`) restricting the area considered 'face'.
        Returns {wrinkle_area_fraction, max_prob, threshold, imgsz, backend}."""
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        prob = self._prob_map(image)
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
            "backend": self.backend,
        }


_lock = threading.Lock()
_instance: WrinkleSegmenter | None = None


def get_wrinkle_segmenter(
    weights_path: str | Path,
    *,
    openvino_ir: str | Path | None = "auto",
    prefer_openvino: bool = True,
    device: str = "cpu",
) -> WrinkleSegmenter:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = WrinkleSegmenter(
                    weights_path, device=device,
                    openvino_ir=openvino_ir, prefer_openvino=prefer_openvino,
                )
    return _instance
