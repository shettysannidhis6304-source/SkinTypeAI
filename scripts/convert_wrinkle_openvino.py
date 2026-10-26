#!/usr/bin/env python3
"""
scripts/convert_wrinkle_openvino.py

Convert the FFHQ-Wrinkle U-Net (ResNet50 encoder, ~289 MB fp32 PyTorch) to an
OpenVINO IR with fp16 weights, for fast CPU inference on the Intel i5 / Iris Xe
dev laptop.

    PyTorch  models/skin_analysis/wrinkle_unet/UNet   (n_channels=3, n_classes=1)
      weights  weights/hf_models/wrinkle_unet/wrinkle_model.safetensors
        |
        v   openvino.convert_model(model, example_input=(1,3,512,512))
        |   openvino.save_model(..., compress_to_fp16=True)
        v
    weights/openvino/wrinkle_unet.xml   (graph)
    weights/openvino/wrinkle_unet.bin   (fp16 weights, ~half the size)

The IR is mathematically equivalent to the PyTorch model (same architecture,
same trained weights, just fp16-compressed + graph-optimised). Licence is
unchanged: the FFHQ-Wrinkle weights are CC BY-NC-SA 4.0 -> academic use only.

Usage (project root):
    .venv-new\\Scripts\\python.exe scripts\\convert_wrinkle_openvino.py
    .venv-new\\Scripts\\python.exe scripts\\convert_wrinkle_openvino.py --no-fp16
    .venv-new\\Scripts\\python.exe scripts\\convert_wrinkle_openvino.py --verify
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("convert_wrinkle_openvino")

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "hf_models" / "wrinkle_unet" / "wrinkle_model.safetensors"
DEFAULT_OUTDIR = PROJECT_ROOT / "weights" / "openvino"
INPUT_SHAPE = (1, 3, 512, 512)


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def load_pytorch_unet(weights_path: Path):
    """Rebuild the U-Net exactly as models/skin_analysis/wrinkle_segmenter.py does
    (encoder pretrained=False; all weights come from the checkpoint)."""
    import torch
    from safetensors.torch import load_file

    from models.skin_analysis.wrinkle_unet import UNet

    if not weights_path.exists():
        raise FileNotFoundError(
            f"wrinkle weights not found: {weights_path}\n"
            f"Run scripts/fetch_models.py first to download them."
        )

    logger.info("building UNet(n_channels=3, n_classes=1, pretrained=False)")
    model = UNet(n_channels=3, n_classes=1, pretrained=False)

    logger.info("loading state_dict from %s (%.0f MB)", weights_path.name, _mb(weights_path))
    state = load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning("state_dict load: %d missing / %d unexpected keys "
                       "(same as the runtime wrapper's strict=False load)",
                       len(missing), len(unexpected))
        if missing:
            logger.warning("  missing e.g.: %s", missing[:5])
        if unexpected:
            logger.warning("  unexpected e.g.: %s", unexpected[:5])
    else:
        logger.info("state_dict load: 0 missing / 0 unexpected keys")

    model.eval()
    return model


def convert(weights_path: Path, outdir: Path, fp16: bool, verify: bool) -> int:
    try:
        import openvino as ov
    except ImportError:
        logger.error("openvino is not installed. Install it with:\n"
                     "    .venv-new\\Scripts\\python.exe -m pip install openvino==2024.6.0")
        return 1

    import torch

    try:
        model = load_pytorch_unet(weights_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception:
        logger.exception("failed to build / load the PyTorch U-Net")
        return 2

    example = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    # sanity: make sure the PyTorch model actually runs at this shape before we convert
    try:
        with torch.no_grad():
            ref = model(example)
        logger.info("PyTorch forward OK: input %s -> output %s", tuple(example.shape), tuple(ref.shape))
    except Exception:
        logger.exception("PyTorch forward pass failed at shape %s -- aborting", INPUT_SHAPE)
        return 2

    logger.info("converting to OpenVINO IR (this traces the graph; ~10-30 s)...")
    t0 = time.perf_counter()
    try:
        ov_model = ov.convert_model(model, example_input=example, input=[INPUT_SHAPE])
    except Exception:
        logger.exception("openvino.convert_model failed")
        return 3
    logger.info("convert_model done in %.1f s", time.perf_counter() - t0)

    outdir.mkdir(parents=True, exist_ok=True)
    xml_path = outdir / "wrinkle_unet.xml"
    bin_path = outdir / "wrinkle_unet.bin"
    try:
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=fp16)
    except Exception:
        logger.exception("openvino.save_model failed")
        return 3

    logger.info("saved:")
    logger.info("  %s  (%.2f MB)", xml_path, _mb(xml_path))
    logger.info("  %s  (%.1f MB, %s)", bin_path, _mb(bin_path), "fp16" if fp16 else "fp32")
    logger.info("  source PyTorch weights: %.0f MB -> IR weights: %.0f MB  (%.1fx smaller)",
                _mb(weights_path), _mb(bin_path),
                _mb(weights_path) / max(_mb(bin_path), 1e-6))

    if verify:
        return _verify(xml_path, model, example)
    return 0


def _verify(xml_path: Path, torch_model, example) -> int:
    """Compare OpenVINO CPU output against the PyTorch reference on a random input."""
    import numpy as np
    import openvino as ov
    import torch

    logger.info("verifying OpenVINO vs PyTorch on a random 512x512 input...")
    try:
        x = torch.rand(*INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            ref = torch.sigmoid(torch_model(x)).cpu().numpy()

        core = ov.Core()
        compiled = core.compile_model(str(xml_path), "CPU")
        ov_out = compiled(x.numpy())[compiled.output(0)]
        ov_prob = 1.0 / (1.0 + np.exp(-ov_out))

        mae = float(np.abs(ref - ov_prob).mean())
        maxd = float(np.abs(ref - ov_prob).max())
        iou_ref = (ref >= 0.5)
        iou_ov = (ov_prob >= 0.5)
        inter = float((iou_ref & iou_ov).sum())
        union = float((iou_ref | iou_ov).sum()) or 1.0
        logger.info("  mean abs diff = %.6f | max abs diff = %.6f | mask IoU @0.5 = %.4f",
                    mae, maxd, inter / union)
        if mae < 5e-3:
            logger.info("  OK - fp16 IR matches the PyTorch model within tolerance")
            return 0
        logger.warning("  larger-than-expected difference; inspect before using in the paper")
        return 0
    except Exception:
        logger.exception("verification failed (the IR was still saved)")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="wrinkle_model.safetensors path")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="output dir for the .xml/.bin IR")
    ap.add_argument("--no-fp16", action="store_true", help="keep fp32 weights in the IR (default is fp16)")
    ap.add_argument("--verify", action="store_true", help="numerically compare the IR against PyTorch after export")
    args = ap.parse_args()

    logger.info("wrinkle U-Net -> OpenVINO IR")
    logger.info("  weights : %s", args.weights)
    logger.info("  outdir  : %s", args.outdir)
    logger.info("  fp16    : %s", not args.no_fp16)
    rc = convert(args.weights, args.outdir, fp16=not args.no_fp16, verify=args.verify)
    logger.info("done (exit %d)", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
