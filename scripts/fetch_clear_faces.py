#!/usr/bin/env python3
"""
scripts/fetch_clear_faces.py

Samples ~100 clean, acne-free face crops from a lightweight public FFHQ subset
(`bitmind/ffhq-256` on the Hugging Face Hub -- FFHQ images are CC BY-NC-SA /
research use; this is an academic project). FFHQ is a curated set of aligned,
high-quality face portraits: the large majority show clear skin, no visible
acne, and even tone -- exactly the "normal / clear" examples missing from the
ACNE04-only dataset.

Output:
    datasets/processed/clear_faces/<idx>.jpg     224x224 face crops
    datasets/processed/clear_faces_manifest.json  provenance

These crops are consumed by scripts/prepare_dataset.py, which assigns them the
fixed ground-truth labels:
    skin_type = "normal"   acne_severity = "clear"   pigmentation = "normal"

Usage (project root, .venv-new; needs internet once):
    .venv-new\\Scripts\\python.exe scripts\\fetch_clear_faces.py
    .venv-new\\Scripts\\python.exe scripts\\fetch_clear_faces.py --count 100 --offset 0
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skin_analysis import heuristics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_clear_faces")

OUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "clear_faces"
MANIFEST = PROJECT_ROOT / "datasets" / "processed" / "clear_faces_manifest.json"

DATASET = "bitmind/ffhq-256"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SPLITS_API = "https://datasets-server.huggingface.co/splits"
CROP_SIZE = 224
_UA = {"User-Agent": "SkinTypeAI-academic/1.0"}


def _get_json(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _download(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _face_crop(bgr: np.ndarray) -> np.ndarray | None:
    """FFHQ images are already aligned/cropped; run the same MediaPipe crop the
    rest of the pipeline uses, falling back to a centre square."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h
    det = heuristics._get_face_detector()
    if det is not None:
        try:
            res = det.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.detections:
                rb = res.detections[0].location_data.relative_bounding_box
                m = 0.12
                x0 = max(0, int((rb.xmin - m) * w))
                y0 = max(0, int((rb.ymin - m) * h))
                x1 = min(w, int((rb.xmin + rb.width + m) * w))
                y1 = min(h, int((rb.ymin + rb.height + m) * h))
        except Exception:
            pass
    if x1 - x0 < 40 or y1 - y0 < 40:
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        x1, y1 = x0 + side, y0 + side
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=100, help="number of clean crops to keep")
    ap.add_argument("--offset", type=int, default=0, help="starting row offset in the FFHQ subset")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    socket.setdefaulttimeout(args.timeout)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        splits = _get_json(f"{SPLITS_API}?dataset={DATASET}")
        cfg = splits["splits"][0]["config"]
        split = splits["splits"][0]["split"]
    except Exception as e:
        logger.error("cannot reach the Hugging Face datasets-server (%s). "
                     "Run this once on a machine with internet.", e)
        return 2

    kept = 0
    row_offset = args.offset
    manifest_rows: list[dict] = []
    page = 100

    while kept < args.count:
        want = min(page, (args.count - kept) + 20)  # over-fetch to cover crop failures
        try:
            data = _get_json(f"{ROWS_API}?dataset={DATASET}&config={cfg}&split={split}"
                             f"&offset={row_offset}&length={want}")
        except Exception as e:
            logger.error("rows fetch failed at offset %d: %s", row_offset, e)
            break
        rows = data.get("rows", [])
        if not rows:
            logger.warning("no more rows at offset %d", row_offset)
            break

        for item in rows:
            if kept >= args.count:
                break
            row_offset += 1
            src = (((item.get("row") or {}).get("image")) or {}).get("src")
            if not src:
                continue
            try:
                raw = _download(src)
                arr = np.frombuffer(raw, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                crop = _face_crop(bgr)
                if crop is None:
                    continue
            except Exception as e:
                logger.debug("row %d failed: %s", row_offset, e)
                continue
            idx = kept
            cv2.imwrite(str(OUT_DIR / f"clear_{idx:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest_rows.append({"file": f"clear_{idx:04d}.jpg", "source_row": row_offset - 1})
            kept += 1
            if kept % 20 == 0:
                logger.info("  kept %d / %d", kept, args.count)

    MANIFEST.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": f"{DATASET} (FFHQ 256px subset, Hugging Face Hub)",
        "source_note": "FFHQ = Flickr-Faces-HQ (Karras et al., NVIDIA). Images CC BY 2.0 / "
                       "CC BY-NC / public-domain mix per the FFHQ LICENSE; academic/research use.",
        "crop_size": CROP_SIZE,
        "count": kept,
        "assigned_labels": {"skin_type": "normal", "acne_severity": "clear", "pigmentation": "normal"},
        "label_basis": "FFHQ portraits are curated, high-quality, predominantly clear-skinned faces. "
                       "Labels are asserted (weak) ground truth for the 'normal/clear' classes that "
                       "ACNE04 cannot provide. Documented for the viva.",
        "rows": manifest_rows,
    }, indent=2), encoding="utf-8")

    print(f"\nfetched {kept} clear face crops -> {OUT_DIR}")
    print(f"manifest -> {MANIFEST}")
    if kept < args.count:
        print(f"WARNING: only {kept}/{args.count} kept (network or crop failures).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
