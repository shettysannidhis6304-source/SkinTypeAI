"""
backend/app/routers/overlay.py

GET /api/images/{image_id}/overlay?mode=gradcam|facemesh_zones&head=acne|wrinkles

Explainability overlays for the Analysis Results image card:
  * gradcam        -- Grad-CAM heatmap (cv2.COLORMAP_TURBO) for `head`, blended
                      onto the photo. Shows the visual patterns that most
                      influenced the from-scratch CNN's classification.
  * facemesh_zones -- the MediaPipe (or geometric-fallback) measurement zones the
                      OpenCV engine reads for redness / pores / dark spots.

Both return a PNG at the ORIGINAL image dimensions, so the frontend can stack the
overlay directly over /images/{image_id} with no re-alignment.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.core.config import resolve_upload_path
from app.core.security import get_current_user
from app.models_db.base import get_db
from app.models_db.image import Image
from app.models_db.user import User

# models/ lives at the project root (sibling of backend/); make it importable
# regardless of the cwd uvicorn was started from -- same shim as analysis_service.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["overlay"])

_GRADCAM_HEADS = ("acne", "wrinkles", "oiliness", "dryness", "pigmentation")
# "zones" is the public alias; "facemesh_zones" kept for back-compat.
_MODE_ALIAS = {"gradcam": "gradcam", "zones": "facemesh_zones", "facemesh_zones": "facemesh_zones"}


def _png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:  # pragma: no cover
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to encode overlay PNG")
    return buf.tobytes()


def _paste_back(original: np.ndarray, rendered_crop: np.ndarray, bbox) -> np.ndarray:
    """Put a crop-space render back onto a copy of the original photo so the
    result lines up with GET /images/{id}."""
    if not bbox:
        h, w = original.shape[:2]
        if rendered_crop.shape[:2] != (h, w):
            rendered_crop = cv2.resize(rendered_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return rendered_crop
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.shape[1], x1), min(original.shape[0], y1)
    out = original.copy()
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = cv2.resize(rendered_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return out


@router.get("/api/images/{image_id}/overlay")
def image_overlay(
    image_id: int,
    mode: str = Query("gradcam", description="gradcam | zones"),
    head: str = Query("acne", description="Grad-CAM target head: acne | wrinkles | ..."),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    mode = _MODE_ALIAS.get(mode, "")
    if not mode:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "mode must be one of ('gradcam', 'zones')")
    if head not in _GRADCAM_HEADS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"head must be one of {_GRADCAM_HEADS}")

    image = db.get(Image, image_id)
    if image is None or image.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    path = resolve_upload_path(image.file_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image file missing on disk")

    from models.skin_analysis import heuristics
    from models.skin_analysis.opencv_engine import OpenCVSkinAnalyzer

    original = cv2.imread(str(path))
    if original is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Could not read the image")
    headers = {"Cache-Control": "private, max-age=300"}

    # Any failure past here becomes a 4xx/5xx *through* CORSMiddleware, so the
    # browser can read the message instead of a bare net::ERR_FAILED.
    try:
        face = heuristics.face_region(str(path))
        crop, bbox = face["crop_bgr"], face.get("bbox")

        if mode == "facemesh_zones":
            rendered = OpenCVSkinAnalyzer().zone_overlay(str(path), face=face)
            return Response(_png(_paste_back(original, rendered, bbox)), media_type="image/png", headers=headers)

        # ---- mode == "gradcam" ----
        from app.services.analysis_service import _get_skin_analysis_model

        loaded = _get_skin_analysis_model()
        if loaded is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "Analysis model unavailable -- train weights/skin_analysis_model.pt.")
        model, ckpt = loaded

        from models.skin_analysis.gradcam import generate_gradcam

        heat = generate_gradcam(model, crop, target_head=head, config=ckpt.get("config"))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("overlay(mode=%s, head=%s) failed for image %s", mode, head, image_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Overlay generation failed: {type(exc).__name__}") from exc

    hm = cv2.resize((np.clip(heat, 0.0, 1.0) * 255).astype(np.uint8), (crop.shape[1], crop.shape[0]))
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(hm_color, 0.5, crop, 0.5, 0.0)
    return Response(_png(_paste_back(original, blended, bbox)), media_type="image/png", headers=headers)
