"""
models/skin_analysis/opencv_engine.py

`OpenCVSkinAnalyzer` -- the deterministic computer-vision half of the 9-characteristic
system. It owns the 4 characteristics that have NO trained model and NO public
labelled facial dataset:

    redness   |   dark_spots   |   pores   |   blackheads

Everything here is classical OpenCV (colour-space stats + morphology) on the
MediaPipe/OpenCV-detected skin region. Nothing is learned; the same image always
gives the same numbers. Each characteristic is surfaced as "estimated"
(never "detected"), with a measurement-reliability figure rather than a model
probability.

Single source of truth: the per-characteristic formulas live in
models.skin_analysis.heuristics (already used by the backend and validated
against the existing response schema). This class is a thin, documented facade
that runs all four in one call and returns a stable dict -- so Stage 4 can route
the OpenCV traits through one object without touching the formulas.

Spatial masking + lighting normalisation (applied inside heuristics, so the
Stage-1 weak-label generation and the served analysis stay consistent):
  * pores      -> nasal wings + inner cheeks + chin  (FaceMesh polygons)
  * dark_spots -> cheeks + forehead                  (FaceMesh polygons)
  * eyes / eyebrows / lips are masked out completely (dilated exclusion mask)
  * Gray-World colour constancy before CIELAB conversion (dark spots)
  * CLAHE(clipLimit=2.0, tileGridSize=(8,8)) on the L* / grayscale channel
    before the dark-spot / blackhead thresholds
Geometric fallback zones are used whenever FaceMesh cannot localise a face.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from models.skin_analysis import heuristics

logger = logging.getLogger(__name__)

# The 4 characteristics this engine is responsible for. The keys match the
# canonical condition keys used by backend/app/services/analysis_service.py
# ("enlarged_pores", not "pores") so Stage 4 integration is a drop-in.
CHARACTERISTICS: tuple[str, ...] = ("redness", "enlarged_pores", "blackheads", "dark_spots")


class OpenCVSkinAnalyzer:
    """Deterministic OpenCV analyser for redness / pores / blackheads / dark spots."""

    def __init__(self, *, min_skin_pixels: int = 500) -> None:
        self.min_skin_pixels = int(min_skin_pixels)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def face_region(image: str | np.ndarray) -> dict[str, Any]:
        """Return {crop_bgr, skin_mask, face_detected, bbox} for a path or a BGR array.
        For a raw array (already-read upload / crop) no face detection is run -- the
        whole array is treated as the region, with a skin-colour mask."""
        if isinstance(image, str):
            return heuristics.face_region(image)
        if isinstance(image, np.ndarray):
            bgr = image
            return {
                "crop_bgr": bgr,
                "skin_mask": heuristics._skin_mask(bgr),
                "face_detected": False,
                "bbox": None,
                "orig_size": [bgr.shape[1], bgr.shape[0]],
            }
        raise TypeError(f"OpenCVSkinAnalyzer.face_region: unsupported type {type(image)}")

    # ------------------------------------------------------------------ analysis
    def analyze(
        self,
        image: str | np.ndarray,
        *,
        face: dict[str, Any] | None = None,
        oiliness_score: float | None = None,
        pigment_dist: dict[str, float] | None = None,
        landmarks: list | None = None,
    ) -> dict[str, dict]:
        """Run all four characteristics on one image.

        Parameters
        ----------
        image          : path, or a BGR ndarray already read by cv2.
        face           : optional pre-computed region dict (from heuristics.face_region
                         or self.face_region) -- pass it to avoid re-detecting the face
                         when the caller already has one.
        oiliness_score : optional 0..1 oiliness signal from SkinAnalysisModel -- refines
                         the pore estimate (pore visibility tracks sebum).
        pigment_dist   : optional {normal,mild,marked} distribution from SkinAnalysisModel
                         -- localises the dark-spot estimate.

        Returns
        -------
        {"redness": {...}, "enlarged_pores": {...}, "blackheads": {...}, "dark_spots": {...}}
        Each value is the canonical 8-key condition entry (status/method/label/score/
        confidence/detail/model/note) already understood by the backend.
        """
        if face is None:
            face = self.face_region(image)
        crop, skin = face["crop_bgr"], face["skin_mask"]

        jobs = {
            "redness": lambda: heuristics.redness_estimate(crop, skin),
            "enlarged_pores": lambda: heuristics.pore_estimate(crop, skin, oiliness_score=oiliness_score,
                                                              landmarks=landmarks),
            "blackheads": lambda: heuristics.blackhead_estimate(crop, skin),
            "dark_spots": lambda: heuristics.dark_spot_estimate(crop, skin, pigment_dist=pigment_dist,
                                                               landmarks=landmarks),
        }

        out: dict[str, dict] = {}
        for key, fn in jobs.items():
            try:
                entry = fn()
                entry.setdefault("model", "cv_analysis_engine")
                if entry.get("status") not in ("estimated", "not_available"):
                    entry["status"] = "estimated"
                out[key] = entry
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("OpenCV characteristic %s failed", key)
                out[key] = {
                    "status": "error", "method": None, "label": None, "score": None,
                    "confidence": None, "detail": {"error": repr(exc)},
                    "model": "cv_analysis_engine", "note": None,
                }
        return out

    __call__ = analyze

    # ------------------------------------------------------------------ zone overlay
    # The measurement zones this engine reads, and how the results view colours them
    # (BGR). Kept here so the frontend legend and this drawing stay in sync.
    ZONE_LEGEND: tuple[dict[str, Any], ...] = (
        {"key": "redness",    "label": "Redness / erythema", "measures": "Full skin region",
         "bgr": (90, 90, 235), "css": "#eb5a5a"},
        {"key": "dark_spots", "label": "Dark spots & pigmentation", "measures": "Cheeks + forehead",
         "bgr": (235, 150, 60), "css": "#3c96eb"},
        {"key": "pores",      "label": "Pores & blackheads", "measures": "Nasal wings, inner cheeks, chin",
         "bgr": (110, 200, 150), "css": "#3ec89a"},
    )

    def zone_overlay(self, image: str | np.ndarray, *, face: dict[str, Any] | None = None,
                     landmarks: list | None = None) -> np.ndarray:
        """Return a BGR image with the MediaPipe / geometric measurement zones drawn
        as translucent, contoured regions:

            full skin mask      -> soft red   (redness / erythema)
            cheeks + forehead   -> blue       (dark spots & pigmentation)
            T-zone / nasal wings -> teal-green (pores & blackheads)
        """
        if face is None:
            face = self.face_region(image)
        crop = face["crop_bgr"]
        skin = face["skin_mask"]
        h, w = crop.shape[:2]

        tr = heuristics._facemesh_trait_regions(crop, skin, landmarks=landmarks)
        if tr is not None:
            pores_m = tr["pores"]
            spots_m = tr["dark_spots"]
        else:                                        # geometric fallback
            zg = heuristics._zone_masks(h, w, skin)
            pores_m = zg["tzone"] | zg["cheeks"]
            spots_m = zg["cheeks"] | zg["forehead"]

        layers = (
            (skin.astype(bool),   (90, 90, 235)),    # redness  -- soft red
            (spots_m.astype(bool), (235, 150, 60)),  # dark spots -- blue
            (pores_m.astype(bool), (110, 200, 150)),  # pores -- teal-green
        )

        out = crop.copy()
        for mask, color in layers:
            if int(mask.sum()) < 50:
                continue
            fill = out.copy()
            fill[mask] = color
            cv2.addWeighted(fill, 0.24, out, 0.76, 0.0, dst=out)
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, max(1, w // 320))
        return out
