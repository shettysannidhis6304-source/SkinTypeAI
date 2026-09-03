"""
models/skin_analysis/heuristics.py

Classical computer-vision ESTIMATES for two conditions that have no defensible
public pretrained model or dataset for facial photos: **redness** and
**enlarged pores**.

These are NOT machine-learning predictions. They are colour / texture
statistics on the skin region. Every value produced here is surfaced to the
user as "estimated" (never "detected"), with no confidence score.

Also provides:
  * face_region(image)  -> a best-effort face crop + a skin-pixel boolean mask,
    used to focus both these heuristics and the wrinkle model on skin.

Approach inspired by the open-source github.com/DurtyDhiana/skin-scan (MIT).
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_mp_face = None


def _get_face_detector():
    global _mp_face
    if _mp_face is None:
        try:
            import mediapipe as mp

            _mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("mediapipe unavailable (%s); face detection disabled", e)
            _mp_face = False
    return _mp_face or None


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Loose skin-colour mask (YCrCb + HSV) to exclude hair/background/clothes.
    Not perfect - just enough to keep the heuristics off obvious non-skin."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    m2 = cv2.inRange(hsv, (0, 15, 40), (25, 170, 255))
    m3 = cv2.inRange(hsv, (160, 15, 40), (180, 170, 255))
    mask = cv2.bitwise_and(m1, cv2.bitwise_or(m2, m3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask > 0


def face_region(image_path: str) -> dict[str, Any]:
    """Returns {crop_bgr, skin_mask, face_detected, bbox}. Falls back to the
    whole image (with a skin-colour mask) when no face is found - common for
    ACNE04-style close-ups."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"cannot read image {image_path}")
    h, w = bgr.shape[:2]

    face_detected = False
    bbox = None
    det = _get_face_detector()
    if det is not None:
        try:
            res = det.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.detections:
                rb = res.detections[0].location_data.relative_bounding_box
                x0 = max(0, int((rb.xmin - 0.06) * w))
                y0 = max(0, int((rb.ymin - 0.06) * h))
                x1 = min(w, int((rb.xmin + rb.width + 0.06) * w))
                y1 = min(h, int((rb.ymin + rb.height + 0.10) * h))
                if x1 - x0 > 40 and y1 - y0 > 40:
                    bgr = bgr[y0:y1, x0:x1]
                    face_detected = True
                    bbox = [x0, y0, x1, y1]
        except Exception as e:  # pragma: no cover
            logger.warning("face detection error: %s", e)

    return {
        "crop_bgr": bgr,
        "skin_mask": _skin_mask(bgr),
        "face_detected": face_detected,
        "bbox": bbox,
        "orig_size": [w, h],
    }


# ---------------------------------------------------------------------------
# CV analysis engine -- deterministic computer-vision measurements for the
# conditions that have no trained model or labelled facial dataset:
# redness, enlarged pores, blackheads, dark spots.
#
# These are NOT machine learning. Each produces:
#   * score        : a normalised 0..1 severity from a documented formula
#   * distribution : soft membership over 3 bands (from distance-to-boundary)
#   * confidence   : how decisively the score sits inside one band, scaled by
#                    signal quality (skin-pixel coverage). NOT a model
#                    probability -- it is a measurement-reliability figure.
# The backend surfaces these with method "cv_analysis" / model
# "cv_analysis_engine" and status "estimated" (never "detected").
# ---------------------------------------------------------------------------
def _band_distribution(
    score: float, names: tuple[str, str, str], quality: float, temp: float = 0.16
) -> tuple[dict[str, float], float, str]:
    """Soft 3-band membership for a 0..1 score. Band centres at 1/6, 1/2, 5/6.
    `quality` in [0,1] blends the distribution toward uniform (lower certainty)
    when the underlying signal is weak. Returns (distribution, confidence, band)."""
    centres = (1 / 6, 1 / 2, 5 / 6)
    d = np.array([abs(score - c) for c in centres], dtype=np.float64)
    logits = -d / max(temp, 1e-3)
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    q = float(np.clip(quality, 0.0, 1.0))
    p = p * q + (1.0 - q) / 3.0
    p = p / p.sum()
    dist = {n: round(float(x), 4) for n, x in zip(names, p)}
    top = int(p.argmax())
    return dist, round(float(p[top]), 4), names[top]


def _zone_masks(h: int, w: int, skin: np.ndarray) -> dict[str, np.ndarray]:
    """Crude geometric facial zones for a roughly-centred face crop, intersected
    with the skin mask. Used when only a face box (no landmarks) is available."""
    z = {}
    forehead = np.zeros((h, w), bool)
    forehead[int(0.04 * h):int(0.30 * h), int(0.18 * w):int(0.82 * w)] = True
    nose = np.zeros((h, w), bool)
    nose[int(0.28 * h):int(0.62 * h), int(0.38 * w):int(0.62 * w)] = True
    lcheek = np.zeros((h, w), bool)
    lcheek[int(0.34 * h):int(0.70 * h), int(0.10 * w):int(0.36 * w)] = True
    rcheek = np.zeros((h, w), bool)
    rcheek[int(0.34 * h):int(0.70 * h), int(0.64 * w):int(0.90 * w)] = True
    z["forehead"] = forehead & skin
    z["tzone"] = (forehead | nose) & skin
    z["cheeks"] = (lcheek | rcheek) & skin
    return z


def redness_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray) -> dict[str, Any]:
    """CIELAB a* on skin pixels. Higher a* => redder. Score maps a* 128 (neutral)
    -> 0 and a* 150 (strong erythema) -> 1, nudged up by the fraction of skin
    that is locally very red."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    a_skin = a[skin_mask]
    a_mean = float(a_skin.mean())
    # local erythema: fraction of skin markedly redder than its own surround
    thr = a_skin.mean() + 2.0 * a_skin.std()
    red_frac = float((a_skin > thr).mean())
    # a* ~140 = neutral skin, ~160 = strong erythema (calibrated on the project's crops)
    score = float(np.clip((a_mean - 140.0) / 20.0 + 0.5 * red_frac, 0.0, 1.0))

    quality = float(np.clip(skin_mask.mean() / 0.25, 0.3, 1.0))
    dist, conf, band = _band_distribution(score, ("low", "moderate", "elevated"), quality)
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": f"{band} redness",
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist, "a_star_mean": round(a_mean, 1),
                   "red_pixel_fraction": round(red_frac, 4),
                   "note": "CIELAB a* colour analysis on the skin region - a measurement, not a trained model"},
        "note": "Computer-vision colour analysis. No trained facial-redness model or dataset exists; "
                "this is a quantitative estimate, not a clinical assessment.",
    }


def _texture_density(crop_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of a skin region flagged by a tight black-hat filter (small dark
    pits = pores / fine texture). Empirically ~0.05 on smooth skin, higher on
    coarse/oily skin."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 512:
        gray = cv2.resize(gray, (512, int(512 * gray.shape[0] / gray.shape[1])))
    m = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.sum() < 200:
        return 0.0
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    _, sp = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(((sp > 0) & m).sum()) / max(int(m.sum()), 1)


def pore_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray,
                  oiliness_score: float | None = None) -> dict[str, Any]:
    """Pore visibility. Pore visibility correlates strongly with sebum, so when
    the model's oiliness score is available it is the primary term, refined by a
    black-hat skin-texture density measurement. Without it, texture only (lower
    confidence)."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    density = _texture_density(crop_bgr, skin_mask)
    tex = float(np.clip((density - 0.05) / 0.12, 0.0, 1.0))     # 0.05 smooth .. 0.17 coarse

    if oiliness_score is not None:
        score = 0.55 * float(np.clip(oiliness_score, 0.0, 1.0)) + 0.45 * tex
        basis = "model oiliness + skin-texture density"
        qual = 0.85
    else:
        score = tex
        basis = "skin-texture density only (oiliness unavailable)"
        qual = 0.5

    quality = float(np.clip(skin_mask.mean() / 0.25, 0.3, 1.0)) * qual
    dist, conf, band = _band_distribution(score, ("low", "moderate", "elevated"), quality)
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": f"{band} pore visibility",
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist, "texture_density": round(density, 5),
                   "basis": basis,
                   "note": "sebum-linked estimate, not a trained pore model"},
        "note": "Computer-vision estimate. Pore visibility tracks sebum output; this combines the model's "
                "oiliness read with a skin-texture measurement. Not a clinical assessment.",
    }


def blackhead_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray) -> dict[str, Any]:
    """Localised comedone estimate: count of small, high-contrast dark round
    points in the T-zone (forehead + nose), from a tight black-hat filter at a
    fixed contrast threshold with a size/shape gate. Empirically monotonic with
    acne grade (clean < mild < severe) on the project's validation crops, but a
    weak signal -- confidence is capped accordingly."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    tz = _zone_masks(h, w, skin_mask)["tzone"]
    tz_area = int(tz.sum())
    if tz_area < 300:
        return {
            "status": "estimated", "method": "cv_analysis", "label": "no blackheads detected",
            "score": 0.0, "confidence": 0.3,
            "detail": {"band": "none", "distribution": {"none": 0.6, "few": 0.25, "many": 0.15},
                       "tzone_visible": False, "count": 0},
            "model": "cv_analysis_engine",
            "note": "Computer-vision estimate: the T-zone was not clearly visible in this photo.",
        }

    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cand = ((bh >= 15) & tz).astype(np.uint8)      # fixed contrast threshold (calibrated)
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(cand, connectivity=8)
    count = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bhh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 2 <= area <= 60 and 0.4 <= (bw / max(bhh, 1)) <= 2.5:
            count += 1

    per_1k = count / (tz_area / 1000.0)
    score = float(np.clip((per_1k - 2.5) / 5.0, 0.0, 1.0))   # ~2.5/1k clean .. ~7.5/1k heavy
    quality = float(np.clip(tz_area / 4000.0, 0.35, 1.0)) * 0.7   # weak signal -> cap confidence
    dist, conf, band = _band_distribution(score, ("none", "few", "many"), quality)
    label = {"none": "no significant blackheads", "few": "a few blackheads in the T-zone",
             "many": "noticeable blackhead density in the T-zone"}[band]
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": label,
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist, "count": count,
                   "tzone_visible": True, "per_1000px": round(per_1k, 3),
                   "note": "T-zone black-hat comedone count - a weak measurement, not a trained model"},
        "model": "cv_analysis_engine",
        "note": "Computer-vision estimate of open comedones in the T-zone. No trained comedone model or "
                "public facial dataset exists; a weak indicator, not a clinical diagnosis.",
    }


def _darkspot_spatial(crop_bgr: np.ndarray, skin_mask: np.ndarray) -> tuple[float, bool, float]:
    """Spatial-clustering signal for dark spots on the cheeks + forehead: area
    fraction of that region that is a *discrete blob* markedly darker (>=10 L*)
    than the region's own median skin lightness. Returns (frac, region_visible,
    quality)."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    h, w = L.shape
    zones = _zone_masks(h, w, skin_mask)
    region = zones["cheeks"] | zones["forehead"]
    r_area = int(region.sum())
    if r_area < 400:
        return 0.0, False, 0.35
    med = float(np.median(L[region]))
    darker = ((med - L) >= 14.0) & region       # markedly darker than the region's own skin
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(darker.astype(np.uint8), connectivity=8)
    blob_px = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                  if 10 <= stats[i, cv2.CC_STAT_AREA] <= 400)
    frac = blob_px / max(r_area, 1)
    return frac, True, float(np.clip(r_area / 6000.0, 0.35, 1.0))


def dark_spot_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray,
                       pigment_dist: dict[str, float] | None = None) -> dict[str, Any]:
    """Localised hyperpigmentation. Primary signal is the trained pigmentation
    head (its 'mild'/'marked' probability), localised to the cheeks + forehead
    by a spatial-clustering check (discrete darker blobs there). Without the
    model, the spatial check alone drives a low-confidence estimate."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    frac, region_ok, qual = _darkspot_spatial(crop_bgr, skin_mask)
    spatial = float(np.clip(frac / 0.04, 0.0, 1.0))

    if pigment_dist:
        pig = 0.4 * float(pigment_dist.get("mild", 0.0)) + 1.0 * float(pigment_dist.get("marked", 0.0))
        score = float(np.clip(0.85 * pig + 0.15 * spatial, 0.0, 1.0))
        basis = "trained pigmentation head + cheek/forehead spatial clustering"
        q = 0.8
    else:
        score = spatial * 0.7
        basis = "cheek/forehead spatial clustering only (pigmentation model unavailable)"
        q = 0.45

    quality = qual * q * (1.0 if region_ok else 0.6)
    dist, conf, band = _band_distribution(score, ("none", "mild", "marked"), quality)
    label = {"none": "no notable dark spots", "mild": "mild localised dark spots (cheeks/forehead)",
             "marked": "marked localised dark spots (cheeks/forehead)"}[band]
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": label,
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist, "spatial_blob_fraction": round(frac, 5),
                   "region_visible": region_ok, "basis": basis,
                   "note": "hyperpigmentation localised from the pigmentation head - not an independent detector"},
        "note": "Computer-vision reading of the trained pigmentation model, localised to sun-exposed zones. "
                "No independent dark-spot model or public facial dataset exists; not a clinical diagnosis.",
    }
