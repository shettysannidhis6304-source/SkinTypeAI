"""
models/skin_analysis/heuristics.py

Classical computer-vision ESTIMATES for conditions that have no defensible
public pretrained model or dataset for facial photos: **redness**, **enlarged
pores**, blackheads and dark spots.

These are NOT machine-learning predictions. They are colour / texture
statistics on the skin region. Every value produced here is surfaced to the
user as "estimated" (never "detected").

Literature-grounded measurements:
  * redness -> the dermatological **Erythema Index**
        EI = 100 * (log10(R_skin) - log10(G_skin))
    on masked skin pixels (see redness_estimate + EI_BASELINE / EI_STRONG).
  * pores   -> black-hat texture density computed STRICTLY over the cheek +
    nasal skin region (landmark-anchored via MediaPipe FaceMesh, with a refined
    geometric fallback) so hairline / jaw / lip texture cannot inflate it.

Also provides:
  * face_region(image)  -> a best-effort face crop + a skin-pixel boolean mask,
    used to focus both these heuristics and the wrinkle model on skin.

Approach inspired by the open-source github.com/DurtyDhiana/skin-scan (MIT).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Dermatological Erythema Index (EI) calibration -------------------------
# EI = 100 * (log10(R_skin) - log10(G_skin))  over mean skin-pixel R,G channels.
# Published EI scales (e.g. Park et al. 2020; Yamamoto et al. 2008) assume
# spectrophotometer / calibrated-reflectance capture and quote higher absolute
# numbers. On uncalibrated 8-bit sRGB face photos the SAME formula yields ~12
# for healthy skin (measured: mean 12.6, p90 16.5 across this project's ACNE04
# + FFHQ skin crops). The healthy-baseline / strong-erythema bounds below are
# therefore calibrated to THIS capture pipeline, not copied from a paper.
EI_BASELINE = 12.0     # ~ mean EI of healthy skin in this pipeline  -> score 0.0
EI_STRONG = 26.0       # ~ 2x baseline: marked diffuse erythema      -> score 1.0

_mp_face = None
_mp_mesh = None


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


def _get_face_mesh():
    """Lazy MediaPipe FaceMesh singleton (468-pt landmarks) for region gating.
    Returns None if mediapipe is missing; callers fall back to geometric zones."""
    global _mp_mesh
    if _mp_mesh is None:
        try:
            import mediapipe as mp

            _mp_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.4,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("mediapipe FaceMesh unavailable (%s); landmark gating disabled", e)
            _mp_mesh = False
    return _mp_mesh or None


def facemesh_landmarks(bgr: np.ndarray) -> list[tuple[float, float]]:
    """Run MediaPipe FaceMesh ONCE and return its 468 normalised (x, y) landmarks
    -- or an EMPTY list when mediapipe is unavailable / no face is found.

    An empty list is a valid "already computed, nothing there" result: callers
    pass it straight back into `_facemesh_trait_regions` / `_head_pose_deg` /
    `_facemesh_region_masks` via `landmarks=` so the mesh never runs twice, even
    for a faceless crop.
    """
    mesh = _get_face_mesh()
    if mesh is None:
        return []
    try:
        res = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("FaceMesh process failed: %s", e)
        return []
    if not getattr(res, "multi_face_landmarks", None):
        return []
    return [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]


def _landmarks_for(bgr: np.ndarray, landmarks: list | None) -> list[tuple[float, float]]:
    """Memoised landmarks if supplied (including an empty list), otherwise a
    single fresh FaceMesh pass. The ONLY place a fallback `.process()` fires."""
    return landmarks if landmarks is not None else facemesh_landmarks(bgr)


# MediaPipe FaceMesh landmark index groups (canonical 468-pt topology).
_LM_LEFT_CHEEK = [116, 117, 118, 119, 100, 142, 203, 206, 216, 212, 123, 147, 187, 205, 36]
_LM_RIGHT_CHEEK = [345, 346, 347, 348, 329, 371, 423, 426, 436, 432, 352, 376, 411, 425, 266]
_LM_NOSE = [8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 45, 275, 220, 440, 218, 438, 48, 278]
_LM_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78]
_LM_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                 234, 127, 162, 21, 54, 103, 67, 109]

# --- Spatial-masking landmark groups (canonical 468-pt FaceMesh topology) ----
# Used by _facemesh_trait_regions() to build the tight per-trait polygons and the
# eyes/eyebrows/lips exclusion mask required by the OpenCV spatial-masking spec.
_LM_LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 246]
_LM_RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 466]
_LM_LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_LM_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_LM_NASAL_WINGS = [48, 115, 220, 45, 4, 275, 440, 344, 278, 79, 309]
_LM_CHIN = [175, 152, 148, 176, 150, 169, 170, 140, 171, 396, 394, 395, 369, 400, 377,
            378, 379, 200, 199, 18, 83, 313, 421, 428]
_LM_FOREHEAD = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 108, 151, 337,
                299, 333, 298, 301, 68, 104, 69, 66, 107, 336, 296, 334, 293, 63]
_LM_INNER_LEFT_CHEEK = [50, 205, 206, 207, 187, 123, 116, 117, 118, 101, 36, 142, 129]
_LM_INNER_RIGHT_CHEEK = [280, 425, 426, 427, 411, 352, 345, 346, 347, 330, 266, 371, 358]


def _poly_mask(pts: list[tuple[float, float]], idxs: list[int], h: int, w: int) -> np.ndarray:
    """Filled convex hull (bool mask) of the given normalised landmark indices."""
    m = np.zeros((h, w), np.uint8)
    arr = np.array([[int(round(pts[i][0] * w)), int(round(pts[i][1] * h))]
                    for i in idxs if 0 <= i < len(pts)], dtype=np.int32)
    if len(arr) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(arr), 1)
    return m.astype(bool)


# ---------------------------------------------------------------------------
# Lighting normalisation -- applied before colour-space analysis so warm / yellow
# indoor lighting does not bias the CIELAB dark-spot / pigmentation reads.
# ---------------------------------------------------------------------------
def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Gray-World colour constancy: scale each channel so its mean equals the
    overall grey mean. Neutralises a global colour cast (uint8 BGR in / out)."""
    b = bgr.astype(np.float32)
    means = b.reshape(-1, 3).mean(axis=0)
    grey = float(means.mean())
    scale = grey / np.clip(means, 1e-6, None)
    return np.clip(b * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def apply_clahe(chan_u8: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE (Contrast-Limited Adaptive Histogram Equalisation) on a single uint8
    channel -- L* or grayscale -- to even out local illumination before adaptive
    thresholding for dark spots / blackheads."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    return clahe.apply(chan_u8)


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
    """Geometric facial zones for a roughly-centred, roughly-frontal face crop,
    intersected with the skin mask. Used when FaceMesh landmarks are not
    available. The boxes deliberately stop above the lip line, subtract a central
    mouth region, and drop an 8% outer border so peripheral shadow / hair / jaw
    edges don't leak into the cheek and T-zone measurements."""
    z: dict[str, np.ndarray] = {}

    # 8% outer-border trim (peripheral shadow / vignette / hairline)
    inner = np.zeros((h, w), bool)
    my, mx = int(0.08 * h), int(0.08 * w)
    inner[my:h - my, mx:w - mx] = True

    # central mouth / lip exclusion
    mouth = np.zeros((h, w), bool)
    mouth[int(0.60 * h):int(0.82 * h), int(0.30 * w):int(0.70 * w)] = True

    forehead = np.zeros((h, w), bool)
    forehead[int(0.06 * h):int(0.27 * h), int(0.22 * w):int(0.78 * w)] = True
    nose = np.zeros((h, w), bool)
    nose[int(0.30 * h):int(0.56 * h), int(0.40 * w):int(0.60 * w)] = True
    lcheek = np.zeros((h, w), bool)
    lcheek[int(0.36 * h):int(0.58 * h), int(0.12 * w):int(0.34 * w)] = True
    rcheek = np.zeros((h, w), bool)
    rcheek[int(0.36 * h):int(0.58 * h), int(0.66 * w):int(0.88 * w)] = True

    base = skin & inner & ~mouth
    z["forehead"] = forehead & base
    z["nose"] = nose & base
    z["tzone"] = (forehead | nose) & base
    z["cheeks"] = (lcheek | rcheek) & base
    z["mouth_excluded"] = mouth
    return z


def _facemesh_region_masks(crop_bgr: np.ndarray, skin: np.ndarray,
                           landmarks: list | None = None) -> dict[str, Any] | None:
    """Best-effort landmark-anchored cheek / nose masks with the lips and the
    face border explicitly removed. Returns None if FaceMesh finds no face."""
    pts = _landmarks_for(crop_bgr, landmarks)
    if not pts:
        return None
    h, w = crop_bgr.shape[:2]

    oval = _poly_mask(pts, _LM_FACE_OVAL, h, w).astype(np.uint8)
    oval = cv2.erode(oval, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, w // 18), max(3, h // 18)))).astype(bool)
    lips = _poly_mask(pts, _LM_LIPS, h, w).astype(np.uint8)
    lips = cv2.dilate(lips, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, w // 22), max(3, h // 22)))).astype(bool)

    base = skin & oval & ~lips
    lcheek = _poly_mask(pts, _LM_LEFT_CHEEK, h, w)
    rcheek = _poly_mask(pts, _LM_RIGHT_CHEEK, h, w)
    nose = _poly_mask(pts, _LM_NOSE, h, w)
    return {
        "cheeks": (lcheek | rcheek) & base,
        "nose": nose & base,
        "cheeks_nose": (lcheek | rcheek | nose) & base,
        "_source": "facemesh",
    }


def _cheek_nose_region(crop_bgr: np.ndarray, skin: np.ndarray,
                       landmarks: list | None = None) -> tuple[np.ndarray, str]:
    """(mask, source) for the cheek + nasal skin region -- FaceMesh landmarks
    when available, refined geometric zones otherwise."""
    fm = _facemesh_region_masks(crop_bgr, skin, landmarks=landmarks)
    if fm is not None and int(fm["cheeks_nose"].sum()) >= 400:
        return fm["cheeks_nose"], "facemesh"
    h, w = crop_bgr.shape[:2]
    zg = _zone_masks(h, w, skin)
    return (zg["cheeks"] | zg["nose"]), "geometric"


# ---------------------------------------------------------------------------
# Tight per-trait spatial masks (spec: OpenCV Spatial Masking).
#   pores      -> nasal wings + inner cheeks + chin
#   dark_spots -> cheeks + forehead
#   exclusion  -> eyes + eyebrows + lips masked out completely
# ---------------------------------------------------------------------------
def _facemesh_trait_regions(crop_bgr: np.ndarray, skin: np.ndarray,
                            landmarks: list | None = None) -> dict[str, Any] | None:
    """Landmark-anchored per-trait polygon masks with an eyes/eyebrows/lips
    exclusion mask removed. Returns None if FaceMesh finds no face."""
    pts = _landmarks_for(crop_bgr, landmarks)
    if not pts:
        return None
    h, w = crop_bgr.shape[:2]

    def poly(idxs: list[int]) -> np.ndarray:
        return _poly_mask(pts, idxs, h, w)

    oval = cv2.erode(poly(_LM_FACE_OVAL).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, w // 20), max(3, h // 20)))).astype(bool)

    # exclusion: eyes + eyebrows + lips, each dilated so lash-line / vermilion
    # borders are safely outside every measurement region.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 24), max(3, h // 24)))
    excl = np.zeros((h, w), bool)
    for grp in (_LM_LEFT_EYE, _LM_RIGHT_EYE, _LM_LEFT_BROW, _LM_RIGHT_BROW, _LM_LIPS):
        excl |= cv2.dilate(poly(grp).astype(np.uint8), k).astype(bool)

    base = skin & oval & ~excl
    cheeks = (poly(_LM_LEFT_CHEEK) | poly(_LM_RIGHT_CHEEK)) & base
    inner_cheeks = (poly(_LM_INNER_LEFT_CHEEK) | poly(_LM_INNER_RIGHT_CHEEK)) & base
    forehead = poly(_LM_FOREHEAD) & base
    nasal_wings = poly(_LM_NASAL_WINGS) & base
    chin = poly(_LM_CHIN) & base
    return {
        "pores": inner_cheeks | nasal_wings | chin,
        "dark_spots": cheeks | forehead,
        "cheeks": cheeks,
        "forehead": forehead,
        "exclusion": excl,
        "_source": "facemesh",
    }


def _trait_region(crop_bgr: np.ndarray, skin: np.ndarray, trait: str,
                  geometric_fallback: tuple[str, ...],
                  landmarks: list | None = None) -> tuple[np.ndarray, str]:
    """(mask, source) for a named trait region -- tight FaceMesh polygons when a
    face is found, else the union of the named _zone_masks() geometric zones."""
    tr = _facemesh_trait_regions(crop_bgr, skin, landmarks=landmarks)
    if tr is not None and int(tr.get(trait, np.zeros(0)).sum()) >= 400:
        return tr[trait], "facemesh"
    h, w = crop_bgr.shape[:2]
    zg = _zone_masks(h, w, skin)
    mask = np.zeros((h, w), bool)
    for z in geometric_fallback:
        mask |= zg.get(z, np.zeros((h, w), bool))
    return mask, "geometric"


# ---------------------------------------------------------------------------
# Head-pose estimate -- consumed by analysis_service.assess_image_quality() for
# the advisory 1-10 rating (no hard reject anywhere).
# ---------------------------------------------------------------------------
# 6 canonical 3D points (mm, generic head model) matched to FaceMesh indices for
# solvePnP head-pose: nose tip, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner.
_POSE_LM_IDX = [1, 152, 33, 263, 61, 291]
_POSE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
], dtype=np.float64)


def _head_pose_deg(crop_bgr: np.ndarray, landmarks: list | None = None) -> dict[str, float] | None:
    """Head pose via solvePnP on 6 FaceMesh points.

    `landmarks` -- normalised (x, y) list from `facemesh_landmarks(crop_bgr)`;
    supply it to avoid a second FaceMesh pass. Returns {"pitch","yaw","roll",
    "reproj_norm"} in degrees, or None when no usable face landmarks exist."""
    pts = _landmarks_for(crop_bgr, landmarks)
    if not pts:
        return None
    h, w = crop_bgr.shape[:2]
    if max(_POSE_LM_IDX) >= len(pts):  # pragma: no cover
        return None
    image_pts = np.array([(pts[i][0] * w, pts[i][1] * h) for i in _POSE_LM_IDX], dtype=np.float64)
    f = float(w)
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(_POSE_MODEL_3D, image_pts, cam, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    proj, _ = cv2.projectPoints(_POSE_MODEL_3D, rvec, tvec, cam, dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - image_pts) ** 2).sum(axis=1).mean()))
    eye_dist = float(np.linalg.norm(image_pts[2] - image_pts[3])) or 1.0
    reproj_norm = rms / eye_dist

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

    # fold each angle into [-90, 90] so a near-frontal face reads ~0
    def _fold(a: float) -> float:
        while a > 90:
            a -= 180
        while a < -90:
            a += 180
        return a
    return {"pitch": _fold(pitch), "yaw": _fold(yaw), "roll": _fold(roll),
            "reproj_norm": round(reproj_norm, 4)}


def redness_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray) -> dict[str, Any]:
    """Dermatological **Erythema Index** on masked skin pixels:

        EI = 100 * (log10(R_skin) - log10(G_skin))

    where R_skin / G_skin are the mean red / green channel over skin pixels,
    clamped to >= 1 to avoid log10(0). EI is normalised against calibrated
    healthy-skin bounds (EI_BASELINE .. EI_STRONG) to a continuous 0..1 severity,
    then nudged up slightly by the fraction of skin showing *focal* erythema
    (CIELAB a* outliers). Returned-dict keys are unchanged."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}

    bgr = crop_bgr.astype(np.float32)
    r_skin = float(np.clip(bgr[:, :, 2][skin_mask].mean(), 1.0, 255.0))
    g_skin = float(np.clip(bgr[:, :, 1][skin_mask].mean(), 1.0, 255.0))
    ei = 100.0 * (float(np.log10(r_skin)) - float(np.log10(g_skin)))
    ei_score = float(np.clip((ei - EI_BASELINE) / (EI_STRONG - EI_BASELINE), 0.0, 1.0))

    # focal erythema: fraction of skin markedly redder (CIELAB a*) than its surround
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    a_skin = lab[:, :, 1].astype(np.float32)[skin_mask]
    a_mean = float(a_skin.mean())
    thr = a_skin.mean() + 2.0 * a_skin.std()
    red_frac = float((a_skin > thr).mean())
    # a Gaussian a* distribution already puts ~2.3% of pixels past +2 sigma, so
    # only the EXCESS above that expected tail counts as focal erythema
    focal = min(max(red_frac - 0.023, 0.0) / 0.05, 1.0)

    score = float(np.clip(0.85 * ei_score + 0.15 * focal, 0.0, 1.0))

    quality = float(np.clip(skin_mask.mean() / 0.25, 0.3, 1.0))
    dist, conf, band = _band_distribution(score, ("low", "moderate", "elevated"), quality)
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": f"{band} redness",
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist,
                   "erythema_index": round(ei, 2),
                   "ei_baseline": EI_BASELINE, "ei_strong": EI_STRONG,
                   "r_skin_mean": round(r_skin, 1), "g_skin_mean": round(g_skin, 1),
                   "a_star_mean": round(a_mean, 1), "red_pixel_fraction": round(red_frac, 4),
                   "note": "Erythema Index EI=100*(log10 R - log10 G) on masked skin pixels, "
                           "normalised to pipeline-calibrated healthy bounds - a measurement, not a trained model"},
        "note": "Computer-vision colour analysis (dermatological Erythema Index). No trained "
                "facial-redness model or dataset exists; a quantitative estimate, not a clinical assessment.",
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
                  oiliness_score: float | None = None,
                  landmarks: list | None = None) -> dict[str, Any]:
    """Pore visibility. Texture density is measured STRICTLY over the cheek +
    nasal skin region (landmark-anchored via MediaPipe FaceMesh, refined
    geometric fallback) -- not the whole skin mask -- so hairline / jaw / lip
    texture cannot inflate it. Pore visibility correlates strongly with sebum,
    so when the model's oiliness score is available it is the primary term,
    refined by that texture density; without it, texture only (lower
    confidence). `landmarks` -- memoised FaceMesh landmarks (see facemesh_landmarks).
    Returned-dict keys are unchanged."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}

    region, region_src = _trait_region(crop_bgr, skin_mask, "pores", ("cheeks", "nose"), landmarks=landmarks)
    if int(region.sum()) < 400:                 # region gating failed -> whole-skin fallback
        region, region_src = skin_mask, "skin_mask_fallback"
    density = _texture_density(crop_bgr, region)
    tex = float(np.clip((density - 0.05) / 0.12, 0.0, 1.0))     # 0.05 smooth .. 0.17 coarse

    if oiliness_score is not None:
        score = 0.55 * float(np.clip(oiliness_score, 0.0, 1.0)) + 0.45 * tex
        basis = f"model oiliness + cheek/nasal texture density ({region_src})"
        qual = 0.85
    else:
        score = tex
        basis = f"cheek/nasal texture density only ({region_src}); oiliness unavailable"
        qual = 0.5
    if region_src == "skin_mask_fallback":
        qual *= 0.7

    region_cov = float(region.mean())
    quality = float(np.clip(region_cov / 0.06, 0.3, 1.0)) * qual
    dist, conf, band = _band_distribution(score, ("low", "moderate", "elevated"), quality)
    return {
        "status": "estimated",
        "method": "cv_analysis",
        "label": f"{band} pore visibility",
        "score": round(score, 3),
        "confidence": conf,
        "detail": {"band": band, "distribution": dist, "texture_density": round(density, 5),
                   "region_source": region_src, "region_coverage": round(region_cov, 4),
                   "basis": basis,
                   "note": "sebum-linked estimate over the cheek/nasal region, not a trained pore model"},
        "note": "Computer-vision estimate. Pore visibility tracks sebum output; this combines the model's "
                "oiliness read with a cheek/nasal skin-texture measurement. Not a clinical assessment.",
    }


def blackhead_estimate(crop_bgr: np.ndarray, skin_mask: np.ndarray) -> dict[str, Any]:
    """Localised comedone estimate: count of small, high-contrast dark round
    points in the T-zone (forehead + nose), from a tight black-hat filter at a
    fixed contrast threshold with a size/shape gate. Empirically monotonic with
    acne grade (clean < mild < severe) on the project's validation crops, but a
    weak signal -- confidence is capped accordingly."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    # CLAHE on the grayscale channel evens out local illumination before the
    # black-hat comedone screen (spec: OpenCV spatial masking / lighting norm).
    gray = apply_clahe(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
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


def _darkspot_spatial(crop_bgr: np.ndarray, skin_mask: np.ndarray,
                      landmarks: list | None = None) -> tuple[float, bool, float]:
    """Spatial-clustering signal for dark spots on the cheeks + forehead: area
    fraction of that region that is a *discrete blob* markedly darker (>=10 L*)
    than the region's own median skin lightness. Returns (frac, region_visible,
    quality)."""
    # Gray-World white balance before CIELAB so warm/yellow room lighting does
    # not bias L*, then CLAHE on L* before the dark-blob threshold. Region is the
    # tight cheeks+forehead FaceMesh polygon (spec: OpenCV spatial masking).
    wb = gray_world_white_balance(crop_bgr)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    L = apply_clahe(lab[:, :, 0]).astype(np.float32)
    h, w = L.shape
    region, _src = _trait_region(crop_bgr, skin_mask, "dark_spots", ("cheeks", "forehead"), landmarks=landmarks)
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
                       pigment_dist: dict[str, float] | None = None,
                       landmarks: list | None = None) -> dict[str, Any]:
    """Localised hyperpigmentation. Primary signal is the trained pigmentation
    head (its 'mild'/'marked' probability), localised to the cheeks + forehead
    by a spatial-clustering check (discrete darker blobs there). Without the
    model, the spatial check alone drives a low-confidence estimate.
    `landmarks` -- memoised FaceMesh landmarks (see facemesh_landmarks)."""
    if skin_mask.sum() < 500:
        return {"status": "not_available", "reason": "insufficient skin pixels detected"}
    frac, region_ok, qual = _darkspot_spatial(crop_bgr, skin_mask, landmarks=landmarks)
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
