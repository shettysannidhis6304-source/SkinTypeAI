"""
backend/app/services/analysis_service.py

ACTIVE inference path -- the 9-characteristic modular system:

    image -> face crop + skin mask + ONE MediaPipe FaceMesh pass (prepare_face)
          -> SkinAnalysisModel  (from-scratch 403,656-param CNN, models/skin_analysis_model.py)
                 acne | wrinkles | oiliness | dryness | pigmentation
          -> skin_type            : DERIVED deterministically from oiliness + dryness
          -> OpenCVSkinAnalyzer  (models/skin_analysis/opencv_engine.py, deterministic OpenCV)
                 redness | dark_spots | enlarged_pores | blackheads   (all "estimated")

Nothing pretrained. SkinAnalysisModel is trained 100% from scratch (no timm, no
ImageNet). The OpenCV engine is not machine learning. Each engine call is
isolated: if one fails the endpoint still returns the rest, with that condition
marked "error". No fabricated detections or scores anywhere.

Image quality is ADVISORY only (`assess_image_quality`): a 1-10 rating is always
computed and embedded in the result; nothing is ever hard-rejected. The router
decides whether to surface an advisory dialog or proceed.

Unified per-condition shape (schemas/prediction.py::ConditionEntry):
    {status, method, label, score, confidence, detail, model, note}
"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from app.core.config import resolve_model_weights, settings

# models/ lives at the project root (sibling of backend/). Ensure it's importable
# regardless of the cwd uvicorn was started from.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

MODEL_VERSIONS = {
    "skin_analysis_model": "SkinTypeAI SkinAnalysisModel -- from-scratch 4-conv CNN + 2 SE blocks, "
                           "403,656 params, Kaiming init, NO pretrained/timm/ImageNet. Heads: acne "
                           "(real ACNE04 grades) + wrinkles/oiliness/dryness/pigmentation (OpenCV weak labels).",
    "derived_skin_type": "skin_type is derived deterministically from the oiliness + dryness heads "
                         "(not a learned head).",
    "opencv_engine": "SkinTypeAI OpenCVSkinAnalyzer -- deterministic OpenCV for redness / dark_spots / "
                     "enlarged_pores / blackheads. Not machine learning; not a clinical assessment.",
}
ANALYSIS_VERSION = "skinintel-scratch-v2"

# --- SkinAnalysisModel label maps -----------------------------------------------
# acne head classes are ACNE04 Hayashi grades "0".."3".
_ACNE_HUMAN = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
# tier -> a 0..1 severity score (soft, via the head's class distribution)
_TIER_CENTER = {"minimal": 0.15, "moderate": 0.5, "prominent": 0.85,
                "low": 0.15, "balanced": 0.5, "high": 0.85}
_PIGMENT_LABEL = {
    "normal": "no significant uneven pigmentation",
    "mild": "mild uneven pigmentation (estimate)",
    "marked": "marked uneven pigmentation (estimate)",
}
# (oiliness_tier, dryness_tier) -> skin type
_SKIN_TYPE_RULE = {
    ("low", "low"): "normal", ("low", "balanced"): "normal", ("low", "high"): "dry",
    ("balanced", "low"): "normal", ("balanced", "balanced"): "normal", ("balanced", "high"): "dry",
    ("high", "low"): "oily", ("high", "balanced"): "oily", ("high", "high"): "combination",
}

_MODEL_CONDITIONS = ("acne", "wrinkles", "oiliness", "dryness", "skin_type", "pigmentation")
_CV_CONDITIONS = ("redness", "enlarged_pores", "blackheads", "dark_spots")

_CANONICAL_KEYS = ("status", "method", "label", "score", "confidence", "detail", "model", "note")


# --------------------------------------------------------------------------- #
# Advisory image-quality rating (1-10). Never rejects -- see module docstring.
# --------------------------------------------------------------------------- #
# score >= lo  ->  (short label, advisory guidance)
_QUALITY_BANDS: list[tuple[int, str, str]] = [
    (8, "Excellent Quality",
     "Sharp capture with clear skin micro-textures. Enlarged pores, fine wrinkles "
     "and blackheads can be assessed at full precision."),
    (6, "Good Quality",
     "Acceptable focus with, at most, minor lighting imbalance. Analysis is reliable; "
     "only the very finest micro-textures may be slightly soft."),
    (4, "Fair / Webcam Softness",
     "Typical built-in laptop webcam. Focus is soft, so fine pores, subtle wrinkle "
     "lines and small blackheads may be estimated with lower precision. The trained "
     "characteristics (acne, skin type, oiliness, dryness, pigmentation) are much less affected."),
    (1, "Suboptimal / Heavy Blur",
     "Severe motion blur or very low light. The micro-texture characteristics (pores, "
     "fine wrinkles, blackheads) may be unreliable -- retaking in brighter, steadier "
     "conditions is recommended."),
]


def _quality_band(score: int) -> tuple[str, str]:
    for lo, label, advisory in _QUALITY_BANDS:
        if score >= lo:
            return label, advisory
    return _QUALITY_BANDS[-1][1], _QUALITY_BANDS[-1][2]


def _lighting_label(mean: float) -> str:
    if mean < 50:
        return "Very dark"
    if mean < 80:
        return "Dim"
    if mean <= 180:
        return "Good"
    if mean <= 210:
        return "Bright"
    return "Overexposed"


def prepare_face(image_path: str) -> dict[str, Any]:
    """One-time per-request face prep so MediaPipe FaceMesh runs exactly ONCE.

      * heuristics.face_region()          -> crop + skin mask (FaceDetection, not FaceMesh)
      * heuristics.facemesh_landmarks()   -> the single 468-pt FaceMesh pass

    Thread the result into assess_image_quality(prep=...) (pose sub-score) and
    run_analysis(prep=...) -> OpenCVSkinAnalyzer (pore / dark-spot region gating),
    so nothing re-runs the mesh.
    """
    from models.skin_analysis import heuristics

    face = None
    try:
        face = heuristics.face_region(image_path)
    except Exception:  # pragma: no cover
        logger.exception("face_region failed")
    landmarks = None
    if face is not None and face.get("crop_bgr") is not None:
        try:
            landmarks = heuristics.facemesh_landmarks(face["crop_bgr"])
        except Exception:  # pragma: no cover
            logger.exception("facemesh_landmarks failed")
    return {"face": face, "landmarks": landmarks}


def assess_image_quality(image: "str | Path | Any", *, prep: dict[str, Any] | None = None) -> dict[str, Any]:
    """Advisory 1-10 image-quality rating. NEVER rejects an image -- the caller
    decides whether to proceed or show an advisory dialog.

    `prep` -- the dict from prepare_face(); when given, the pose sub-score reuses
    its memoised FaceMesh landmarks instead of running the mesh again.

    Combines three sub-scores into an integer Q in [1, 10]:
      * Sharpness    -- scaled log-Laplacian variance:
                        min(10, max(1, round(log10(var + 1) * 3.2)))
      * Illumination -- grayscale mean; 80-180 is optimal, <50 or >210 penalised.
      * Pose         -- MediaPipe FaceMesh + solvePnP; |pitch|,|yaw| within +/-15 deg ideal.

    Returns {score, max_score, rating, label, message, is_suboptimal, advisory,
             metrics{sharpness_score, laplacian_variance, brightness_mean, lighting,
                     illumination_score, pose, pose_score, pose_deviation_deg,
                     face_pose_confident}}.  is_suboptimal is True when Q <= 6.
    """
    import cv2
    import numpy as np

    from models.skin_analysis import heuristics

    bgr = cv2.imread(str(image)) if isinstance(image, (str, Path)) else np.asarray(image)
    if bgr is None or getattr(bgr, "size", 0) == 0:
        label, advisory = _quality_band(1)
        return {"score": 1, "max_score": 10, "rating": f"1/10 - {label}", "label": label,
                "message": "The image could not be read for a quality assessment.",
                "is_suboptimal": True, "advisory": advisory, "metrics": {}}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # --- sharpness ---
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = int(min(10, max(1, round(math.log10(lap_var + 1.0) * 3.2))))

    # --- illumination ---
    brightness = float(gray.mean())
    if 80.0 <= brightness <= 180.0:
        illum = 10
    elif brightness < 50.0 or brightness > 210.0:
        illum = 2
    elif brightness < 80.0:                       # 50..80 -> ramp 2..10
        illum = int(round(2 + 8 * (brightness - 50.0) / 30.0))
    else:                                          # 180..210 -> ramp 10..2
        illum = int(round(2 + 8 * (210.0 - brightness) / 30.0))
    illum = min(10, max(1, illum))

    # --- pose (reuse the single memoised FaceMesh pass when prep is supplied) ---
    if prep is not None and prep.get("face") is not None and prep["face"].get("crop_bgr") is not None:
        pose = heuristics._head_pose_deg(prep["face"]["crop_bgr"], landmarks=prep.get("landmarks"))
    else:
        pose = heuristics._head_pose_deg(bgr)
    if pose is None:
        pose_score, pose_label, pose_dev, pose_conf = 6, "Unknown (no face landmarks)", None, False
    else:
        pose_dev = round(max(abs(pose["pitch"]), abs(pose["yaw"])), 1)
        pose_conf = bool(pose["reproj_norm"] <= 0.12)
        if not pose_conf:
            pose_score, pose_label = 6, "Uncertain"
        elif pose_dev <= 15.0:
            pose_score, pose_label = 10, "Frontal"
        elif pose_dev <= 30.0:
            pose_score, pose_label = 6, "Slightly turned"
        else:
            pose_score, pose_label = 3, "Off-angle"

    q = min(10, max(1, int(round(0.60 * sharp + 0.25 * illum + 0.15 * pose_score))))
    label, advisory = _quality_band(q)
    message = (
        f"Image quality is {q}/10. "
        + ("Laptop webcams produce soft focus that can reduce the precision of pore, "
           "blackhead, and fine wrinkle analysis. " if q <= 5 else "")
        + "Camera clarity directly influences the micro-texture characteristics "
          "(enlarged pores, fine wrinkles, blackheads)."
    )
    return {
        "score": q,
        "max_score": 10,
        "rating": f"{q}/10 - {label}",
        "label": label,
        "message": message,
        "is_suboptimal": bool(q <= 6),
        "advisory": advisory,
        "metrics": {
            "sharpness_score": sharp,
            "laplacian_variance": round(lap_var, 1),
            "brightness_mean": round(brightness, 1),
            "lighting": _lighting_label(brightness),
            "illumination_score": illum,
            "pose": pose_label,
            "pose_score": pose_score,
            "pose_deviation_deg": pose_dev,
            "face_pose_confident": pose_conf,
        },
    }


def _err(model: str, exc: Exception) -> dict[str, Any]:
    logger.exception("analysis engine %s failed", model)
    return {"status": "error", "method": None, "label": None, "score": None,
            "confidence": None, "detail": {"error": repr(exc)}, "model": model, "note": None}


def _na(model: str | None, note: str) -> dict[str, Any]:
    return {"status": "not_available", "method": None, "label": None, "score": None,
            "confidence": None, "detail": {}, "model": model, "note": note}


def _normalize(block: dict[str, Any]) -> dict[str, Any]:
    """Guarantee every condition entry has exactly the canonical keys."""
    out = {k: block.get(k) for k in _CANONICAL_KEYS}
    if out["detail"] is None:
        out["detail"] = {}
    return out


def _tier_score(distribution: dict[str, float]) -> float:
    """Soft 0..1 score = sum_k P(tier_k) * centre_k."""
    return round(sum(float(p) * _TIER_CENTER.get(k, 0.5) for k, p in distribution.items()), 3)


# ---------------------------------------------------------------------------
# SkinAnalysisModel -- lazy singleton, loaded offline (nothing pretrained to fetch)
# ---------------------------------------------------------------------------
_sam = None                # (model, checkpoint_dict) once loaded
_sam_load_failed = False


def _get_skin_analysis_model():
    global _sam, _sam_load_failed
    if _sam is not None:
        return _sam
    if _sam_load_failed:
        return None
    ckpt_path = resolve_model_weights()
    if not ckpt_path.exists():
        logger.warning("SkinAnalysisModel checkpoint missing: %s", ckpt_path)
        _sam_load_failed = True
        return None
    try:
        import torch
        from models.skin_analysis_model import load_skin_analysis_model

        torch.set_num_threads(settings.ml_torch_threads)
        model, ckpt = load_skin_analysis_model(str(ckpt_path), device=settings.ml_device)
        _sam = (model, ckpt)
        logger.info("SkinAnalysisModel loaded from %s (trained_at=%s, params=%s)",
                    ckpt_path, ckpt.get("trained_at"),
                    (ckpt.get("metrics") or {}).get("model", {}).get("params"))
        return _sam
    except Exception:
        logger.exception("failed to load SkinAnalysisModel")
        _sam_load_failed = True
        return None


def _predict_skin_analysis(face: dict | None, image_path: str) -> dict[str, dict] | None:
    """Run the 5-head model once on the face crop. Returns the per-head dict from
    models.skin_analysis_model.predict(), or None if the model is unavailable."""
    loaded = _get_skin_analysis_model()
    if loaded is None:
        return None
    model, ckpt = loaded
    try:
        from models.skin_analysis_model import predict as _predict

        crop = face["crop_bgr"] if face and face.get("crop_bgr") is not None else image_path
        return _predict(model, crop, ckpt.get("config"))
    except Exception:
        logger.exception("SkinAnalysisModel inference failed for this image")
        return None


# ---------------------------------------------------------------------------
# Skin Health Index -- deterministic, documented, never random
# ---------------------------------------------------------------------------
SHI_FORMULA = (
    "SHI = 100 - 30*(acne_severity_index/4) - 15*wrinkle_score "
    "- 12*clamp((dryness_score-0.34)/0.66, 0, 1) - 10*redness_score - 8*pore_score, "
    "clamped to [0,100]. acne_severity_index rescales the 0..3 model grade to 0..4. "
    "Redness/pores are down-weighted because they are computer-vision estimates, not "
    "trained models. Pigmentation, blackheads and dark spots are reported separately "
    "but do not affect this score. If no model-backed condition succeeded, SHI is null."
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Max points each condition can subtract from 100. Single source of truth for
# compute_skin_health_index() AND xai_explainer's driver breakdown.
SHI_PENALTY_SPEC: dict[str, dict[str, Any]] = {
    "acne":           {"max_points": 30.0, "kind": "acne_index", "model_backed": True},
    "wrinkles":       {"max_points": 15.0, "kind": "score", "model_backed": True},
    "dryness":        {"max_points": 12.0, "kind": "dryness_clamp", "model_backed": True},
    "redness":        {"max_points": 10.0, "kind": "estimated_score", "model_backed": False},
    "enlarged_pores": {"max_points": 8.0, "kind": "estimated_score", "model_backed": False},
}


def shi_penalty_breakdown(conditions: dict[str, dict]) -> list[dict[str, Any]]:
    """Per-condition points subtracted from 100 by compute_skin_health_index().
    Returns [{condition, points, max_points, weight_fraction, driver_value, basis,
    model_backed, contributed}], sorted by points descending."""
    out: list[dict[str, Any]] = []
    total_max = sum(s["max_points"] for s in SHI_PENALTY_SPEC.values())
    for cond, spec in SHI_PENALTY_SPEC.items():
        e = conditions.get(cond, {}) or {}
        status = e.get("status")
        score = e.get("score")
        pts = 0.0
        drv: float | None = None

        if spec["kind"] == "acne_index" and status == "detected":
            idx = float((e.get("detail") or {}).get("severity_index") or 0.0)   # 0..4 scale
            drv = _clamp(idx / 4.0, 0.0, 1.0)
            pts = spec["max_points"] * drv
        elif spec["kind"] == "score" and status == "detected" and isinstance(score, (int, float)):
            drv = float(score)
            pts = spec["max_points"] * drv
        elif spec["kind"] == "dryness_clamp" and status == "detected" and isinstance(score, (int, float)):
            drv = _clamp((float(score) - 0.34) / 0.66, 0.0, 1.0)
            pts = spec["max_points"] * drv
        elif spec["kind"] == "estimated_score" and status == "estimated" and isinstance(score, (int, float)):
            drv = float(score)
            pts = spec["max_points"] * drv

        out.append({
            "condition": cond,
            "points": round(pts, 2),
            "max_points": spec["max_points"],
            "weight_fraction": round(spec["max_points"] / total_max, 3),
            "driver_value": None if drv is None else round(drv, 3),
            "basis": f"{cond} (estimate)" if not spec["model_backed"] else cond,
            "model_backed": spec["model_backed"],
            "contributed": pts > 0.0,
        })
    out.sort(key=lambda d: d["points"], reverse=True)
    return out


def compute_skin_health_index(conditions: dict[str, dict]) -> dict[str, Any]:
    breakdown = shi_penalty_breakdown(conditions)
    penalty = sum(d["points"] for d in breakdown)
    basis = [d["basis"] for d in breakdown if d["contributed"]]
    model_backed_ok = any(d["contributed"] and d["model_backed"] for d in breakdown)

    if not model_backed_ok:
        return {"value": None, "band": "not_enough_data", "basis": basis, "formula": SHI_FORMULA}

    value = int(round(_clamp(100.0 - penalty, 0.0, 100.0)))
    band = ("excellent" if value >= 80 else "good" if value >= 65
            else "fair" if value >= 50 else "needs_attention")
    return {"value": value, "band": band, "basis": basis, "formula": SHI_FORMULA}


# ---------------------------------------------------------------------------
# SkinAnalysisModel head -> condition entry
# ---------------------------------------------------------------------------
def _acne_block(pred: dict[str, dict]) -> dict[str, Any]:
    ac = pred["acne"]                                  # label "0".."3", index 0..3
    grade = int(ac["index"])
    sev_index = round(grade * 4.0 / 3.0, 2)            # rescale 0..3 -> 0..4 for the SHI + frontend
    human = _ACNE_HUMAN.get(grade, ac["label"])
    label = (f"clear / minimal - no acne-like severity"
             if grade == 0 else f"severity estimate: {human} (whole-face, grade {grade}/3)")
    return {
        "status": "detected", "method": "classification",
        "label": label,
        "score": round(sev_index / 4.0, 3),
        "confidence": ac["confidence"],
        "detail": {
            "severity_grade": grade,                   # raw 0..3 model output (authoritative)
            "severity_index": sev_index,               # 0..4 rescale (SHI / frontend compat)
            "severity_level": ac["label"],
            "severity_human": human,
            "severity_distribution": ac["distribution"],
            "lesion_count": 0,                          # no lesion detector on this path
            "detections": [],
            "top_box_confidence": None,
        },
        "model": "skin_analysis_model",
        "note": "Whole-face Hayashi-grade estimate from the from-scratch model (real ACNE04 grade "
                "labels, 0..3). No lesion bounding boxes on this path.",
    }


def _wrinkle_block(pred: dict[str, dict]) -> dict[str, Any]:
    w = pred["wrinkles"]                               # minimal / moderate / prominent
    score = _tier_score(w["distribution"])
    band = "low" if score < 0.34 else ("moderate" if score < 0.67 else "elevated")
    return {
        "status": "detected", "method": "classification",
        "label": f"{band} wrinkle severity",
        "score": score, "confidence": w["confidence"],
        "detail": {"tier": w["label"], "distribution": w["distribution"],
                   "wrinkle_area_fraction": None, "max_prob": None, "backend": "skin_analysis_model",
                   "source": "SkinAnalysisModel wrinkles head (Stage-1 edge/line-frequency weak labels)"},
        "model": "skin_analysis_model",
        "note": "Whole-face wrinkle tier from the from-scratch model. Trained on OpenCV edge/line "
                "weak labels (not the FFHQ-Wrinkle U-Net) - indicative only.",
    }


def _oiliness_dryness_blocks(pred: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for head in ("oiliness", "dryness"):
        h = pred[head]                                 # low / balanced / high
        score = _tier_score(h["distribution"])
        lead = ("oily-leaning" if head == "oiliness" and score >= 0.34
                else "dry-leaning" if head == "dryness" and score >= 0.34
                else f"{h['label']}")
        signal = "specular-highlight / luminance" if head == "oiliness" else "texture-roughness"
        out[head] = {
            "status": "detected", "method": "classification",
            "label": f"{lead} ({h['label']})", "score": score, "confidence": h["confidence"],
            "detail": {"tier": h["label"], "distribution": h["distribution"],
                       "source": f"SkinAnalysisModel {head} head (Stage-1 {signal} weak labels)"},
            "model": "skin_analysis_model",
            "note": f"Whole-face {head} tier from the from-scratch model - not a localised map.",
        }
    return out


def _skin_type_block(pred: dict[str, dict]) -> dict[str, Any]:
    """DERIVED, not a learned head: a fixed rule over the predicted oiliness +
    dryness tiers, plus a soft distribution over {dry, normal, oily, combination}."""
    oil, dry = pred["oiliness"], pred["dryness"]
    o_tier, d_tier = oil["label"], dry["label"]
    label = _SKIN_TYPE_RULE.get((o_tier, d_tier), "normal")

    p_oh = float(oil["distribution"].get("high", 0.0))
    p_dh = float(dry["distribution"].get("high", 0.0))
    p_oily = p_oh * (1.0 - p_dh)
    p_dry = p_dh * (1.0 - p_oh)
    p_combo = p_oh * p_dh
    p_normal = max(0.0, 1.0 - p_oily - p_dry - p_combo)
    tot = (p_oily + p_dry + p_combo + p_normal) or 1.0
    dist = {"dry": round(p_dry / tot, 4), "normal": round(p_normal / tot, 4),
            "oily": round(p_oily / tot, 4), "combination": round(p_combo / tot, 4)}

    return {
        "status": "detected", "method": "derived",
        "label": label, "score": None,
        "confidence": round(min(float(oil["confidence"]), float(dry["confidence"])), 4),
        "detail": {"distribution": dist, "oiliness_tier": o_tier, "dryness_tier": d_tier,
                   "rule": f"oiliness={o_tier} x dryness={d_tier} -> {label}",
                   "source": "deterministic rule over the SkinAnalysisModel oiliness + dryness heads"},
        "model": "derived(oiliness,dryness)",
        "note": "Skin type is not a learned head - it is derived deterministically from the "
                "predicted oiliness and dryness tiers.",
    }


def _pigmentation_block(pred: dict[str, dict]) -> dict[str, Any]:
    pg = pred["pigmentation"]                          # normal / mild / marked
    dist = pg["distribution"]
    return {
        "status": "detected", "method": "classification",
        "label": _PIGMENT_LABEL.get(pg["label"], pg["label"]),
        "score": round(1.0 - float(dist.get("normal", 0.0)), 3),
        "confidence": pg["confidence"],
        "detail": {"distribution": dist, "tier": pg["label"],
                   "source": "SkinAnalysisModel pigmentation head"},
        "model": "skin_analysis_model",
        "note": "Trained on Stage-1 OpenCV weak labels (CIELAB L* unevenness + dark-spot area "
                "fraction), not a clinically validated pigmentation grade. Indicative only.",
    }


def _model_blocks(pred: dict[str, dict]) -> dict[str, dict]:
    """All 6 model-backed condition entries (5 heads + derived skin_type)."""
    blocks: dict[str, dict] = {}
    blocks["acne"] = _acne_block(pred)
    blocks["wrinkles"] = _wrinkle_block(pred)
    blocks.update(_oiliness_dryness_blocks(pred))
    blocks["skin_type"] = _skin_type_block(pred)
    blocks["pigmentation"] = _pigmentation_block(pred)
    return blocks


# ---------------------------------------------------------------------------
def run_analysis(image_path: str, *, quality: dict[str, Any] | None = None,
                 prep: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full 9-characteristic modular stack on one uploaded image.

    `prep` is the dict from prepare_face() -- the face crop + the SINGLE
    per-request MediaPipe FaceMesh pass. Its landmarks are threaded into the
    OpenCV pore / dark-spot region gating so the mesh runs at most once.
    `quality` is the advisory 1-10 rating from assess_image_quality(). Pass
    pre-computed ones (the router already has them) or let this compute them.
    Quality is embedded in the result under `quality`; nothing is hard-rejected."""
    from models.skin_analysis.opencv_engine import OpenCVSkinAnalyzer

    conditions: dict[str, Any] = {}
    used_models: list[str] = []
    timings: dict[str, float] = {}
    _t_pipeline = time.perf_counter()

    # ---- face crop + skin mask + the single FaceMesh pass (shared everywhere) ----
    _t = time.perf_counter()
    if prep is None:
        prep = prepare_face(image_path)
    face = prep.get("face")
    landmarks = prep.get("landmarks")
    timings["crop_ms"] = round((time.perf_counter() - _t) * 1000, 1)

    # ---- advisory image-quality rating (never rejects; reuses the FaceMesh pass) ----
    _t = time.perf_counter()
    if quality is None:
        try:
            quality = assess_image_quality(image_path, prep=prep)
        except Exception:  # pragma: no cover - quality must never 500 the request
            logger.exception("assess_image_quality failed; continuing without a rating")
            quality = None
    timings["quality_check_ms"] = round((time.perf_counter() - _t) * 1000, 1)

    # ---- SkinAnalysisModel: acne / wrinkles / oiliness / dryness / pigmentation ----
    _t = time.perf_counter()
    pred = _predict_skin_analysis(face, image_path)
    timings["cnn_forward_ms"] = round((time.perf_counter() - _t) * 1000, 1)
    if pred is not None:
        try:
            conditions.update(_model_blocks(pred))
            used_models.append("skin_analysis_model")
            used_models.append("derived_skin_type")
        except Exception as e:
            for k in _MODEL_CONDITIONS:
                conditions[k] = _err("skin_analysis_model", e)
    else:
        for k in _MODEL_CONDITIONS:
            conditions[k] = _na("skin_analysis_model",
                                "SkinAnalysisModel checkpoint unavailable "
                                f"({resolve_model_weights()}). Train it with "
                                "training/train_skin_analysis.py.")

    # ---- OpenCV engine: redness / dark_spots / enlarged_pores / blackheads ----
    _t = time.perf_counter()
    if face is not None:
        _oil = (conditions.get("oiliness") or {}).get("score")
        _oil = _oil if isinstance(_oil, (int, float)) else None
        _pig = (conditions.get("pigmentation") or {}).get("detail", {}).get("distribution")
        _pig = _pig if isinstance(_pig, dict) else None
        try:
            cv = OpenCVSkinAnalyzer().analyze(image_path, face=face, oiliness_score=_oil,
                                              pigment_dist=_pig, landmarks=landmarks)
            conditions.update(cv)
            used_models.append("opencv_engine")
        except Exception as e:
            for k in _CV_CONDITIONS:
                conditions[k] = _err("opencv_engine", e)
    else:
        for k in _CV_CONDITIONS:
            conditions[k] = _na("opencv_engine", "Skin region unavailable - OpenCV analysis skipped.")
    timings["opencv_heuristics_ms"] = round((time.perf_counter() - _t) * 1000, 1)

    # ---- defensive: guarantee all 10 condition keys exist ----
    for k in (*_MODEL_CONDITIONS, *_CV_CONDITIONS):
        conditions.setdefault(k, _na(None, "Not produced by any engine."))

    # image size
    from PIL import Image
    with Image.open(image_path) as im:
        image_size = [im.width, im.height]

    normalized = {k: _normalize(v) for k, v in conditions.items()}
    shi = compute_skin_health_index(normalized)

    # ---- Explainable-AI layer: plain-language rationale + SHI driver breakdown ----
    try:
        from app.services.xai_explainer import generate_diagnostic_explanations

        xai_insights = generate_diagnostic_explanations(normalized, shi)
    except Exception:  # pragma: no cover - XAI must never break the analysis
        logger.exception("xai_explainer failed; returning analysis without xai_insights")
        xai_insights = None

    model_versions = {}
    if "skin_analysis_model" in used_models:
        model_versions["skin_analysis_model"] = MODEL_VERSIONS["skin_analysis_model"]
        model_versions["derived_skin_type"] = MODEL_VERSIONS["derived_skin_type"]
    if "opencv_engine" in used_models:
        model_versions["opencv_engine"] = MODEL_VERSIONS["opencv_engine"]

    # ---- latency logging (spec: observability) ----
    timings["total_pipeline_ms"] = round((time.perf_counter() - _t_pipeline) * 1000, 1)
    logger.info(
        "pipeline timings ms | crop+facemesh=%.1f quality_check=%.1f cnn_forward=%.1f "
        "opencv_heuristics=%.1f total=%.1f  (facemesh landmarks: %s)",
        timings.get("crop_ms", 0.0), timings.get("quality_check_ms", 0.0),
        timings.get("cnn_forward_ms", 0.0), timings.get("opencv_heuristics_ms", 0.0),
        timings["total_pipeline_ms"],
        "1 pass, face" if landmarks else ("1 pass, no face" if landmarks is not None else "skipped"),
    )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "face_detected": bool(face and face.get("face_detected")),
        "image_size": image_size,
        "conditions": normalized,
        "skin_health_index": shi,
        "xai_insights": xai_insights,
        "model_versions": model_versions,
        "quality": quality,
    }
