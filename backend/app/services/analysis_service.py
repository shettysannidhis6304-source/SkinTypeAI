"""
backend/app/services/analysis_service.py

ACTIVE inference path: multi-condition facial-skin analysis from a stack of
PUBLIC pretrained models (nothing trained by this project).

    image -> face crop + skin mask (mediapipe/opencv)
          -> skin-type ViT        (dima806)          -> skin_type, dryness, oiliness
          -> acne-severity ViT     (skintelligent)    -> acne severity level
          -> acne detector YOLOv8s (glowlytics)       -> acne boxes + count
          -> wrinkle U-Net         (FFHQ-Wrinkle)     -> wrinkle area fraction   [CC BY-NC-SA]
          -> redness / pore ESTIMATES (classical CV)  -> labelled "estimated"
    pigmentation / blackheads / dark_spots           -> "not_available" (no trustworthy model)

Every model is a lazily-loaded module-level singleton. Each model call is
isolated: if one fails the endpoint still returns the rest, with that
condition marked "error". No fabricated detections or scores anywhere.

Unified per-condition shape:
    {
      "status": "detected" | "estimated" | "not_available" | "error",
      "method": "classification" | "detection" | "segmentation" | "heuristic" | null,
      "label":  str | null,          # human-readable finding
      "score":  float | null,        # 0-1 severity / presence
      "confidence": float | null,    # model confidence where meaningful
      "detail": {...},               # method-specific extras
      "model":  str | null,          # registry key
      "note":   str | null,
    }
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings

# models/ lives at the project root (sibling of backend/). Ensure it's importable
# regardless of the cwd uvicorn was started from.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

MODEL_VERSIONS = {
    "skin_type": "dima806/skin_types_image_detection (ViT, Apache-2.0)",
    "acne_severity": "imfarzanansari/skintelligent-acne (ViT, MIT)",
    "acne_detector": "mufasabrownie/glowlytics acne_detector (YOLOv8s, MIT)",
    "wrinkles": "rmsandu/ffhq-wrinkle-unet (U-Net, CC BY-NC-SA - academic only)",
    "cv_analysis_engine": "SkinTypeAI CV analysis engine (this project) - deterministic "
                          "colour/texture measurements for redness, pores, blackheads, dark spots. "
                          "Not machine learning; not a clinical assessment.",
    "custom_skin_model": "SkinTypeAI custom multi-task (mobilenetv3_small_100 frozen "
                         "trunk + 3 trained heads; trained by this project, no runtime HF)",
}
ANALYSIS_VERSION = "skinintel-multimodel-v1"

# --- custom acne head -> the 0..4 severity_index the SHI formula + recommendation
#     engine already expect. "clear" (FFHQ clean faces) is grade 0.
_CUSTOM_ACNE_INDEX = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3, "very_severe": 4}
_CUSTOM_ACNE_HUMAN = {"clear": "clear", "mild": "mild", "moderate": "moderate",
                      "severe": "severe", "very_severe": "very severe"}
_CUSTOM_PIGMENT_LABEL = {
    "normal": "no significant uneven pigmentation",
    "mild": "mild uneven pigmentation (estimate)",
    "marked": "marked uneven pigmentation (estimate)",
}

# Defensive fallback only. pigmentation comes from the custom model; blackheads
# and dark_spots come from the CV analysis engine. An entry lands here only if
# the custom model is unavailable (pigmentation) or the face crop failed.
_NOT_AVAILABLE = {
    "pigmentation": "Pigmentation is produced by the custom model; it is unavailable because the "
                    "checkpoint (weights/custom_skin_model.pt) is missing.",
    "blackheads": "Blackhead analysis unavailable - the face region could not be read.",
    "dark_spots": "Dark-spot analysis unavailable - the face region could not be read.",
}


_CANONICAL_KEYS = ("status", "method", "label", "score", "confidence", "detail", "model", "note")


def _err(model: str, exc: Exception) -> dict[str, Any]:
    logger.exception("analysis model %s failed", model)
    return {"status": "error", "method": None, "label": None, "score": None,
            "confidence": None, "detail": {"error": repr(exc)}, "model": model, "note": None}


def _normalize(block: dict[str, Any]) -> dict[str, Any]:
    """Guarantee every condition entry has exactly the canonical keys."""
    out = {k: block.get(k) for k in _CANONICAL_KEYS}
    if out["detail"] is None:
        out["detail"] = {}
    return out


# ---------------------------------------------------------------------------
# Custom multi-task model (trained by this project) -- optional, replaces the
# HF ViT calls for skin_type / dryness / oiliness / acne-severity / pigmentation
# when settings.use_custom_model is on and the checkpoint exists. The HF path is
# the automatic fallback. NO Hugging Face Hub access at load time.
# ---------------------------------------------------------------------------
_custom_model = None          # (model, checkpoint_dict) once loaded
_custom_load_failed = False   # don't retry a broken checkpoint every request


def _custom_enabled() -> bool:
    return bool(getattr(settings, "use_custom_model", False)) and not _custom_load_failed and \
        Path(settings.custom_model_weights_path).exists()


def _get_custom_model():
    """Lazy singleton. Returns (model, ckpt) or None if unavailable/broken."""
    global _custom_model, _custom_load_failed
    if _custom_model is not None:
        return _custom_model
    if not _custom_enabled():
        return None
    try:
        import torch
        from models.custom_multi_task import load_custom_model

        torch.set_num_threads(settings.ml_torch_threads)
        model, ckpt = load_custom_model(settings.custom_model_weights_path, device=settings.ml_device)
        _custom_model = (model, ckpt)
        logger.info("custom multi-task model loaded from %s (trained_at=%s)",
                    settings.custom_model_weights_path, ckpt.get("trained_at"))
        return _custom_model
    except Exception:
        logger.exception("failed to load custom model -- falling back to the HF path")
        _custom_load_failed = True
        return None


def _predict_custom(face: dict | None, image_path: str) -> dict[str, dict] | None:
    """Run the custom model once on the face crop. Returns the per-head dict from
    models.custom_multi_task.predict(), or None on any failure (-> HF fallback)."""
    loaded = _get_custom_model()
    if loaded is None:
        return None
    model, ckpt = loaded
    try:
        from models.custom_multi_task import predict as _predict

        crop = face["crop_bgr"] if face and face.get("crop_bgr") is not None else image_path
        return _predict(model, crop, ckpt.get("config"))
    except Exception:
        logger.exception("custom model inference failed -- falling back to the HF path for this image")
        return None


# ---------------------------------------------------------------------------
# Skin Health Index -- deterministic, documented, never random
# ---------------------------------------------------------------------------
SHI_FORMULA = (
    "SHI = 100 - 30*(acne_severity_index/4) - 15*wrinkle_score "
    "- 12*clamp((dryness_score-0.34)/0.66, 0, 1) - 10*redness_score - 8*pore_score, "
    "clamped to [0,100]. Redness/pores are down-weighted because they are computer-vision "
    "estimates, not trained models. Pigmentation, blackheads and dark spots are reported "
    "separately but do not affect this score. If no model-backed condition succeeded, SHI is null."
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_skin_health_index(conditions: dict[str, dict]) -> dict[str, Any]:
    penalty = 0.0
    basis: list[str] = []
    model_backed_ok = False

    ac = conditions.get("acne", {})
    if ac.get("status") == "detected":
        idx = int((ac.get("detail") or {}).get("severity_index") or 0)  # 0..4
        penalty += 30.0 * (idx / 4.0)
        basis.append("acne (severity)")
        model_backed_ok = True

    wr = conditions.get("wrinkles", {})
    if wr.get("status") == "detected" and isinstance(wr.get("score"), (int, float)):
        penalty += 15.0 * float(wr["score"])
        basis.append("wrinkles")
        model_backed_ok = True

    dr = conditions.get("dryness", {})
    if dr.get("status") == "detected" and isinstance(dr.get("score"), (int, float)):
        penalty += 12.0 * _clamp((float(dr["score"]) - 0.34) / 0.66, 0.0, 1.0)
        basis.append("dryness")
        model_backed_ok = True

    for cond, weight in (("redness", 10.0), ("enlarged_pores", 8.0)):
        e = conditions.get(cond, {})
        if e.get("status") == "estimated" and isinstance(e.get("score"), (int, float)):
            penalty += weight * float(e["score"])
            basis.append(f"{cond} (estimate)")

    if not model_backed_ok:
        return {"value": None, "band": "not_enough_data", "basis": basis, "formula": SHI_FORMULA}

    value = int(round(_clamp(100.0 - penalty, 0.0, 100.0)))
    band = ("excellent" if value >= 80 else "good" if value >= 65
            else "fair" if value >= 50 else "needs_attention")
    return {"value": value, "band": band, "basis": basis, "formula": SHI_FORMULA}


def _skin_type_block_custom(custom: dict[str, dict]) -> dict[str, dict]:
    """Same output shape as _skin_type_block(), sourced from our trained model.
    'combination' probability is split half to dryness, half to oiliness so the
    downstream dryness/oiliness scores stay comparable to the HF path."""
    st = custom["skin_type"]
    dist = st["distribution"]                       # dry / normal / oily / combination
    combo = dist.get("combination", 0.0)
    dry = dist.get("dry", 0.0) + 0.5 * combo
    oily = dist.get("oily", 0.0) + 0.5 * combo
    return {
        "skin_type": {
            "status": "detected", "method": "classification",
            "label": st["label"], "score": None, "confidence": st["confidence"],
            "detail": {"distribution": dist}, "model": "custom_skin_model",
            "note": "Whole-face classification from the project's own trained multi-task model "
                    "(frozen ImageNet trunk + trained head). Skin-type labels were distilled from a "
                    "teacher and are domain-biased; treat as indicative.",
        },
        "dryness": {
            "status": "detected", "method": "classification",
            "label": "dry-leaning" if dry >= 0.34 else "not dry-leaning",
            "score": round(dry, 3), "confidence": round(dry, 3),
            "detail": {"source": "'dry' (+ half of 'combination') probability from the custom skin-type head"},
            "model": "custom_skin_model",
            "note": "Derived from the custom model's skin-type head - not a localised dryness map.",
        },
        "oiliness": {
            "status": "detected", "method": "classification",
            "label": "oily-leaning" if oily >= 0.34 else "not oily-leaning",
            "score": round(oily, 3), "confidence": round(oily, 3),
            "detail": {"source": "'oily' (+ half of 'combination') probability from the custom skin-type head"},
            "model": "custom_skin_model",
            "note": "Derived from the custom model's skin-type head.",
        },
    }


def _pigmentation_block_custom(custom: dict[str, dict]) -> dict[str, Any]:
    """New: pigmentation from the custom model's third head. Same 8-key
    ConditionEntry shape as every other condition. compute_skin_health_index()
    does not read pigmentation, so the SHI is unaffected."""
    pg = custom["pigmentation"]                     # normal / mild / marked
    dist = pg["distribution"]
    return {
        "status": "detected", "method": "classification",
        "label": _CUSTOM_PIGMENT_LABEL.get(pg["label"], pg["label"]),
        "score": round(1.0 - dist.get("normal", 0.0), 3),
        "confidence": pg["confidence"],
        "detail": {"distribution": dist, "source": "custom multi-task model, pigmentation head"},
        "model": "custom_skin_model",
        "note": "Trained on self-generated heuristic weak labels (CIELAB L* unevenness + dark-spot "
                "area fraction), not a clinically validated pigmentation grade. Indicative only.",
    }


def _skin_type_block(image_path: str, custom: dict[str, dict] | None = None) -> dict[str, dict]:
    if custom is not None:
        return _skin_type_block_custom(custom)

    from models.skin_analysis.classifier import get_classifier
    from PIL import Image

    clf = get_classifier(settings.skin_type_model_dir, "skin_type")
    r = clf.predict(Image.open(image_path))
    dist = r["distribution"]
    dry = dist.get("dry", 0.0)
    oily = dist.get("oily", 0.0)
    return {
        "skin_type": {
            "status": "detected", "method": "classification",
            "label": r["label"], "score": None, "confidence": r["confidence"],
            "detail": {"distribution": dist}, "model": "skin_type",
            "note": "Skin type is a whole-face classification (~65% reported accuracy); treat as indicative.",
        },
        "dryness": {
            "status": "detected", "method": "classification",
            "label": "dry-leaning" if dry >= 0.34 else "not dry-leaning",
            "score": round(dry, 3), "confidence": round(dry, 3),
            "detail": {"source": "'dry' probability from the skin-type model"},
            "model": "skin_type",
            "note": "Derived from the skin-type classifier's 'dry' class - not a localised dryness map.",
        },
        "oiliness": {
            "status": "detected", "method": "classification",
            "label": "oily-leaning" if oily >= 0.34 else "not oily-leaning",
            "score": round(oily, 3), "confidence": round(oily, 3),
            "detail": {"source": "'oily' probability from the skin-type model"},
            "model": "skin_type", "note": "Derived from the skin-type classifier's 'oily' class.",
        },
    }


def _acne_block(image_path: str, custom: dict[str, dict] | None = None) -> dict[str, Any]:
    """Combines a whole-face severity estimate + YOLO lesion localisation into
    one 'acne' entry. The severity source is either our custom model (when
    `custom` is given) or the HF ViT; the YOLO detector is used either way, so
    `detail.detections` / `lesion_count` are unchanged."""
    if custom is not None:
        ac = custom["acne"]                              # mild / moderate / severe / very_severe
        level_idx = _CUSTOM_ACNE_INDEX.get(ac["label"], 1)   # -> 1..4 on the existing 0..4 scale
        sev_label = ac["label"]
        sev_human = _CUSTOM_ACNE_HUMAN.get(ac["label"], ac["label"])
        sev_dist = ac["distribution"]
        sev_conf = ac["confidence"]
        sev_model = "custom_skin_model"
        sev_note = ("Severity is a whole-face estimate from the project's own trained model "
                    "(ACNE04 Hayashi grades; grade 0 == 'mild', there is no acne-free class). ")
    else:
        from models.skin_analysis.classifier import get_classifier
        from PIL import Image

        sev = get_classifier(settings.acne_severity_model_dir, "acne_severity").predict(Image.open(image_path))
        level_idx = sev["index"]                      # 0..4  (level -1 .. level 3)
        sev_label = sev["label"]
        sev_human = {0: "clear", 1: "occasional spots", 2: "mild", 3: "moderate",
                     4: "more pronounced"}.get(level_idx, sev["label"])
        sev_dist = sev["distribution"]
        sev_conf = sev["confidence"]
        sev_model = "acne_severity+acne_detector"
        sev_note = ("Severity is a coarse whole-face estimate (source model trained 1 epoch). ")

    sev_score = round(level_idx / 4.0, 3)

    d = _get_acne_detector().detect(image_path)
    n = d["counts"].get("acne", 0)
    top_conf = max((x["confidence"] for x in d["detections"]), default=None)

    if n == 0 and level_idx <= 1:
        label = "clear / minimal - no acne-like lesions localised"
    elif n == 0:
        label = f"severity estimate: {sev_human} (whole-face); no discrete lesions localised at the current threshold"
    else:
        label = f"severity estimate: {sev_human}; {n} acne-like lesion(s) localised"

    return {
        "status": "detected", "method": "classification+detection",
        "label": label,
        "score": max(sev_score, min(n / 10.0, 1.0)),
        "confidence": sev_conf,
        "detail": {
            "severity_level": sev_label, "severity_index": level_idx, "severity_human": sev_human,
            "severity_distribution": sev_dist,
            "lesion_count": n, "top_box_confidence": top_conf,
            "detections": d["detections"], "image_size": d["image_size"],
        },
        "model": f"{sev_model}+acne_detector" if custom is not None else sev_model,
        "note": sev_note + "Boxes are 'acne-like lesions' from a separate detector (author-reported "
                "mAP50 ~0.47), not a diagnosis. The two signals can disagree.",
    }


_acne_det = None


def _get_acne_detector():
    global _acne_det
    if _acne_det is None:
        from models.yolo_detector import YoloDetector

        _acne_det = YoloDetector(
            settings.acne_detector_weights,
            conf=settings.acne_detector_conf,
            iou=settings.acne_detector_iou,
            imgsz=settings.acne_detector_imgsz,
            device=settings.ml_device,
        )
    return _acne_det


def _wrinkle_block(image_path: str, face: dict) -> dict[str, Any]:
    from models.skin_analysis.wrinkle_segmenter import get_wrinkle_segmenter
    from PIL import Image
    import cv2

    seg = get_wrinkle_segmenter(settings.wrinkle_weights)
    crop = Image.fromarray(cv2.cvtColor(face["crop_bgr"], cv2.COLOR_BGR2RGB))
    r = seg.predict(crop, threshold=settings.wrinkle_threshold, skin_mask=face["skin_mask"])
    frac = r["wrinkle_area_fraction"]
    # map: 0 -> 0 ; 0.02 (2% of skin) -> 1.0
    score = round(min(frac / 0.02, 1.0), 3)
    band = "low" if score < 0.33 else ("moderate" if score < 0.66 else "elevated")
    return {
        "status": "detected", "method": "segmentation",
        "label": f"{band} wrinkle area", "score": score, "confidence": None,
        "detail": r,
        "model": "wrinkle_segmenter",
        "note": "Wrinkle-probability mask area fraction. Weights are CC BY-NC-SA (academic use only); "
                "positive-case accuracy not independently validated.",
    }


def run_analysis(image_path: str) -> dict[str, Any]:
    """Run the full pretrained-model stack on one uploaded image."""
    from models.skin_analysis import heuristics

    conditions: dict[str, Any] = {}
    used_models: list[str] = []

    # ---- face crop + skin mask (shared) ----
    try:
        face = heuristics.face_region(image_path)
    except Exception as e:  # pragma: no cover
        logger.exception("face_region failed")
        face = None

    # ---- custom multi-task model (skin_type + acne severity + pigmentation) ----
    # Runs once; None means "not enabled / checkpoint missing / failed" -> the
    # blocks below fall back to the HF ViT path automatically.
    custom = _predict_custom(face, image_path)
    using_custom = custom is not None

    # ---- skin type / dryness / oiliness ----
    try:
        conditions.update(_skin_type_block(image_path, custom))
        used_models.append("custom_skin_model" if using_custom else "skin_type")
    except Exception as e:
        for k in ("skin_type", "dryness", "oiliness"):
            conditions[k] = _err("custom_skin_model" if using_custom else "skin_type", e)

    # ---- acne (severity + detection) ----
    try:
        conditions["acne"] = _acne_block(image_path, custom)
        used_models += (["custom_skin_model", "acne_detector"] if using_custom
                        else ["acne_severity", "acne_detector"])
    except Exception as e:
        conditions["acne"] = _err("acne_severity+acne_detector", e)

    # ---- wrinkles ----
    if settings.wrinkle_enabled and face is not None:
        try:
            conditions["wrinkles"] = _wrinkle_block(image_path, face)
            used_models.append("wrinkle_segmenter")
        except Exception as e:
            conditions["wrinkles"] = _err("wrinkle_segmenter", e)
    else:
        conditions["wrinkles"] = {
            "status": "not_available", "method": None, "label": None, "score": None,
            "confidence": None, "model": None,
            "detail": {}, "note": "Wrinkle model disabled (settings.wrinkle_enabled=False) or face crop unavailable.",
        }

    # ---- pigmentation: produced by the custom model when available ----
    # (Set here, before the CV engine, so dark_spots can localise from it.
    #  compute_skin_health_index does not read pigmentation -> SHI unaffected.)
    if using_custom and isinstance(custom.get("pigmentation"), dict):
        try:
            conditions["pigmentation"] = _pigmentation_block_custom(custom)
        except Exception as e:
            conditions["pigmentation"] = _err("custom_skin_model", e)

    # ---- CV analysis engine: redness, pores, blackheads, dark spots ----
    # Deterministic colour/texture measurements (NOT ML). Each returns a 0..1
    # score, a soft 3-band distribution and a measurement-reliability confidence.
    # Pores and dark_spots are refined with the trained model's oiliness /
    # pigmentation outputs when those succeeded.
    if settings.heuristics_enabled and face is not None:
        crop, sm = face["crop_bgr"], face["skin_mask"]
        _oil = (conditions.get("oiliness") or {}).get("score")
        _oil = _oil if isinstance(_oil, (int, float)) else None
        _pig = (conditions.get("pigmentation") or {}).get("detail", {}).get("distribution")
        _pig = _pig if isinstance(_pig, dict) else None
        _cv_jobs = {
            "redness": lambda: heuristics.redness_estimate(crop, sm),
            "enlarged_pores": lambda: heuristics.pore_estimate(crop, sm, oiliness_score=_oil),
            "blackheads": lambda: heuristics.blackhead_estimate(crop, sm),
            "dark_spots": lambda: heuristics.dark_spot_estimate(crop, sm, pigment_dist=_pig),
        }
        for key, fn in _cv_jobs.items():
            try:
                entry = fn()
                entry.setdefault("model", "cv_analysis_engine")
                if entry.get("status") not in ("estimated", "not_available"):
                    entry["status"] = "estimated"
                conditions[key] = entry
            except Exception as e:
                conditions[key] = _err("cv_analysis_engine", e)
        used_models.append("cv_analysis_engine")
    else:
        for k in ("redness", "enlarged_pores", "blackheads", "dark_spots"):
            conditions[k] = {"status": "not_available", "method": None, "label": None,
                             "score": None, "confidence": None, "model": None, "detail": {},
                             "note": "CV analysis disabled or skin region unavailable."}

    # ---- any remaining uncovered condition -> not_available (defensive) ----
    for cond, reason in _NOT_AVAILABLE.items():
        conditions.setdefault(cond, {
            "status": "not_available", "method": None, "label": None, "score": None,
            "confidence": None, "model": None, "detail": {}, "note": reason,
        })

    # image size (from the acne detector's read, or PIL)
    image_size = conditions.get("acne", {}).get("detail", {}).get("image_size")
    if not image_size:
        from PIL import Image
        with Image.open(image_path) as im:
            image_size = [im.width, im.height]

    normalized = {k: _normalize(v) for k, v in conditions.items()}
    return {
        "analysis_version": ANALYSIS_VERSION,
        "face_detected": bool(face and face.get("face_detected")),
        "image_size": image_size,
        "conditions": normalized,
        "skin_health_index": compute_skin_health_index(normalized),
        "model_versions": {k: MODEL_VERSIONS[k] for k in dict.fromkeys(
            (["custom_skin_model", "acne_detector"] if "custom_skin_model" in used_models
             else ["skin_type", "acne_severity", "acne_detector"])
            + (["wrinkles"] if "wrinkle_segmenter" in used_models else [])
            + (["cv_analysis_engine"] if "cv_analysis_engine" in used_models else [])
        )},
    }
