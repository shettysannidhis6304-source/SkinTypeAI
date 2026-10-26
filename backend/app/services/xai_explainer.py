"""
backend/app/services/xai_explainer.py

Explainable-AI (XAI) diagnostic layer.

Turns the numeric multi-condition output of analysis_service.run_analysis() into
human-readable rationale. It performs NO new inference -- it only re-describes
values the trained models and the CV engine already produced, and it makes no
medical / diagnostic claims.

Output (attached to the analysis response under `xai_insights`):

  {
    "version": "xai-v1",
    "condition_rationales": {
        "<condition>": {
            "status":  "detected" | "estimated" | "not_available" | "error",
            "severity": "<band / label>" | null,
            "observation": "<plain-language clinical observation>",
            "metrics": { ...the exact numbers behind the observation... }
        }, ...
    },
    "shi_drivers": {
        "shi_value": 86, "shi_band": "excellent",
        "total_penalty_points": 5.1,
        "primary_drivers": [ {condition, penalty_points, share_pct, rationale}, ... ],  # top 2
        "all_penalties": [ {condition, points}, ... ],
        "note": "..."
    },
    "method_note": "..."
  }
"""
from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- helpers
def _na(entry: dict) -> dict[str, Any]:
    return {
        "status": entry.get("status"),
        "severity": None,
        "observation": entry.get("note") or "Not assessed for this image.",
        "metrics": {},
    }


def _pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------- per-condition
def _redness(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    ei = d.get("erythema_index")
    base = d.get("ei_baseline", 12.0)
    strong = d.get("ei_strong", 26.0)
    if entry.get("status") != "estimated" or ei is None:
        return _na(entry)

    if ei >= base + 6:
        obs = (f"Elevated red-channel absorbance in skin ROIs: Erythema Index of {ei} exceeds "
               f"baseline {base} (strong-erythema reference {strong}). Mean skin R/G reflectance "
               f"= {d.get('r_skin_mean')}/{d.get('g_skin_mean')}.")
    elif ei >= base:
        obs = (f"Erythema Index {ei} sits just above the healthy baseline of {base} - "
               f"mild diffuse redness at most.")
    else:
        obs = (f"Erythema Index {ei} is at or below the healthy baseline of {base}; "
               f"no measurable diffuse erythema.")
    frf = d.get("red_pixel_fraction") or 0.0
    if frf > 0.03:
        obs += f" Focal red patches cover ~{_pct(frf)} of skin pixels."
    return {
        "status": "estimated",
        "severity": d.get("band"),
        "observation": obs,
        "metrics": {
            "erythema_index": ei, "baseline": base, "strong_reference": strong,
            "r_skin_mean": d.get("r_skin_mean"), "g_skin_mean": d.get("g_skin_mean"),
            "focal_red_fraction": d.get("red_pixel_fraction"),
            "severity_score_0_1": entry.get("score"), "band": d.get("band"),
        },
    }


def _acne(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "detected":
        return _na(entry)
    sev = d.get("severity_human") or d.get("severity_level")
    grade = d.get("severity_grade")
    if grade is None:                                   # legacy record
        grade = d.get("severity_index")
    conf = entry.get("confidence")
    dist = d.get("severity_distribution") or {}

    obs = (f"Whole-face ACNE04 Hayashi-grade estimate: {sev} (grade {grade}/3, model confidence {conf}). "
           f"From-scratch classifier, no lesion bounding boxes on this path. "
           f"Grade probabilities: " + ", ".join(f"{k}:{v}" for k, v in dist.items()) + ".")
    return {
        "status": "detected",
        "severity": sev,
        "observation": obs,
        "metrics": {
            "severity_grade": grade, "severity_label": sev, "severity_confidence": conf,
            "severity_distribution": dist, "lesion_count": 0,
        },
    }


def _blackheads(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "estimated":
        return _na(entry)
    cnt, per1k, band = d.get("count"), d.get("per_1000px"), d.get("band")
    obs = (f"T-zone black-hat comedone screen: {cnt} candidate point(s) "
           f"(~{per1k} per 1000 px of visible T-zone) -> {band} density. Weak signal; "
           f"no trained comedone model exists.")
    if not d.get("tzone_visible", True):
        obs = "The T-zone was not clearly visible in this photo; blackhead screen inconclusive."
    return {
        "status": "estimated", "severity": band, "observation": obs,
        "metrics": {"tzone_count": cnt, "per_1000px": per1k, "band": band,
                    "tzone_visible": d.get("tzone_visible"), "score_0_1": entry.get("score")},
    }


def _dark_spots(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "estimated":
        return _na(entry)
    frac, band = d.get("spatial_blob_fraction"), d.get("band")
    obs = (f"Cheek/forehead dark-blob clustering covers {frac} of that region, localised from "
           f"the pigmentation head -> {band}. Basis: {d.get('basis')}.")
    return {
        "status": "estimated", "severity": band, "observation": obs,
        "metrics": {"spatial_blob_fraction": frac, "region_visible": d.get("region_visible"),
                    "band": band, "basis": d.get("basis"), "score_0_1": entry.get("score")},
    }


def _pores(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "estimated":
        return _na(entry)
    src = d.get("region_source")
    cov = d.get("region_coverage") or 0.0
    dens = d.get("texture_density")
    band = d.get("band")
    region_txt = {
        "facemesh": "MediaPipe FaceMesh cheek + nasal region (lips and face border removed)",
        "geometric": "geometric cheek + nasal zones (no landmarks available)",
        "skin_mask_fallback": "whole skin mask (landmark + geometric region gating both failed)",
    }.get(src, str(src))
    obs = (f"Black-hat texture density measured strictly over the {region_txt}, "
           f"covering ~{_pct(cov)} of the crop: {dens} of that region flags as pore / "
           f"fine-texture pits -> {band} pore visibility.")
    if "oiliness" in (d.get("basis") or ""):
        obs += " Combined with the model's oiliness read (sebum is the primary driver of pore visibility)."
    return {
        "status": "estimated", "severity": band, "observation": obs,
        "metrics": {"region_source": src, "region_coverage": cov, "texture_density": dens,
                    "band": band, "basis": d.get("basis"), "score_0_1": entry.get("score")},
    }


def _skin_barrier(conditions: dict) -> dict[str, Any] | None:
    st = conditions.get("skin_type", {}) or {}
    oil = conditions.get("oiliness", {}) or {}
    dry = conditions.get("dryness", {}) or {}
    if st.get("status") != "detected":
        return None
    dist = (st.get("detail") or {}).get("distribution") or {}
    o = oil.get("score")
    dr = dry.get("score")
    label = st.get("label")

    parts = [f"Skin type '{label}' is derived from the oiliness + dryness heads "
             f"(rule: {(st.get('detail') or {}).get('rule')}; confidence {st.get('confidence')})."]
    if isinstance(o, (int, float)) and isinstance(dr, (int, float)):
        if o >= 0.34 and dr >= 0.34:
            state = "combination / dehydrated-oily - both the sebum and the dryness signals are raised"
        elif o >= 0.34:
            state = "oil-dominant - sebum signal leads, barrier hydration looks adequate"
        elif dr >= 0.34:
            state = "dry-leaning / barrier-stressed - dryness signal leads, low sebum"
        else:
            state = "balanced - neither the sebum nor the dryness signal is prominent"
        parts.append(f"Oiliness proportion {o:.2f} vs dryness proportion {dr:.2f} -> {state}.")
    parts.append("Oiliness and dryness are whole-face tier estimates from the from-scratch "
                 "SkinAnalysisModel (trained on OpenCV weak labels), not localised maps.")
    return {
        "status": "detected", "severity": None, "observation": " ".join(parts),
        "metrics": {"skin_type": label, "confidence": st.get("confidence"), "distribution": dist,
                    "oiliness_score": o, "dryness_score": dr},
    }


def _wrinkles(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "detected":
        return _na(entry)
    tier = d.get("tier")
    dist = d.get("distribution") or {}
    obs = (f"SkinAnalysisModel wrinkles head reads '{tier}' "
           f"({', '.join(f'{k}:{v}' for k, v in dist.items())}); normalised severity "
           f"{entry.get('score')}. Trained on OpenCV edge/line-frequency weak labels "
           f"(not the FFHQ-Wrinkle U-Net) - indicative only.")
    return {
        "status": "detected", "severity": entry.get("label"), "observation": obs,
        "metrics": {"tier": tier, "distribution": dist, "score_0_1": entry.get("score"),
                    "backend": "skin_analysis_model"},
    }


def _pigmentation(entry: dict) -> dict[str, Any]:
    d = entry.get("detail") or {}
    if entry.get("status") != "detected":
        return _na(entry)
    dist = d.get("distribution") or {}
    obs = (f"SkinAnalysisModel pigmentation head: {entry.get('label')} "
           f"(P(normal)={dist.get('normal')}, P(mild)={dist.get('mild')}, P(marked)={dist.get('marked')}). "
           f"Trained on Stage-1 OpenCV weak labels - indicative only.")
    return {
        "status": "detected", "severity": entry.get("label"), "observation": obs,
        "metrics": {"distribution": dist, "confidence": entry.get("confidence"),
                    "score_0_1": entry.get("score")},
    }


# --------------------------------------------------------------------------- SHI drivers
def _driver_rationale(d: dict, entry: dict) -> str:
    cond, pts, mx = d["condition"], d["points"], d["max_points"]
    det = entry.get("detail") or {}
    if cond == "acne":
        return (f"Acne Hayashi grade {det.get('severity_grade', det.get('severity_index'))}/3 "
                f"-> -{pts} of a possible -{mx} pts.")
    if cond == "wrinkles":
        return f"Wrinkle-area severity {d['driver_value']} -> -{pts} of -{mx} pts."
    if cond == "dryness":
        return (f"Dryness score {entry.get('score')} past the 0.34 barrier threshold "
                f"-> -{pts} of -{mx} pts.")
    if cond in ("redness", "enlarged_pores"):
        return (f"Estimated {cond.replace('_', ' ')} severity {entry.get('score')} "
                f"-> -{pts} of -{mx} pts (CV estimate, down-weighted in the SHI).")
    return f"-{pts} of -{mx} pts."


def _shi_drivers(conditions: dict, shi: dict) -> dict[str, Any]:
    try:
        from app.services.analysis_service import shi_penalty_breakdown  # lazy: avoid import cycle
        breakdown = shi_penalty_breakdown(conditions)
    except Exception:  # pragma: no cover - defensive
        breakdown = []

    contributing = [d for d in breakdown if d["points"] > 0]
    total = sum(d["points"] for d in contributing)
    drivers: list[dict[str, Any]] = []
    for d in contributing[:2]:
        drivers.append({
            "condition": d["condition"],
            "penalty_points": d["points"],
            "share_pct": round(100.0 * d["points"] / total, 1) if total else 0.0,
            "driver_value_0_1": d["driver_value"],
            "model_backed": d["model_backed"],
            "rationale": _driver_rationale(d, conditions.get(d["condition"], {}) or {}),
        })

    if not contributing:
        note = "No condition is currently reducing the Skin Health Index."
    elif len(drivers) == 1:
        note = f"{drivers[0]['condition']} accounts for the entire SHI reduction."
    else:
        note = (f"{drivers[0]['condition']} and {drivers[1]['condition']} account for "
                f"{round(drivers[0]['share_pct'] + drivers[1]['share_pct'], 1)}% of the "
                f"{round(total, 1)}-point SHI reduction.")

    return {
        "shi_value": shi.get("value"),
        "shi_band": shi.get("band"),
        "total_penalty_points": round(total, 2),
        "primary_drivers": drivers,
        "all_penalties": [{"condition": d["condition"], "points": d["points"]} for d in breakdown],
        "note": note,
    }


# --------------------------------------------------------------------------- public entry
_DISPATCH = {
    "redness": _redness,
    "acne": _acne,
    "blackheads": _blackheads,
    "dark_spots": _dark_spots,
    "enlarged_pores": _pores,
    "wrinkles": _wrinkles,
    "pigmentation": _pigmentation,
}


def generate_diagnostic_explanations(conditions: dict[str, dict], shi: dict) -> dict[str, Any]:
    """Build the `xai_insights` block. `conditions` is the normalised
    results.conditions dict; `shi` is results.skin_health_index."""
    conditions = conditions or {}
    shi = shi or {}
    rationales: dict[str, Any] = {}

    for name, fn in _DISPATCH.items():
        entry = conditions.get(name)
        if entry is None:
            continue
        try:
            rationales[name] = fn(entry)
        except Exception as e:  # pragma: no cover - never break the response
            rationales[name] = {"status": entry.get("status"), "severity": None,
                                "observation": "Explanation unavailable.", "metrics": {},
                                "error": repr(e)}

    barrier = _skin_barrier(conditions)
    if barrier is not None:
        rationales["skin_type"] = barrier

    return {
        "version": "xai-v1",
        "condition_rationales": rationales,
        "shi_drivers": _shi_drivers(conditions, shi),
        "method_note": ("Rule-based translation of the trained-model and CV-engine outputs into "
                        "plain-language observations. No new inference is performed; not a diagnosis."),
    }
