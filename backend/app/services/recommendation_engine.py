"""
backend/app/services/recommendation_engine.py

Rule-based skincare guidance driven by the REAL multi-condition analysis output
from analysis_service.run_analysis(), plus the optional questionnaire +
environment inputs (temperature, humidity, lifestyle -- see ScanContext).

BOUNDARIES (unchanged, extended to the new conditions)
-----------------------------------------------------
* Nothing is invented. Guidance is only added for conditions the models
  actually flagged ("detected") or estimated ("estimated"). "not_available"
  and "error" conditions produce no guidance.
* No medical/diagnostic language. Not "you have severe acne" but
  "acne-like lesions were detected -- consider a professional evaluation if
  persistent". Redness/pores guidance is explicitly framed as "estimated".
* `context` never reaches any model. It only layers standard
  climate/lifestyle tips (temperature + humidity included) on top.

BACK-COMPAT: also accepts the two older `results` shapes
  * YOLO-only:      {"detections": [...], "counts": {...}, "image_size": [...]}
  * EfficientNet:   {"skin_type": {"label": ..., "confidence": ...}}
so replaying an old /history record never crashes.
"""
from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "This is an academic skincare-guidance tool, not a medical device. It does not "
    "diagnose any condition. For persistent, painful, spreading or concerning skin "
    "changes, consult a qualified dermatologist."
)

# --- per-condition guidance, keyed by a coarse band -------------------------
CONDITION_RULES: dict[str, dict[str, dict[str, list[str]]]] = {
    "acne": {
        "clear":    {"routine": [], "ingredients": []},
        "mild":     {"routine": ["Gentle cleanser AM/PM", "Spot-treat flagged areas with a BHA or benzoyl peroxide gel",
                                 "Lightweight, non-comedogenic moisturizer", "Daily SPF 30+"],
                     "ingredients": ["Salicylic acid (BHA)", "Benzoyl peroxide (2.5%)", "Niacinamide"]},
        "moderate": {"routine": ["BHA cleanser", "Leave-on acne treatment 3-4x/week (build tolerance)",
                                 "Non-comedogenic moisturizer to support the barrier", "Daily SPF 30+"],
                     "ingredients": ["Salicylic acid", "Adapalene (OTC retinoid)", "Azelaic acid", "Niacinamide"]},
        "high":     {"routine": ["Consider a dermatologist for prescription options",
                                 "Gentle, fragrance-free cleanser AM/PM", "Non-comedogenic moisturizer", "Daily SPF 30+"],
                     "ingredients": ["Adapalene", "Azelaic acid", "Prescription-strength options (consult a professional)"]},
    },
    "wrinkles": {
        "low":      {"routine": ["Daily SPF 30+ (the biggest lever for skin ageing)"], "ingredients": []},
        "moderate": {"routine": ["Daily SPF 30+", "Introduce a nightly retinoid, low strength, 2-3x/week"],
                     "ingredients": ["Retinol / retinaldehyde", "Peptides", "Vitamin C (AM)"]},
        "elevated": {"routine": ["Daily SPF 30+", "Nightly retinoid (build up as tolerated)",
                                 "Rich moisturizer to reduce the look of fine lines"],
                     "ingredients": ["Retinol", "Peptides", "Hyaluronic acid", "Vitamin C (AM)"]},
    },
    "redness": {  # estimated only
        "low":      {"routine": [], "ingredients": []},
        "moderate": {"routine": ["Fragrance-free products only", "Patch-test new actives"],
                     "ingredients": ["Centella asiatica", "Niacinamide"]},
        "elevated": {"routine": ["Fragrance-free, minimal routine", "Avoid physical scrubs and hot water",
                                 "If redness is persistent or flushing, a dermatologist can rule out rosacea"],
                     "ingredients": ["Azelaic acid", "Centella asiatica", "Niacinamide"]},
    },
    "enlarged_pores": {  # estimated only
        "low":      {"routine": [], "ingredients": []},
        "moderate": {"routine": ["BHA exfoliation 2-3x/week"], "ingredients": ["Salicylic acid", "Niacinamide"]},
        "elevated": {"routine": ["BHA exfoliation 2-3x/week", "Weekly clay mask", "Non-comedogenic everything"],
                     "ingredients": ["Salicylic acid", "Niacinamide", "Retinoid (also helps texture)"]},
    },
    "blackheads": {  # CV-analysis estimate only
        "low":      {"routine": [], "ingredients": []},
        "moderate": {"routine": ["BHA (salicylic acid) cleanser or leave-on 2-3x/week in the T-zone"],
                     "ingredients": ["Salicylic acid (BHA)", "Niacinamide"]},
        "elevated": {"routine": ["Daily BHA in the T-zone (build tolerance)", "Weekly clay or charcoal mask",
                                 "Non-comedogenic sunscreen and moisturiser"],
                     "ingredients": ["Salicylic acid (BHA)", "Adapalene (OTC retinoid)", "Niacinamide"]},
    },
    "dark_spots": {  # CV-analysis estimate only
        "low":      {"routine": [], "ingredients": []},
        "moderate": {"routine": ["Daily broad-spectrum SPF 30+ (essential - UV drives pigmentation)",
                                 "Introduce a brightening active at night"],
                     "ingredients": ["Vitamin C (AM)", "Niacinamide", "Alpha arbutin"]},
        "elevated": {"routine": ["Daily SPF 30+ and reapply", "Nightly brightening + gentle exfoliation",
                                 "For stubborn or spreading marks, a dermatologist can advise on stronger options"],
                     "ingredients": ["Vitamin C", "Alpha arbutin", "Azelaic acid", "Tranexamic acid"]},
    },
}

SKIN_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    "oily":   {"routine": ["Oil-free gel cleanser AM/PM", "Lightweight, oil-free moisturizer"],
               "ingredients": ["Niacinamide", "Salicylic acid (BHA)"]},
    "dry":    {"routine": ["Cream/lotion cleanser", "Richer moisturizer AM/PM", "Optional facial oil at night"],
               "ingredients": ["Hyaluronic acid", "Ceramides", "Squalane"]},
    "normal": {"routine": ["Gentle cleanser AM/PM", "Lightweight moisturizer"],
               "ingredients": ["Hyaluronic acid", "Vitamin C (AM)"]},
}


def _u(target: list[str], items: list[str]) -> None:
    for it in items:
        if it not in target:
            target.append(it)


def _band_from_score(score: float | None) -> str:
    if score is None:
        return "low"
    return "low" if score < 0.34 else ("moderate" if score < 0.67 else "elevated")


def _acne_band(entry: dict) -> str:
    d = entry.get("detail") or {}
    n = int(d.get("lesion_count") or 0)
    idx = int(d.get("severity_index") or 0)  # 0..4
    if n == 0 and idx <= 1:
        return "clear"
    if n <= 3 and idx <= 2:
        return "mild"
    if n <= 8 and idx <= 3:
        return "moderate"
    return "high"


def _context_layer(context: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (routine, ingredients, notes). Standard climate/lifestyle tips."""
    routine: list[str] = []
    ingredients: list[str] = []
    notes: list[str] = []

    temp = context.get("temperature_c")
    hum = context.get("humidity_pct")
    if isinstance(hum, (int, float)):
        if hum < 35:
            notes.append(f"Low humidity ({hum:.0f}%) - skin loses water faster; layer a humectant under moisturizer.")
            _u(ingredients, ["Hyaluronic acid", "Glycerin", "Ceramides"])
            _u(routine, ["Consider a humidifier indoors"])
        elif hum > 70:
            notes.append(f"High humidity ({hum:.0f}%) - lighter, non-comedogenic textures; a mattifying SPF helps.")
            _u(ingredients, ["Niacinamide"])
    if isinstance(temp, (int, float)):
        if temp <= 5:
            notes.append(f"Cold conditions ({temp:.0f} C) - protect the barrier; avoid very hot water.")
            _u(ingredients, ["Ceramides", "Squalane"])
        elif temp >= 30:
            notes.append(f"Hot conditions ({temp:.0f} C) - reapply SPF more often; blot rather than re-wash.")
            _u(routine, ["Reapply SPF every 2-3 hours outdoors"])

    uv = context.get("uv_index")
    if isinstance(uv, (int, float)):
        if uv >= 6:
            notes.append(f"UV index {uv:.0f} (high) - SPF 30+ every morning and reapply; seek shade midday.")
            _u(routine, ["Daily SPF 30+ and reapply every 2-3 hours"])
        elif uv >= 3:
            notes.append(f"UV index {uv:.0f} (moderate) - daily SPF 30+ is still worthwhile.")
            _u(routine, ["Daily SPF 30+"])
    aqi = context.get("air_quality_aqi")
    if isinstance(aqi, (int, float)) and aqi > 100:
        notes.append(f"Air quality is poor (AQI {aqi:.0f}) - double cleanse in the evening; antioxidants in the morning.")
        _u(routine, ["Double cleanse in the evening (pollutant removal)"])
        _u(ingredients, ["Vitamin C", "Vitamin E"])

    loc = context.get("location")
    if loc in ("cold_dry", "desert"):
        _u(routine, ["Dry climate: humidifier + occlusive at night"]); _u(ingredients, ["Ceramides", "Glycerin"])
    elif loc == "humid_tropical":
        _u(routine, ["Humid climate: lightweight, mattifying sunscreen"]); _u(ingredients, ["Niacinamide"])
    elif loc == "urban_polluted":
        _u(routine, ["Double cleanse in the evening (pollutant removal)"]); _u(ingredients, ["Vitamin C", "Vitamin E"])

    if context.get("job_type") == "outdoor":
        _u(routine, ["Reapply SPF every 2-3 hours during extended outdoor time"])
    elif context.get("job_type") == "indoor_ac":
        _u(routine, ["Hydrating mist during the day - A/C is dehydrating"])
    if context.get("sun_exposure") == "high":
        _u(routine, ["Daily SPF 30+ (non-negotiable at your sun-exposure level)"])
    age = context.get("age")
    if isinstance(age, (int, float)) and age >= 35:
        _u(routine, ["A nightly retinoid/peptide serum supports early prevention"]); _u(ingredients, ["Retinol", "Peptides"])
    if context.get("stress_level") == "high":
        _u(routine, ["Weekly barrier-repair / calming mask for stress-related flare-ups"]); _u(ingredients, ["Centella asiatica"])
    sh = context.get("sleep_hours")
    if isinstance(sh, (int, float)) and sh < 6:
        notes.append("Short sleep - most barrier repair happens overnight; prioritise where possible.")
    wi = context.get("water_intake_liters")
    if isinstance(wi, (int, float)) and wi < 1.5:
        notes.append("Low reported water intake - hydration supports overall skin comfort.")
    return routine, ingredients, notes


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_recommendation(results: dict, context: dict | None = None) -> dict:
    context = context or {}

    # ---- back-compat shims ------------------------------------------------
    if "conditions" not in results:
        return _legacy_build(results, context)

    conditions: dict[str, dict] = results.get("conditions", {})
    summary: list[str] = []
    per_condition: dict[str, list[str]] = {}
    routine: list[str] = []
    ingredients: list[str] = []

    # skin type (drives the base routine)
    st = conditions.get("skin_type", {})
    if st.get("status") == "detected" and st.get("label") in SKIN_TYPE_RULES:
        rule = SKIN_TYPE_RULES[st["label"]]
        _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
        summary.append(f"Skin type looks {st['label']} (model confidence {st.get('confidence')}).")
        per_condition["skin_type"] = [f"Base routine tuned for {st['label']} skin."]

    # acne
    ac = conditions.get("acne", {})
    if ac.get("status") == "detected":
        band = _acne_band(ac)
        rule = CONDITION_RULES["acne"][band]
        _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
        n = int((ac.get("detail") or {}).get("lesion_count") or 0)
        if band == "clear":
            per_condition["acne"] = ["No acne-like lesions localised and severity reads as clear/occasional. "
                                     "A model cannot confirm skin is problem-free."]
        else:
            msg = f"{n} acne-like lesion(s) localised; severity estimate: {band}."
            if band == "high":
                msg += " Consider a professional dermatological evaluation."
            per_condition["acne"] = [msg]
            summary.append(msg)

    # wrinkles
    wr = conditions.get("wrinkles", {})
    if wr.get("status") == "detected":
        band = _band_from_score(wr.get("score"))
        rule = CONDITION_RULES["wrinkles"][band]
        _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
        per_condition["wrinkles"] = [f"Wrinkle-area estimate: {band}."]

    # CV-analysis estimates: redness / pores / blackheads / dark spots
    _cv_nice = {"redness": "redness", "enlarged_pores": "pore visibility",
                "blackheads": "blackhead density", "dark_spots": "localised dark spots"}
    for cond in ("redness", "enlarged_pores", "blackheads", "dark_spots"):
        e = conditions.get(cond, {})
        if e.get("status") == "estimated":
            band = _band_from_score(e.get("score"))
            rule = CONDITION_RULES[cond][band]
            _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
            per_condition[cond] = [f"Estimated {_cv_nice[cond]}: {band} "
                                   f"(computer-vision measurement, not a trained model)."]

    # pigmentation (custom model) or any still-uncovered condition
    for cond in ("pigmentation", "blackheads", "dark_spots"):
        e = conditions.get(cond, {})
        if e.get("status") == "not_available":
            per_condition[cond] = ["Not assessed - the analysis engine could not read this from the photo."]

    # context layer
    cr, ci, notes = _context_layer(context)
    _u(routine, cr); _u(ingredients, ci)

    if not summary:
        summary.append("No prominent concerns flagged by the available models. See per-condition notes.")

    return {
        "summary": summary,
        "per_condition": per_condition,
        "routine": routine,
        "ingredients": ingredients,
        "context_notes": notes,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# legacy shapes
# ---------------------------------------------------------------------------
def _legacy_build(results: dict, context: dict) -> dict:
    routine: list[str] = []
    ingredients: list[str] = []
    summary: list[str] = []

    if "detections" in results:  # YOLO-only record
        n = int((results.get("counts") or {}).get("acne", 0))
        band = "clear" if n == 0 else ("mild" if n <= 3 else ("moderate" if n <= 8 else "high"))
        rule = CONDITION_RULES["acne"][band]
        _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
        summary.append(f"(legacy YOLO record) {n} acne-like lesion(s).")
    elif "skin_type" in results:  # EfficientNet record
        label = (results.get("skin_type") or {}).get("label")
        rule = SKIN_TYPE_RULES.get(label)
        if rule:
            _u(routine, rule["routine"]); _u(ingredients, rule["ingredients"])
        summary.append(f"(legacy skin-type record: {label})")

    cr, ci, notes = _context_layer(context)
    _u(routine, cr); _u(ingredients, ci)
    return {"summary": summary, "per_condition": {}, "routine": routine,
            "ingredients": ingredients, "context_notes": notes, "disclaimer": DISCLAIMER}
