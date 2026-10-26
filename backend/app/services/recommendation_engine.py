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
    "combination": {"routine": ["Gentle gel or cream cleanser AM/PM",
                                "Lighter moisturizer on the T-zone, richer on the cheeks",
                                "Optional BHA 2x/week in the T-zone only"],
                    "ingredients": ["Niacinamide", "Hyaluronic acid", "Salicylic acid (BHA)"]},
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
    # Current path: SkinAnalysisModel emits a raw ACNE04 Hayashi grade 0..3.
    g = d.get("severity_grade")
    if g is not None:
        return {0: "clear", 1: "mild", 2: "moderate", 3: "high"}.get(int(g), "moderate")
    # Legacy /history records: 0..4 severity_index (+ optional lesion_count).
    n = int(d.get("lesion_count") or 0)
    idx = int(d.get("severity_index") or 0)
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


# ===========================================================================
# Evidence-based active-ingredient library + AM/PM routine engine
# ---------------------------------------------------------------------------
# Each active maps to the conditions it has established cosmetic-dermatology
# evidence for, when to use it (AM / PM / AM-PM), and a one-line caution. This is
# skincare guidance, not prescribing -- strengths quoted are common OTC ranges.
# ===========================================================================
ACTIVE_LIBRARY: dict[str, dict[str, Any]] = {
    "Azelaic acid 10%": {
        "class": "dicarboxylic acid", "use": "AM/PM",
        "targets": ["redness", "pigmentation", "dark_spots", "acne", "blackheads"],
        "evidence": "Trial-supported for papulopustular rosacea, mild-moderate acne and post-inflammatory "
                    "hyperpigmentation; tolerated on sensitised / barrier-compromised skin.",
        "caution": "Mild transient tingling in the first week.",
        "exfoliant_acid": False, "retinoid": False,
    },
    "Niacinamide 5%": {
        "class": "vitamin B3", "use": "AM/PM",
        "targets": ["redness", "pigmentation", "oiliness", "enlarged_pores", "barrier"],
        "evidence": "Reduces transepidermal water loss and sebum, evens tone, calms erythema; very well tolerated.",
        "caution": "Rare flushing above ~10%.", "exfoliant_acid": False, "retinoid": False,
    },
    "Salicylic acid (BHA) 2%": {
        "class": "beta-hydroxy acid", "use": "PM",
        "targets": ["blackheads", "enlarged_pores", "acne", "oiliness"],
        "evidence": "Lipophilic exfoliant; clears follicular plugging (comedones) and refines pore appearance.",
        "caution": "Photosensitising; dryness/irritation if overused. Not with a compromised barrier.",
        "exfoliant_acid": True, "retinoid": False,
    },
    "PHA (gluconolactone) 4%": {
        "class": "polyhydroxy acid", "use": "PM",
        "targets": ["enlarged_pores", "dark_spots", "texture"],
        "evidence": "Large-molecule exfoliant; gentler alternative to AHA/BHA for reactive or red skin.",
        "caution": "Still mildly photosensitising - keep SPF up.", "exfoliant_acid": True, "retinoid": False,
    },
    "Benzoyl peroxide 2.5%": {
        "class": "antimicrobial oxidiser", "use": "AM",
        "targets": ["acne"],
        "evidence": "Reduces C. acnes load and inflammatory lesions; 2.5% is as effective as higher % with less irritation.",
        "caution": "Bleaches fabric; drying. Do not layer with a retinoid in the same routine.",
        "exfoliant_acid": False, "retinoid": False,
    },
    "Adapalene 0.1%": {
        "class": "topical retinoid (OTC)", "use": "PM",
        "targets": ["acne", "blackheads", "enlarged_pores", "wrinkles"],
        "evidence": "Normalises follicular keratinisation; first-line OTC retinoid for comedonal + inflammatory acne.",
        "caution": "Retinisation (dryness/peeling) for 2-4 weeks; photosensitising; avoid in pregnancy.",
        "exfoliant_acid": False, "retinoid": True,
    },
    "Retinol 0.3%": {
        "class": "retinoid", "use": "PM",
        "targets": ["wrinkles", "pigmentation", "texture"],
        "evidence": "Converts to retinoic acid in skin; improves fine lines, tone and texture over 12+ weeks.",
        "caution": "Start low and slow; photosensitising; avoid in pregnancy.",
        "exfoliant_acid": False, "retinoid": True,
    },
    "Retinaldehyde 0.05%": {
        "class": "retinoid", "use": "PM",
        "targets": ["wrinkles", "acne", "texture"],
        "evidence": "One step from retinoic acid; efficacy near retinol/adapalene with good tolerability.",
        "caution": "Photosensitising; avoid in pregnancy.", "exfoliant_acid": False, "retinoid": True,
    },
    "Peptides": {
        "class": "signal peptides", "use": "AM/PM",
        "targets": ["wrinkles", "barrier"],
        "evidence": "Support dermal matrix synthesis; useful adjunct or retinoid-alternative for sensitive skin.",
        "caution": "None notable.", "exfoliant_acid": False, "retinoid": False,
    },
    "Vitamin C (L-ascorbic acid 10-15%, AM)": {
        "class": "antioxidant", "use": "AM",
        "targets": ["pigmentation", "dark_spots", "wrinkles"],
        "evidence": "Antioxidant photoprotection adjunct; inhibits tyrosinase, brightens over 8-12 weeks.",
        "caution": "Can sting reactive skin; oxidises - store cool, replace if it turns brown.",
        "exfoliant_acid": False, "retinoid": False,
    },
    "Alpha arbutin 2%": {
        "class": "tyrosinase inhibitor", "use": "AM/PM",
        "targets": ["pigmentation", "dark_spots"],
        "evidence": "Hydroquinone-derivative that reduces melanin synthesis with a good safety profile.",
        "caution": "None notable.", "exfoliant_acid": False, "retinoid": False,
    },
    "Tranexamic acid 3%": {
        "class": "plasmin inhibitor", "use": "AM/PM",
        "targets": ["pigmentation", "dark_spots", "redness"],
        "evidence": "Topical evidence for melasma and post-inflammatory erythema/pigmentation.",
        "caution": "None notable topically.", "exfoliant_acid": False, "retinoid": False,
    },
    "Centella asiatica": {
        "class": "botanical (madecassoside)", "use": "AM/PM",
        "targets": ["redness", "barrier"],
        "evidence": "Supports barrier repair and reduces erythema in sensitive-skin trials.",
        "caution": "None notable.", "exfoliant_acid": False, "retinoid": False,
    },
    "Hyaluronic acid": {
        "class": "humectant", "use": "AM/PM", "targets": ["dryness", "barrier"],
        "evidence": "Multi-weight HA binds water in the stratum corneum; apply to damp skin, seal with moisturiser.",
        "caution": "In very dry air, always seal - unsealed HA can draw water from the skin.",
        "exfoliant_acid": False, "retinoid": False,
    },
    "Glycerin": {
        "class": "humectant", "use": "AM/PM", "targets": ["dryness", "barrier"],
        "evidence": "Best-evidenced humectant for stratum-corneum hydration and barrier recovery.",
        "caution": "None.", "exfoliant_acid": False, "retinoid": False,
    },
    "Ceramides": {
        "class": "barrier lipid", "use": "AM/PM", "targets": ["dryness", "barrier", "redness"],
        "evidence": "Replaces the lipids of a compromised barrier; reduces TEWL and sensitivity.",
        "caution": "None.", "exfoliant_acid": False, "retinoid": False,
    },
    "Squalane": {
        "class": "emollient", "use": "PM", "targets": ["dryness", "barrier"],
        "evidence": "Non-comedogenic emollient that softens skin and reduces water loss.",
        "caution": "None.", "exfoliant_acid": False, "retinoid": False,
    },
}

# condition -> band -> ordered active names. Bands: acne uses clear/mild/moderate/high;
# everything else uses low/moderate/elevated (pigmentation reuses moderate/elevated).
CONDITION_ACTIVES: dict[str, dict[str, list[str]]] = {
    "acne": {
        "mild":     ["Salicylic acid (BHA) 2%", "Benzoyl peroxide 2.5%", "Niacinamide 5%"],
        "moderate": ["Adapalene 0.1%", "Benzoyl peroxide 2.5%", "Azelaic acid 10%", "Niacinamide 5%"],
        "high":     ["Adapalene 0.1%", "Azelaic acid 10%", "Niacinamide 5%"],
    },
    "blackheads": {
        "moderate": ["Salicylic acid (BHA) 2%", "Niacinamide 5%"],
        "elevated": ["Salicylic acid (BHA) 2%", "Adapalene 0.1%", "Niacinamide 5%"],
    },
    "enlarged_pores": {
        "moderate": ["Salicylic acid (BHA) 2%", "Niacinamide 5%"],
        "elevated": ["Salicylic acid (BHA) 2%", "Niacinamide 5%", "Adapalene 0.1%"],
    },
    "redness": {
        "moderate": ["Azelaic acid 10%", "Niacinamide 5%", "Centella asiatica"],
        "elevated": ["Azelaic acid 10%", "Centella asiatica", "Niacinamide 5%", "Tranexamic acid 3%"],
    },
    "pigmentation": {
        "moderate": ["Niacinamide 5%", "Alpha arbutin 2%", "Vitamin C (L-ascorbic acid 10-15%, AM)"],
        "elevated": ["Azelaic acid 10%", "Vitamin C (L-ascorbic acid 10-15%, AM)", "Alpha arbutin 2%", "Tranexamic acid 3%"],
    },
    "dark_spots": {
        "moderate": ["Vitamin C (L-ascorbic acid 10-15%, AM)", "Niacinamide 5%", "Alpha arbutin 2%"],
        "elevated": ["Vitamin C (L-ascorbic acid 10-15%, AM)", "Azelaic acid 10%", "Alpha arbutin 2%", "PHA (gluconolactone) 4%"],
    },
    "wrinkles": {
        "moderate": ["Retinol 0.3%", "Peptides", "Vitamin C (L-ascorbic acid 10-15%, AM)"],
        "elevated": ["Retinaldehyde 0.05%", "Peptides", "Vitamin C (L-ascorbic acid 10-15%, AM)", "Hyaluronic acid"],
    },
}

SKIN_TYPE_ACTIVES: dict[str, list[str]] = {
    "dry":         ["Hyaluronic acid", "Glycerin", "Ceramides", "Squalane"],
    "oily":        ["Niacinamide 5%", "Salicylic acid (BHA) 2%"],
    "normal":      ["Hyaluronic acid", "Peptides"],
    "combination": ["Niacinamide 5%", "Hyaluronic acid", "Ceramides"],
}

_MOISTURISER_ACTIVES = ("Ceramides", "Hyaluronic acid", "Squalane", "Glycerin", "Peptides")


def _is_exfoliant(a: str) -> bool:
    return bool(ACTIVE_LIBRARY.get(a, {}).get("exfoliant_acid"))


def _is_retinoid(a: str) -> bool:
    return bool(ACTIVE_LIBRARY.get(a, {}).get("retinoid"))


def _is_photosensitising(a: str) -> bool:
    """True if the active raises UV sensitivity (retinoids, AHA/BHA, or an
    explicit 'photosensitising' caution). Such actives are routed to PM only."""
    if _is_retinoid(a) or _is_exfoliant(a):
        return True
    return "photosensitis" in (ACTIVE_LIBRARY.get(a, {}).get("caution") or "").lower()


# --- pairwise ingredient contraindication matrix ---------------------------
# Each rule: two predicates over an active name; if the picked set contains an
# active matching each side, the rule fires. `action`:
#   "separate_am_pm"  -> the two must not share a routine (one AM, one PM)
#   "alternate_nights" -> use on different evenings for the first few weeks
CONTRAINDICATION_MATRIX: list[dict[str, Any]] = [
    {
        "id": "retinoid_x_exfoliant_acid",
        "left": _is_retinoid,
        "right": _is_exfoliant,
        "severity": "high",
        "action": "alternate_nights",
        "message": "Retinoid + AHA/BHA in the same night is a common irritation / barrier-damage "
                   "combination. Alternate them on different evenings (retinoid nights vs acid nights) "
                   "for at least the first 4-6 weeks.",
    },
    {
        "id": "vitamin_c_x_benzoyl_peroxide",
        "left": lambda a: "Vitamin C" in a,
        "right": lambda a: "Benzoyl peroxide" in a,
        "severity": "high",
        "action": "separate_am_pm",
        "message": "Benzoyl peroxide oxidises L-ascorbic acid (vitamin C), degrading both. Keep vitamin C "
                   "in the AM and benzoyl peroxide in the PM (or on alternate days).",
    },
]


def _apply_contraindication_matrix(picked: list[str]) -> tuple[list[dict], set[str]]:
    """Scan the picked actives against CONTRAINDICATION_MATRIX. Returns
    (flags, force_pm) where force_pm is the set of actives that must move to PM."""
    flags: list[dict] = []
    force_pm: set[str] = set()
    for rule in CONTRAINDICATION_MATRIX:
        left = [a for a in picked if rule["left"](a)]
        right = [a for a in picked if rule["right"](a)]
        if not (left and right):
            continue
        flags.append({
            "rule": rule["id"], "severity": rule["severity"], "action": rule["action"],
            "actives": sorted({*left, *right}),
            "message": rule["message"],
        })
        if rule["action"] == "separate_am_pm":
            # keep the AM-preferred member in AM, push the other to PM
            for a in right:
                if "AM" not in ACTIVE_LIBRARY.get(a, {}).get("use", "AM/PM"):
                    force_pm.add(a)
    return flags, force_pm


def _pigmentation_band(entry: dict) -> str:
    lbl = (entry.get("label") or "").lower()
    if "marked" in lbl:
        return "elevated"
    if "mild" in lbl:
        return "moderate"
    sc = entry.get("score")
    return _band_from_score(sc) if isinstance(sc, (int, float)) else "low"


def _bands_from_conditions(conditions: dict) -> tuple[dict[str, str], str | None]:
    bands: dict[str, str] = {}
    st = conditions.get("skin_type", {}) or {}
    skin_type_label = st.get("label") if st.get("status") == "detected" else None

    ac = conditions.get("acne", {}) or {}
    if ac.get("status") == "detected":
        bands["acne"] = _acne_band(ac)
    wr = conditions.get("wrinkles", {}) or {}
    if wr.get("status") == "detected":
        bands["wrinkles"] = _band_from_score(wr.get("score"))
    pg = conditions.get("pigmentation", {}) or {}
    if pg.get("status") == "detected":
        bands["pigmentation"] = _pigmentation_band(pg)
    for c in ("dryness", "oiliness"):
        e = conditions.get(c, {}) or {}
        if e.get("status") == "detected":
            bands[c] = _band_from_score(e.get("score"))
    for c in ("redness", "enlarged_pores", "blackheads", "dark_spots"):
        e = conditions.get(c, {}) or {}
        if e.get("status") == "estimated":
            bands[c] = _band_from_score(e.get("score"))
    return bands, skin_type_label


def _collect_actives(bands: dict[str, str], skin_type_label: str | None) -> list[str]:
    picked: list[str] = []
    for cond in ("acne", "blackheads", "enlarged_pores", "redness",
                 "pigmentation", "dark_spots", "wrinkles"):
        band = bands.get(cond)
        if not band or band in ("low", "clear"):
            continue
        for a in CONDITION_ACTIVES.get(cond, {}).get(band, []):
            if a not in picked:
                picked.append(a)
    for a in SKIN_TYPE_ACTIVES.get(skin_type_label or "", []):
        if a not in picked:
            picked.append(a)
    return picked


def _barrier_repair_protocol(bands: dict[str, str], context: dict) -> dict | None:
    """Trigger a Barrier Repair protocol when the dryness signal is high AND the
    environment is low-humidity -- the classic dehydrated / compromised-barrier
    setup where piling on actives makes things worse."""
    dry_high = bands.get("dryness") == "elevated"
    hum = context.get("humidity_pct")
    low_humidity = isinstance(hum, (int, float)) and hum < 40
    if not (dry_high and low_humidity):
        return None
    return {
        "triggered": True,
        "reason": f"High dryness signal with low ambient humidity ({hum:.0f}%).",
        "protocol": [
            "Pause all exfoliating acids and retinoids for 5-7 days.",
            "Cleanse with a non-foaming cream or oil cleanser, lukewarm water only.",
            "Apply a humectant (glycerin / hyaluronic acid) to damp skin, then seal immediately.",
            "Layer a ceramide + fatty-acid barrier cream; add an occlusive (e.g. petrolatum) at night.",
            "Reintroduce one active at a time once skin feels comfortable, starting 1-2x/week.",
        ],
        "ingredients": ["Glycerin", "Hyaluronic acid", "Ceramides", "Squalane"],
    }


def _contraindication_engine(
    bands: dict[str, str], picked: list[str], context: dict | None = None
) -> tuple[list[dict], list[str], dict, dict]:
    """Returns (flags, mutated_actives, spf_block, state)."""
    context = context or {}
    flags: list[dict] = []
    picked = list(picked)
    redness_elevated = bands.get("redness") == "elevated"
    redness_present = bands.get("redness") in ("moderate", "elevated")

    if redness_elevated:
        removed = [a for a in picked if _is_exfoliant(a) and "PHA" not in a]
        picked = [a for a in picked if a not in removed]
        for sub in ("Azelaic acid 10%", "PHA (gluconolactone) 4%"):
            if sub not in picked:
                picked.append(sub)
        msg = ("Elevated erythema detected - "
               + (f"removed {', '.join(removed)}; " if removed else "")
               + "avoid AHA/BHA above 2% and at-home peels on a reactive barrier. "
                 "Azelaic acid and a low-% PHA are the safer exfoliation route; patch-test first.")
        flags.append({"rule": "avoid_high_strength_acids_with_erythema", "severity": "high", "message": msg})

    has_exf = any(_is_exfoliant(a) for a in picked)
    has_ret = any(_is_retinoid(a) for a in picked)

    if has_exf or has_ret:
        flags.append({
            "rule": "spf50_mandatory_with_photosensitisers", "severity": "high",
            "message": "Exfoliating acids and/or retinoids in this routine raise UV sensitivity - "
                       "Broad-Spectrum SPF 50 every morning is mandatory, reapplied every 2-3 hours of "
                       "daylight exposure.",
        })
    if has_ret and any("Benzoyl peroxide" in a for a in picked):
        flags.append({
            "rule": "separate_bpo_and_retinoid", "severity": "info",
            "message": "Use benzoyl peroxide in the AM and the retinoid in the PM - together they can "
                       "oxidise / degrade some retinoid formulations.",
        })
    # ---- pairwise contraindication matrix (Retinol+AHA/BHA, Vit C + BPO, ...) ----
    matrix_flags, force_pm = _apply_contraindication_matrix(picked)
    flags.extend(matrix_flags)

    # ---- photosensitive actives are routed to PM exclusively ----
    photosensitisers = sorted({a for a in picked if _is_photosensitising(a)})
    force_pm |= set(photosensitisers)
    if photosensitisers:
        flags.append({
            "rule": "photosensitisers_pm_only", "severity": "info",
            "actives": photosensitisers,
            "message": "Photosensitising actives (" + ", ".join(photosensitisers) + ") are scheduled for "
                       "the PM routine only; never apply them before sun exposure.",
        })

    # ---- Barrier Repair protocol (high dryness + low humidity) ----
    barrier_repair = _barrier_repair_protocol(bands, context)
    if barrier_repair:
        picked = [a for a in picked if not (_is_exfoliant(a) or _is_retinoid(a))]
        for a in barrier_repair["ingredients"]:
            if a not in picked:
                picked.append(a)
        photosensitisers = [a for a in photosensitisers if a in picked]
        force_pm &= set(picked)
        flags.append({
            "rule": "barrier_repair_protocol", "severity": "high",
            "message": barrier_repair["reason"] + " Started a Barrier Repair protocol: exfoliants and "
                       "retinoids are paused; focus on humectants, ceramides and an occlusive until the "
                       "barrier recovers.",
        })
        has_exf = any(_is_exfoliant(a) for a in picked)
        has_ret = any(_is_retinoid(a) for a in picked)

    spf = {
        "required": bool(has_exf or has_ret),
        "product": "Broad-Spectrum SPF 50" if (has_exf or has_ret) else "Broad-Spectrum SPF 30+",
        "reason": ("Mandatory - exfoliating acids / retinoids in this routine increase photosensitivity."
                   if (has_exf or has_ret)
                   else "Daily broad-spectrum SPF is the single biggest lever against photoageing and pigmentation."),
    }
    state = {"has_exfoliant": has_exf, "has_retinoid": has_ret,
             "redness_elevated": redness_elevated, "redness_present": redness_present,
             "force_pm": force_pm, "photosensitisers": photosensitisers,
             "barrier_repair": barrier_repair}
    return flags, picked, spf, state


def _active_record(name: str) -> dict:
    r = ACTIVE_LIBRARY.get(name, {})
    return {"name": name, "class": r.get("class"), "targets": r.get("targets", []),
            "use": r.get("use", "AM/PM"), "evidence": r.get("evidence"), "caution": r.get("caution")}


def _split_am_pm(picked: list[str], skin_type_label: str | None,
                 spf: dict, state: dict) -> tuple[list[dict], list[dict]]:
    def use_of(a: str) -> str:
        return ACTIVE_LIBRARY.get(a, {}).get("use", "AM/PM")

    mdry = skin_type_label == "dry"
    force_pm: set[str] = set(state.get("force_pm") or ())
    moist_actives = [a for a in picked if a in _MOISTURISER_ACTIVES]

    am_treat = [a for a in picked
                if "AM" in use_of(a) and not _is_photosensitising(a)
                and a not in force_pm and a not in _MOISTURISER_ACTIVES]
    am_treat.sort(key=lambda a: (0 if "Vitamin C" in a else 1))

    def prio(a: str) -> int:
        if _is_retinoid(a):
            return 0
        if _is_exfoliant(a):
            return 1
        if "Benzoyl peroxide" in a:
            return 2
        return 9

    pm_pool = [a for a in picked if "PM" in use_of(a) or a in force_pm]
    pm_primary = sorted([a for a in pm_pool if prio(a) < 9], key=prio)[:1]
    pm_secondary = [a for a in pm_pool
                    if a not in pm_primary and a not in am_treat and a not in _MOISTURISER_ACTIVES]

    am: list[dict] = [{"step": "Cleanser",
                       "detail": "Gentle, fragrance-free cleanser (or just water if skin feels comfortable).",
                       "actives": []}]
    if am_treat:
        am.append({"step": "Treatment serum",
                   "detail": "Apply to clean, dry skin; wait ~1 minute before the next layer.",
                   "actives": am_treat})
    am.append({"step": "Moisturiser",
               "detail": "Rich cream with ceramides." if mdry else "Lightweight, non-comedogenic lotion / gel-cream.",
               "actives": moist_actives})
    am.append({"step": "Sunscreen",
               "detail": f"{spf['product']} as the final AM step. {spf['reason']}", "actives": []})

    pm: list[dict] = [{"step": "Cleanser",
                       "detail": "Double cleanse if you wore SPF or make-up; otherwise one gentle cleanse.",
                       "actives": []}]
    if pm_primary:
        a = pm_primary[0]
        ramp = _is_retinoid(a) or _is_exfoliant(a)
        freq = "2-3x/week, building to nightly as tolerated" if ramp else "nightly"
        detail = f"Apply {freq} to fully dry skin."
        caution = ACTIVE_LIBRARY.get(a, {}).get("caution")
        if caution:
            detail += f" {caution}"
        pm.append({"step": f"Primary active - {a}", "detail": detail.strip(), "actives": [a]})
    if pm_secondary:
        detail = "Layer after the primary active, or use on its 'off' nights."
        if state["redness_present"]:
            detail += " Buffer with moisturiser first if skin is reactive."
        pm.append({"step": "Supporting serum", "detail": detail, "actives": pm_secondary})
    pm.append({"step": "Moisturiser",
               "detail": ("Rich barrier cream; add a facial oil on top if very dry."
                          if mdry else "Barrier-supporting moisturiser."),
               "actives": moist_actives})
    return am, pm


_EVIDENCE_BASIS = (
    "Conditions the models / CV engine flagged are mapped to actives with established "
    "cosmetic-dermatology evidence: azelaic acid & niacinamide for erythema and post-inflammatory "
    "pigmentation; salicylic acid (BHA) for comedones and enlarged pores; retinoids & peptides for "
    "photoageing; hyaluronic acid & ceramides for a dry-leaning barrier. This is guidance, not a "
    "prescription - introduce one new active at a time and patch-test."
)


def _evidence_routines(conditions: dict, context: dict | None = None) -> dict:
    """The upgraded structured output: AM/PM routines, active list, contraindication
    flags and the SPF rule. Additive - the legacy keys are built separately."""
    bands, skin_type_label = _bands_from_conditions(conditions)
    picked = _collect_actives(bands, skin_type_label)
    flags, picked, spf, state = _contraindication_engine(bands, picked, context or {})
    am, pm = _split_am_pm(picked, skin_type_label, spf, state)
    return {
        "am_routine": am,
        "pm_routine": pm,
        "actives": [_active_record(a) for a in picked],
        "contraindications": flags,
        "contraindication_matrix": [
            {"rule": r["id"], "action": r["action"], "severity": r["severity"], "message": r["message"]}
            for r in CONTRAINDICATION_MATRIX
        ],
        "photosensitisers_pm_only": state.get("photosensitisers", []),
        "barrier_repair": state.get("barrier_repair"),
        "spf": spf,
        "evidence_basis": _EVIDENCE_BASIS,
    }


_EMPTY_EVIDENCE = {
    "am_routine": [], "pm_routine": [], "actives": [], "contraindications": [],
    "contraindication_matrix": [], "photosensitisers_pm_only": [], "barrier_repair": None,
    "spf": {"required": False, "product": "Broad-Spectrum SPF 30+",
            "reason": "Daily broad-spectrum SPF is the single biggest lever against photoageing."},
    "evidence_basis": "Legacy record - structured AM/PM routine not available for this result shape.",
}


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

    # ---- evidence-based AM/PM routine + contraindication engine (additive) ----
    evidence = _evidence_routines(conditions, context)
    _u(ingredients, [a["name"] for a in evidence["actives"]])
    _u(routine, [s["step"] for s in evidence["am_routine"]])
    _u(routine, [s["step"] for s in evidence["pm_routine"]])
    if evidence["spf"]["required"]:
        _u(routine, [f"{evidence['spf']['product']} every morning (mandatory with this routine)"])
    for f in evidence["contraindications"]:
        if f["severity"] == "high":
            notes.append(f"Safety: {f['message']}")
    if evidence.get("barrier_repair"):
        br = evidence["barrier_repair"]
        per_condition["barrier_repair"] = [br["reason"], *br["protocol"]]
        summary.append("Barrier Repair protocol started: actives paused while the skin barrier recovers.")

    return {
        "summary": summary,
        "per_condition": per_condition,
        "routine": routine,
        "ingredients": ingredients,
        "context_notes": notes,
        "disclaimer": DISCLAIMER,
        **evidence,
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
            "ingredients": ingredients, "context_notes": notes, "disclaimer": DISCLAIMER,
            **_EMPTY_EVIDENCE}
