"""
backend/app/schemas/prediction.py

Response shapes for /predict and /history.

ACTIVE pipeline = multi-condition analysis from a stack of PUBLIC pretrained
models (see app.services.analysis_service). A prediction's `results` now looks
like:

    {
      "analysis_version": "skinintel-multimodel-v1",
      "face_detected": true,
      "image_size": [w, h],
      "conditions": {
        "acne":         {"status": "detected", "method": "classification+detection",
                         "label": "...", "score": 0.6, "confidence": 0.49,
                         "detail": {"detections": [...], "lesion_count": 6, ...},
                         "model": "acne_severity+acne_detector", "note": "..."},
        "skin_type":    {"status": "detected", "method": "classification", ...},
        "dryness":      {...}, "oiliness": {...}, "wrinkles": {...},
        "redness":      {"status": "estimated", "method": "heuristic", ...},
        "enlarged_pores": {...},
        "pigmentation": {"status": "not_available", ...},
        "blackheads":   {"status": "not_available", ...},
        "dark_spots":   {"status": "not_available", ...}
      },
      "model_versions": {...}
    }

`results` is a free-form dict on the response models on purpose: /history may
contain THREE historical shapes -- multi-model (current), YOLO-only
({detections, counts, image_size}), and legacy EfficientNet
({skin_type: {label, confidence}}). All pass through untouched.

`ConditionEntry` / `SkinAnalysis` below document + validate the CURRENT shape
where it is created. `Detection` is still used inside `conditions.acne.detail`.

`ScanContext` is the optional questionnaire + environment inputs. It never
reaches any model; it only tailors the rule-based recommendation. See
app.services.recommendation_engine.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConditionStatus = Literal["detected", "estimated", "not_available", "error"]


class Detection(BaseModel):
    """One acne bounding box. Serialised with key "class" (not "class_")."""
    class_: str = Field(serialization_alias="class", validation_alias="class")
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2] in source-image pixels")

    model_config = ConfigDict(populate_by_name=True)


class ConditionEntry(BaseModel):
    """Canonical per-condition result in `results.conditions`."""
    status: ConditionStatus
    method: str | None = None                 # classification | detection | segmentation | heuristic | ...
    label: str | None = None
    score: float | None = None                # 0-1 severity / presence
    confidence: float | None = None           # model confidence where meaningful
    detail: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    note: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class SkinHealthIndex(BaseModel):
    """Deterministic 0-100 score derived from the model-backed + estimated
    conditions (see analysis_service.compute_skin_health_index for the exact
    formula). `value` is None when no model-backed condition succeeded."""
    value: int | None = None
    band: str                                 # "excellent" | "good" | "fair" | "needs_attention" | "not_enough_data"
    basis: list[str] = Field(default_factory=list)   # which conditions contributed
    formula: str = ""


class SkinAnalysis(BaseModel):
    """The CURRENT `results` payload (multi-condition pretrained-model stack)."""
    analysis_version: str
    face_detected: bool
    image_size: list[int] = Field(min_length=2, max_length=2)
    conditions: dict[str, ConditionEntry]
    skin_health_index: SkinHealthIndex | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())


class ConditionResult(BaseModel):
    """LEGACY -- one head of the old EfficientNet skin-type classifier. Retained
    only so old /history records remain documented; nothing produces it now."""
    label: str
    confidence: float


class ScanContext(BaseModel):
    """Optional questionnaire + environment. Never reaches any model; only
    tailors the rule-based recommendation. Every field optional; missing
    environment values stay null (never invented)."""
    age: int | None = None
    gender: Literal["female", "male", "nonbinary", "prefer_not_to_say"] | None = None
    location: Literal["temperate", "humid_tropical", "cold_dry", "desert", "urban_polluted"] | None = None
    job_type: Literal["indoor_ac", "outdoor", "hybrid"] | None = None
    sleep_hours: float | None = None
    water_intake_liters: float | None = None
    stress_level: Literal["low", "medium", "high"] | None = None
    sun_exposure: Literal["low", "moderate", "high"] | None = None
    # --- environment (hybrid: current-location / select-location / manual) ---
    env_mode: Literal["current_location", "select_location", "manual"] | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    uv_index: float | None = None
    air_quality_aqi: float | None = None
    place_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    model_config = ConfigDict(extra="ignore")


class PredictionResult(BaseModel):
    """Response of the live /predict call -- the CURRENT multi-condition shape."""
    id: int
    image_id: int
    image_url: str | None = None
    results: SkinAnalysis
    recommendation: dict | None = None
    context: ScanContext | None = None
    model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class HistoryRecord(BaseModel):
    """Response row of /history. `results` is free-form: history may hold
    multi-model, YOLO-only, or legacy EfficientNet shapes -- all returned
    untouched so old rows never break the endpoint."""
    id: int
    image_id: int
    image_url: str | None = None
    results: dict
    recommendation: dict | None = None
    context: ScanContext | None = None
    model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class UserProfile(BaseModel):
    """GET /me -- authenticated user + their most recent stored context."""
    id: int
    email: str
    full_name: str | None = None
    created_at: datetime | None = None
    scan_count: int = 0
    latest_context: ScanContext | None = None

    model_config = ConfigDict(from_attributes=True)


class WeatherReading(BaseModel):
    """Normalised environment reading. Any field may be null (never invented)."""
    source: str                               # "open-meteo" | "manual" | "unavailable"
    place_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    uv_index: float | None = None
    air_quality_aqi: float | None = None
    air_quality_label: str | None = None
    fetched_at: datetime | None = None
    note: str | None = None
