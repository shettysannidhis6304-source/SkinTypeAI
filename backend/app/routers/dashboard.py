"""
backend/app/routers/dashboard.py

Small additive endpoints the SkinIntelli dashboard needs. No DB schema change.

  GET  /me                        - authenticated user + their latest stored context
  GET  /images/{image_id}         - the user's own uploaded photo (thumbnail / overlay)
  GET  /environment/weather       - live temp/humidity/UV/air-quality via Open-Meteo
                                     (keyless). ?lat=&lon=  OR  ?place=City,Country
  GET  /environment/geocode       - Country/State/City search -> coordinates (Open-Meteo)

Environment data is fetched server-side (avoids browser CORS + keeps it optional).
Any value Open-Meteo does not return stays null -- never invented.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import resolve_upload_path, settings
from app.core.security import get_current_user
from app.models_db.base import get_db
from app.models_db.image import Image
from app.models_db.prediction import Prediction
from app.models_db.user import User
from app.schemas.prediction import ScanContext, UserProfile, WeatherReading

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])

_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_AQI = "https://air-quality-api.open-meteo.com/v1/air-quality"
_OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
_HTTP_TIMEOUT = 6.0


@router.get("/me", response_model=UserProfile)
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UserProfile:
    q = db.query(Prediction).filter(Prediction.user_id == user.id)
    latest = q.order_by(Prediction.created_at.desc()).first()
    latest_ctx = None
    if latest is not None and latest.context:
        try:
            latest_ctx = ScanContext(**latest.context)
        except Exception:  # pragma: no cover - defensive
            latest_ctx = None
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        created_at=getattr(user, "created_at", None),
        scan_count=q.count(),
        latest_context=latest_ctx,
    )


@router.get("/images/{image_id}")
def get_image(
    image_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    image = db.get(Image, image_id)
    if image is None or image.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    p = resolve_upload_path(image.file_path)
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image file missing on disk")
    return FileResponse(str(p))


def _aqi_label(aqi: float | None) -> str | None:
    if aqi is None:
        return None
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy (sensitive groups)"
    if aqi <= 200:
        return "Unhealthy"
    return "Very unhealthy"


@router.get("/environment/weather", response_model=WeatherReading)
async def weather(
    lat: float | None = Query(default=None),
    lon: float | None = Query(default=None),
    place: str | None = Query(default=None, description="e.g. 'Bengaluru, India'"),
    user: User = Depends(get_current_user),
) -> WeatherReading:
    """Live environment reading. Provide lat/lon OR a place string. Returns
    whatever Open-Meteo provides; missing fields stay null."""
    place_name = place
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            if (lat is None or lon is None) and place:
                g = await client.get(_OPEN_METEO_GEO, params={"name": place.split(",")[0].strip(), "count": 1})
                g.raise_for_status()
                hits = g.json().get("results") or []
                if not hits:
                    return WeatherReading(source="unavailable", place_name=place,
                                          note="Location not found by the geocoder.")
                lat, lon = hits[0]["latitude"], hits[0]["longitude"]
                place_name = ", ".join(x for x in (hits[0].get("name"), hits[0].get("admin1"),
                                                   hits[0].get("country")) if x)
            if lat is None or lon is None:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    "Provide lat & lon, or a place name.")

            w = await client.get(_OPEN_METEO, params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,uv_index",
            })
            w.raise_for_status()
            cur = w.json().get("current", {})

            aqi = None
            try:
                a = await client.get(_OPEN_METEO_AQI, params={
                    "latitude": lat, "longitude": lon, "current": "us_aqi",
                })
                a.raise_for_status()
                aqi = a.json().get("current", {}).get("us_aqi")
            except Exception as e:  # pragma: no cover
                logger.info("AQI lookup failed: %s", e)

        return WeatherReading(
            source="open-meteo",
            place_name=place_name,
            latitude=lat, longitude=lon,
            temperature_c=cur.get("temperature_2m"),
            humidity_pct=cur.get("relative_humidity_2m"),
            uv_index=cur.get("uv_index"),
            air_quality_aqi=aqi,
            air_quality_label=_aqi_label(aqi),
            fetched_at=datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("weather lookup failed: %s", e)
        return WeatherReading(source="unavailable", place_name=place_name,
                              latitude=lat, longitude=lon,
                              note="Live weather is unavailable right now (offline or the "
                                   "weather service did not respond). Enter values manually if needed.")


@router.get("/environment/geocode")
async def geocode(
    q: str = Query(min_length=2, description="City or place name"),
    user: User = Depends(get_current_user),
) -> dict:
    """Lightweight place search for the 'Select Location' mode."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.get(_OPEN_METEO_GEO, params={"name": q, "count": 8})
            r.raise_for_status()
            hits = r.json().get("results") or []
    except Exception as e:
        logger.warning("geocode failed: %s", e)
        return {"results": [], "note": "Geocoder unavailable (offline?)."}
    return {"results": [
        {
            "name": h.get("name"),
            "admin1": h.get("admin1"),
            "country": h.get("country"),
            "latitude": h.get("latitude"),
            "longitude": h.get("longitude"),
            "label": ", ".join(x for x in (h.get("name"), h.get("admin1"), h.get("country")) if x),
        }
        for h in hits
    ]}
