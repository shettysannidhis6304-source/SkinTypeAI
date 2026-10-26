"""
backend/app/routers/upload.py

/upload                 - save a photo, create an Image row.
/predict/{id}           - run the ACTIVE pipeline (multi-condition analysis --
                          see app.services.analysis_service) on a previously
                          uploaded image and store the unified result.

                          Image quality is ADVISORY (a 1-10 rating), never a hard
                          reject. When ?force_proceed=false (default) and the
                          rating is <= 6/10, this returns HTTP 200 with
                          {"status": "advisory", "quality": {...},
                           "temp_image_id": "<id>"} instead of running analysis.
                          The client shows a "proceed / retake" dialog and, to
                          proceed, re-calls with ?force_proceed=true using the
                          same id (the uploaded frame is the cache -- no re-upload).

                          Returns 503 if the model file is missing. Nothing is
                          fabricated: conditions with no model are "not_available";
                          heuristics are "estimated".
/history                - list this user's past predictions.
"""
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import resolve_model_weights, resolve_upload_path, settings
from app.core.security import get_current_user
from app.models_db.base import get_db
from app.models_db.image import Image
from app.models_db.prediction import Prediction
from app.models_db.user import User
from app.schemas.prediction import HistoryRecord, PredictionResult, ScanContext

router = APIRouter(tags=["scan"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# The advisory dialog appears at or below this rating (assess_image_quality is 1-10).
QUALITY_ADVISORY_AT_OR_BELOW = 6

# A dedicated cache dir for uploaded/captured frames pending an advisory decision.
# (The Image row + file already persist the frame, so temp_image_id == image id;
#  this dir just satisfies the "uploads/temp/" contract for tooling that looks.)
_TEMP_DIR = settings.upload_dir / "temp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_image(
    file: UploadFile,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type: {file.content_type}")

    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    stored_name = f"{user.id}_{uuid.uuid4().hex}{ext}"
    dest_path = settings.upload_dir / stored_name

    with dest_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Store only the filename (portable across machines); resolve via
    # resolve_upload_path() on read. Legacy rows with absolute paths still work.
    image = Image(user_id=user.id, file_path=stored_name, original_filename=file.filename or stored_name)
    db.add(image)
    db.commit()
    db.refresh(image)
    return {"image_id": image.id, "filename": image.original_filename}


@router.post("/predict/{image_id}", response_model=None)
def predict(
    image_id: int,
    context: ScanContext | None = None,
    force_proceed: bool = Query(
        False, description="Skip the advisory quality check and run analysis regardless."),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionResult | dict[str, Any]:
    """`context` is the optional questionnaire + environment inputs -- entirely
    skippable, and even when filled in it never reaches any model. It only
    tailors the rule-based recommendation. Pass no body / null to skip.

    Quality is advisory: if `force_proceed` is false and the image rates
    <= 6/10, this returns 200 with an advisory payload (no analysis run).
    Re-call with `force_proceed=true` and the same `image_id` to proceed.

    Idempotent: `predictions.image_id` is UNIQUE, so a repeat /predict for an
    already-analysed image returns the existing prediction (no 500).
    """
    image = db.get(Image, image_id)
    if image is None or image.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")

    model_ckpt = resolve_model_weights()
    if not model_ckpt.exists():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Analysis model not available (missing {model_ckpt}). "
            "Train it with training/train_skin_analysis.py.",
        )

    # Imported lazily so the app still boots (and /upload still works) before
    # the model files exist.
    from app.services.analysis_service import (
        ANALYSIS_VERSION, assess_image_quality, prepare_face, run_analysis,
    )
    from app.services.recommendation_engine import build_recommendation

    context_dict = context.model_dump(exclude_none=True) if context else None
    img_path = str(resolve_upload_path(image.file_path))

    def _as_result(pred: Prediction) -> PredictionResult:
        try:
            rec = build_recommendation(pred.results, pred.context) if isinstance(pred.results, dict) else None
        except Exception:  # pragma: no cover - defensive
            rec = None
        return PredictionResult(
            id=pred.id, image_id=image.id, image_url=f"/images/{image.id}",
            results=pred.results, recommendation=rec, context=pred.context,
            model_version=pred.model_version, created_at=pred.created_at,
        )

    # ---- idempotency: image already analysed -> return the stored prediction ----
    existing = db.query(Prediction).filter(Prediction.image_id == image.id).first()
    if existing is not None:
        return _as_result(existing)

    # One FaceMesh pass, shared by the quality pose check and the OpenCV heuristics.
    prep = prepare_face(img_path)
    quality = assess_image_quality(img_path, prep=prep)

    # ---- advisory gate: never rejects, just asks the user to confirm ----
    if not force_proceed and int(quality.get("score", 10)) <= QUALITY_ADVISORY_AT_OR_BELOW:
        return {
            "status": "advisory",
            "quality": {
                "score": quality["score"],
                "max_score": quality["max_score"],
                "rating": quality["rating"],
                "message": quality["message"],
                "advisory": quality.get("advisory"),
                "is_suboptimal": quality["is_suboptimal"],
                "metrics": quality.get("metrics", {}),
            },
            "temp_image_id": str(image.id),
            "image_id": image.id,
            "image_url": f"/images/{image.id}",
        }

    results = run_analysis(img_path, quality=quality, prep=prep)   # quality embedded in results
    recommendation = build_recommendation(results, context_dict)

    prediction = Prediction(
        user_id=user.id,
        image_id=image.id,
        results=results,           # unified SkinAnalysis dict -> JSON column, no schema change
        context=context_dict,
        model_version=ANALYSIS_VERSION,
    )
    db.add(prediction)
    try:
        db.commit()
        db.refresh(prediction)
    except IntegrityError:
        # a concurrent /predict for the same image won the race -> return theirs
        db.rollback()
        raced = db.query(Prediction).filter(Prediction.image_id == image.id).first()
        if raced is not None:
            return _as_result(raced)
        raise

    return PredictionResult(
        id=prediction.id,
        image_id=image.id,
        image_url=f"/images/{image.id}",
        results=results,
        recommendation=recommendation,
        context=context_dict,
        model_version=prediction.model_version,
        created_at=prediction.created_at,
    )


@router.get("/history", response_model=list[HistoryRecord])
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[HistoryRecord]:
    predictions = (
        db.query(Prediction)
        .filter(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    # The recommendation is not persisted (no schema change); it is a pure
    # rule-based function of the stored `results` + `context`, so we recompute
    # it here. That keeps the Plan / Products screens working when a scan is
    # opened from history (e.g. after a page reload). Only attempted for the
    # current unified result shape; legacy blobs get recommendation=None.
    try:
        from app.services.recommendation_engine import build_recommendation
    except Exception:  # pragma: no cover - defensive
        build_recommendation = None

    def _recommend(p: Prediction) -> dict | None:
        if build_recommendation is None or not isinstance(p.results, dict):
            return None
        if "conditions" not in p.results:
            return None
        try:
            return build_recommendation(p.results, p.context)
        except Exception:
            return None

    # `results` passed straight through: new YOLO {detections,...} and legacy
    # EfficientNet {skin_type:{...}} blobs both serialise fine (results is a free dict here).
    return [
        HistoryRecord(
            id=p.id,
            image_id=p.image_id,
            image_url=f"/images/{p.image_id}",
            results=p.results,
            recommendation=_recommend(p),
            context=p.context,
            model_version=p.model_version,
            created_at=p.created_at,
        )
        for p in predictions
    ]
