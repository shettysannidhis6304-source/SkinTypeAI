"""
backend/app/routers/upload.py

/upload          - save a photo, create an Image row.
/predict/{id}    - run the ACTIVE pipeline (multi-condition pretrained-model
                    analysis -- see app.services.analysis_service) on a
                    previously uploaded image and store the unified result.
                    Returns 503 if the model files are missing. Nothing is
                    fabricated: conditions with no model are marked
                    "not_available"; heuristics are marked "estimated".
/history         - list this user's past predictions. Round-trips all three
                    historical shapes (multi-model / YOLO-only / EfficientNet).
"""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import resolve_upload_path, settings
from app.core.security import get_current_user
from app.models_db.base import get_db
from app.models_db.image import Image
from app.models_db.prediction import Prediction
from app.models_db.user import User
from app.schemas.prediction import HistoryRecord, PredictionResult, ScanContext

router = APIRouter(tags=["scan"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


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


@router.post("/predict/{image_id}", response_model=PredictionResult)
def predict(
    image_id: int,
    context: ScanContext | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionResult:
    """`context` is the optional questionnaire + environment inputs -- entirely
    skippable, and even when filled in it never reaches any model. It only
    tailors the rule-based recommendation. Pass no body / null to skip.
    """
    image = db.get(Image, image_id)
    if image is None or image.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")

    missing = [
        p for p in (
            settings.skin_type_model_dir / "config.json",
            settings.acne_severity_model_dir / "config.json",
            settings.acne_detector_weights,
        ) if not p.exists()
    ]
    if missing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Analysis models not available. Missing: "
            + ", ".join(str(m) for m in missing)
            + ". See weights/skin_analysis_models.metadata.json for how to fetch them.",
        )

    # Imported lazily so the app still boots (and /upload still works) before
    # the model files exist.
    from app.services.analysis_service import ANALYSIS_VERSION, run_analysis
    from app.services.recommendation_engine import build_recommendation

    context_dict = context.model_dump(exclude_none=True) if context else None

    results = run_analysis(str(resolve_upload_path(image.file_path)))
    recommendation = build_recommendation(results, context_dict)

    prediction = Prediction(
        user_id=user.id,
        image_id=image.id,
        results=results,           # unified SkinAnalysis dict -> JSON column, no schema change
        context=context_dict,
        model_version=ANALYSIS_VERSION,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

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
