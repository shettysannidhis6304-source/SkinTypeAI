"""
backend/app/models_db/prediction.py

One row per inference run. `results` stores the full multi-head model output
(per-condition predicted class + confidence) as JSON -- see
app.schemas.prediction for the shape written here. Individual condition
columns are NOT duplicated out of the JSON blob for v1; add typed columns
later if you need to query/filter by a specific condition efficiently.

`context` is the optional questionnaire (age/gender/location/lifestyle) the
user filled in before this scan, or None if they skipped it -- stored as-is
so a past recommendation stays reproducible/auditable. It never influenced
`results` (the model only ever sees the photo); it only ever influenced the
rule-based recommendation computed at the time.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models_db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), unique=True, nullable=False)
    results: Mapped[dict] = mapped_column(JSON, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="predictions")
    image: Mapped["Image"] = relationship(back_populates="prediction")
