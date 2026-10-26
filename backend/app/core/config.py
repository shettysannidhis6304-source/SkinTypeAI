"""
backend/app/core/config.py

Central app settings, loaded from environment variables / an optional .env
file (SkinTypeAI/.env) via pydantic-settings. Import `settings` everywhere
instead of reading os.environ directly, so there's one source of truth for
paths, secrets, and tunables.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[1]      # SkinTypeAI/backend/app
BACKEND_DIR = APP_DIR.parent                        # SkinTypeAI/backend
PROJECT_ROOT = BACKEND_DIR.parent                    # SkinTypeAI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),   # allow `model_weights_path` (not a pydantic model attr)
    )

    # --- Auth ---
    jwt_secret_key: str = "dev-secret-change-me"  # override via .env for anything beyond local dev
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 day

    # --- Database ---
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'database' / 'skintype.db'}"

    # --- Uploads ---
    upload_dir: Path = BACKEND_DIR / "uploads"
    max_upload_size_mb: int = 10

    ml_device: str = "cpu"
    ml_torch_threads: int = 4

    # --- ACTIVE: from-scratch SkinAnalysisModel (models/skin_analysis_model.py) ---
    # 403,656-param CNN (4 conv stages + 2 SE blocks) trained 100% from scratch
    # (training/train_skin_analysis.py) -- NO pretrained weights, NO timm, NO
    # ImageNet. Produces 5 characteristics: acne | wrinkles | oiliness | dryness |
    # pigmentation. skin_type is DERIVED deterministically from oiliness + dryness.
    # The remaining 4 (redness / dark_spots / enlarged_pores / blackheads) come
    # from models/skin_analysis/opencv_engine.py::OpenCVSkinAnalyzer (deterministic
    # OpenCV, not machine learning).
    #
    # The retired stacks (MobileNetV3 multi-task ViT, the dima806 / skintelligent
    # ViTs, the glowlytics YOLOv8 acne detector, the FFHQ-Wrinkle U-Net, the
    # ACNE04 YOLO11 proof) have been removed -- their code lives under
    # legacy_models/ and scripts/benchmark_system.py hardcodes its own paths.
    model_weights_path: str = "weights/skin_analysis_model.pt"   # resolved via resolve_model_weights()


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)


def resolve_model_weights() -> Path:
    """Absolute path to the active SkinAnalysisModel checkpoint. `model_weights_path`
    may be relative (to the project root) or absolute."""
    p = Path(settings.model_weights_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_upload_path(stored: str) -> Path:
    """Turn an ``Image.file_path`` value into an absolute path on this machine.

    New rows store just the filename (portable); legacy rows stored a full
    absolute path. Both resolve here, and an absolute legacy path from another
    machine falls back to its basename under ``upload_dir``.
    """
    p = Path(stored)
    if not p.is_absolute():
        return settings.upload_dir / p
    if p.exists():
        return p
    return settings.upload_dir / p.name
