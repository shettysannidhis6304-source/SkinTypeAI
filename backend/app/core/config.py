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

    # --- ACTIVE: multi-condition pretrained-model analysis ---
    # Every model below is a PUBLIC pretrained checkpoint (nothing trained here).
    # Provenance/licences: weights/skin_analysis_models.metadata.json + models/skin_analysis/registry.py
    # CPU only on this machine. Model dirs live under weights/hf_models/ (git-ignored).
    ml_device: str = "cpu"
    ml_torch_threads: int = 4

    # --- Custom multi-task model (trained by this project; replaces the HF ViTs) ---
    # When use_custom_model is True AND the checkpoint exists, skin_type / dryness /
    # oiliness / acne-severity / pigmentation are produced by our own trained model
    # (models/custom_multi_task.py). The HF path stays as an automatic fallback for
    # when the checkpoint is missing. See training/train_custom.py.
    use_custom_model: bool = True
    custom_model_weights_path: str = str(PROJECT_ROOT / "weights" / "custom_skin_model.pt")

    skin_type_model_dir: Path = PROJECT_ROOT / "weights" / "hf_models" / "skin_type_vit"
    acne_severity_model_dir: Path = PROJECT_ROOT / "weights" / "hf_models" / "acne_severity_vit"
    acne_detector_weights: Path = PROJECT_ROOT / "weights" / "hf_models" / "acne_detector_yolo" / "acne_detector_best.pt"
    acne_detector_conf: float = 0.25
    acne_detector_iou: float = 0.45
    acne_detector_imgsz: int = 640
    wrinkle_weights: Path = PROJECT_ROOT / "weights" / "hf_models" / "wrinkle_unet" / "wrinkle_model.safetensors"
    wrinkle_enabled: bool = True          # CC BY-NC-SA weights; academic use only. Set false to disable.
    wrinkle_threshold: float = 0.5
    heuristics_enabled: bool = True       # redness + pores ESTIMATES (never labelled "detected")

    # --- Legacy: standalone ACNE04 YOLO11 detector (own-trained proof artifact) ---
    # Not on the active path (the acne detector is acne_detector_weights above);
    # weights/skin_yolo11n.pt is kept as an academic-integrity record only.
    yolo_weights_path: Path = PROJECT_ROOT / "weights" / "skin_yolo11n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_imgsz: int = 640
    yolo_device: str = "cpu"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)


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
