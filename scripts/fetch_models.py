"""
scripts/fetch_models.py

Downloads the PUBLIC pretrained models used by the multi-condition analysis
into weights/hf_models/ (git-ignored). Run once after `pip install -r requirements.txt`.

Nothing is trained. Provenance + licences: weights/skin_analysis_models.metadata.json

    .venv-new\\Scripts\\python.exe scripts\\fetch_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "weights" / "hf_models"

# (hf repo id, local subdir, [allow_patterns] or None for a single file, single_file_name)
JOBS = [
    ("dima806/skin_types_image_detection", "skin_type_vit",
     ["config.json", "model.safetensors", "preprocessor_config.json", "README.md"], None),
    ("imfarzanansari/skintelligent-acne", "acne_severity_vit",
     ["config.json", "model.safetensors", "preprocessor_config.json", "README.md"], None),
    ("mufasabrownie/glowlytics-skin-models", "acne_detector_yolo", None, "acne_detector_best.pt"),
    ("rmsandu/ffhq-wrinkle-unet", "wrinkle_unet", None, "wrinkle_model.safetensors"),
    ("rmsandu/ffhq-wrinkle-unet", "wrinkle_unet", None, "face_segmentation.safetensors"),
]


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub missing -- run: pip install -r requirements.txt")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    for repo, sub, patterns, single in JOBS:
        target = DEST / sub
        try:
            if single:
                p = hf_hub_download(repo, single, local_dir=str(target))
            else:
                p = snapshot_download(repo, local_dir=str(target), allow_patterns=patterns)
            size = sum(f.stat().st_size for f in Path(p if single else target).rglob("*") if f.is_file()) \
                if not single else Path(p).stat().st_size
            print(f"OK  {repo}:{single or '(snapshot)'}  -> {target}  ({size / 1e6:.0f} MB)")
        except Exception as e:  # pragma: no cover
            print(f"FAIL {repo}:{single or ''}  {e!r}")
            return 2

    print("\nAll models fetched into weights/hf_models/ (git-ignored).")
    print("Start the app:  start-servers.bat   (or run-backend.bat + run-frontend.bat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
