"""
models/skin_analysis/registry.py

Central provenance + configuration for every pretrained model used by
SkinTypeAI's multi-condition analysis. This is the single source of truth for
"which model, from where, under what licence, covering which condition".

NOTHING here was trained by this project. Every entry points at publicly
downloadable weights. Metrics recorded here are ATTRIBUTED to their source and
are NOT independently reproduced unless the `verified` note says so.

The same content is mirrored (human-readable) in
weights/skin_analysis_models.metadata.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_DIR = PROJECT_ROOT / "weights" / "hf_models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    conditions: list[str]                 # which of the 9 targets this feeds
    task: str                             # classification | detection | segmentation | heuristic
    source_url: str
    architecture: str
    classes: list[str]
    local_path: Path
    weight_files: list[str]
    weight_format: str
    approx_size_mb: int
    framework: str
    input_size: int
    license_: str
    academic_use: str
    commercial_use: str
    training_data: str
    reported_metrics: str                 # verbatim from source, tagged
    metric_provenance: str                # "author-reported" | "from repo results file" | "verified this project"
    evidence: str
    notes: str = ""
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    "skin_type": ModelSpec(
        key="skin_type",
        conditions=["skin_type", "dryness", "oiliness"],
        task="classification",
        source_url="https://huggingface.co/dima806/skin_types_image_detection",
        architecture="ViT-base-patch16-224 (ViTForImageClassification)",
        classes=["dry", "normal", "oily"],
        local_path=HF_DIR / "skin_type_vit",
        weight_files=["model.safetensors", "config.json", "preprocessor_config.json"],
        weight_format="safetensors",
        approx_size_mb=343,
        framework="transformers / torch (CPU)",
        input_size=224,
        license_="Apache-2.0",
        academic_use="yes",
        commercial_use="yes",
        training_data="Kaggle 'Oily, Dry and Normal Skin Types' dataset (facial photos)",
        reported_metrics="accuracy 0.653, macro-F1 0.653 on a 1529-image test split (per-class P/R ~0.64-0.68)",
        metric_provenance="author-reported (classification report in model card + linked Kaggle notebook)",
        evidence="565 downloads / 17 likes; config + trainer_args present; task is inherently fuzzy so ~65% is plausible",
        notes="Domain: ordinary facial selfies. 'dry' output also used as the dryness signal; 'oily' as an oiliness indicator.",
    ),
    "acne_severity": ModelSpec(
        key="acne_severity",
        conditions=["acne"],
        task="classification",
        source_url="https://huggingface.co/imfarzanansari/skintelligent-acne",
        architecture="ViT-base-patch16-224 (ViTForImageClassification)",
        classes=["level -1", "level 0", "level 1", "level 2", "level 3"],
        local_path=HF_DIR / "acne_severity_vit",
        weight_files=["model.safetensors", "config.json", "preprocessor_config.json"],
        weight_format="safetensors",
        approx_size_mb=343,
        framework="transformers / torch (CPU)",
        input_size=224,
        license_="MIT",
        academic_use="yes",
        commercial_use="yes",
        training_data="not named in card; severity levels align with ACNE04 Hayashi grading",
        reported_metrics="eval precision 0.756 (single epoch, from trainer_state.json)",
        metric_provenance="from repo results file (trainer_state.json)",
        evidence="716 downloads / 18 likes (most-used acne model found); on this project's spot checks it returned "
                 "'level 3' for a severe ACNE04 test image and 'level -1' (clear) for a clean selfie",
        notes="level -1 == clear skin, level 3 == most severe. Only 1 training epoch -> treat as a coarse severity estimate.",
    ),
    "acne_detector": ModelSpec(
        key="acne_detector",
        conditions=["acne"],
        task="detection",
        source_url="https://huggingface.co/mufasabrownie/glowlytics-skin-models (acne_detector_best.pt)",
        architecture="YOLOv8s (Ultralytics)",
        classes=["acne"],
        local_path=HF_DIR / "acne_detector_yolo",
        weight_files=["acne_detector_best.pt"],
        weight_format="pt (ultralytics)",
        approx_size_mb=22,
        framework="ultralytics / torch (CPU)",
        input_size=640,
        license_="MIT",
        academic_use="yes",
        commercial_use="yes",
        training_data="'acne_unified' (~1,843 images / 18,717 boxes; box count ~= ACNE04's 18,983 -> likely ACNE04-derived)",
        reported_metrics="mAP50 0.473, precision 0.503, recall 0.486",
        metric_provenance="author-reported (model card)",
        evidence="on this project's spot checks at conf=0.25 it produced 4-6 boxes on severe ACNE04 test images and "
                 "0 boxes on a clean selfie (no fabricated detections). Replaces the abandoned 3-epoch ACNE04 proof model.",
        notes="Provides the bounding-box / lesion-count signal + the visual explanation for acne.",
    ),
    "wrinkle_segmenter": ModelSpec(
        key="wrinkle_segmenter",
        conditions=["wrinkles"],
        task="segmentation",
        source_url="https://huggingface.co/rmsandu/ffhq-wrinkle-unet",
        architecture="U-Net with ResNet50 encoder (vendored model code, MIT)",
        classes=["wrinkle"],
        local_path=HF_DIR / "wrinkle_unet",
        weight_files=["wrinkle_model.safetensors", "face_segmentation.safetensors"],
        weight_format="safetensors",
        approx_size_mb=335,
        framework="torch + torchvision (CPU)",
        input_size=512,
        license_="code MIT; WEIGHTS + training data CC BY-NC-SA 4.0 (FFHQ-Wrinkle)",
        academic_use="yes",
        commercial_use="NO (CC BY-NC-SA) - academic / research use only",
        training_data="FFHQ-Wrinkle: 1,000 manually-labelled wrinkle masks (+ 50k weak), on FFHQ face images",
        reported_metrics="no headline IoU/Dice published by the repo",
        metric_provenance="unknown / unverified",
        evidence="state_dict loads with 0 missing/0 unexpected keys; returns ~0 wrinkle-pixel fraction on young faces "
                 "(this project's spot check). Positive-case accuracy not independently validated.",
        notes="Output is a wrinkle probability mask -> we derive a wrinkle-area fraction as the score. "
              "NON-COMMERCIAL licence - fine for this academic project, must not ship as a product.",
    ),
    "redness_heuristic": ModelSpec(
        key="redness_heuristic",
        conditions=["redness"],
        task="heuristic",
        source_url="classical CV (this project) - approach inspired by github.com/DurtyDhiana/skin-scan (MIT)",
        architecture="OpenCV colour analysis on the MediaPipe-detected face/skin region (CIELAB a* + normalised redness)",
        classes=["redness_estimate"],
        local_path=PROJECT_ROOT,
        weight_files=[],
        weight_format="n/a",
        approx_size_mb=0,
        framework="opencv + mediapipe (CPU)",
        input_size=0,
        license_="n/a (own code)",
        academic_use="yes",
        commercial_use="yes",
        training_data="none - not a trained model",
        reported_metrics="none - heuristic, not benchmarked",
        metric_provenance="n/a",
        evidence="NOT a machine-learning model. A colour-based ESTIMATE. Surfaced to the user as 'estimated', never 'detected'.",
        notes="No public facial-redness detection model or dataset exists (verified over 3 audit rounds). "
              "This is a clearly-labelled estimate only.",
    ),
    "pores_heuristic": ModelSpec(
        key="pores_heuristic",
        conditions=["enlarged_pores"],
        task="heuristic",
        source_url="classical CV (this project) - approach inspired by github.com/DurtyDhiana/skin-scan (MIT)",
        architecture="OpenCV local-contrast / black-hat blob density on the detected skin region",
        classes=["pore_density_estimate"],
        local_path=PROJECT_ROOT,
        weight_files=[],
        weight_format="n/a",
        approx_size_mb=0,
        framework="opencv + mediapipe (CPU)",
        input_size=0,
        license_="n/a (own code)",
        academic_use="yes",
        commercial_use="yes",
        training_data="none - not a trained model",
        reported_metrics="none - heuristic",
        metric_provenance="n/a",
        evidence="NOT a machine-learning model. A texture-based ESTIMATE. Surfaced as 'estimated', never 'detected'.",
        notes="No public facial-pore detection model or dataset exists. Clearly-labelled estimate only.",
    ),
}

# Conditions with NO defensible pretrained model or dataset (verified over 3 audit rounds).
UNAVAILABLE_CONDITIONS: dict[str, str] = {
    "pigmentation": "No public facial-photo model or box/mask dataset for pigmentation exists. "
                    "Everything found is dermoscopy (wrong domain), image-level only, or unlicensed.",
    "blackheads": "No working public comedone/blackhead detector for facial photos "
                  "(the one candidate scored mAP50 ~0.14; the ACNE-DET dataset is 276 images, not public).",
    "dark_spots": "Same data gap as pigmentation - no validated facial-photo dataset or model.",
}

CONDITION_LABELS: dict[str, str] = {
    "acne": "Acne",
    "pigmentation": "Pigmentation",
    "blackheads": "Blackheads / comedones",
    "wrinkles": "Wrinkles",
    "dark_spots": "Dark spots",
    "dryness": "Dryness",
    "redness": "Redness",
    "enlarged_pores": "Enlarged pores",
    "skin_type": "Skin type",
}
