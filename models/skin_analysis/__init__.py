"""
models.skin_analysis

Computer-vision layer for SkinTypeAI's multi-condition facial analysis:
  * heuristics.py    -- deterministic OpenCV estimates (redness / pores / blackheads /
                        dark spots) + the shared MediaPipe face crop + single
                        FaceMesh landmark pass (facemesh_landmarks).
  * opencv_engine.py -- OpenCVSkinAnalyzer facade + the zone-overlay renderer.
  * gradcam.py       -- Grad-CAM for the from-scratch models.skin_analysis_model.

`registry.py` / `weights/skin_analysis_models.metadata.json` are a historical
provenance record of the pretrained stack that was RETIRED in favour of the
from-scratch SkinAnalysisModel -- kept for academic-integrity reference only.
"""
from models.skin_analysis.registry import (  # noqa: F401
    CONDITION_LABELS,
    MODELS,
    UNAVAILABLE_CONDITIONS,
    ModelSpec,
)
