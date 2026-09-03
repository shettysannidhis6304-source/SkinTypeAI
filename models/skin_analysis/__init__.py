"""
models.skin_analysis

Pretrained-model layer for SkinTypeAI's multi-condition facial analysis.
No model here was trained by this project - see registry.MODELS for provenance.
"""
from models.skin_analysis.registry import (  # noqa: F401
    CONDITION_LABELS,
    MODELS,
    UNAVAILABLE_CONDITIONS,
    ModelSpec,
)
