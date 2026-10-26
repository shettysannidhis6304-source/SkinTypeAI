# legacy_models/ — archived, intact, not on any active path

These files were the **previous** model stack. They were replaced in the
from-scratch rebuild (Stage 4) by:

- `models/skin_analysis_model.py` — `SkinAnalysisModel` (403,656 params, 100% from
  scratch, 5 heads: acne / wrinkles / oiliness / dryness / pigmentation)
- `models/skin_analysis/opencv_engine.py` — `OpenCVSkinAnalyzer` (redness, dark
  spots, pores, blackheads)
- `training/train_skin_analysis.py` — trainer for the above

Nothing in the running backend imports anything here. Kept for reproducibility
and the paper's ablation history.

| File | What it was |
|---|---|
| `custom_multi_task.py` | `CustomSkinModel` — frozen (later `blocks[-1]`-fine-tuned) `timm mobilenetv3_small_100` trunk + 3 heads (skin_type, acne, pigmentation) |
| `train_custom.py` | its trainer (cached-feature + end-to-end trunk fine-tune modes, ordinal acne loss, QWK/MAE) |
| `custom_skin_model.pt` | the fine-tuned checkpoint (val mean macro-F1 0.686, acne QWK 0.799) |

Frozen-trunk / ordinal / CE ablation checkpoints remain under `runs/ablation/`.
