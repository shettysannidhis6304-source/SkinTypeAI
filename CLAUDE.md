# SkinType AI — Project Context for Claude Code

## What this is
An AI-powered skin analysis platform. Real trained ML model (no fake/hardcoded
predictions), integrated into an existing HTML/CSS/JS website via a FastAPI backend.

## Environment
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6GB VRAM
- Driver: 576.52, supports up to CUDA 12.9 (confirmed via `nvidia-smi`)
- Python: needs a clean install — previous install was broken
  (PATH pointed to a missing C:\python\python.exe, error 0x80070003).
  Fix this first: check for stray PATH entries pointing to C:\python,
  remove them, do a fresh Python 3.11 install from python.org with
  "Add python.exe to PATH" checked.
- PyTorch must be installed from the CUDA 12.1 index, NOT plain PyPI
  (plain PyPI gives a CPU-only build):
  `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`
  Then: `pip install -r requirements.txt`

## Timeline & scope (IMPORTANT — do not over-build)
User needs a WORKING END-TO-END SKELETON in 2 days. Accuracy does not matter
right now — that's a separate future phase ("next case"). Do not gold-plate.

### In scope for v1 (this 2-day build):
- Model: EfficientNetV2-S only (via `timm`, ImageNet-pretrained, fine-tuned).
  NO YOLO in v1 — deferred to a future phase. Classifier only.
- Multi-label output head covering 10 targets (as severity scores /
  classes, not bounding boxes):
  Skin Type, Acne, Acne Severity, Oiliness, Dryness, Pigmentation,
  Dark Spots, Redness, Pores, Wrinkles
- Input: photo upload only for v1 (live camera = same endpoint, later,
  just captures one frame and posts it — not a separate pipeline)
- Backend: FastAPI — /register, /login, /upload, /predict, /history
- Auth: JWT
- DB: SQLite — Users, Predictions, Images tables
- Recommendation engine: rule-based logic driven by real prediction
  outputs (ingredients/routine suggestions) — not fake/random
- Frontend: existing HTML/CSS/JS site — DO NOT REDESIGN. Only wire up
  the Upload button via `frontend/js/api.js` to call the backend and
  populate existing UI elements (skin score, skin type, routine, etc.)

### Explicitly OUT of scope for v1 (defer to later phase):
- YOLOv11 detection (acne/dark spot/redness bounding boxes)
- Grad-CAM heatmaps
- PDF report generation
- /retrain endpoint
- Multi-dataset merging (Fitzpatrick17k + ACNE04 + HAM10000 + DermNet)
- Docker deployment
- Full test suite
- TensorBoard dashboard

## Build principles (carried over from original spec)
- No placeholder/fake code — every prediction comes from the trained model
- Modular architecture, PEP8, type hints, logging, environment variables
- Go step by step: architecture -> requirements.txt -> dataset/training
  scripts -> model -> backend -> frontend integration
- Explain each file's purpose before writing it
- Don't rewrite previous files unless necessary; maintain consistency

## Dataset plan
User is assembling small labeled folders under `datasets/raw/` in this
structure (one subfolder tree per condition):
```
datasets/raw/
├── skin_type/{oily,dry,normal}/
├── acne/{none,mild,moderate,severe}/
├── pigmentation/{low,high}/
... same per-condition pattern
```
Starting point: Kaggle "Oily, Dry, Normal Skin Types Dataset" for the
skin_type folder. Other condition folders are small hand-gathered sets
(15-30 images per class is fine for a skeleton run — accuracy doesn't
matter yet, just that the pipeline trains and predicts end-to-end).

## Already built (do not redo, build on top of it)
- Full folder scaffold (see project tree)
- `requirements.txt` (pinned versions, CUDA 12.1 torch instructions inside)

## Next steps in order
1. Fix the Python install / PATH issue on this machine
2. Run the two pip install commands above, verify with:
   ```python
   import torch
   print(torch.cuda.is_available())      # expect True
   print(torch.cuda.get_device_name(0))  # expect "NVIDIA GeForce RTX 4050 Laptop GPU"
   ```
3. `training/dataset.py` + `training/augment.py`
4. `models/efficientnet_classifier.py`
5. `training/train.py` (batch size 8-16, AMP enabled — 6GB VRAM is tight),
   `training/evaluate.py`, `training/predict.py`
6. Run a short training run (few epochs, small data) to produce a `.pt`
   weights file
7. `database/init_db.py` + SQLAlchemy models, JWT auth, FastAPI routers
8. `backend/app/services/inference_service.py` to load the trained model
9. Wire `frontend/js/api.js` into the existing HTML (no redesign)
