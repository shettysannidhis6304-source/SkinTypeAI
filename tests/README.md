# tests/

Scaffold for a future pytest suite (`tests/api/`, `tests/model/`, `tests/unit/`).

For now the end-to-end verification lives in **`../test_inference.py`** (project
root). It runs against a throwaway SQLite DB (the real `database/skintype.db` is
never touched) and covers:

- **A. direct model** — loads `weights/skin_analysis_model.pt` offline (no network,
  nothing pretrained) and runs the 5 heads (acne / wrinkles / oiliness / dryness /
  pigmentation) on sample crops; asserts the param budget (< 450k).
- **B. OpenCV engine** — `OpenCVSkinAnalyzer` returns the 4 deterministic traits
  (redness / dark_spots / enlarged_pores / blackheads), all `"estimated"`.
- **C. API contract** — `/register → /login → /upload → /predict?force_proceed=true`
  and asserts the response schema (10 condition keys, 8 canonical sub-keys each,
  valid Skin Health Index, `xai_insights`, `quality`, no `not_available`).
- **D. Grad-CAM** — `models.skin_analysis.gradcam.GradCAM` produces a normalised
  `[0, 1]` heatmap for every head.
- **E. advisory quality** — a soft image with `force_proceed=false` returns a
  `status: "advisory"` payload (score ≤ 6 + `temp_image_id`); re-calling with
  `force_proceed=true` runs the full pipeline.
- **F. QR session** — `detect_local_ip()` + `GET /api/session/create-qr-session`
  (+ the full mobile-upload → status `ready` lifecycle).
- **G. image overlays** — `GET /api/images/{id}/overlay?mode=gradcam|zones` returns
  a PNG at the original dimensions; bad `mode`/`head` → 422, missing image → 404.

Run it from the project root:

```
.venv-new\Scripts\python.exe test_inference.py
```
