# tests/

Scaffold for a future pytest suite (`tests/api/`, `tests/model/`, `tests/unit/`).

For now the end-to-end verification lives in **`../test_inference.py`** (project
root), which covers:

- **A. direct model** — loads `weights/custom_skin_model.pt` offline and runs the
  three heads on sample crops;
- **B. API contract** — spins up FastAPI against a throwaway SQLite DB, runs
  `/register → /login → /upload → /predict`, and asserts the response schema
  (10 condition keys, 8 canonical sub-keys each, valid Skin Health Index, no
  `not_available`);
- **C. fallback** — forces `use_custom_model=False` and checks the Hugging Face
  ViT path still returns the identical shape.

Run it from the project root:

```
.venv-new\Scripts\python.exe test_inference.py
```
