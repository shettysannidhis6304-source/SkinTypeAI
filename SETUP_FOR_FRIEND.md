# SkinType AI — Setup

An academic facial-skin-analysis web app: FastAPI backend + a stack of
**publicly available pretrained models** + a plain HTML/CSS/JS frontend.
Nothing is trained here — every result comes from a public model or a
clearly-labelled heuristic estimate. Not a medical device.

## 1. Install Python 3.11

If you don't already have Python 3.11, on Windows:
```powershell
winget install Python.PythonInstallManager
pymanager install 3.11
```
(Or install Python 3.11 from python.org — either works.)

## 2. Create the virtual environment and install dependencies

From this folder, in PowerShell:
```powershell
py -3.11 -m venv .venv-new
.venv-new\Scripts\python.exe -m pip install -r requirements.txt
```
> Use `.venv-new` (not `.venv`). Installs PyTorch (CPU is fine — this app runs
> on CPU), FastAPI, transformers, ultralytics, etc. Takes several minutes and
> downloads ~3 GB.

## 3. Download the pretrained model weights (~1 GB, one-time)

```powershell
.venv-new\Scripts\python.exe scripts\fetch_models.py
```
This pulls the four public models into `weights\hf_models\` (git-ignored).
Provenance and licences: `weights\skin_analysis_models.metadata.json`.
One of them (the wrinkle model) is **CC BY-NC-SA — academic use only**; to
skip it set `wrinkle_enabled: false` in `backend\app\core\config.py`.

## 4. Create your local database

```powershell
.venv-new\Scripts\python.exe database\init_db.py
```
Creates an empty `database\skintype.db` — you register your own account
through the website.

## 5. Start it

Double-click **`start-servers.bat`** (backend on `:8000`, frontend on `:5500`,
opens the site automatically). Stop with `stop-servers.bat` or by closing the
two windows.

The first scan takes ~30 s (models loading); after that ~4 s per scan on CPU.

## What's covered

| Model-backed | Estimated | Not available |
|---|---|---|
| Acne (severity + lesion boxes), skin type (dry/normal/oily), dryness, oiliness, wrinkles | Redness, enlarged pores (heuristic estimates) | Pigmentation, blackheads, dark spots (no validated public model exists) |

The old EfficientNet skin-type classifier and the ACNE04 YOLO experiment are
kept in the repo for reference/rollback but are not on the active path.
