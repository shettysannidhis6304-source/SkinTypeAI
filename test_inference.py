#!/usr/bin/env python3
"""
test_inference.py

Step 5 verification for the custom multi-task model integration.

  A. Direct model check -- load weights/custom_skin_model.pt with NO network and
     run all three heads on a few ACNE04 face crops; print the distributions.

  B. End-to-end API check -- spin up the FastAPI app against a THROWAWAY SQLite
     database (the real database/skintype.db is never touched), register a user,
     upload a face image, call /predict, print the full `results` JSON, and
     assert its shape is byte-for-byte the contract the frontend +
     compute_skin_health_index() + recommendation_engine.py expect.

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe test_inference.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (PROJECT_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CROPS_DIR = PROJECT_ROOT / "datasets" / "processed" / "crops"
CUSTOM_WEIGHTS = PROJECT_ROOT / "weights" / "custom_skin_model.pt"

# The 8 canonical keys every condition entry must have (schemas/prediction.py::ConditionEntry)
CANONICAL_KEYS = {"status", "method", "label", "score", "confidence", "detail", "model", "note"}
# The 10 condition keys run_analysis() must always emit
EXPECTED_CONDITIONS = {
    "acne", "skin_type", "dryness", "oiliness", "wrinkles",
    "redness", "enlarged_pores", "pigmentation", "blackheads", "dark_spots",
}
TOP_LEVEL_KEYS = {"analysis_version", "face_detected", "image_size", "conditions",
                  "skin_health_index", "model_versions"}


def _sample_crops(n: int = 3) -> list[Path]:
    if not CROPS_DIR.is_dir():
        return []
    crops = sorted(CROPS_DIR.glob("*.jpg"))
    # spread across acne grades if possible
    picks: list[Path] = []
    for prefix in ("levle0", "levle1", "levle2", "levle3"):
        for c in crops:
            if c.name.startswith(prefix):
                picks.append(c)
                break
    return (picks or crops)[:n]


# ============================================================ A. direct model
def test_direct_model() -> bool:
    print("=" * 70)
    print("A. DIRECT MODEL CHECK  (no network, load weights/custom_skin_model.pt)")
    print("=" * 70)
    if not CUSTOM_WEIGHTS.exists():
        print(f"  SKIP: {CUSTOM_WEIGHTS} not found -- run training/train_custom.py first.")
        return False

    from models.custom_multi_task import load_custom_model, predict

    model, ckpt = load_custom_model(CUSTOM_WEIGHTS)
    cfg = ckpt.get("config", {})
    print(f"  backbone      : {cfg.get('backbone')}  ({cfg.get('feature_dim')}-d, frozen)")
    print(f"  heads         : {cfg.get('heads')}")
    print(f"  trained_at    : {ckpt.get('trained_at')}")
    m = ckpt.get("metrics", {}).get("final_val", {})
    if m:
        print(f"  val mean F1   : {m.get('mean_macro_f1')}  "
              f"(skin_type {m.get('skin_type', {}).get('macro_f1')}, "
              f"acne {m.get('acne', {}).get('macro_f1')}, "
              f"pigmentation {m.get('pigmentation', {}).get('macro_f1')})")

    crops = _sample_crops(4)
    if not crops:
        print("  SKIP: no crops in datasets/processed/crops/")
        return True
    for c in crops:
        out = predict(model, str(c), cfg)
        print(f"\n  {c.name}")
        for head, r in out.items():
            dist = " ".join(f"{k}:{v:.2f}" for k, v in r["distribution"].items())
            print(f"    {head:13s} -> {r['label']:12s} (conf {r['confidence']:.2f})   [{dist}]")
    return True


# ============================================================ B. end-to-end API
def test_api_contract() -> bool:
    print("\n" + "=" * 70)
    print("B. END-TO-END API CHECK  (throwaway DB, real /upload + /predict)")
    print("=" * 70)

    tmpdb = Path(tempfile.gettempdir()) / "skintypeai_test_inference.db"
    if tmpdb.exists():
        tmpdb.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdb.as_posix()}"

    from fastapi.testclient import TestClient

    from app.models_db import Base
    from app.models_db.base import engine
    from app.main import app

    Base.metadata.create_all(bind=engine)
    client = TestClient(app)

    # ---- model files present? (predict returns 503 otherwise) ----
    from app.core.config import settings
    need = [settings.skin_type_model_dir / "config.json",
            settings.acne_severity_model_dir / "config.json",
            settings.acne_detector_weights]
    missing = [str(p) for p in need if not Path(p).exists()]
    if missing:
        print(f"  SKIP: analysis models missing: {missing}")
        return False
    print(f"  use_custom_model = {settings.use_custom_model} ; checkpoint exists = {CUSTOM_WEIGHTS.exists()}")

    # ---- auth ----
    email = "test_inference@example.com"
    r = client.post("/register", json={"email": email, "password": "test-pw-123456", "full_name": "Test Inference"})
    assert r.status_code in (201, 400), r.text
    r = client.post("/login", json={"email": email, "password": "test-pw-123456"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ---- pick a face image ----
    crops = _sample_crops(1)
    img_path = crops[0] if crops else None
    if img_path is None:
        for cand in sorted((PROJECT_ROOT / "backend" / "uploads").glob("*.jp*g")):
            img_path = cand
            break
    if img_path is None:
        print("  SKIP: no test image available")
        return False
    print(f"  test image     : {img_path.name}")

    with open(img_path, "rb") as fh:
        r = client.post("/upload", files={"file": (img_path.name, fh, "image/jpeg")}, headers=headers)
    assert r.status_code == 201, r.text
    image_id = r.json()["image_id"]

    # remember the file this test wrote into backend/uploads/ so we can delete it
    from app.core.config import resolve_upload_path
    from app.models_db.base import SessionLocal
    from app.models_db.image import Image as _Image
    with SessionLocal() as _s:
        _uploaded_file = resolve_upload_path(_s.get(_Image, image_id).file_path)

    ctx = {"age": 24, "stress_level": "medium", "sleep_hours": 6, "sun_exposure": "moderate",
           "env_mode": "manual", "temperature_c": 30, "humidity_pct": 65, "uv_index": 6}
    r = client.post(f"/predict/{image_id}", json=ctx, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    results = body["results"]

    print("\n  ---- results JSON ----")
    print("\n".join("  " + ln for ln in json.dumps(results, indent=2).splitlines()))

    # ---- CONTRACT ASSERTIONS ----
    errors: list[str] = []
    if set(results) != TOP_LEVEL_KEYS:
        errors.append(f"top-level keys {set(results)} != {TOP_LEVEL_KEYS}")
    conds = results.get("conditions", {})
    if set(conds) != EXPECTED_CONDITIONS:
        errors.append(f"condition keys {set(conds)} != {EXPECTED_CONDITIONS}")
    for name, entry in conds.items():
        if set(entry) != CANONICAL_KEYS:
            errors.append(f"conditions.{name} keys {set(entry)} != {CANONICAL_KEYS}")
        if entry["status"] not in ("detected", "estimated", "not_available", "error"):
            errors.append(f"conditions.{name}.status invalid: {entry['status']}")
    shi = results.get("skin_health_index")
    if not isinstance(shi, dict) or set(shi) < {"value", "band", "basis", "formula"}:
        errors.append(f"skin_health_index shape wrong: {shi}")
    if not (isinstance(results.get("image_size"), list) and len(results["image_size"]) == 2):
        errors.append(f"image_size wrong: {results.get('image_size')}")
    if not isinstance(results.get("face_detected"), bool):
        errors.append("face_detected not bool")

    # ---- COVERAGE: nothing should be not_available with all engines active ----
    na = [k for k, v in conds.items() if v["status"] == "not_available"]
    if na:
        errors.append(f"conditions still not_available: {na}")

    # ---- CV-analysis conditions must carry a score, confidence and distribution
    for k in ("redness", "enlarged_pores", "blackheads", "dark_spots"):
        e = conds[k]
        if e["status"] != "estimated":
            errors.append(f"{k}.status = {e['status']} (expected 'estimated')")
        if not isinstance(e.get("confidence"), (int, float)):
            errors.append(f"{k}.confidence missing/invalid: {e.get('confidence')}")
        if not isinstance(e.get("score"), (int, float)):
            errors.append(f"{k}.score missing/invalid: {e.get('score')}")
        if not isinstance((e.get("detail") or {}).get("distribution"), dict):
            errors.append(f"{k}.detail.distribution missing")
        if e.get("model") != "cv_analysis_engine":
            errors.append(f"{k}.model = {e.get('model')} (expected 'cv_analysis_engine')")

    # ---- pigmentation must be an active scan from the custom model ----
    if conds["pigmentation"]["status"] != "detected":
        errors.append(f"pigmentation.status = {conds['pigmentation']['status']} (expected 'detected')")

    print("\n  ---- engine routing ----")
    print(f"    skin_type.model     = {conds['skin_type']['model']}")
    print(f"    acne.model          = {conds['acne']['model']}  (class '{conds['acne']['detail'].get('severity_level')}')")
    for k in ("pigmentation", "redness", "enlarged_pores", "blackheads", "dark_spots"):
        e = conds[k]
        d = (e.get("detail") or {}).get("distribution")
        print(f"    {k:15s} {e['status']:9s} score={e['score']}  conf={e['confidence']}  "
              f"model={e['model']}  dist={d}")
    print(f"    SHI                 = {shi['value']} ({shi['band']})")
    print(f"    model_versions keys = {list(results['model_versions'])}")
    print(f"    recommendation keys = {list((body.get('recommendation') or {}).keys())}")

    # clean up: the uploaded file (real backend/uploads/) + the throwaway db
    try:
        Path(_uploaded_file).unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (note: could not remove upload {_uploaded_file}: {e})")
    try:
        engine.dispose()          # release the SQLite handle (Windows lock)
        tmpdb.unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (note: could not remove {tmpdb.name}: {e})")

    if errors:
        print("\n  CONTRACT FAILURES:")
        for e in errors:
            print(f"    - {e}")
        return False
    print("\n  CONTRACT OK: same top-level + 10 condition keys, same 8 canonical sub-keys, "
          "valid SHI shape. All 10 conditions active (0 not_available).")
    return True


def test_fallback() -> bool:
    """C. With use_custom_model forced off, the HF ViT path must still run and
    still produce the identical output shape."""
    print("\n" + "=" * 70)
    print("C. FALLBACK CHECK  (use_custom_model = False -> HF ViT path)")
    print("=" * 70)
    from app.core.config import settings
    from app.services import analysis_service as A

    need = [settings.skin_type_model_dir / "config.json",
            settings.acne_severity_model_dir / "config.json", settings.acne_detector_weights]
    if any(not Path(p).exists() for p in need):
        print("  SKIP: HF model files missing")
        return True

    crops = _sample_crops(1)
    if not crops:
        print("  SKIP: no test image")
        return True

    saved = settings.use_custom_model
    A._custom_model = None  # force a clean decision
    settings.use_custom_model = False
    try:
        res = A.run_analysis(str(crops[0]))
    finally:
        settings.use_custom_model = saved
        A._custom_model = None

    c = res["conditions"]
    st = c["skin_type"]["model"]
    ac = c["acne"]["model"]
    pg = c["pigmentation"]["status"]
    cv_ok = all(c[k]["status"] == "estimated" and c[k]["model"] == "cv_analysis_engine"
                and isinstance(c[k]["confidence"], (int, float))
                for k in ("redness", "enlarged_pores", "blackheads", "dark_spots"))
    ok = (st == "skin_type" and ac == "acne_severity+acne_detector" and pg == "not_available"
          and cv_ok and set(c) == EXPECTED_CONDITIONS)
    print(f"  skin_type.model     = {st}   (expected 'skin_type')")
    print(f"  acne.model          = {ac}   (expected 'acne_severity+acne_detector')")
    print(f"  pigmentation.status = {pg}   (expected 'not_available' - needs the custom model)")
    print(f"  CV conditions still active (redness/pores/blackheads/dark_spots): {cv_ok}")
    print(f"  SHI                 = {res['skin_health_index']['value']} ({res['skin_health_index']['band']})")
    print(f"  {'FALLBACK OK' if ok else 'FALLBACK FAILED'}")
    return ok


def main() -> int:
    ok_a = test_direct_model()
    ok_b = test_api_contract()
    ok_c = test_fallback()
    print("\n" + "=" * 70)
    print(f"RESULT:  direct-model={'PASS' if ok_a else 'FAIL/SKIP'}   "
          f"api-contract={'PASS' if ok_b else 'FAIL'}   "
          f"fallback={'PASS' if ok_c else 'FAIL'}")
    print("=" * 70)
    return 0 if (ok_b and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
