#!/usr/bin/env python3
"""
test_inference.py

Stage 4 verification for the from-scratch 9-characteristic system.

  A. Direct model check -- load weights/skin_analysis_model.pt with NO network and
     run all 5 heads (acne / wrinkles / oiliness / dryness / pigmentation) on a
     few ACNE04 face crops; print the distributions.

  B. OpenCV engine check -- run OpenCVSkinAnalyzer on a face image and confirm it
     returns the 4 deterministic traits (redness / dark_spots / enlarged_pores /
     blackheads), all "estimated".

  C. End-to-end API check -- spin up FastAPI against a THROWAWAY SQLite database
     (the real database/skintype.db is never touched), register a user, upload a
     face image, call /predict, print the full `results` JSON and assert its shape
     is exactly the contract the frontend + compute_skin_health_index() +
     recommendation_engine.py expect (10 conditions, 8 canonical keys, SHI + XAI
     shape, skin_type derived, 4 OpenCV traits estimated).

  D. Grad-CAM check -- load the model offline and confirm models.skin_analysis.
     gradcam.GradCAM produces a normalised [0,1] heatmap for every head.

  E. Quality-gate check -- POST a synthetically blurred image to /predict and
     confirm the pre-analysis gate returns HTTP 422 Unprocessable Entity.

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
MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "skin_analysis_model.pt"

CANONICAL_KEYS = {"status", "method", "label", "score", "confidence", "detail", "model", "note"}
EXPECTED_CONDITIONS = {
    "acne", "skin_type", "dryness", "oiliness", "wrinkles",
    "redness", "enlarged_pores", "pigmentation", "blackheads", "dark_spots",
}
MODEL_HEADS = {"acne", "wrinkles", "oiliness", "dryness", "pigmentation"}
CV_TRAITS = {"redness", "enlarged_pores", "blackheads", "dark_spots"}
TOP_LEVEL_KEYS = {"analysis_version", "face_detected", "image_size", "conditions",
                  "skin_health_index", "xai_insights", "model_versions", "quality"}


def _sample_crops(n: int = 3) -> list[Path]:
    if not CROPS_DIR.is_dir():
        return []
    crops = sorted(CROPS_DIR.glob("*.jpg"))
    picks: list[Path] = []
    for prefix in ("levle0", "levle1", "levle2", "levle3"):
        for c in crops:
            if c.name.startswith(prefix):
                picks.append(c)
                break
    return (picks or crops)[:n]


def _any_face_image() -> Path | None:
    for c in _sample_crops(1):
        return c
    for cand in sorted((PROJECT_ROOT / "backend" / "uploads").glob("*.jp*g")):
        return cand
    return None


# ============================================================ A. direct model
def test_direct_model() -> bool:
    print("=" * 72)
    print("A. DIRECT MODEL CHECK  (offline, load weights/skin_analysis_model.pt)")
    print("=" * 72)
    if not MODEL_WEIGHTS.exists():
        print(f"  FAIL: {MODEL_WEIGHTS} not found -- run training/train_skin_analysis.py.")
        return False

    from models.skin_analysis_model import HEAD_CLASSES, load_skin_analysis_model, predict

    model, ckpt = load_skin_analysis_model(MODEL_WEIGHTS)
    cfg = ckpt.get("config", {})
    mm = ckpt.get("metrics", {})
    print(f"  from_scratch   : {cfg.get('from_scratch')}   params: {mm.get('model', {}).get('params')}")
    print(f"  heads          : {list(cfg.get('heads', {}))}")
    print(f"  trained_at     : {ckpt.get('trained_at')}")
    fv = mm.get("final_val", {})
    if fv:
        print(f"  val mean F1    : {fv.get('mean_macro_f1')}  (acne QWK {fv.get('acne', {}).get('qwk')}, "
              f"MAE {fv.get('acne', {}).get('mae')})")

    total_params = sum(p.numel() for p in model.parameters())
    ok = total_params < 450_000
    print(f"  param budget   : {total_params:,} < 450,000 ? {ok}")

    crops = _sample_crops(4)
    for c in crops:
        out = predict(model, str(c), cfg)
        assert set(out) == MODEL_HEADS, set(out)
        print(f"\n  {c.name}")
        for head in ("acne", "wrinkles", "oiliness", "dryness", "pigmentation"):
            r = out[head]
            assert len(r["distribution"]) == len(HEAD_CLASSES[head])
            dist = " ".join(f"{k}:{v:.2f}" for k, v in r["distribution"].items())
            print(f"    {head:13s} -> {r['label']:10s} (conf {r['confidence']:.2f})   [{dist}]")
    return ok


# ============================================================ B. opencv engine
def test_opencv_engine() -> bool:
    print("\n" + "=" * 72)
    print("B. OPENCV ENGINE CHECK  (OpenCVSkinAnalyzer -> redness/dark_spots/pores/blackheads)")
    print("=" * 72)
    img = _any_face_image()
    if img is None:
        print("  SKIP: no test image")
        return True

    from models.skin_analysis.opencv_engine import CHARACTERISTICS, OpenCVSkinAnalyzer

    res = OpenCVSkinAnalyzer().analyze(str(img), oiliness_score=0.6,
                                       pigment_dist={"normal": 0.3, "mild": 0.4, "marked": 0.3})
    if set(res) != set(CHARACTERISTICS):
        print(f"  FAIL: keys {set(res)} != {set(CHARACTERISTICS)}")
        return False
    ok = True
    for k, v in res.items():
        good = (v["status"] == "estimated" and isinstance(v.get("score"), (int, float))
                and v.get("model") == "cv_analysis_engine")
        ok &= good
        print(f"    {k:15s} status={v['status']:10s} score={v['score']}  model={v['model']}  {'ok' if good else 'BAD'}")
    return ok


# ============================================================ C. end-to-end API
def test_api_contract() -> bool:
    print("\n" + "=" * 72)
    print("C. END-TO-END API CHECK  (throwaway DB, real /upload + /predict -> 200)")
    print("=" * 72)

    tmpdb = Path(tempfile.gettempdir()) / "skintypeai_test_inference.db"
    if tmpdb.exists():
        tmpdb.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdb.as_posix()}"

    from fastapi.testclient import TestClient

    from app.core.config import resolve_model_weights, resolve_upload_path
    from app.main import app
    from app.models_db import Base
    from app.models_db.base import SessionLocal, engine
    from app.models_db.image import Image as _Image

    if not resolve_model_weights().exists():
        print(f"  FAIL: {resolve_model_weights()} missing")
        return False

    Base.metadata.create_all(bind=engine)
    client = TestClient(app)

    email = "test_inference@example.com"
    r = client.post("/register", json={"email": email, "password": "test-pw-123456", "full_name": "Test"})
    assert r.status_code in (201, 400), r.text
    r = client.post("/login", json={"email": email, "password": "test-pw-123456"})
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    img_path = _any_face_image()
    if img_path is None:
        print("  SKIP: no test image available")
        return False
    print(f"  test image     : {img_path.name}")

    with open(img_path, "rb") as fh:
        r = client.post("/upload", files={"file": (img_path.name, fh, "image/jpeg")}, headers=headers)
    assert r.status_code == 201, r.text
    image_id = r.json()["image_id"]
    with SessionLocal() as _s:
        uploaded_file = resolve_upload_path(_s.get(_Image, image_id).file_path)

    ctx = {"age": 24, "stress_level": "medium", "sleep_hours": 6, "sun_exposure": "moderate",
           "env_mode": "manual", "temperature_c": 30, "humidity_pct": 65, "uv_index": 6}
    # force_proceed=true so the advisory quality gate never intercepts this contract check
    r = client.post(f"/predict/{image_id}?force_proceed=true", json=ctx, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    results = body["results"]

    print("\n  ---- results JSON ----")
    print("\n".join("  " + ln for ln in json.dumps(results, indent=2).splitlines()))

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

    qual = results.get("quality")
    if not isinstance(qual, dict) or not (1 <= int(qual.get("score", 0)) <= 10) or qual.get("max_score") != 10:
        errors.append(f"results.quality shape/score wrong: {qual}")

    xai = results.get("xai_insights")
    if not isinstance(xai, dict) or "condition_rationales" not in xai or "shi_drivers" not in xai:
        errors.append(f"xai_insights shape wrong: {type(xai)}")
    else:
        drv = xai["shi_drivers"].get("primary_drivers")
        if not isinstance(drv, list) or len(drv) > 2:
            errors.append(f"xai_insights.shi_drivers.primary_drivers must be <=2 list: {drv}")

    na = [k for k, v in conds.items() if v["status"] == "not_available"]
    if na:
        errors.append(f"conditions still not_available (all engines active): {na}")

    # 5 model-backed characteristics
    for k in MODEL_HEADS:
        if conds[k]["status"] != "detected":
            errors.append(f"{k}.status = {conds[k]['status']} (expected 'detected')")
        if conds[k]["model"] != "skin_analysis_model":
            errors.append(f"{k}.model = {conds[k]['model']} (expected 'skin_analysis_model')")
    # acne detail carries the raw grade + rescaled index + distribution
    ad = conds["acne"]["detail"]
    if not (isinstance(ad.get("severity_grade"), int) and 0 <= ad["severity_grade"] <= 3):
        errors.append(f"acne.detail.severity_grade invalid: {ad.get('severity_grade')}")
    if not isinstance(ad.get("severity_distribution"), dict):
        errors.append("acne.detail.severity_distribution missing")

    # skin_type is DERIVED
    st = conds["skin_type"]
    if st["status"] != "detected" or st["method"] != "derived":
        errors.append(f"skin_type not derived: status={st['status']} method={st['method']}")
    if st["model"] != "derived(oiliness,dryness)":
        errors.append(f"skin_type.model = {st['model']}")
    if not isinstance((st.get("detail") or {}).get("distribution"), dict):
        errors.append("skin_type.detail.distribution missing")

    # 4 OpenCV traits
    for k in CV_TRAITS:
        e = conds[k]
        if e["status"] != "estimated":
            errors.append(f"{k}.status = {e['status']} (expected 'estimated')")
        if not isinstance(e.get("score"), (int, float)):
            errors.append(f"{k}.score invalid: {e.get('score')}")
        if not isinstance(e.get("confidence"), (int, float)):
            errors.append(f"{k}.confidence invalid: {e.get('confidence')}")
        if not isinstance((e.get("detail") or {}).get("distribution"), dict):
            errors.append(f"{k}.detail.distribution missing")
        if e.get("model") != "cv_analysis_engine":
            errors.append(f"{k}.model = {e.get('model')} (expected 'cv_analysis_engine')")

    rec = body.get("recommendation") or {}
    for key in ("summary", "routine", "ingredients", "am_routine", "pm_routine", "disclaimer"):
        if key not in rec:
            errors.append(f"recommendation missing '{key}'")

    # idempotency: a repeat /predict for the same (already-analysed) image must
    # return the SAME prediction, not a 500 from the UNIQUE(image_id) constraint.
    r2 = client.post(f"/predict/{image_id}?force_proceed=true", json=ctx, headers=headers)
    if r2.status_code != 200 or r2.json().get("id") != body.get("id"):
        errors.append(f"repeat /predict not idempotent: HTTP {r2.status_code}, id={r2.json().get('id')} vs {body.get('id')}")
    else:
        print(f"  repeat /predict -> HTTP 200, same prediction id {r2.json().get('id')}  (idempotent)")

    print("\n  ---- engine routing ----")
    print(f"    skin_type           = {st['label']}  (rule: {st['detail'].get('rule')})")
    print(f"    acne                = grade {ad.get('severity_grade')}/3  index {ad.get('severity_index')}  "
          f"human '{ad.get('severity_human')}'")
    for k in ("wrinkles", "oiliness", "dryness", "pigmentation"):
        print(f"    {k:15s} {conds[k]['status']:9s} score={conds[k]['score']}  "
              f"tier={conds[k]['detail'].get('tier')}  model={conds[k]['model']}")
    for k in sorted(CV_TRAITS):
        print(f"    {k:15s} {conds[k]['status']:9s} score={conds[k]['score']}  model={conds[k]['model']}")
    print(f"    SHI                 = {shi['value']} ({shi['band']})")
    print(f"    model_versions      = {list(results['model_versions'])}")
    print(f"    recommendation keys = {list(rec)}")

    try:
        Path(uploaded_file).unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (note: could not remove upload {uploaded_file}: {e})")
    try:
        engine.dispose()
        tmpdb.unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (note: could not remove {tmpdb.name}: {e})")

    if errors:
        print("\n  CONTRACT FAILURES:")
        for e in errors:
            print(f"    - {e}")
        return False
    print("\n  CONTRACT OK: 7 top-level keys, 10 condition keys, 8 canonical sub-keys, valid SHI + "
          "XAI shape. 5 heads via SkinAnalysisModel, skin_type derived, 4 traits via OpenCV. "
          "0 not_available.")
    return True


# ============================================================ D. grad-cam
def test_gradcam() -> bool:
    print("\n" + "=" * 72)
    print("D. GRAD-CAM CHECK  (GradCAM -> normalised [0,1] heatmap per head)")
    print("=" * 72)
    if not MODEL_WEIGHTS.exists():
        print(f"  FAIL: {MODEL_WEIGHTS} not found")
        return False
    img = _any_face_image()
    if img is None:
        print("  SKIP: no test image")
        return True

    from models.skin_analysis.gradcam import GradCAM
    from models.skin_analysis_model import load_skin_analysis_model, preprocess

    model, ckpt = load_skin_analysis_model(MODEL_WEIGHTS)
    x = preprocess(str(img), ckpt.get("config", {}).get("img_size", 128))
    cam = GradCAM(model)
    ok = True
    try:
        for head in ("acne", "wrinkles", "oiliness", "dryness", "pigmentation"):
            heat = cam.generate(x, head=head)
            good = (heat.shape == (x.shape[-2], x.shape[-1])
                    and float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0 + 1e-5
                    and float(heat.std()) > 0.0)
            ok &= good
            print(f"    {head:13s} heatmap {heat.shape}  range [{heat.min():.3f}, {heat.max():.3f}]  "
                  f"{'ok' if good else 'BAD'}")
    finally:
        cam.remove()
    return ok


# ============================================================ E. advisory quality gate
def _blur_copy(src: Path) -> Path:
    import cv2

    im = cv2.imread(str(src))
    blurred = cv2.GaussianBlur(im, (31, 31), 0)
    dst = Path(tempfile.gettempdir()) / "skintypeai_test_blurry.jpg"
    cv2.imwrite(str(dst), blurred)
    return dst


def _client_with_user(db_name: str, email: str):
    """(client, headers) against a throwaway SQLite DB."""
    tmpdb = Path(tempfile.gettempdir()) / db_name
    if tmpdb.exists():
        tmpdb.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdb.as_posix()}"

    from fastapi.testclient import TestClient

    from app.main import app
    from app.models_db import Base
    from app.models_db.base import engine

    Base.metadata.create_all(bind=engine)
    client = TestClient(app)
    client.post("/register", json={"email": email, "password": "test-pw-123456", "full_name": "T"})
    tok = client.post("/login", json={"email": email, "password": "test-pw-123456"}).json()["access_token"]
    return client, {"Authorization": f"Bearer {tok}"}, tmpdb, engine


def test_advisory_quality() -> bool:
    print("\n" + "=" * 72)
    print("E. ADVISORY QUALITY CHECK  (soft image -> advisory 200; force_proceed -> full analysis)")
    print("=" * 72)

    from app.core.config import resolve_model_weights

    if not resolve_model_weights().exists():
        print(f"  FAIL: {resolve_model_weights()} missing")
        return False
    src = _any_face_image()
    if src is None:
        print("  SKIP: no test image")
        return True

    client, headers, tmpdb, engine = _client_with_user("skintypeai_test_advisory.db", "test_adv@example.com")
    ok = True

    blurry = _blur_copy(src)
    with open(blurry, "rb") as fh:
        up = client.post("/upload", files={"file": (blurry.name, fh, "image/jpeg")}, headers=headers)
    assert up.status_code == 201, up.text
    image_id = up.json()["image_id"]

    # 1. force_proceed=false on a soft image -> advisory payload, no analysis
    r = client.post(f"/predict/{image_id}?force_proceed=false", json=None, headers=headers)
    adv = r.json()
    a_ok = (r.status_code == 200 and adv.get("status") == "advisory"
            and isinstance(adv.get("quality"), dict)
            and 1 <= int(adv["quality"].get("score", 99)) <= 6
            and adv["quality"].get("max_score") == 10
            and str(adv.get("temp_image_id") or "") != "")
    ok &= a_ok
    print(f"  soft + force_proceed=false -> HTTP {r.status_code}  status={adv.get('status')}  "
          f"score={adv.get('quality', {}).get('score')}  temp_image_id={adv.get('temp_image_id')}  "
          f"{'ok' if a_ok else 'BAD'}")
    if a_ok:
        print(f"    rating  : {adv['quality'].get('rating')}")
        print(f"    message : {adv['quality'].get('message')}")

    # 2. force_proceed=true (reusing temp_image_id) -> full analysis, quality embedded
    temp_id = adv.get("temp_image_id") or image_id
    r2 = client.post(f"/predict/{temp_id}?force_proceed=true", json=None, headers=headers)
    body = r2.json()
    res = body.get("results", {}) if isinstance(body, dict) else {}
    conds = res.get("conditions", {})
    b_ok = (r2.status_code == 200 and body.get("status") != "advisory"
            and set(conds) == EXPECTED_CONDITIONS
            and isinstance(res.get("quality"), dict)
            and 1 <= int(res["quality"].get("score", 0)) <= 10)
    ok &= b_ok
    print(f"  temp_image_id + force_proceed=true -> HTTP {r2.status_code}  "
          f"conditions={len(conds)}/9(+skin_type)  quality={res.get('quality', {}).get('score')}/10  "
          f"{'ok' if b_ok else 'BAD'}")

    try:
        blurry.unlink(missing_ok=True)
        engine.dispose()
        tmpdb.unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (cleanup note: {e})")
    return ok


# ============================================================ F. QR session
def test_qr_session() -> bool:
    print("\n" + "=" * 72)
    print("F. QR SESSION CHECK  (local-IP detection + GET /api/session/create-qr-session)")
    print("=" * 72)

    # --- unit: detect_local_ip() ---
    from app.routers.mobile import detect_local_ip

    ip0, confident0, note0 = detect_local_ip()
    import ipaddress
    try:
        ipaddress.ip_address(ip0)
        ip_valid = True
    except ValueError:
        ip_valid = False
    ip_ok = ip_valid and isinstance(confident0, bool) and isinstance(note0, str) \
        and (confident0 == (not ip0.startswith("127.")))
    print(f"  detect_local_ip() -> ip={ip0}  confident={confident0}  note={note0[:60] or '(none)'}  {'ok' if ip_ok else 'BAD'}")

    client, headers, tmpdb, engine = _client_with_user("skintypeai_test_qr.db", "test_qr@example.com")

    r = client.get("/api/session/create-qr-session", headers=headers)
    j = r.json() if r.status_code == 200 else {}
    sid = j.get("session_id", "")
    url = j.get("capture_url", "")
    ok = (r.status_code == 200 and len(sid) >= 16
          and url.startswith("http://") and "/mobile-capture?session=" in url and sid in url
          and isinstance(j.get("ip_confident"), bool) and isinstance(j.get("warnings"), list)
          and str(j.get("local_ip") or "") in url)
    ok &= ip_ok
    qr_png = j.get("qr_png") or ""
    ok &= qr_png.startswith("data:image/png;base64,")
    print(f"  HTTP {r.status_code}  session_id={sid[:12]}...  capture_url={url}  {'ok' if ok else 'BAD'}")
    print(f"  local_ip={j.get('local_ip')}  ip_confident={j.get('ip_confident')}  port={j.get('port')}  "
          f"qr_png={'present (%d chars)' % len(qr_png) if qr_png else 'MISSING'}")
    for w in (j.get("warnings") or []):
        print(f"    warning: {w}")

    # manual IP override
    ovr = client.get("/api/session/create-qr-session?ip=192.168.1.50", headers=headers).json()
    ovr_ok = ovr.get("local_ip") == "192.168.1.50" and "192.168.1.50:" in ovr.get("capture_url", "") \
        and ovr.get("ip_confident") is True
    ok &= ovr_ok
    print(f"  ?ip=192.168.1.50 override -> local_ip={ovr.get('local_ip')}  {'ok' if ovr_ok else 'BAD'}")

    # the mobile capture page must be served
    page = client.get(f"/mobile-capture?session={sid}")
    page_ok = page.status_code == 200 and "Take High-Res Selfie" in page.text
    ok &= page_ok
    print(f"  GET /mobile-capture -> HTTP {page.status_code}  {'ok' if page_ok else 'BAD'}")

    # status: pending before any phone upload
    st = client.get(f"/api/session/{sid}/status", headers=headers)
    st_ok = st.status_code == 200 and st.json().get("status") == "pending"
    ok &= st_ok
    print(f"  status (pre-upload) -> {st.json() if st.status_code == 200 else st.status_code}  {'ok' if st_ok else 'BAD'}")

    # --- full mobile-upload lifecycle: phone POSTs (no auth) -> status flips to ready ---
    src = _any_face_image()
    if src is None:
        print("  SKIP: no sample image for the mobile-upload lifecycle")
    else:
        with open(src, "rb") as fh:
            up = client.post(f"/api/session/{sid}/mobile-upload",
                             files={"file": ("selfie.jpg", fh, "image/jpeg")})   # no Authorization header
        up_ok = up.status_code == 201 and up.json().get("status") == "ready"
        st2 = client.get(f"/api/session/{sid}/status", headers=headers).json()
        temp_id = st2.get("temp_image_id")
        flip_ok = st2.get("status") == "ready" and str(temp_id or "").isdigit()
        ok &= up_ok and flip_ok
        print(f"  mobile-upload (no auth) -> HTTP {up.status_code}   status now '{st2.get('status')}'  "
              f"temp_image_id={temp_id}  {'ok' if (up_ok and flip_ok) else 'BAD'}")

        # the phone frame must be analysable via the normal pipeline
        if flip_ok:
            pr = client.post(f"/predict/{temp_id}?force_proceed=true", json=None, headers=headers)
            body = pr.json()
            pr_ok = pr.status_code == 200 and len((body.get("results", {}) or {}).get("conditions", {})) == 10
            ok &= pr_ok
            print(f"  analyse phone frame -> HTTP {pr.status_code}  conditions={len((body.get('results', {}) or {}).get('conditions', {}))}  {'ok' if pr_ok else 'BAD'}")

        # expired / bogus session -> 404 with a helpful message
        bad = client.post("/api/session/deadbeef/mobile-upload",
                          files={"file": ("x.jpg", b"x", "image/jpeg")})
        exp_ok = bad.status_code == 404
        ok &= exp_ok
        print(f"  mobile-upload to unknown session -> HTTP {bad.status_code}  {'ok' if exp_ok else 'BAD'}")

    try:
        engine.dispose()
        tmpdb.unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (cleanup note: {e})")
    return ok


# ============================================================ G0. FaceMesh single-pass
def test_facemesh_single_pass() -> bool:
    print("\n" + "=" * 72)
    print("G0. FACEMESH MEMOISATION  (one .process() per run_analysis request)")
    print("=" * 72)
    from app.core.config import resolve_model_weights

    if not resolve_model_weights().exists():
        print("  SKIP: model weights missing")
        return True
    crops = _sample_crops(1)
    if not crops:
        print("  SKIP: no sample crop")
        return True

    from models.skin_analysis import heuristics
    from app.services import analysis_service

    real = heuristics.facemesh_landmarks
    calls = {"n": 0}

    def counting(bgr):
        calls["n"] += 1
        return real(bgr)

    heuristics.facemesh_landmarks = counting
    try:
        prep = analysis_service.prepare_face(str(crops[0]))
        analysis_service.run_analysis(str(crops[0]), prep=prep)
    finally:
        heuristics.facemesh_landmarks = real

    ok = calls["n"] == 1
    print(f"  facemesh_landmarks() called {calls['n']}x for one run_analysis  {'ok' if ok else 'BAD (expected 1)'}")
    return ok


# ============================================================ G. XAI image overlays
def test_overlays() -> bool:
    print("\n" + "=" * 72)
    print("G. IMAGE OVERLAY CHECK  (GET /api/images/{id}/overlay -- gradcam + facemesh_zones)")
    print("=" * 72)

    import numpy as np
    import cv2

    from app.core.config import resolve_model_weights

    if not resolve_model_weights().exists():
        print(f"  FAIL: {resolve_model_weights()} missing")
        return False
    src = _any_face_image()
    if src is None:
        print("  SKIP: no test image")
        return True

    client, headers, tmpdb, engine = _client_with_user("skintypeai_test_overlay.db", "test_ov@example.com")
    with open(src, "rb") as fh:
        up = client.post("/upload", files={"file": (src.name, fh, "image/jpeg")}, headers=headers)
    assert up.status_code == 201, up.text
    image_id = up.json()["image_id"]
    orig = cv2.imread(str(src))
    oh, ow = orig.shape[:2]

    ok = True
    for mode, head, label in (
        ("zones", "acne", "measurement zones"),
        ("facemesh_zones", "acne", "zones (legacy alias)"),
        ("gradcam", "acne", "Grad-CAM · acne"),
        ("gradcam", "wrinkles", "Grad-CAM · wrinkles"),
    ):
        r = client.get(f"/api/images/{image_id}/overlay?mode={mode}&head={head}", headers=headers)
        png_ok = (r.status_code == 200 and r.headers.get("content-type") == "image/png"
                  and len(r.content) > 800)
        dims = None
        if png_ok:
            im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            dims = None if im is None else im.shape[:2]
            png_ok = dims == (oh, ow)           # overlay must match the original size
        ok &= png_ok
        print(f"  {label:22s} -> HTTP {r.status_code}  {r.headers.get('content-type')}  "
              f"{len(r.content)}B  dims={dims} vs orig=({oh},{ow})  {'ok' if png_ok else 'BAD'}")

    bad_mode = client.get(f"/api/images/{image_id}/overlay?mode=bogus", headers=headers)
    bad_head = client.get(f"/api/images/{image_id}/overlay?mode=gradcam&head=nope", headers=headers)
    missing = client.get("/api/images/999999/overlay?mode=gradcam", headers=headers)
    err_ok = bad_mode.status_code == 422 and bad_head.status_code == 422 and missing.status_code == 404
    ok &= err_ok
    print(f"  bad mode->{bad_mode.status_code}  bad head->{bad_head.status_code}  "
          f"missing image->{missing.status_code}  {'ok' if err_ok else 'BAD'}")

    try:
        engine.dispose()
        tmpdb.unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover
        print(f"  (cleanup note: {e})")
    return ok


def main() -> int:
    ok_a = test_direct_model()
    ok_b = test_opencv_engine()
    ok_c = test_api_contract()
    ok_d = test_gradcam()
    ok_e = test_advisory_quality()
    ok_f = test_qr_session()
    ok_g0 = test_facemesh_single_pass()
    ok_g = test_overlays()
    print("\n" + "=" * 72)
    print(f"RESULT:  direct-model={'PASS' if ok_a else 'FAIL'}   "
          f"opencv-engine={'PASS' if ok_b else 'FAIL'}   "
          f"api-contract={'PASS' if ok_c else 'FAIL'}")
    print(f"         grad-cam={'PASS' if ok_d else 'FAIL'}   "
          f"advisory-quality={'PASS' if ok_e else 'FAIL'}   "
          f"qr-session={'PASS' if ok_f else 'FAIL'}")
    print(f"         facemesh-1pass={'PASS' if ok_g0 else 'FAIL'}   "
          f"image-overlays={'PASS' if ok_g else 'FAIL'}")
    print("=" * 72)
    return 0 if all((ok_a, ok_b, ok_c, ok_d, ok_e, ok_f, ok_g0, ok_g)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
