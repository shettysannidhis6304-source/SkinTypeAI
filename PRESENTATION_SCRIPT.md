# SkinType AI / SkinIntelli — Technical Architecture Presentation Script

> Spoken, first-person, panel-facing. Stage cues in `[brackets]`. Every claim is
> anchored to a file in this repository.

---

## 0. Opening (30 seconds)

[Show title slide: "SkinType AI — an evidence-gated facial skin analysis pipeline"]

"Good morning. What I'm presenting is a full-stack computer-vision system that takes
a single face photograph and returns a structured, per-condition skin analysis plus
a rule-based skincare plan. I'm going to spend the entire time on engineering: the
component topology, why each technology was chosen over its obvious alternative, how
a request flows byte-by-byte from the browser to a model and back into SQLite, the
data model, and the concurrency and security envelope.

The one architectural principle I want you to hold onto is this: **the system never
fabricates output.** Every field the API returns is traceable to either a named model
checkpoint, a documented deterministic formula, or an explicit `not_available` marker.
That constraint drove most of the design decisions you'll see."

---

## 1. System Architecture & Design Patterns

[Show architecture diagram: four boxes — Browser SPA / FastAPI app / SQLite / Model layer — plus an external "Open-Meteo" cloud]

### 1.1 The four components

"There are four moving parts.

**One — the frontend.** [Point to `frontend/index.html`] It's a zero-build,
vanilla-JavaScript single-page application. Three script tags:
`js/config.js`, `js/api.js`, `js/app.js` — that's the whole client. No React,
no bundler, no `node_modules`. `app.js` is a hash-router SPA: routes like
`#/dashboard`, `#/scan`, `#/result/:id` are matched with a regex in the `route()`
method [point to `frontend/js/app.js:98`] and each maps to a render function that
swaps `innerHTML` on a root node.

**Two — the backend.** [Point to `backend/app/main.py`] A FastAPI application
served by Uvicorn on port 8000. It mounts three routers — `auth`, `upload`,
`dashboard` — and one bare `/health` endpoint. That's it. FastAPI's dependency
injection is the main structural pattern: `get_db` yields a per-request SQLAlchemy
session [point to `backend/app/models_db/base.py:25`], and `get_current_user`
[point to `backend/app/core/security.py:49`] is a dependency that every protected
route lists in its signature, so authentication is declarative rather than
middleware-based.

**Three — persistence.** A single SQLite file, accessed through SQLAlchemy 2.0's
typed ORM. Three tables: `users`, `images`, `predictions`
[point to `backend/app/models_db/`].

**Four — the model layer.** [Point to `models/`] This lives *outside* the backend
package, at the repository root, and is imported lazily. It's a stack of
independently-loaded models, each wrapped in a thin adapter class that returns
plain Python dicts — never a torch tensor — so the backend has no compile-time
dependency on torch, transformers, or ultralytics.

There is also one **external service**: Open-Meteo, a keyless weather / air-quality /
geocoding API, called server-side from the `dashboard` router
[point to `backend/app/routers/dashboard.py:36`]."

### 1.2 How they communicate, and why

[Show "Protocol" table: REST/JSON, multipart, JWT bearer, no WS/gRPC/queue]

"Every hop is **HTTP/1.1 with JSON bodies**. Specifically:

- The browser talks to the API over `fetch` with a JSON content type for
  normal calls and `multipart/form-data` for the one file upload
  [point to `frontend/js/api.js:89`].
- Auth is a **stateless JWT bearer token** in the `Authorization` header.
- Uploaded images are served back to the browser as **binary blobs** — the client
  does an authenticated `fetch`, gets an `image/*` response, and wraps it in
  `URL.createObjectURL` [point to `frontend/js/api.js:105`], because a plain
  `<img src>` can't carry the bearer token.
- The backend calls Open-Meteo over HTTPS with `httpx.AsyncClient` and a
  6-second timeout [point to `backend/app/routers/dashboard.py:39`].

**What is deliberately absent: no WebSockets, no gRPC, no message queue, no Redis,
no pub/sub, no Celery.** I want to defend that directly because it's the first thing
a reviewer challenges.

The workload is: one user, uploading one photo, waiting for one synchronous result.
There is no streaming data, no fan-out, no server-push requirement, no
multi-service graph that would justify gRPC's schema/transport overhead. A
message queue is the textbook answer for slow inference — and I'll come back to it
as the number-one scaling change — but for the current target, which is an
academic single-node deployment, a queue would add a broker process, a worker
pool, a result backend, and a polling protocol on the client, to solve a problem
(concurrent slow requests) that the current user model doesn't have. The
architecture is intentionally the simplest thing that satisfies the 'no fabricated
output, real model, end-to-end' requirement. Every complexity I left out has a
named trigger condition for adding it back."

### 1.3 Design patterns actually in the code

"Concretely, the patterns you'll see when you read it:

- **Adapter / façade per model.** `HFImageClassifier`, `YoloDetector`,
  `WrinkleSegmenter`, and `CustomSkinModel` all expose a `predict()` or `detect()`
  that returns a dict. [Point to `models/skin_analysis/classifier.py`,
  `models/yolo_detector.py`]
- **Lazy singleton with double-checked locking.** [Point to
  `models/skin_analysis/classifier.py:66`] `get_classifier()` builds the model on
  first use under a `threading.Lock`, then every later request reuses it. Same in
  `wrinkle_segmenter.py:90`. Models are 300–350 MB each; you load them once per
  process lifetime.
- **Strategy with fallback.** [Point to
  `backend/app/services/analysis_service.py:139`] The skin-type / acne-severity /
  pigmentation predictions come from *our own trained model* when
  `weights/custom_skin_model.pt` exists, and **automatically fall back** to the
  Hugging Face ViTs when it doesn't. `_custom_load_failed` is a latch so a broken
  checkpoint is not retried on every request.
- **Registry / single-source-of-truth.** [Point to
  `models/skin_analysis/registry.py`] Every model's provenance, licence, input
  size, and reported metrics is a frozen dataclass in one file, mirrored as JSON
  in `weights/skin_analysis_models.metadata.json`.
- **Pipeline with per-stage isolation.** [Point to
  `analysis_service.py:402` `run_analysis`] Each model call is in its own
  `try/except`; a failure writes `{"status": "error"}` for that one condition and
  the endpoint still returns everything else.
- **Repository-free ORM.** Routers use the SQLAlchemy session directly. For three
  tables and five queries, a repository layer would be ceremony."

---

## 2. Tech Stack Justification

[Show stack table: layer → choice → the alternative rejected]

### 2.1 Backend framework — FastAPI + Uvicorn + Pydantic

"**FastAPI 0.111 on Uvicorn 0.30.** [Point to `requirements.txt`]

Why not Flask or Django? Three reasons that are load-bearing here:

1. **Pydantic v2 request/response models give me the 'no fabrication' guarantee
   for free.** [Point to `backend/app/schemas/prediction.py`] The `SkinAnalysis`
   model declares that `conditions` is a dict of `ConditionEntry`, each with a
   `status` that is a `Literal["detected","estimated","not_available","error"]`.
   If my service layer ever tried to return a malformed or partial result, the
   response serializer raises. The schema *is* the contract with the frontend and
   with the recommendation engine.
2. **Declarative dependency injection.** Auth, DB session, and current-user
   lookup are function parameters, not middleware. That makes each endpoint's
   security posture readable in its signature.
3. **Native `async` for the one place it matters** — the Open-Meteo calls, which
   are I/O-bound and benefit from `httpx.AsyncClient`. The ML endpoints are
   deliberately sync `def`, which FastAPI runs in a threadpool.

Django was rejected because its ORM, admin, template, and migration machinery are
dead weight for a JSON API with three tables. Flask was rejected because I'd be
hand-rolling request validation and OpenAPI docs that FastAPI generates from the
type hints — and the interactive `/docs` page is how the whole API was tested
during development."

### 2.2 Database — SQLite now, Postgres-ready

"**SQLite via SQLAlchemy 2.0.** [Point to `backend/app/models_db/base.py`]

The ORM is configured so the engine URL is the *only* thing that changes between
SQLite and Postgres — `connect_args={"check_same_thread": False}` is applied
conditionally on the `sqlite` prefix [point to `base.py:16`], and the
`.env.example` file documents the exact `postgresql+psycopg://` URL to switch to
[point to `.env.example`].

Why SQLite for v1? Zero configuration, ships with Python, the entire database is
one file that `database/init_db.py` creates from the ORM metadata, and the write
volume is one row per scan. Why *not* SQLite in production? Single-writer locking —
I'll cover that under concurrency. The point is the migration path is a
one-line `.env` change, not a rewrite, because no raw SQL and no SQLite-specific
column types are used. The `results` and `context` fields are SQLAlchemy `JSON`
columns [point to `backend/app/models_db/prediction.py:30`], which map to native
`JSONB` on Postgres."

### 2.3 The ML stack — and why it's a *stack*, not one model

[Show the model table from `SETUP_FOR_FRIEND.md`: model-backed vs estimated vs not-available]

"This is the core of the engineering story. There are **six inference paths**,
chosen per condition based on what evidence actually exists:

| Condition | Path | Technology | Licence |
|---|---|---|---|
| Skin type, dryness, oiliness | Classification | Our `CustomSkinModel` (MobileNetV3-Small frozen trunk + trained head) — or HF ViT `dima806/skin_types_image_detection` as fallback | Ours / Apache-2.0 |
| Acne severity | Classification | Our custom head — or HF ViT `imfarzanansari/skintelligent-acne` | Ours / MIT |
| Acne lesions | Object detection | YOLOv8s (`glowlytics`), Ultralytics runtime | MIT |
| Wrinkles | Semantic segmentation | U-Net with ResNet50 encoder (`ffhq-wrinkle-unet`) | **CC BY-NC-SA — academic only** |
| Pigmentation | Classification | Our custom head (weak-label trained) | Ours |
| Redness, pores, blackheads, dark spots | Deterministic CV | OpenCV colour/texture math — **labelled 'estimated', never 'detected'** | Ours |
| Blackheads/dark-spots (as ML) | — | `not_available` — no validated public model or dataset exists | — |

**Why a stack:** these are genuinely different computer-vision problems. Acne
severity is whole-image classification. Acne lesions are localisation — you need
boxes to draw an overlay. Wrinkles are a per-pixel mask. And for redness and pores,
**there is no trained model or labelled facial dataset that survives scrutiny**
[point to `registry.py:189` `UNAVAILABLE_CONDITIONS` and the audit note], so those
are explicit classical-CV estimates with a reliability figure, not a fake
confidence score.

**Why `timm` for the backbone:** [point to `requirements.txt` and
`models/custom_multi_task.py:86`] `timm.create_model("mobilenetv3_small_100",
pretrained=True, num_classes=0)` gives me an ImageNet-pretrained feature extractor
with a pooled output vector in one line, and the identical call with
`pretrained=False` rebuilds the architecture at load time with **no network
access** — the weights come entirely from our `.pt`. That offline guarantee is
why the backend can boot on a machine that has never touched Hugging Face.

**Why MobileNetV3-Small and not EfficientNetV2-S:** the original spec called for
EfficientNetV2-S [point to `models/efficientnet_classifier.py`, which is still in
the repo for rollback]. I moved to MobileNetV3-Small for the custom model because
the dev machine is **CPU-only**. MobileNetV3-Small emits a 1024-d vector, the
trunk is ~2.5M parameters, and — critically — because the trunk is **frozen**, I
run it over the whole dataset *once*, cache the feature vectors
[point to `training/train_custom.py:96` `build_feature_cache`], and then train only
the three linear heads on cached features. End-to-end training is **4.6 seconds**
[point to `runs/training_report.txt`]. EfficientNetV2-S at 300px on CPU would be
minutes per epoch for no accuracy benefit at this data scale.

**Why Ultralytics for YOLO:** it's the reference implementation of YOLOv8/v11,
the checkpoint format is a single `.pt`, and inference is one `model.predict()`
call. `models/yolo_detector.py` wraps it so boxes come back as
`[x1,y1,x2,y2]` in *source-image pixel coordinates* — the frontend needs exactly
that to stroke rectangles on a `<canvas>` [point to `frontend/js/app.js:767`
`drawResultCanvas`].

**Why MediaPipe + OpenCV for preprocessing:** [point to
`models/skin_analysis/heuristics.py:60` `face_region`] MediaPipe's face detector is
fast on CPU and gives a relative bounding box; OpenCV does the crop and a
YCrCb+HSV skin-colour mask so the heuristics and the wrinkle model operate on skin
pixels, not hair and background."

### 2.4 Auth libraries

"**`python-jose[cryptography]` for JWT, `passlib[bcrypt]` + `bcrypt` for
password hashing.** [Point to `requirements.txt`, `backend/app/core/security.py`]
Standard, audited choices. HS256 symmetric signing because there's one service
signing and verifying its own tokens — RS256's asymmetric keypair solves a
multi-party verification problem we don't have."

### 2.5 Frontend — the deliberate non-choice

"No framework was a **decision**, per the project brief [point to `CLAUDE.md`:
'existing HTML/CSS/JS site — DO NOT REDESIGN']. The integration surface was one
`api.js` module. Adding a framework would have meant a build pipeline and a
rewrite for zero functional gain — the UI is a dozen screens rendered from JSON.
`config.js` centralises the one thing that changes between environments: the API
base URL [point to `frontend/js/config.js:4`]."

---

## 3. Data Flow & State Lifecycle

[Show sequence diagram for the scan operation, 8 numbered arrows]

### 3.1 Trace one: a full skin scan, client to database and back

"Let me trace the expensive operation end to end.

**Step 1 — capture.** [Point to `frontend/js/app.js:421` `viewScan`] The user
picks a file or captures a webcam frame. A webcam frame is drawn to an offscreen
`<canvas>` and exported with `canvas.toBlob(..., 'image/jpeg', 0.92)` into a
`File` object [point to `app.js:507`]. There is deliberately **one** code path
from here — camera and upload converge on the same `File`.

**Step 2 — `POST /upload`.** [Point to `frontend/js/api.js:89`] The client builds
a `FormData`, appends the file, and POSTs it with the bearer token. On the server
[point to `backend/app/routers/upload.py:34`]:
- `get_current_user` runs first as a dependency: it pulls the token from the
  `Authorization` header via FastAPI's `HTTPBearer`, calls
  `decode_access_token` [point to `security.py:40`], which does
  `jwt.decode(token, secret, algorithms=["HS256"])`, extracts `sub` (the user id
  as a string), and does `db.get(User, int(user_id))`. No user → 401.
- The route checks `file.content_type` against
  `{"image/jpeg","image/png","image/webp"}` → 415 if not.
- It generates a **collision-proof, traversal-proof** filename:
  `f"{user.id}_{uuid4().hex}{ext}"` [point to `upload.py:44`].
- `shutil.copyfileobj(file.file, out)` streams the bytes to
  `backend/uploads/`.
- An `Image` row is inserted storing **only the filename** (not an absolute path,
  so the DB is portable across machines — `resolve_upload_path` reconstructs the
  full path on read [point to `config.py:82`]).
- Response: `{"image_id": ..., "filename": ...}`.

**Step 3 — `POST /predict/{image_id}`** with an optional `ScanContext` JSON body
(the questionnaire + weather). [Point to `upload.py:59`]
- Auth again (stateless — every request re-verifies).
- **Ownership check:** `image.user_id != user.id` → 404 (not 403 — we don't
  confirm the row exists to a non-owner).
- **Model-availability gate:** it stats the three required model files; if any is
  missing it returns **503 with the list of missing files** [point to
  `upload.py:74`]. The app can boot and serve `/upload` without models present;
  only `/predict` hard-requires them.
- **Lazy import:** `from app.services.analysis_service import run_analysis` happens
  *inside the function* [point to `upload.py:91`] so importing torch is deferred
  until the first prediction.

**Step 4 — the inference pipeline.** [Point to `analysis_service.py:402`]
1. `heuristics.face_region(path)` — `cv2.imread`, MediaPipe detect, crop with a
   6–10% margin, build the skin mask. Falls back to the whole image if no face
   (common for tight ACNE04-style crops).
2. `_predict_custom(face, path)` — runs our multi-task model once on the crop.
   Returns `None` if the checkpoint is absent → downstream blocks switch to the
   HF ViT path transparently.
3. `_skin_type_block` — softmax distribution over {dry, normal, oily,
   combination}; `combination` probability is split half to dryness, half to
   oiliness so the downstream scores stay comparable across both paths
   [point to `analysis_service.py:210`].
4. `_acne_block` — **fuses two signals**: a whole-face severity index (0–4) from
   the classifier, and a lesion count + boxes from YOLO. The final `score` is
   `max(severity/4, min(count/10, 1))` [point to `analysis_service.py:347`]. The
   note explicitly says the two signals can disagree.
5. `_wrinkle_block` — U-Net sigmoid mask ≥ 0.5, intersected with the skin mask,
   reduced to `wrinkle_area_fraction`, then mapped so 2% of skin area → score 1.0
   [point to `analysis_service.py:380`].
6. CV engine — `redness_estimate`, `pore_estimate`, `blackhead_estimate`,
   `dark_spot_estimate` from `heuristics.py`. Each returns a 0–1 score, a soft
   3-band distribution, and a *measurement-reliability* confidence scaled by
   skin-pixel coverage [point to `heuristics.py:111` `_band_distribution`].
7. Every entry is forced through `_normalize` to exactly eight canonical keys
   [point to `analysis_service.py:92`].
8. `compute_skin_health_index` — a **documented deterministic formula**
   [point to `analysis_service.py:158` `SHI_FORMULA`]:
   `SHI = 100 − 30·(acne_severity/4) − 15·wrinkle − 12·clamp((dryness−0.34)/0.66)
   − 10·redness − 8·pores`, clamped to [0,100]. If no model-backed condition
   succeeded, SHI is `null` — never a guessed number.

**Step 5 — recommendation.** [Point to
`backend/app/services/recommendation_engine.py:176`] `build_recommendation(results,
context)` is a **pure function**. It only emits guidance for conditions with
status `detected` or `estimated`; `not_available` and `error` produce nothing. It
bands each score, looks up `CONDITION_RULES[condition][band]` — a static dict of
routine steps and ingredients — dedupes, then layers climate/lifestyle tips from
the questionnaire on top [point to `recommendation_engine.py:107` `_context_layer`].
The `context` **never touches a model** — that boundary is stated in three
docstrings.

**Step 6 — persist.** [Point to `upload.py:99`] One `Prediction` row:
`results` (the full analysis dict → JSON column), `context` (the questionnaire →
JSON column), `model_version`. The recommendation is **not** persisted — it's
recomputed on read because it's a pure function of the two stored fields.

**Step 7 — response.** `PredictionResult` — validated against the Pydantic schema —
with `image_url: "/images/{id}"`.

**Step 8 — render.** [Point to `frontend/js/app.js:642` `viewResult`] The client
routes to `#/result/:id`, fetches the image blob, draws it to a `<canvas>`, and
strokes the YOLO boxes from `results.conditions.acne.detail.detections` using the
source-pixel coordinates scaled by the canvas ratio [point to `app.js:767`]."

### 3.2 Trace two: authentication state lifecycle

"Shorter, but it's the security-critical path.

- **Register** [point to `backend/app/routers/auth.py:18`]: Pydantic validates
  `email` is an `EmailStr` and `password` is ≥ 8 chars. Duplicate email → 400.
  `hash_password` runs bcrypt (cost factor default 12) and stores only the hash.
- **Login** [point to `auth.py:33`]: fetch by email, `verify_password` does a
  constant-time bcrypt compare. Success → `create_access_token(subject=str(user.id))`
  → `{"sub": user_id, "exp": now + 1 day}` signed HS256.
- **Client storage** [point to `frontend/js/api.js:9`]: token goes to
  `localStorage` under `skinintelli_token`. Every `_request` attaches it as
  `Bearer`.
- **Expiry / 401 handling** [point to `api.js:48`]: any 401 on a non-auth route
  clears the token and dispatches an `api:unauthorized` event; `app.js` catches it,
  nulls `App.state.user`, and redirects to login [point to `app.js:83`].
- **Logout**: `localStorage.removeItem` + revoke any object URLs. No server call —
  the token just isn't sent anymore and expires on its own.

There is **no server-side session, no refresh token, and no revocation list.** The
token is valid until `exp`. That's the classic stateless-JWT tradeoff and I'll
defend it and its limits in the grilling section."

---

## 4. Data Models & Schemas

[Show ER diagram: users 1─∞ images, users 1─∞ predictions, images 1─1 predictions]

### 4.1 The three tables

"[Point to `backend/app/models_db/`]

**`users`** — `id` PK, `email` (`String(255)`, `unique=True`, `index=True`),
`hashed_password`, `full_name` nullable, `created_at`.

**`images`** — `id` PK, `user_id` FK → users, `file_path` (the stored filename),
`original_filename`, `uploaded_at`.

**`predictions`** — `id` PK, `user_id` FK, `image_id` FK **with `unique=True`**,
`results` (`JSON`, not null), `context` (`JSON`, nullable), `model_version`,
`created_at`.

Relationships [point to `user.py:24`]: `User.predictions` cascades
`all, delete-orphan`; `Image.prediction` is `uselist=False` — a strict one-to-one."

### 4.2 The normalization / denormalization decision

"This is the schema question I most expect to be pushed on.

**`users` and `images` are normalized.** Standard 3NF — an image belongs to a
user, a prediction references both.

**`predictions.results` is deliberately denormalized into a JSON blob.**
[Point to the docstring in `backend/app/models_db/prediction.py`]. I did *not*
break out `acne_severity`, `skin_type`, `wrinkle_score` etc. into typed columns.
Three reasons:

1. **Schema stability across three historical result shapes.** [Point to
   `backend/app/schemas/prediction.py:30`] `/history` has to round-trip records
   from the current multi-model pipeline, an older YOLO-only shape, and a legacy
   EfficientNet shape. A JSON column absorbs all three; a wide typed table would
   need a migration and nullable columns for every retired field.
2. **The result is a document, consumed whole.** The frontend renders the entire
   `conditions` map; nothing queries `WHERE acne_severity > 3`.
3. **v1 timeline.** The brief [point to `CLAUDE.md`] explicitly defers typed
   condition columns: *"add typed columns later if you need to query/filter by a
   specific condition efficiently."*

**The cost I'm accepting:** no efficient aggregate query like "average skin health
index over time" — that requires loading rows and parsing JSON in Python. On
SQLite the `JSON` column is just `TEXT`; on Postgres it's `JSONB` and I *could*
index into it with an expression index. That's the documented upgrade trigger."

### 4.3 Indexing strategy

"Current indexes: **primary keys, `users.email` unique index, and the implicit
index SQLite creates for `predictions.image_id` because it's `UNIQUE`.**

**What is NOT indexed and should be:** `predictions.user_id` and
`predictions.created_at`. The `/history` query is
`WHERE user_id = ? ORDER BY created_at DESC` [point to `upload.py:124`] — that's a
full scan plus a sort today. At a few dozen rows per user it's sub-millisecond, so
it wasn't worth a migration in v1, but a composite index on
`(user_id, created_at DESC)` is the first thing I'd add. I'm flagging it before
you do."

### 4.4 Consistency & transactions

"**Transaction boundary = one HTTP request.** [Point to
`backend/app/models_db/base.py:18`] `sessionmaker(autocommit=False,
autoflush=False)`. Each route does its work and calls `db.commit()` exactly once;
`get_db`'s `finally` closes the session.

The `/predict` handler [point to `upload.py:96`] runs inference *before* opening
any write, then does a single `add` + `commit` + `refresh`. So inference failure
leaves **zero** partial rows — the `Image` row from `/upload` already exists and
is simply left without a `Prediction`, which is a valid state.

**Consistency model:** SQLite is fully serializable — it takes a database-level
write lock. There are no multi-statement race windows because there are no
multi-row writes. The one integrity risk is the filesystem: `/upload` writes the
file *then* the row, non-atomically. If the process is killed between them you get
an orphan file with no DB row. Harmless (it's just disk), and a periodic sweep of
`uploads/` against the `images` table would reconcile it. There is no
write-ahead-log configuration or connection pool tuning because a single writer
doesn't need it."

### 4.5 The request/response schemas

"[Point to `backend/app/schemas/prediction.py`] Worth noting the schema does real
work beyond documentation:

- `ConditionEntry.status` is a `Literal` of four values — an invalid status is a
  serialization error.
- `Detection` serializes its `class_` field as `"class"` via
  `serialization_alias` because `class` is a Python keyword [point to
  `prediction.py:52`].
- `ScanContext` has `extra="ignore"` — unknown questionnaire fields are dropped,
  not rejected, so the frontend can send more than the model uses.
- `HistoryRecord.results` is typed as a bare `dict`, not `SkinAnalysis`, *because*
  of the three-shapes problem — history intentionally relaxes validation on the
  way out."

---

## 5. Security, Performance & Concurrency

[Show "Security envelope" slide: what's covered / what's explicitly deferred]

### 5.1 Authentication & authorization

"**AuthN:** stateless JWT, HS256, `python-jose`. Secret from
`JWT_SECRET_KEY` env var [point to `backend/app/core/config.py:26`], with a
loud `dev-secret-change-me` default and a `.env.example` note to generate a real
one with `secrets.token_urlsafe(48)`.

**AuthZ:** two layers. `get_current_user` gates *access* [point to
`security.py:49`]. Then every resource route does an explicit **owner check** —
`image.user_id != user.id` on `/predict`, `/images/{id}` [point to
`dashboard.py:69`]. There are no roles; every user sees only their own rows.

**Password storage:** bcrypt via passlib, work factor 12, per-password salt.
Plaintext is never persisted or logged."

### 5.2 Known security limitations — stated, not hidden

"I'll enumerate these because you'll find them anyway:

1. **CORS is `allow_origins=["*"]`** [point to `backend/app/main.py:25`] with a
   code comment saying 'tighten before any real deployment.' Fine for
   local-origin dev, must become an allow-list in production.
2. **Token in `localStorage`**, so an XSS bug would expose it. Mitigation posture:
   the frontend never uses `eval` or injects un-escaped user HTML (there's an
   `esc()` helper), and there are no third-party script tags except Google Fonts
   and Font Awesome from CDNs. `httpOnly` cookies would be the hardening step,
   at the cost of needing CSRF protection.
3. **Upload size is configured but not enforced.** [Point to
   `config.py:35` `max_upload_size_mb`] `shutil.copyfileobj` streams without a
   cap. A hostile client could POST a huge file. Real bug; the fix is a
   `MAX_UPLOAD_SIZE` check on `file.size` or a chunked read with a running total.
4. **Content-type is trusted, not verified.** The `/upload` allow-list checks the
   declared MIME type, not magic bytes. `cv2.imread` / `PIL.Image.open` will fail
   on a non-image later, but validating the signature at the door is better.
5. **No rate limiting** anywhere — no login throttle, no per-user scan quota.
6. **No token revocation.** A leaked token is valid for up to 24 hours.

None of these is load-bearing for the academic deployment target, and each has a
one-paragraph fix. What I did *not* do is ship fake mitigations."

### 5.3 SQL injection, path traversal, SSRF

"**SQLi:** not possible via the ORM — every query is parameterized; there is no
string-built SQL anywhere.

**Path traversal on upload:** the stored filename is
`{user_id}_{uuid4().hex}{ext}` — the client's filename is never used for the path,
only recorded as `original_filename` metadata [point to `upload.py:43`].

**Path traversal on read:** `resolve_upload_path` [point to `config.py:82`]
joins under `upload_dir`; legacy absolute paths from other machines fall back to
`upload_dir / p.name`.

**SSRF via the weather endpoint:** [point to `dashboard.py:91`] the user supplies
`lat`/`lon`/`place`, and those only ever become query params to the three
hard-coded Open-Meteo hostnames — the user can't control the target host."

### 5.4 Concurrency model

[Show "Concurrency" diagram: Uvicorn → threadpool → GIL-bound CPU inference → SQLite single writer]

"Honest description of what actually happens:

- **Uvicorn runs one worker process** by default [point to `run-backend.bat:19` —
  no `--workers` flag].
- The ML endpoints are **sync `def`**, so FastAPI dispatches them to an
  `anyio` threadpool (default 40 threads).
- But inference is **CPU-bound Python**, so the **GIL serializes the actual
  compute.** `torch.set_num_threads(4)` [point to `analysis_service.py:127`] lets
  a single inference use 4 cores for the tensor ops, but two concurrent scans
  don't run in parallel — the second waits.
- **Model loading is guarded** by double-checked `threading.Lock`s [point to
  `classifier.py:66`, `wrinkle_segmenter.py:90`] so a burst of first-requests
  builds each model once, not N times.
- **SQLite** allows concurrent reads but **serializes writes** with a file lock.
  `check_same_thread=False` is safe here *only because* every session is
  request-scoped and never shared across threads [point to `base.py:13` comment].

**Measured behaviour** [point to `SETUP_FOR_FRIEND.md`]: first scan ≈ 30 s (cold
model load), ≈ 4 s per scan after. That 4 s is the concurrency unit — the server
comfortably handles one active user and degrades linearly (queue-behind-the-GIL)
under concurrent load. It does **not** fall over; it slows down."

### 5.5 Error handling

"Layered:

- **Pipeline level** [point to `analysis_service.py:_err`]: each model in its own
  `try/except`; failure → `{"status":"error","detail":{"error": repr(exc)}}` for
  that condition, `logger.exception` for the trace, and the endpoint returns the
  rest.
- **Route level:** `HTTPException` with correct status codes — 401, 404, 415, 503.
- **External calls** [point to `dashboard.py:145`]: every Open-Meteo failure is
  caught and returns a `WeatherReading` with `source: "unavailable"` and a
  human note — missing fields stay `null`, never invented.
- **Client level** [point to `frontend/js/api.js:27`]: `AbortController` with a
  120-second timeout, and `_friendly()` maps status codes to readable messages.
  A 503 surfaces the server's 'models not available' text verbatim."

### 5.6 Caching policy

"**There is no caching layer, and that's a conscious choice with one exception
I'd add first.**

- No Redis, no in-memory response cache.
- The **models themselves are the cache** — loaded once, held for process
  lifetime as module singletons. That's the expensive thing and it *is* cached.
- Image blobs are re-fetched from disk on every result view [point to
  `app.js:412` `hydrateImages`] — a browser `Cache-Control` header on
  `/images/{id}` is a trivial win, since an uploaded image is immutable.
- Predictions are **recomputed on `/history`** for the recommendation only —
  the `results` are stored, but `build_recommendation` re-runs per row [point to
  `upload.py:141`]. Deliberate (keeps the schema unchanged) but it's O(n) pure-Python
  work on history load. Persisting the recommendation JSON, or memoizing on
  `(results_hash, context_hash)`, is the documented optimization."

### 5.7 Scalability bottlenecks — ranked

[Show "Bottleneck ladder" slide]

"In the order they'd bite:

1. **Single synchronous inference process.** No queue. Concurrent users serialize
   behind the GIL. **Fix:** move `run_analysis` to a worker pool (Celery/RQ/Arq +
   Redis), make `/predict` enqueue-and-poll or push over SSE. This is the
   number-one production change and the architecture already isolates
   `run_analysis` as a single pure-ish function to make that lift small.
2. **SQLite single writer.** **Fix:** the one-line `DATABASE_URL` switch to
   Postgres — already wired.
3. **`predictions.image_id UNIQUE`** [point to `prediction.py:29`] means you
   cannot re-run a prediction on the same image without deleting the old row.
   Intentional 1:1 for v1; drop the constraint and add a `created_at` sort if
   re-analysis is needed.
4. **Model memory footprint.** Two ViTs at 343 MB + a ResNet50 U-Net at 335 MB +
   YOLOv8s — well over 1 GB resident once all paths warm. On a 6 GB-VRAM / 8 GB-RAM
   box that's the ceiling on how many model variants can coexist. The custom
   MobileNet path exists partly to shrink this.
5. **No DB indexes on the history query path** (§4.3).
6. **Unbounded upload** (§5.2).
7. **Lazy imports on the request path** — the first `/predict` after boot pays the
   torch/transformers import cost (~seconds) on top of model load."

---

## 6. Close (20 seconds)

"To summarise the engineering: a deliberately flat REST architecture — vanilla-JS
SPA, FastAPI, SQLite-to-Postgres-ready ORM, and a six-path model stack behind a
single pipeline function. Every technology choice trades away capability the
current workload doesn't need — a queue, a framework, a cache — in exchange for a
system small enough to reason about completely. And the invariant that shaped all
of it: the API returns evidence or it returns `not_available`. It never guesses.

Happy to go deep on any of it."

---
---

# HOSTILE DEFENSE PREP — Professor Grilling Section

> 12 questions the panel is most likely to ask, with the exact defense, the
> limitation acknowledged, and the mitigation already in the code.

---

### Q1. "Your skin-type labels are distilled from the same Hugging Face ViT you claim your custom model *replaces*. Isn't that circular — your model can't be better than its teacher, so what was the point?"

**Answer.** "You've identified it correctly and it's documented as the weakest
link — `training/train_custom.py`'s viva notes and `label_provenance.json` both
state skin-type labels are distilled and domain-biased.

Three defenses. **First**, only the skin-type head is distilled. The **acne head
is trained on real ACNE04 Hayashi grades** — ground truth parsed from the
`levleN_` filename prefix [`scripts/prepare_dataset.py:92`] — and the pigmentation
head on heuristic weak labels. So the model as a whole is not a pure copy of one
teacher.

**Second**, the point of the custom model isn't accuracy — the brief
[`CLAUDE.md`] explicitly says accuracy is a later phase. The point is
**removing the runtime dependency on Hugging Face**: `load_custom_model` rebuilds
the backbone with `pretrained=False` and loads a 5 MB `.pt` with zero network
access [`models/custom_multi_task.py:153`]. The three 343 MB ViTs become one
small local checkpoint.

**Third**, distillation *can* beat the teacher on latency and can regularize —
the student sees the teacher's soft disagreements averaged over augmentation. But
I'm not claiming that here. The honest position is: skin-type is indicative,
labelled as such in the API `note` field, and the HF ViT remains the automatic
fallback if the checkpoint is missing."

---

### Q2. "Two concurrent users hit `/predict`. Walk me through exactly what happens, and prove you don't corrupt data."

**Answer.** "Uvicorn is one process, one worker [`run-backend.bat`]. Both
requests are sync `def`, so FastAPI hands each to a separate thread in the anyio
threadpool.

**Compute:** both call into torch. The GIL means only one executes Python
bytecode at a time; the tensor kernels release the GIL so there's some real
overlap, but effectively user B's scan starts finishing after user A's. Latency
roughly doubles for B. No parallelism benefit — acknowledged.

**Data safety:** each thread gets its **own** SQLAlchemy session from `get_db`
[`base.py:25`] — sessions are never shared. Each writes exactly one `Prediction`
row with one `commit`. SQLite takes a write lock, so the two commits serialize;
the second blocks microseconds for the first. `image_id` is `UNIQUE` and each
user is predicting on their own distinct image, so there's no key collision.
**No shared mutable state is written** — the model singletons are read-only after
load, and their construction is behind a `threading.Lock` with a double-check
[`classifier.py:66`]. So: slower, yes; corrupt, no."

---

### Q3. "The wrinkle model is CC BY-NC-SA. You're presenting this as a system. Isn't the whole thing legally un-shippable?"

**Answer.** "For a **commercial product, yes** — and that's why it's isolated and
switchable. `WRINKLE_ENABLED` is a config flag [`config.py:59`]; set it false and
the wrinkle condition returns `not_available` with a clear note, and nothing else
in the pipeline changes. The licence is recorded in three places:
`registry.py`, `skin_analysis_models.metadata.json`, and the runtime `note` on
every wrinkle result.

For **this academic project**, CC BY-NC-SA non-commercial use is exactly the
permitted case. Every *other* model in the stack — the two ViTs (Apache-2.0 /
MIT), the YOLO detector (MIT), our own code — is commercial-safe. So the shippable
subset is: disable one flag, lose one condition. The architecture was built
anticipating this question."

---

### Q4. "Your YOLO acne detector reports mAP50 of 0.47 from the model card, and your *own* earlier YOLO experiment scored mAP50 0.048. How is a detector that's wrong more than half the time acceptable in a 'no fabrication' system?"

**Answer.** "Because the system is explicit about exactly that uncertainty.

The 0.048 model was our own 3-epoch CPU proof run — it's in the repo
(`weights/skin_yolo11n.pt.version` literally says `-proof`) purely for rollback,
and it is **not on the active path** [`config.py:63`]. The active detector is the
glowlytics YOLOv8s at author-reported mAP50 0.47.

'No fabrication' does not mean 'always correct.' It means the output is a real
model inference, surfaced with its provenance, and **not dressed up**. The acne
result `note` says verbatim that boxes are 'acne-like lesions' from a detector
with 'author-reported mAP50 ~0.47, not a diagnosis' and that 'the two signals can
disagree' [`analysis_service.py:357`]. We spot-checked it: 4–6 boxes on severe
ACNE04 images, **0 boxes on a clean selfie** — no phantom detections
[`registry.py:114`].

The design mitigation is **signal fusion**: the acne score is
`max(severity_index/4, lesion_count/10)` [`analysis_service.py:347`], so a bad
box count can't by itself dominate — the whole-face classifier is a check on it,
and vice versa."

---

### Q5. "You store the full prediction as a JSON blob. That's a denormalized anti-pattern. Defend it, and tell me what query you *can't* run."

**Answer.** "It's denormalized on purpose and the tradeoff is explicit in
`prediction.py`'s docstring.

**Why:** `/history` must serve three incompatible historical result shapes —
current multi-model, legacy YOLO-only, legacy EfficientNet [`schemas/prediction.py:30`].
A JSON column absorbs all three with zero migrations. And the result is a
document the frontend consumes whole — nothing needs column-level access.

**What I can't do efficiently:** any server-side aggregate — 'average SHI per
month', 'how many users have detected acne' — because that means loading rows and
parsing JSON in Python rather than `SELECT AVG(...)`. Also can't put a plain B-tree
index on `acne_severity`.

**Mitigation path, already identified:** on Postgres the column is `JSONB` and
supports GIN and expression indexes — `CREATE INDEX ON predictions ((results->'skin_health_index'->>'value'))`.
Or promote the handful of hot fields to real columns in a migration. The brief
defers this explicitly; it's not an oversight, it's sequencing."

---

### Q6. "Stateless JWT with no revocation and a 24-hour expiry. A token leaks. What's your incident response — 'wait a day'?"

**Answer.** "Today, the blunt options are: rotate `JWT_SECRET_KEY`, which
invalidates **every** token instantly [noted in `.env.example`], or wait for
`exp`. That's the acknowledged cost of pure-stateless auth and I chose it for v1
because there's one service and no session infrastructure.

The **proper fix is small and I can describe it precisely**: add a
`token_version` integer column to `users`, include it as a claim at issue time,
and check it in `get_current_user` [`security.py:49`] — one extra field on a row
we already load. Bumping one user's `token_version` revokes just their tokens.
That's a denylist-free revocation scheme and it's maybe 15 lines.

Shorter expiry + a refresh token is the other axis; I'd pair a 15-minute access
token with a refresh token once there's a reason to. For a single-user academic
deployment behind localhost, 24 hours was the pragmatic call, not an ideological
one."

---

### Q7. "`max_upload_size_mb` is in your config and your code never checks it. Isn't that exactly the kind of dead config that hides a DoS?"

**Answer.** "Yes — that's a real bug and I'm not going to talk around it.
`shutil.copyfileobj` in `/upload` [`upload.py:47`] streams with no cap, so a
client can POST an arbitrarily large body and fill the disk or exhaust memory in
downstream `cv2.imread`.

**The fix is three lines**: check `file.size` against
`settings.max_upload_size_mb * 1024 * 1024` before the copy, or read in fixed
chunks and abort past the threshold, returning 413. Uvicorn can also be given a
body-size limit at the ASGI layer.

**Why it's not yet exploited in context:** the deployment target is localhost,
single user, brief explicitly scoped to a working skeleton
[`CLAUDE.md`: 'do not gold-plate']. But you're right that config which implies a
control that doesn't exist is worse than no config — I should either enforce it
or comment it as TODO. I'll enforce it."

---

### Q8. "Your redness and pore 'estimates' are just OpenCV colour math with hand-tuned constants — `a_star` 128→0, 150→1, `density/0.03`. Those magic numbers aren't validated against anything. Why is this in the product at all?"

**Answer.** "Because the alternative was worse. We did three audit rounds looking
for a facial-redness or facial-pore model or a labelled dataset, and there isn't
a defensible one [`registry.py:189`]. The choices were: (a) omit the conditions,
(b) fake a model, or (c) ship a **clearly-labelled deterministic estimate**.

We chose (c) with guardrails:
- Status is always `"estimated"`, **never** `"detected"` — enforced in
  `run_analysis` [`analysis_service.py:466`] and in the frontend tier system
  [`config.js:23`].
- The 'confidence' is explicitly a **measurement-reliability** figure scaled by
  skin-pixel coverage, not a model probability — said so in the docstring
  [`heuristics.py:101`].
- In the Skin Health Index formula, redness and pores are **down-weighted** (10
  and 8 points) precisely *because* they're heuristic [`SHI_FORMULA`,
  `analysis_service.py:158`].
- The recommendation engine frames them as '(heuristic estimate, not a trained
  model)' in the user-facing text [`recommendation_engine.py:230`].

The constants are calibration anchors from the CIELAB literature (a* ≈ 128 is
neutral; strong erythema pushes a* up) and visual tuning, and they're all in one
readable function. It's a documented estimate, not a hidden guess. If a validated
model appears, it's one registry entry to swap in."

---

### Q9. "You lazy-import torch inside the request handler. So your first user after every deploy eats a multi-second import *plus* a 30-second model load. Why not just load at startup?"

**Answer.** "Deliberate tradeoff. Loading at startup means the app can't boot —
and can't serve `/health`, `/register`, `/login`, or `/upload` — until 1 GB of
models is in memory, and it means the process **crashes on startup** if a model
file is missing rather than returning a clean 503.

With lazy import [`upload.py:91`] and the file-existence gate [`upload.py:74`],
the app boots in under a second, every non-prediction route works immediately,
and a missing model produces an actionable 503 listing the missing files instead
of a stack trace.

**Mitigations for the cold-start hit:**
- The frontend already expects it — `REQUEST_TIMEOUT_MS` is 120 s and the UI
  literally says 'up to ~30s on the first scan' [`config.js:14`, `app.js:532`].
- The models are singletons, so it's a **one-time** cost per process, not per
  request.
- If I wanted warm starts, a startup `@app.on_event("startup")` hook that fires
  `run_analysis` on a bundled test image in a background task gets the best of
  both — boot fast, warm the models asynchronously. That's a clean addition the
  current structure supports."

---

### Q10. "The recommendation engine is a static dictionary of if-statements. That's not 'AI'. And you recompute it on every history read. Defend both."

**Answer.** "**On 'not AI':** correct, and intentional — the brief calls for
'rule-based logic driven by real prediction outputs, not fake/random'
[`CLAUDE.md`]. Skincare guidance is a domain where a transparent, auditable
rule table (`CONDITION_RULES` keyed by condition and severity band,
`recommendation_engine.py:35`) is *better* than a black box, because every
suggestion is traceable and every clause is medically conservative — it never
says 'you have acne', it says 'acne-like lesions were detected, consider a
professional evaluation if persistent' [`recommendation_engine.py:14`]. The
*intelligence* in the system is the vision pipeline; the recommender is a
deterministic renderer of its output plus climate context.

**On recomputation:** [`upload.py:141`] the recommendation is a **pure function**
of `results` + `context`, both stored. Not persisting it keeps the DB schema
frozen — no `recommendation` column, no migration when the rules change (old
scans get the improved rules automatically). The cost is O(n) pure-Python dict
work per `/history` call. At real history sizes that's negligible; if it mattered
I'd memoize on a hash of `(results, context)` or add the column. It's a
considered tradeoff between schema stability and compute, and I picked stability
for v1."

---

### Q11. "CORS is wide open, the token's in localStorage, and there's no rate limiting or CSRF story. If I deployed this tomorrow, I'd be owned. Why should the panel trust your security judgment?"

**Answer.** "Because I can enumerate every one of those and its fix without
looking — which is the difference between deferred and missed.

- **CORS `*`** [`main.py:25`]: has a code comment saying tighten before
  deployment. Fix: `allow_origins=[FRONTEND_ORIGIN]` from config. One line.
- **Token in localStorage**: XSS-exposed. The mitigation *in place* is that the
  frontend does no `eval`, no `innerHTML` of un-escaped user data (there's an
  `esc()` helper used throughout `app.js`), and loads no third-party JS except
  fonts/icons. The hardening step is `httpOnly` + `SameSite=Strict` cookies,
  which then requires a CSRF token — a coupled change I'd do together.
- **CSRF**: currently a non-issue *because* auth is a bearer header, not a
  cookie — a cross-site form can't attach the `Authorization` header. It becomes
  a real requirement only if I move to cookies.
- **Rate limiting**: none. Fix: `slowapi` (a FastAPI limiter) on `/login` and
  `/predict`, backed by Redis. That's also the same Redis I'd add for the
  inference queue, so it comes for free with bottleneck #1.

The brief scoped v1 to a working end-to-end skeleton on localhost and said
explicitly 'do not gold-plate'. Security that's *stubbed* would be worse than
security that's *scoped and documented*. Nothing here claims to be production-
hardened."

---

### Q12. "Your custom model's validation macro-F1 is 0.39 for skin type, 0.55 for acne, and train scores are 20+ points higher — that's overfitting on 2,272 images. Why is a model this weak wired into the live path as the *primary*?"

**Answer.** "The numbers are exactly as you say and they're in
`runs/training_report.txt`, printed, not hidden — including the train-vs-val gap
as an explicit overfit check.

**Why it's primary:** the selection criterion isn't accuracy, it's
**dependency footprint and offline operation** (see Q1). The brief defers
accuracy to a later phase and the report's viva notes say so.

**Why the weakness is contained:**
- Skin-type F1 0.39 is partly a **support artefact** — the val set has *zero*
  'normal' examples because the ACNE04 source has no acne-free faces
  [`training_report.txt` confusion matrix], so that class is unlearnable from this
  data. The FFHQ clean-face crops were added [`prepare_dataset.py:347`] to
  partially fix it.
- Every custom-model output carries a `note` saying 'treat as indicative' and
  'domain-biased' [`analysis_service.py:224`].
- The **HF ViT fallback is automatic** — delete the checkpoint and the stronger
  (0.65-accuracy) public model takes over with no code change
  [`analysis_service.py:_predict_custom`].
- The acne head, which feeds the SHI most heavily, is trained on **real grades**
  and reaches 0.55 macro-F1 across 4 classes — well above the 0.25 chance
  baseline.

So: the primary path optimizes for the v1 goal (self-contained, offline, real
inference), the fallback path optimizes for accuracy, and the switch is a file's
presence. That's the design working as intended, not a model being oversold."

---

### Bonus rapid-fire answers (if time allows)

- **"Why no Docker?"** — Explicitly out of scope for v1 [`CLAUDE.md`]. The app is
  `pip install -r requirements.txt` + two `.bat` files. Dockerizing is a
  `Dockerfile` with a CUDA or slim-Python base; nothing in the code resists it
  (all paths are config-relative via `pathlib`).
- **"Why no tests?"** — There's `test_inference.py` (end-to-end: throwaway SQLite,
  register→upload→predict, asserts the response shape) and
  `training/yolo/validate_dataset.py`. A full pytest suite is deferred in the
  brief. `httpx` + `pytest` are already pinned for `TestClient`.
- **"Why `threading.Lock` and not `asyncio.Lock`?"** — The model loaders run in
  the sync threadpool, not the event loop, so a threading primitive is correct.
- **"What happens if MediaPipe fails to find a face?"** —
  `face_region` falls back to the whole image with a skin-colour mask
  [`heuristics.py:88`]; `face_detected: false` is reported in the result and the
  wrinkle model (which needs a crop) returns `not_available`.
- **"Image coordinate system for the boxes?"** — YOLO returns `[x1,y1,x2,y2]` in
  **source-image pixels** [`yolo_detector.py:18`]; the canvas scales by
  `canvasWidth / originalWidth` [`app.js:767`]. No letterbox math leaks to the
  client.
