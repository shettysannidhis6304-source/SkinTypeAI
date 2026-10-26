#!/usr/bin/env python3
"""
scripts/prepare_dataset.py

Offline, one-time dataset preparation for the custom multi-task skin model.

Produces, from the LOCAL ACNE04 dataset only (no network, no Hugging Face Hub
downloads at run time -- the teacher weights are already on disk under
weights/hf_models/):

    datasets/processed/crops/<stem>.jpg        224x224 face crops
    datasets/processed/train_labels.csv        image_path,skin_type,acne_severity,pigmentation
    datasets/processed/val_labels.csv          (same columns, 20% stratified split)
    datasets/processed/label_provenance.json   how every label column was produced

Label sources (documented for the viva -- see label_provenance.json):
  * acne_severity : REAL. ACNE04 Hayashi grade, parsed from the "levleN_" filename
                    prefix (0..3 -> mild / moderate / severe / very_severe).
  * skin_type     : DISTILLED. Pseudo-labels from the local ViT teacher
                    weights/hf_models/skin_type_vit/ (dima806, Apache-2.0), run
                    ONCE here. "combination" is synthesised by a fixed rule when
                    the teacher splits its mass between "dry" and "oily".
  * pigmentation  : HEURISTIC (weak). Our own classical-CV score
                    = 0.5*norm(L* std over skin) + 0.5*(dark-spot area fraction),
                    tertile-bucketed into normal / mild / marked. No trained
                    model exists for this condition; these are self-generated
                    weak labels, clearly labelled as such.

Memory safety (Dell Inspiron, CPU, ~8 GB RAM):
  * torch.set_num_threads(4)
  * the teacher is loaded with low_cpu_mem_usage=True, run under
    torch.inference_mode() in batches of 16, then explicitly released with
    `del teacher; gc.collect()` before anything else happens.
  * images are down-scaled before face detection; full-res frames are never held.

Usage (from the project root, with the .venv-new environment):
    .venv-new\\Scripts\\python.exe scripts\\prepare_dataset.py
    .venv-new\\Scripts\\python.exe scripts\\prepare_dataset.py --limit 200   # quick dry run
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skin_analysis import heuristics  # noqa: E402  (face detector + skin mask)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_dataset")

# --------------------------------------------------------------------------- paths
ACNE04_IMAGES = PROJECT_ROOT / "datasets" / "external" / "_extracted" / "Classification" / "JPEGImages"
NNEW_DIR = PROJECT_ROOT / "datasets" / "external" / "_extracted" / "Classification"
OUT_DIR = PROJECT_ROOT / "datasets" / "processed"
CROPS_DIR = OUT_DIR / "crops"
CLEAR_FACES_DIR = OUT_DIR / "clear_faces"     # produced by scripts/fetch_clear_faces.py
HARMONIZED_DIR = OUT_DIR / "clear_faces_harmonized"   # domain-matched copies (see harmonize_clear_faces)
SKIN_TYPE_TEACHER_DIR = PROJECT_ROOT / "weights" / "hf_models" / "skin_type_vit"

CONVERSION_MANIFEST = PROJECT_ROOT / "datasets" / "skin_detect" / "conversion_manifest.json"

# --------------------------------------------------------------------------- labels
# ACNE04 Hayashi grades 0..3 -> our "mild".."very_severe". "clear" (grade -1) is
# supplied separately by the FFHQ clean-face crops.
ACNE_GRADE_TO_LABEL = {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"}
ACNE_SEVERITY_CLASSES = ["clear", "mild", "moderate", "severe", "very_severe"]
SKIN_TYPE_CLASSES = ["dry", "normal", "oily", "combination"]
PIGMENTATION_CLASSES = ["normal", "mild", "marked"]

# Fixed ground-truth labels for the FFHQ clean-face crops.
CLEAR_FACE_LABELS = {"skin_type": "normal", "acne_severity": "clear", "pigmentation": "normal"}

CROP_SIZE = 224
FACE_MARGIN = 0.12          # expand the detected face box by this fraction each side
DOWNSCALE_MAX = 1000        # longest side before face detection
SEED = 42


# ============================================================ image / crop helpers
def parse_acne_grade(stem: str) -> int | None:
    """'levle2_137' -> 2. Returns None if the prefix is not recognised."""
    if not stem.startswith("levle"):
        return None
    try:
        return int(stem[len("levle")])
    except (ValueError, IndexError):
        return None


def face_crop(path: Path) -> np.ndarray | None:
    """Read an ACNE04 photo, down-scale, detect the face (MediaPipe, shared with
    the backend), crop with a margin, and return a CROP_SIZE x CROP_SIZE BGR
    array. Falls back to a centre square crop when no face is found."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = DOWNSCALE_MAX / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]

    x0, y0, x1, y1 = 0, 0, w, h
    det = heuristics._get_face_detector()
    if det is not None:
        try:
            res = det.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.detections:
                rb = res.detections[0].location_data.relative_bounding_box
                x0 = max(0, int((rb.xmin - FACE_MARGIN) * w))
                y0 = max(0, int((rb.ymin - FACE_MARGIN) * h))
                x1 = min(w, int((rb.xmin + rb.width + FACE_MARGIN) * w))
                y1 = min(h, int((rb.ymin + rb.height + FACE_MARGIN) * h))
        except Exception as e:  # pragma: no cover
            logger.debug("face detect failed on %s: %s", path.name, e)

    if x1 - x0 < 40 or y1 - y0 < 40:
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        x1, y1 = x0 + side, y0 + side

    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


# ============================================================ domain harmonization
# The FFHQ clean-face crops are studio-grade (sharp, neutral white balance); the
# ACNE04 crops are clinical-camera photos. Left alone, the model can separate the
# two datasets -- and therefore predict "clear / normal" -- from sensor sharpness
# and colour temperature instead of skin content. We push the FFHQ crops toward
# ACNE04 capture statistics (blur + downscale + WB jitter + noise + JPEG) so that
# shortcut is closed. Applied to the FFHQ subset ONLY; ACNE04 crops are untouched.
def _laplacian_var(bgr: np.ndarray) -> float:
    """Variance of the Laplacian = a standard image-sharpness proxy."""
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _domain_stats(paths: list[Path], sample: int = 150) -> dict:
    """Mean sharpness + mean per-channel value over a sample -- the two axes a
    model would exploit to tell FFHQ from ACNE04."""
    ps = list(paths)
    if len(ps) > sample:
        ps = random.Random(SEED).sample(ps, sample)
    sharp: list[float] = []
    means: list[np.ndarray] = []
    for p in ps:
        im = cv2.imread(str(p))
        if im is None:
            continue
        sharp.append(_laplacian_var(im))
        means.append(im.reshape(-1, 3).mean(axis=0))
    if not sharp:
        return {"n": 0}
    return {
        "n": len(sharp),
        "sharpness_lapvar_mean": round(float(np.mean(sharp)), 1),
        "channel_mean_bgr": [round(float(x), 1) for x in np.mean(means, axis=0)],
    }


def _harmonize_one(bgr: np.ndarray, rng: np.random.Generator,
                   target_bgr_mean: np.ndarray) -> np.ndarray:
    """One FFHQ studio crop -> ACNE04-like statistics. Deterministic given rng.
    TARGET-AWARE: because both subsets already pass through the same 224px crop
    pipeline their sharpness is close, so the real studio-vs-clinic gap is colour
    temperature / exposure. We move this crop's channel means toward the ACNE04
    mean (with jitter so it is not a perfect giveaway the other way), then add a
    light clinical-camera blur + noise + JPEG. Deliberately subtle: the
    'clear / normal' labels must stay visually valid."""
    out = bgr.astype(np.float32)

    # 1) colour-temperature / exposure match toward the ACNE04 channel means
    cur = out.reshape(-1, 3).mean(axis=0)
    tgt = target_bgr_mean.astype(np.float32) * rng.uniform(0.97, 1.03, size=3).astype(np.float32)
    out *= np.clip(tgt / np.maximum(cur, 1.0), 0.85, 1.18)[None, None, :]

    # 2) light optical blur toward the clinical sharpness level (never far below it)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(rng.uniform(0.3, 0.6)))

    # 3) mild sensor noise
    out += rng.normal(0.0, float(rng.uniform(1.0, 3.0)), size=out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 4) clinical-grade JPEG recompression
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 95))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else out


def harmonize_clear_faces(clear_files: list[Path], acne04_crop_dir: Path,
                          seed: int) -> tuple[list[Path], dict]:
    """Degrade every FFHQ clear-face crop toward ACNE04 capture statistics, write
    the results to HARMONIZED_DIR, and run a before/after safety check against
    the ACNE04 crop statistics. Returns (harmonized_paths, provenance_meta).
    Always regenerates from the pristine clear_faces/ source -> idempotent."""
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    acne_stats = _domain_stats(sorted(acne04_crop_dir.glob("*.jpg")))
    before = _domain_stats(clear_files)
    target_bgr = np.array(acne_stats.get("channel_mean_bgr") or [70.0, 90.0, 117.0], dtype=np.float32)

    out_paths: list[Path] = []
    for f in clear_files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        seed_int = int(hashlib.sha1(f"{seed}:{f.name}".encode()).hexdigest()[:8], 16)
        dst = HARMONIZED_DIR / f.name
        cv2.imwrite(str(dst), _harmonize_one(bgr, np.random.default_rng(seed_int), target_bgr),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        out_paths.append(dst)

    after = _domain_stats(out_paths)

    def _sharp_gap(a: dict) -> float | None:
        if not a.get("sharpness_lapvar_mean") or not acne_stats.get("sharpness_lapvar_mean"):
            return None
        return round(abs(a["sharpness_lapvar_mean"] - acne_stats["sharpness_lapvar_mean"])
                     / acne_stats["sharpness_lapvar_mean"], 3)

    def _color_gap(a: dict) -> float | None:
        if not a.get("channel_mean_bgr") or not acne_stats.get("channel_mean_bgr"):
            return None
        av, rv = np.array(a["channel_mean_bgr"]), np.array(acne_stats["channel_mean_bgr"])
        return round(float(np.abs(av - rv).mean() / max(rv.mean(), 1e-6)), 3)

    s_before, s_after = _sharp_gap(before), _sharp_gap(after)
    c_before, c_after = _color_gap(before), _color_gap(after)
    reduced = (
        (s_after is not None and s_before is not None and s_after <= s_before + 0.02)
        and (c_after is not None and c_before is not None and c_after < c_before)
    )

    meta = {
        "applied_to": "FFHQ clear_faces subset ONLY (ACNE04 crops untouched)",
        "reason": "prevent lighting / resolution shortcuts: without this the model can tell "
                  "FFHQ (studio) from ACNE04 (clinical) by white balance / exposure (and, to a "
                  "lesser extent, sensor sharpness) and predict 'clear / normal' from the domain "
                  "rather than the skin.",
        "transforms": ["per-channel colour match toward the ACNE04 mean (gain 0.85-1.18) + jitter",
                       "gaussian blur sigma 0.3-0.6 (p=0.5)", "gaussian sensor noise std 1-3",
                       "JPEG recompress q82-94"],
        "deterministic": "per-file rng seeded by sha1(f'{seed}:{filename}') -> byte-reproducible",
        "source_dir": CLEAR_FACES_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "output_dir": HARMONIZED_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "safety_check": {
            "metric": "relative gap of FFHQ vs ACNE04 on (a) mean Laplacian-variance sharpness "
                      "and (b) mean per-channel value",
            "acne04_crops": acne_stats,
            "ffhq_before": before,
            "ffhq_after": after,
            "sharpness_relgap_before": s_before,
            "sharpness_relgap_after": s_after,
            "color_relgap_before": c_before,
            "color_relgap_after": c_after,
            "shortcut_gap_reduced": bool(reduced),
        },
    }
    logger.info("clear-face harmonization [%s]: sharpness rel-gap %s->%s | colour rel-gap %s->%s",
                "OK" if reduced else "WARN", s_before, s_after, c_before, c_after)
    return out_paths, meta


# ============================================================ pigmentation heuristic
def pigmentation_raw_score(crop_bgr: np.ndarray) -> tuple[float, float]:
    """Weak, self-generated pigmentation signal on the face crop. Returns the
    two RAW (un-clipped) components:

        l_std     : std of CIELAB L* over skin pixels        (tone unevenness)
        spot_frac : dark-spot area fraction from black-hat    (discrete marks)

    The caller z-normalises each across the whole dataset, averages them, and
    tertile-buckets the result -> normal / mild / marked. This is NOT a trained
    model and NOT distilled from any network -- it is a documented colour /
    texture statistic. Higher => more uneven / more spots.
    """
    skin = heuristics._skin_mask(crop_bgr)
    if skin.sum() < 500:
        return float("nan"), float("nan")

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_star = lab[:, :, 0].astype(np.float32)
    l_std = float(l_star[skin].std())

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, spots = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    spot_frac = float(((spots > 0) & skin).sum()) / float(max(int(skin.sum()), 1))

    return l_std, spot_frac


def tertile_labels(components: dict[str, tuple[float, float]]) -> tuple[dict[str, str], dict]:
    """Combine the two raw pigmentation components (L* std, dark-spot fraction)
    into one score by z-normalising each across the dataset and averaging, then
    bucket into normal/mild/marked by the 33rd & 66th percentiles. Rows with
    insufficient skin (NaN) -> 'normal'."""
    stems = list(components)
    l = np.array([components[s][0] for s in stems], dtype=np.float64)
    sp = np.array([components[s][1] for s in stems], dtype=np.float64)
    finite = np.isfinite(l) & np.isfinite(sp)
    if finite.sum() == 0:
        return {s: "normal" for s in stems}, {"note": "no finite scores"}

    def z(a: np.ndarray) -> np.ndarray:
        m, sd = float(np.nanmean(a[finite])), float(np.nanstd(a[finite]) or 1.0)
        return (a - m) / sd

    combined = 0.5 * z(l) + 0.5 * z(sp)
    vals = combined[finite]
    q1, q2 = np.percentile(vals, [33.333, 66.667]).tolist()

    out: dict[str, str] = {}
    for i, s in enumerate(stems):
        v = combined[i]
        if not np.isfinite(v):
            out[s] = "normal"
        elif v <= q1:
            out[s] = "normal"
        elif v <= q2:
            out[s] = "mild"
        else:
            out[s] = "marked"
    meta = {
        "components": ["cielab_L_std", "darkspot_fraction"],
        "combine": "0.5*zscore(L_std) + 0.5*zscore(darkspot_fraction)",
        "percentile_cuts_p33_p66": [round(q1, 4), round(q2, 4)],
        "raw_means": {"L_std": round(float(np.nanmean(l[finite])), 3),
                      "darkspot_fraction": round(float(np.nanmean(sp[finite])), 5)},
    }
    return out, meta


# ============================================================ skin-type teacher
def distill_skin_type(stems: list[str], batch_size: int) -> tuple[dict[str, str], dict]:
    """Load the LOCAL ViT teacher once, pseudo-label every crop, release it.

    "combination" rule: only when the teacher splits its mass hard between the
    two extremes -- P(dry) >= 0.40 AND P(oily) >= 0.40. Otherwise plain argmax
    of {dry, normal, oily} (honest distillation -- the student mimics the
    teacher, including its domain bias, which is documented in the report).
    """
    COMBO_THR = 0.40
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(4)

    if not (SKIN_TYPE_TEACHER_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Skin-type teacher not found at {SKIN_TYPE_TEACHER_DIR}. "
            "Run scripts/fetch_models.py first."
        )

    logger.info("loading skin-type teacher (once) from %s", SKIN_TYPE_TEACHER_DIR)
    processor = AutoImageProcessor.from_pretrained(str(SKIN_TYPE_TEACHER_DIR))
    teacher = AutoModelForImageClassification.from_pretrained(
        str(SKIN_TYPE_TEACHER_DIR), low_cpu_mem_usage=True
    ).eval()
    id2label = {int(k): v for k, v in teacher.config.id2label.items()}  # 0 dry / 1 normal / 2 oily

    labels: dict[str, str] = {}
    dist_counter: Counter[str] = Counter()
    combo_count = 0

    with torch.inference_mode():
        for start in range(0, len(stems), batch_size):
            chunk = stems[start:start + batch_size]
            imgs = [Image.open(CROPS_DIR / f"{s}.jpg").convert("RGB") for s in chunk]
            inputs = processor(imgs, return_tensors="pt")
            probs = teacher(**inputs).logits.softmax(-1).numpy()
            for s, p in zip(chunk, probs):
                pd = {id2label[i]: float(p[i]) for i in range(len(p))}
                if pd.get("dry", 0.0) >= COMBO_THR and pd.get("oily", 0.0) >= COMBO_THR:
                    lbl = "combination"
                    combo_count += 1
                else:
                    lbl = max(pd, key=pd.get)
                labels[s] = lbl
                dist_counter[lbl] += 1
            for im in imgs:
                im.close()
            if (start // batch_size) % 20 == 0:
                logger.info("  teacher: %d / %d crops", start, len(stems))

    meta = {
        "teacher": "dima806/skin_types_image_detection (local, Apache-2.0)",
        "teacher_classes": list(id2label.values()),
        "combination_rule": f"P(dry) >= {COMBO_THR} and P(oily) >= {COMBO_THR}",
        "combination_assigned": combo_count,
        "label_distribution": dict(dist_counter),
    }

    # ---- explicit release (memory safety) ----
    del teacher, processor
    gc.collect()
    logger.info("skin-type teacher released (del + gc.collect)")
    return labels, meta


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=16, help="teacher inference batch size (8 or 16)")
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all) -- for a quick dry run")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not ACNE04_IMAGES.is_dir():
        logger.error("ACNE04 images not found at %s", ACNE04_IMAGES)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # known byte-identical duplicates flagged by the YOLO conversion audit
    dup_stems: set[str] = set()
    if CONVERSION_MANIFEST.is_file():
        try:
            dup_stems = set(json.loads(CONVERSION_MANIFEST.read_text()).get("duplicate_copies_dropped_stems", []))
        except Exception:
            pass
    logger.info("will drop %d known duplicate stems", len(dup_stems))

    all_paths = sorted(p for p in ACNE04_IMAGES.glob("*.jpg") if p.stem not in dup_stems)
    if args.limit:
        all_paths = all_paths[: args.limit]
    logger.info("candidate images: %d", len(all_paths))

    # ---- Phase 1: crop + acne grade + pigmentation raw score (NO models loaded) ----
    acne_label: dict[str, str] = {}
    pigment_raw: dict[str, tuple[float, float]] = {}
    skipped = {"unreadable": 0, "no_grade": 0, "crop_failed": 0}

    for i, path in enumerate(all_paths):
        stem = path.stem
        grade = parse_acne_grade(stem)
        if grade is None or grade not in ACNE_GRADE_TO_LABEL:
            skipped["no_grade"] += 1
            continue
        crop = face_crop(path)
        if crop is None:
            skipped["crop_failed"] += 1
            continue
        cv2.imwrite(str(CROPS_DIR / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        acne_label[stem] = ACNE_GRADE_TO_LABEL[grade]
        pigment_raw[stem] = pigmentation_raw_score(crop)
        if i % 200 == 0:
            logger.info("  phase 1: %d / %d  (kept %d)", i, len(all_paths), len(acne_label))

    stems = sorted(acne_label)
    logger.info("phase 1 done: %d usable crops (skipped %s)", len(stems), skipped)
    if not stems:
        logger.error("no usable images -- aborting")
        return 1

    # ---- Phase 2: skin-type distillation (teacher loaded here, released here) ----
    skin_type_label, skin_type_meta = distill_skin_type(stems, args.batch_size)

    # ---- Phase 3: pigmentation tertile buckets (ACNE04 crops only) ----
    pigment_label, pigment_meta = tertile_labels({s: pigment_raw[s] for s in stems})
    logger.info("pigmentation %s ; dist = %s",
                pigment_meta.get("percentile_cuts_p33_p66"), dict(Counter(pigment_label.values())))

    # ---- Phase 3b: FFHQ clean-face crops with fixed ground-truth labels ----
    # (produced by scripts/fetch_clear_faces.py; give ACNE04 its missing
    #  "clear" acne / "normal" skin-type / "normal" pigmentation examples.)
    clear_records: list[tuple[str, str, str, str]] = []   # (rel_path, skin_type, acne, pigment)
    clear_files = sorted(CLEAR_FACES_DIR.glob("*.jpg")) if CLEAR_FACES_DIR.is_dir() else []

    # ---- domain harmonization: match FFHQ studio crops to ACNE04 capture stats ----
    harmonization_meta: dict = {}
    if clear_files:
        clear_files, harmonization_meta = harmonize_clear_faces(clear_files, CROPS_DIR, args.seed)

    for f in clear_files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        clear_records.append((rel, CLEAR_FACE_LABELS["skin_type"],
                              CLEAR_FACE_LABELS["acne_severity"], CLEAR_FACE_LABELS["pigmentation"]))
    logger.info("clean-face crops ingested: %d (harmonized, labels fixed: %s)",
                len(clear_records), CLEAR_FACE_LABELS)

    # ---- Phase 4: build unified records, stratified split, write CSVs ----
    records: list[tuple[str, str, str, str]] = [
        ((CROPS_DIR / f"{s}.jpg").relative_to(PROJECT_ROOT).as_posix(),
         skin_type_label[s], acne_label[s], pigment_label[s])
        for s in stems
    ] + clear_records

    by_acne: dict[str, list[tuple]] = {}
    for rec in records:
        by_acne.setdefault(rec[2], []).append(rec)
    train_recs: list[tuple] = []
    val_recs: list[tuple] = []
    for _grade, members in by_acne.items():
        random.shuffle(members)
        n_val = max(1, int(round(len(members) * args.val_frac)))
        val_recs.extend(members[:n_val])
        train_recs.extend(members[n_val:])
    random.shuffle(train_recs)
    random.shuffle(val_recs)

    def write_csv(name: str, subset: list[tuple]) -> None:
        lines = ["image_path,skin_type,acne_severity,pigmentation"]
        lines += [",".join(r) for r in subset]
        (OUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote %s (%d rows)", name, len(subset))

    write_csv("train_labels.csv", train_recs)
    write_csv("val_labels.csv", val_recs)

    # ---- provenance ----
    def col_dist(recs: list[tuple], idx: int) -> dict[str, int]:
        return dict(Counter(r[idx] for r in recs))

    n_acne04 = len(stems)
    n_clear = len(clear_records)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_datasets": [
            "ACNE04 (Wu et al., ICCV 2019) -- academic use only; local copy",
            f"FFHQ 256px clean-face subset ({n_clear} crops) -- see clear_faces_manifest.json",
        ],
        "num_crops": n_acne04 + n_clear,
        "num_acne04": n_acne04,
        "num_clear_faces": n_clear,
        "crop_size": CROP_SIZE,
        "split": {"train": len(train_recs), "val": len(val_recs), "val_frac": args.val_frac,
                  "seed": args.seed, "stratified_by": "acne_severity"},
        "columns": {
            "acne_severity": {
                "source": "REAL ground truth (ACNE04) + asserted 'clear' (FFHQ)",
                "method": "ACNE04 Hayashi grade from the 'levleN_' filename prefix; FFHQ crops "
                          "assigned 'clear' (curated clean faces).",
                "classes": ACNE_SEVERITY_CLASSES,
                "mapping": {"-1": "clear (FFHQ)", **{str(k): v for k, v in ACNE_GRADE_TO_LABEL.items()}},
                "distribution_train": col_dist(train_recs, 2),
                "distribution_val": col_dist(val_recs, 2),
            },
            "skin_type": {
                "source": "DISTILLED pseudo-labels (ACNE04) + asserted 'normal' (FFHQ)",
                "method": "local ViT teacher inference (once), argmax + combination rule; FFHQ "
                          "crops assigned 'normal'.",
                "classes": SKIN_TYPE_CLASSES,
                **skin_type_meta,
                "domain_limitation": "teacher was applied to ACNE04 acne-patient faces; the FFHQ crops "
                                     "add the 'normal' class the teacher never produced. Documented for the viva.",
                "distribution_train": col_dist(train_recs, 1),
                "distribution_val": col_dist(val_recs, 1),
            },
            "pigmentation": {
                "source": "HEURISTIC weak labels (ACNE04) + asserted 'normal' (FFHQ)",
                "method": "z(std of CIELAB L* over skin) + z(black-hat dark-spot area fraction), "
                          "averaged, tertile-bucketed on ACNE04; FFHQ crops assigned 'normal'.",
                "classes": PIGMENTATION_CLASSES,
                **pigment_meta,
                "note": "No trained pigmentation model or labelled facial dataset exists. These are "
                        "documented weak labels; the resulting head is the weakest of the three.",
                "distribution_train": col_dist(train_recs, 3),
                "distribution_val": col_dist(val_recs, 3),
            },
        },
        "clear_face_labels": CLEAR_FACE_LABELS,
        "dataset_harmonization": harmonization_meta or {
            "applied": False, "reason": "no FFHQ clear_faces crops present at prepare time",
        },
        "skipped": skipped,
    }
    (OUT_DIR / "label_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("wrote label_provenance.json")

    print("\n=== prepare_dataset summary ===")
    print(f"crops:        {n_acne04} ACNE04 + {n_clear} clean-face  -> {OUT_DIR}")
    print(f"train / val:  {len(train_recs)} / {len(val_recs)}")
    print(f"acne_severity: train {col_dist(train_recs, 2)}")
    print(f"skin_type:     train {col_dist(train_recs, 1)}")
    print(f"pigmentation:  train {col_dist(train_recs, 3)}")
    print(f"provenance:   {OUT_DIR / 'label_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
