#!/usr/bin/env python3
"""
scripts/expand_and_prepare_dataset.py  --  STAGE 1 of the from-scratch rebuild

Takes the 1,419 local ACNE04 face crops (datasets/processed/crops/, 224x224,
named "levleN_*.jpg" where N = Hayashi grade 0..3) and produces a compact,
128x128, augmented training set for the from-scratch SkinAnalysisModel:

    datasets/processed_scratch/
        images/<name>.jpg                 128x128 RGB (base + augmented)
        train_manifest.csv                image_path,acne_severity,wrinkles,oiliness,
        val_manifest.csv                    dryness,pigmentation,parent,is_aug
        label_provenance_scratch.json     how every label column was produced

LABELS (5 characteristics the model will learn)
----------------------------------------------
  acne_severity  REAL. ACNE04 Hayashi grade 0..3, parsed from the filename prefix.
  wrinkles       HEURISTIC tier {minimal,moderate,prominent} -- elongated black-hat
                 line response + Canny edge density on skin pixels.
  oiliness       HEURISTIC tier {low,balanced,high} -- specular-highlight fraction
                 (high V, low S in HSV) + mean luminance on skin pixels.
  dryness        HEURISTIC tier {low,balanced,high} -- mean local 5x5 std (texture
                 roughness) on skin pixels.
  pigmentation   HEURISTIC tier {normal,mild,marked} -- std of CIELAB L* + black-hat
                 dark-spot area fraction on skin pixels.

Only `acne_severity` is ground truth. The other four are dataset-relative tertile
buckets of deterministic OpenCV statistics (self-generated weak labels). The
tertile cut points are fitted on the TRAIN originals only and then applied to
val, so val labels never leak into the cut points.

LEAKAGE CONTROL
---------------
The 80/20 split is done on the 1,419 ORIGINAL crops (stratified by acne grade).
Augmented copies always follow their parent crop into the same split, and only
the TRAIN originals are augmented -- val is left as clean, un-augmented 128x128
images so validation metrics are honest.

Usage (project root, .venv-new):
    .venv-new\\Scripts\\python.exe scripts\\expand_and_prepare_dataset.py
    .venv-new\\Scripts\\python.exe scripts\\expand_and_prepare_dataset.py --per-class-target 950 --max-aug 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skin_analysis import heuristics  # noqa: E402  (skin mask only)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("expand_dataset")

CROPS_SRC = PROJECT_ROOT / "datasets" / "processed" / "crops"
OUT_DIR = PROJECT_ROOT / "datasets" / "processed_scratch"
IMG_DIR = OUT_DIR / "images"

CROP_SIZE = 128
SEED = 42
VAL_FRAC = 0.20

ACNE_CLASSES = ["0", "1", "2", "3"]                       # Hayashi grades
WRINKLE_TIERS = ["minimal", "moderate", "prominent"]
OILINESS_TIERS = ["low", "balanced", "high"]
DRYNESS_TIERS = ["low", "balanced", "high"]
PIGMENT_TIERS = ["normal", "mild", "marked"]

MANIFEST_COLS = ["image_path", "acne_severity", "wrinkles", "oiliness",
                 "dryness", "pigmentation", "parent", "is_aug"]


# ============================================================ helpers
def parse_grade(stem: str) -> int | None:
    if not stem.startswith("levle"):
        return None
    try:
        g = int(stem[len("levle")])
    except (ValueError, IndexError):
        return None
    return g if g in (0, 1, 2, 3) else None


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    m = heuristics._skin_mask(bgr)
    if m.sum() < 300:                                    # tiny/failed mask -> use whole crop
        m = np.ones(bgr.shape[:2], dtype=bool)
    return m


# ---- deterministic heuristic raw scores (computed on the 128x128 base image) ----
def wrinkle_raw(bgr: np.ndarray, skin: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    line = np.maximum(bh_h, bh_v).astype(np.float32)
    line_score = float(line[skin].mean())
    edges = cv2.Canny(gray, 40, 120)
    edge_frac = float((edges[skin] > 0).mean())
    return line_score, edge_frac


def oiliness_raw(bgr: np.ndarray, skin: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    spec = ((v > 220) & (s < 60)) & skin
    spec_frac = float(spec.sum()) / float(max(int(skin.sum()), 1))
    v_mean = float(v[skin].mean())
    return spec_frac, v_mean


def dryness_raw(bgr: np.ndarray, skin: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (5, 5))
    mu2 = cv2.blur(gray * gray, (5, 5))
    local_std = np.sqrt(np.clip(mu2 - mu * mu, 0.0, None))
    return float(local_std[skin].mean())


def pigment_raw(bgr: np.ndarray, skin: np.ndarray) -> tuple[float, float]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_star = lab[:, :, 0].astype(np.float32)
    l_std = float(l_star[skin].std())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    _, spots = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    spot_frac = float(((spots > 0) & skin).sum()) / float(max(int(skin.sum()), 1))
    return l_std, spot_frac


# ---- augmentation (h-flip, +/-10 deg rotation, +/-10% brightness/contrast) ----
def augment(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = bgr
    if rng.random() < 0.5:
        out = out[:, ::-1].copy()
    angle = float(rng.uniform(-10.0, 10.0))
    if abs(angle) > 0.5:
        h, w = out.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha = float(rng.uniform(0.90, 1.10))               # contrast
    beta = float(rng.uniform(0.90, 1.10))                # brightness (multiplicative)
    out = np.clip((out.astype(np.float32) - 128.0) * alpha + 128.0, 0, 255)
    out = np.clip(out * beta, 0, 255).astype(np.uint8)
    return out


def to_128(bgr: np.ndarray) -> np.ndarray:
    if bgr.shape[:2] != (CROP_SIZE, CROP_SIZE):
        bgr = cv2.resize(bgr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return bgr


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class-target", type=int, default=950,
                    help="target TRAIN images per acne grade after augmentation")
    ap.add_argument("--max-aug", type=int, default=6, help="max augmented copies per train original")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if not CROPS_SRC.is_dir():
        logger.error("ACNE04 crops not found at %s", CROPS_SRC)
        return 2

    # ---- 1. collect + grade ----
    crops: list[tuple[str, int]] = []
    skipped = 0
    for p in sorted(CROPS_SRC.glob("*.jpg")):
        g = parse_grade(p.stem)
        if g is None:
            skipped += 1
            continue
        crops.append((p.stem, g))
    logger.info("ACNE04 crops: %d usable (%d skipped, no grade)", len(crops), skipped)
    logger.info("grade distribution: %s", dict(sorted(Counter(g for _, g in crops).items())))

    # ---- 2. stratified 80/20 split on ORIGINALS (by grade) ----
    by_grade: dict[int, list[str]] = {}
    for stem, g in crops:
        by_grade.setdefault(g, []).append(stem)
    train_orig: list[tuple[str, int]] = []
    val_orig: list[tuple[str, int]] = []
    for g, stems in sorted(by_grade.items()):
        stems = sorted(stems)
        rng.shuffle(stems)
        n_val = max(1, round(len(stems) * VAL_FRAC))
        val_orig += [(s, g) for s in stems[:n_val]]
        train_orig += [(s, g) for s in stems[n_val:]]
    logger.info("split originals: train %d | val %d", len(train_orig), len(val_orig))

    # ---- 3. heuristic raw scores for every original crop ----
    raw: dict[str, dict[str, float]] = {}
    all_orig = train_orig + val_orig
    for i, (stem, _g) in enumerate(all_orig):
        bgr = to_128(cv2.imread(str(CROPS_SRC / f"{stem}.jpg")))
        skin = _skin_mask(bgr)
        w_line, w_edge = wrinkle_raw(bgr, skin)
        o_spec, o_v = oiliness_raw(bgr, skin)
        d_rough = dryness_raw(bgr, skin)
        p_lstd, p_spot = pigment_raw(bgr, skin)
        raw[stem] = {"w_line": w_line, "w_edge": w_edge, "o_spec": o_spec, "o_v": o_v,
                     "d_rough": d_rough, "p_lstd": p_lstd, "p_spot": p_spot}
        if i % 300 == 0:
            logger.info("  heuristics %d / %d", i, len(all_orig))

    # ---- 4. z-score on TRAIN originals, tertile-bucket -> tiers ----
    train_stems = [s for s, _ in train_orig]

    def zparams(key: str) -> tuple[float, float]:
        vals = np.array([raw[s][key] for s in train_stems], dtype=np.float64)
        return float(vals.mean()), float(vals.std() or 1.0)

    zp = {k: zparams(k) for k in ("w_line", "w_edge", "o_spec", "o_v", "d_rough", "p_lstd", "p_spot")}

    def z(stem: str, key: str) -> float:
        m, sd = zp[key]
        return (raw[stem][key] - m) / sd

    def combined(stem: str) -> dict[str, float]:
        return {
            "wrinkles": 0.5 * z(stem, "w_line") + 0.5 * z(stem, "w_edge"),
            "oiliness": 0.5 * z(stem, "o_spec") + 0.5 * z(stem, "o_v"),
            "dryness": z(stem, "d_rough"),
            "pigmentation": 0.5 * z(stem, "p_lstd") + 0.5 * z(stem, "p_spot"),
        }

    comb = {s: combined(s) for s in raw}
    tiers_for = {"wrinkles": WRINKLE_TIERS, "oiliness": OILINESS_TIERS,
                 "dryness": DRYNESS_TIERS, "pigmentation": PIGMENT_TIERS}
    cuts: dict[str, tuple[float, float]] = {}
    for head in tiers_for:
        tvals = np.array([comb[s][head] for s in train_stems], dtype=np.float64)
        q1, q2 = np.percentile(tvals, [33.333, 66.667]).tolist()
        cuts[head] = (q1, q2)

    def tier(stem: str, head: str) -> str:
        q1, q2 = cuts[head]
        v = comb[stem][head]
        names = tiers_for[head]
        return names[0] if v <= q1 else (names[1] if v <= q2 else names[2])

    labels: dict[str, dict[str, str]] = {}
    for stem, g in all_orig:
        labels[stem] = {"acne_severity": str(g),
                        **{h: tier(stem, h) for h in tiers_for}}

    # ---- 5. write base images ----
    if IMG_DIR.exists():
        for f in IMG_DIR.glob("*.jpg"):
            f.unlink()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    def write_img(name: str, bgr: np.ndarray) -> str:
        cv2.imwrite(str(IMG_DIR / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return f"datasets/processed_scratch/images/{name}.jpg"

    train_stem_set = {s for s, _ in train_orig}
    base_cache: dict[str, np.ndarray] = {}
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for stem, _g in all_orig:
        bgr = to_128(cv2.imread(str(CROPS_SRC / f"{stem}.jpg")))
        base_cache[stem] = bgr
        rel = write_img(stem, bgr)
        row = {"image_path": rel, **labels[stem], "parent": stem, "is_aug": 0}
        (train_rows if stem in train_stem_set else val_rows).append(row)

    # ---- 6. augment TRAIN originals to balance per acne grade ----
    train_by_grade: dict[int, list[str]] = {}
    for stem, g in train_orig:
        train_by_grade.setdefault(g, []).append(stem)

    aug_counter = 0
    for g, stems in sorted(train_by_grade.items()):
        need = max(0, args.per_class_target - len(stems))
        if need == 0:
            continue
        # distribute `need` copies across this grade's images, capped at --max-aug
        per = [0] * len(stems)
        idx = 0
        while need > 0 and any(c < args.max_aug for c in per):
            if per[idx] < args.max_aug:
                per[idx] += 1
                need -= 1
            idx = (idx + 1) % len(stems)
        for stem, n_copies in zip(stems, per):
            stem_seed = int.from_bytes(hashlib.sha1(stem.encode()).digest()[:4], "big")
            for k in range(n_copies):
                arng = np.random.default_rng(args.seed + stem_seed + k)
                aug = augment(base_cache[stem], arng)
                name = f"{stem}__aug{k}"
                rel = write_img(name, aug)
                train_rows.append({"image_path": rel, **labels[stem],
                                   "parent": stem, "is_aug": 1})
                aug_counter += 1
    logger.info("wrote %d augmented train images", aug_counter)

    # ---- 7. manifests ----
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    def write_manifest(name: str, rows: list[dict]) -> None:
        lines = [",".join(MANIFEST_COLS)]
        lines += [",".join(str(r[c]) for c in MANIFEST_COLS) for r in rows]
        (OUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote %s (%d rows)", name, len(rows))

    write_manifest("train_manifest.csv", train_rows)
    write_manifest("val_manifest.csv", val_rows)

    # ---- provenance ----
    def dist(rows: list[dict], col: str) -> dict[str, int]:
        return dict(sorted(Counter(r[col] for r in rows).items()))

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "1 -- dataset expansion & label preparation (from-scratch rebuild)",
        "source": {"dir": "datasets/processed/crops", "n_acne04_crops": len(crops),
                   "skipped_no_grade": skipped, "source_resolution": 224},
        "output": {"dir": "datasets/processed_scratch", "resolution": CROP_SIZE,
                   "n_train": len(train_rows), "n_val": len(val_rows),
                   "n_augmented_train": aug_counter, "total_images": len(train_rows) + len(val_rows)},
        "split": {"val_frac": VAL_FRAC, "seed": args.seed, "stratified_by": "acne_severity",
                  "unit": "original crop (augmented copies follow their parent; val is NOT augmented)"},
        "augmentation": {"transforms": ["horizontal flip p=0.5", "rotation +/-10 deg (reflect border)",
                                        "contrast x[0.90,1.10]", "brightness x[0.90,1.10]"],
                         "applied_to": "train originals only",
                         "per_class_target": args.per_class_target, "max_aug_per_image": args.max_aug,
                         "label_inheritance": "augmented copies inherit their parent crop's 5 labels"},
        "labels": {
            "acne_severity": {"source": "REAL ground truth", "classes": ACNE_CLASSES,
                              "method": "ACNE04 Hayashi grade from the 'levleN_' filename prefix",
                              "distribution_train": dist(train_rows, "acne_severity"),
                              "distribution_val": dist(val_rows, "acne_severity")},
            "wrinkles": {"source": "HEURISTIC weak label (dataset-relative tertile)",
                         "classes": WRINKLE_TIERS,
                         "method": "0.5*z(elongated black-hat line response) + 0.5*z(Canny edge fraction) "
                                   "on skin pixels; tertile cuts fitted on TRAIN originals",
                         "tertile_cuts_p33_p66": [round(c, 4) for c in cuts["wrinkles"]],
                         "distribution_train": dist(train_rows, "wrinkles"),
                         "distribution_val": dist(val_rows, "wrinkles")},
            "oiliness": {"source": "HEURISTIC weak label (dataset-relative tertile)",
                         "classes": OILINESS_TIERS,
                         "method": "0.5*z(specular fraction: HSV V>220 & S<60) + 0.5*z(mean V) on skin pixels",
                         "tertile_cuts_p33_p66": [round(c, 4) for c in cuts["oiliness"]],
                         "distribution_train": dist(train_rows, "oiliness"),
                         "distribution_val": dist(val_rows, "oiliness")},
            "dryness": {"source": "HEURISTIC weak label (dataset-relative tertile)",
                        "classes": DRYNESS_TIERS,
                        "method": "z(mean local 5x5 std / texture roughness) on skin pixels",
                        "tertile_cuts_p33_p66": [round(c, 4) for c in cuts["dryness"]],
                        "distribution_train": dist(train_rows, "dryness"),
                        "distribution_val": dist(val_rows, "dryness")},
            "pigmentation": {"source": "HEURISTIC weak label (dataset-relative tertile)",
                             "classes": PIGMENT_TIERS,
                             "method": "0.5*z(std of CIELAB L*) + 0.5*z(black-hat dark-spot area fraction) "
                                       "on skin pixels",
                             "tertile_cuts_p33_p66": [round(c, 4) for c in cuts["pigmentation"]],
                             "distribution_train": dist(train_rows, "pigmentation"),
                             "distribution_val": dist(val_rows, "pigmentation")},
        },
        "caveats": [
            "Only acne_severity is ground truth. wrinkles/oiliness/dryness/pigmentation are "
            "self-generated weak labels: a CNN trained on them learns to imitate the OpenCV "
            "statistic, it does not add new clinical information.",
            "Source images are ACNE04 acne-patient faces only (no demographic diversity, "
            "mostly young adults) -- the wrinkle/oiliness/dryness tiers are narrow-domain.",
            "128x128 discards fine texture that wrinkle/pore analysis normally needs.",
        ],
    }
    (OUT_DIR / "label_provenance_scratch.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("wrote label_provenance_scratch.json")

    # ---- summary ----
    print("\n" + "=" * 64)
    print("STAGE 1 SUMMARY -- datasets/processed_scratch/")
    print("=" * 64)
    print(f"source ACNE04 crops : {len(crops)}  (grades "
          f"{dict(sorted(Counter(g for _, g in crops).items()))})")
    print(f"resolution          : {CROP_SIZE}x{CROP_SIZE}")
    print(f"train images        : {len(train_rows)}  ({aug_counter} augmented + "
          f"{len(train_rows) - aug_counter} originals)")
    print(f"val images          : {len(val_rows)}  (originals only, no augmentation)")
    print(f"total               : {len(train_rows) + len(val_rows)}")
    print("-" * 64)
    for col in ("acne_severity", "wrinkles", "oiliness", "dryness", "pigmentation"):
        print(f"{col:14s} train {dist(train_rows, col)}")
        print(f"{'':14s} val   {dist(val_rows, col)}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
