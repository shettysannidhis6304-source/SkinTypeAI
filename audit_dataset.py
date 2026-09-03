#!/usr/bin/env python3
"""
audit_dataset.py  --  standalone dataset inspector for the SkinTypeAI training migration.

Point it at a dataset folder (any size / layout) and it reports, in the terminal:

  * directory tree shape (folders, depth, where the images actually live)
  * image counts per folder and per file extension
  * class distribution for folder-per-class layouts (with train/val/test split awareness)
  * which of the 9 project "heads" appear to have data, vs. single-task-only coverage
  * annotation files found (CSV / JSON / JSONL / COCO / Pascal-VOC XML / YOLO .txt)
    and, for CSV/JSON, their columns / top-level keys
  * corrupted images and zero-byte images

It is deliberately dependency-light: Python standard library only, plus Pillow
if it is installed (used for the pixel-level corruption check -- the script still
runs without it, just skipping that one check).

    python audit_dataset.py  "D:/datasets/skin_v2"
    python audit_dataset.py  ./my_data  --quick --json audit_report.json
    python audit_dataset.py  ./my_data  --ext .jpg .jpeg .png  --top 50

Nothing is written or modified inside the scanned folder.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ Pillow (optional)
try:
    from PIL import Image  # type: ignore

    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
ANNOTATION_EXTS = {".csv", ".json", ".jsonl", ".xml", ".txt", ".yaml", ".yml"}
SPLIT_WORDS = {"train", "training", "val", "valid", "validation", "test", "testing", "holdout"}

# Keywords used to guess which project head a folder relates to. A folder can
# match several heads (e.g. "dry" hits both skin_type and dryness) -- that
# ambiguity is reported, not hidden.
HEAD_KEYWORDS: dict[str, list[str]] = {
    "skin_type":     ["skin_type", "skintype", "skin-type", "combination", "normal_skin"],
    "acne":          ["acne", "pimple", "comedone", "lesion", "blackhead", "whitehead", "papule", "pustule"],
    "acne_severity": ["severity", "grade", "grading", "hayashi", "level", "mild", "moderate", "severe"],
    "oiliness":      ["oili", "oily", "sebum", "shine"],
    "dryness":       ["dry", "dryness", "dehydrat", "flaky", "texture", "rough"],
    "pigmentation":  ["pigment", "melasma", "hyperpigment", "hypopigment", "discolor", "discolour"],
    "dark_spots":    ["dark_spot", "darkspot", "dark-spot", "spot", "lentigo", "freckle", "blemish"],
    "redness":       ["redness", "erythema", "rosacea", "flush", "irritation"],
    "pores":         ["pore", "pores"],
    "wrinkles":      ["wrinkle", "fine_line", "fineline", "aging", "ageing", "crow", "nasolabial"],
}


# ============================================================ small terminal table
def render_table(headers: list[str], rows: list[list[Any]], title: str | None = None) -> str:
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = "+".join("-" * (w + 2) for w in widths)

    def fmt(cells: list[str]) -> str:
        return "|".join(f" {c.ljust(widths[i])} " for i, c in enumerate(cells))

    out: list[str] = []
    if title:
        out.append("")
        out.append(f"== {title} ==")
    out.append(f"+{line}+")
    out.append(f"|{fmt(headers)}|")
    out.append(f"+{line}+")
    for r in rows:
        out.append(f"|{fmt(r)}|")
    out.append(f"+{line}+")
    return "\n".join(out)


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


# ============================================================ scanning
class Scan:
    def __init__(self, root: Path, exts: set[str], quick: bool) -> None:
        self.root = root
        self.exts = exts
        self.quick = quick

        self.total_files = 0
        self.total_bytes = 0
        self.image_count = 0
        self.image_bytes = 0
        self.max_depth = 0

        self.ext_counter: Counter[str] = Counter()
        self.dir_image_counts: dict[str, int] = defaultdict(int)   # rel dir -> image count (direct children)
        self.leaf_class_dirs: list[Path] = []                      # dirs that directly hold images

        self.annotations: list[dict[str, Any]] = []
        self.zero_byte: list[str] = []
        self.corrupted: list[str] = []
        self.checked_for_corruption = 0

        self._t0 = time.time()

    # -------------------------------------------------- walk
    def run(self) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            d = Path(dirpath)
            rel = d.relative_to(self.root)
            depth = 0 if rel == Path(".") else len(rel.parts)
            self.max_depth = max(self.max_depth, depth)

            dir_images = 0
            for name in sorted(filenames):
                fp = d / name
                try:
                    size = fp.stat().st_size
                except OSError:
                    continue
                self.total_files += 1
                self.total_bytes += size
                ext = fp.suffix.lower()

                if ext in self.exts:
                    self.image_count += 1
                    self.image_bytes += size
                    self.ext_counter[ext] += 1
                    dir_images += 1
                    if size == 0:
                        self.zero_byte.append(str(rel / name))
                    else:
                        self._corruption_check(fp, rel / name)
                elif ext in ANNOTATION_EXTS:
                    self._inspect_annotation(fp, rel / name, size)

            if dir_images:
                self.dir_image_counts[str(rel)] = dir_images
                self.leaf_class_dirs.append(rel)

            if self.total_files and self.total_files % 5000 == 0:
                self._progress()

        self._progress(final=True)

    def _progress(self, final: bool = False) -> None:
        dt = time.time() - self._t0
        end = "\n" if final else "\r"
        sys.stderr.write(
            f"  scanned {self.total_files:>8,} files "
            f"({self.image_count:,} images, {len(self.corrupted)} corrupt) "
            f"in {dt:5.1f}s{end}"
        )
        sys.stderr.flush()

    # -------------------------------------------------- corruption
    def _corruption_check(self, fp: Path, rel: Path) -> None:
        if not _HAVE_PIL:
            return
        try:
            with Image.open(fp) as im:
                if self.quick:
                    im.draft("RGB", (64, 64))
                    im.load() if False else None  # header only in quick mode
                else:
                    im.verify()
            if not self.quick:
                # verify() invalidates the object; reopen + load for a real decode
                with Image.open(fp) as im2:
                    im2.load()
            self.checked_for_corruption += 1
        except Exception:
            self.corrupted.append(str(rel))

    # -------------------------------------------------- annotations
    def _inspect_annotation(self, fp: Path, rel: Path, size: int) -> None:
        ext = fp.suffix.lower()
        info: dict[str, Any] = {"path": str(rel), "ext": ext, "size": size, "kind": ext.lstrip(".")}
        try:
            if ext == ".csv":
                with fp.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    n = sum(1 for _ in reader)
                info["kind"] = "csv"
                info["columns"] = header
                info["rows"] = n
            elif ext == ".jsonl":
                keys: Counter[str] = Counter()
                n = 0
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    for i, ln in enumerate(f):
                        ln = ln.strip()
                        if not ln:
                            continue
                        n += 1
                        if i < 200:
                            try:
                                obj = json.loads(ln)
                                if isinstance(obj, dict):
                                    keys.update(obj.keys())
                            except Exception:
                                pass
                info["kind"] = "jsonl"
                info["records"] = n
                info["keys"] = [k for k, _ in keys.most_common(15)]
            elif ext == ".json":
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and {"images", "annotations", "categories"} <= set(obj):
                    info["kind"] = "coco"
                    info["images"] = len(obj.get("images", []))
                    info["boxes"] = len(obj.get("annotations", []))
                    info["categories"] = [c.get("name") for c in obj.get("categories", [])][:20]
                elif isinstance(obj, dict):
                    info["kind"] = "json(dict)"
                    info["keys"] = list(obj.keys())[:20]
                elif isinstance(obj, list):
                    info["kind"] = "json(list)"
                    info["length"] = len(obj)
                    if obj and isinstance(obj[0], dict):
                        info["item_keys"] = list(obj[0].keys())[:20]
            elif ext == ".xml":
                head = fp.read_text(encoding="utf-8", errors="replace")[:400].lower()
                info["kind"] = "voc-xml" if ("<annotation" in head and "<bndbox" in head or "<object" in head) else "xml"
            elif ext == ".txt":
                info["kind"] = self._classify_txt(fp)
            elif ext in (".yaml", ".yml"):
                info["kind"] = "yaml"
        except Exception as e:
            info["parse_error"] = repr(e)
        self.annotations.append(info)

    @staticmethod
    def _classify_txt(fp: Path) -> str:
        """Guess whether a .txt is a YOLO label file (`cls cx cy w h` per line)."""
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").strip().splitlines()[:20]
        except Exception:
            return "txt"
        if not lines:
            return "txt(empty)"
        yolo = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) in (5, 6):
                try:
                    float_parts = [float(p) for p in parts]
                    if float_parts[0].is_integer() and all(0.0 <= v <= 1.0 for v in float_parts[1:5]):
                        yolo += 1
                except ValueError:
                    pass
        return "yolo-labels" if yolo >= max(1, len(lines) // 2) else "txt"


# ============================================================ analysis / reporting
def guess_heads(dirs: list[Path]) -> dict[str, list[str]]:
    """Map each head -> list of directories whose path mentions one of its keywords."""
    hits: dict[str, list[str]] = defaultdict(list)
    for rel in dirs:
        low = str(rel).lower().replace("\\", "/")
        for head, words in HEAD_KEYWORDS.items():
            if any(w in low for w in words):
                hits[head].append(str(rel))
    return hits


def class_distribution(dir_image_counts: dict[str, int]) -> list[tuple[str, str, str, int]]:
    """Return (group, class, split, count). `group` = parent chain minus split & class,
    `class` = leaf folder, `split` = a train/val/test component if present."""
    out: list[tuple[str, str, str, int]] = []
    for rel, count in dir_image_counts.items():
        parts = [p for p in Path(rel).parts if p not in (".", "")]
        if not parts:
            out.append(("(root)", "(root)", "-", count))
            continue
        cls = parts[-1]
        split = "-"
        remaining = parts[:-1]
        for i, p in enumerate(remaining):
            if p.lower() in SPLIT_WORDS:
                split = p
                remaining = remaining[:i] + remaining[i + 1:]
                break
        group = "/".join(remaining) if remaining else "(top level)"
        out.append((group, cls, split, count))
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    return out


def report(scan: Scan, top: int) -> None:
    print()
    print("#" * 78)
    print(f"# DATASET AUDIT  --  {scan.root}")
    print("#" * 78)

    # ---- overview
    print(render_table(
        ["Metric", "Value"],
        [
            ["Root path", str(scan.root)],
            ["Total files", f"{scan.total_files:,}"],
            ["Total size", human_size(scan.total_bytes)],
            ["Image files", f"{scan.image_count:,}"],
            ["Image size on disk", human_size(scan.image_bytes)],
            ["Max folder depth", scan.max_depth],
            ["Folders holding images", f"{len(scan.leaf_class_dirs):,}"],
            ["Annotation files", f"{len(scan.annotations):,}"],
            ["Zero-byte images", f"{len(scan.zero_byte):,}"],
            ["Corrupted images", f"{len(scan.corrupted):,}"],
            ["Pixel-decoded for check", "n/a (Pillow not installed)" if not _HAVE_PIL
             else f"{scan.checked_for_corruption:,}" + (" (quick/header only)" if scan.quick else "")],
        ],
        title="1. OVERVIEW",
    ))

    # ---- extensions
    print(render_table(
        ["Extension", "Count", "% of images"],
        [[e, f"{c:,}", f"{100 * c / max(scan.image_count, 1):.1f}%"]
         for e, c in scan.ext_counter.most_common()],
        title="2. IMAGE FILE EXTENSIONS",
    ))

    # ---- class distribution
    dist = class_distribution(scan.dir_image_counts)
    shown = dist[:top]
    rows = [[g, c, s, f"{n:,}"] for (g, c, s, n) in shown]
    print(render_table(["Group / condition", "Class (leaf folder)", "Split", "Images"], rows,
                       title=f"3. CLASS DISTRIBUTION  (folder-per-class; showing {len(shown)}/{len(dist)})"))
    if len(dist) > top:
        print(f"   ... {len(dist) - top} more rows -- use --top {len(dist)} or --json to see all.")

    # per-group class balance summary
    by_group: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (g, c, s, n) in dist:
        by_group[g].append((f"{c}" + (f" [{s}]" if s != "-" else ""), n))
    balance_rows = []
    for g, items in sorted(by_group.items()):
        agg: Counter[str] = Counter()
        for label, n in items:
            agg[label.split(" [")[0]] += n
        total = sum(agg.values())
        spread = ", ".join(f"{k}:{v:,}" for k, v in sorted(agg.items()))
        imbalance = (max(agg.values()) / max(min(agg.values()), 1)) if agg else 0
        balance_rows.append([g, len(agg), f"{total:,}", f"{imbalance:.1f}x", spread[:90]])
    print(render_table(
        ["Group / condition", "#classes", "Images", "Imbalance", "Per-class counts"],
        balance_rows, title="4. CLASS BALANCE PER GROUP",
    ))

    # ---- head coverage
    head_hits = guess_heads(scan.leaf_class_dirs)
    head_rows = []
    for head in HEAD_KEYWORDS:
        dirs = head_hits.get(head, [])
        imgs = sum(scan.dir_image_counts.get(d, 0) for d in dirs)
        sample = "; ".join(dirs[:3]) + (" ..." if len(dirs) > 3 else "")
        head_rows.append([
            head,
            "YES" if dirs else "-- none --",
            f"{len(dirs)}",
            f"{imgs:,}" if dirs else "0",
            sample,
        ])
    print(render_table(
        ["Project head", "Data found?", "#folders", "~Images", "Example folders"],
        head_rows, title="5. PROJECT-HEAD COVERAGE  (keyword match on folder paths -- verify manually)",
    ))
    covered = [h for h in HEAD_KEYWORDS if head_hits.get(h)]
    missing = [h for h in HEAD_KEYWORDS if not head_hits.get(h)]
    print(f"\n   Heads with candidate data ({len(covered)}/9): {', '.join(covered) or 'NONE'}")
    print(f"   Heads with NO detectable data          : {', '.join(missing) or 'none'}")
    if len(covered) <= 1:
        print("   >> Looks like a SINGLE-TASK dataset (one head only). Multi-task training "
              "would need labels for the other heads.")
    else:
        print("   >> Multiple heads have candidate data -- confirm the labels are real "
              "(folder names / CSV columns), not just keyword collisions.")

    # ---- annotations
    if scan.annotations:
        akind: Counter[str] = Counter(a["kind"] for a in scan.annotations)
        print(render_table(
            ["Annotation kind", "Files"],
            [[k, f"{v:,}"] for k, v in akind.most_common()],
            title="6. ANNOTATION FILES -- BY KIND",
        ))
        detail_rows = []
        for a in scan.annotations:
            if a["kind"] in ("yolo-labels", "txt", "txt(empty)", "voc-xml", "xml"):
                continue  # too many, summarised above
            extra_bits = []
            for key in ("columns", "keys", "item_keys", "categories", "rows", "records",
                        "images", "boxes", "length", "parse_error"):
                if key in a:
                    val = a[key]
                    if isinstance(val, list):
                        val = ", ".join(map(str, val))
                    extra_bits.append(f"{key}={val}")
            detail_rows.append([a["path"], a["kind"], human_size(a["size"]), "; ".join(extra_bits)[:110]])
        if detail_rows:
            print(render_table(["File", "Kind", "Size", "Details"], detail_rows[:top],
                               title="7. CSV / JSON / COCO ANNOTATION DETAIL"))
        # YOLO / VOC pairing sanity
        yolo_labels = [a for a in scan.annotations if a["kind"] == "yolo-labels"]
        voc = [a for a in scan.annotations if a["kind"] == "voc-xml"]
        if yolo_labels or voc:
            print(render_table(
                ["Detection-label type", "Count", "Note"],
                [
                    ["YOLO .txt label files", f"{len(yolo_labels):,}",
                     "expect ~1 per image; pair by filename stem"],
                    ["Pascal-VOC .xml files", f"{len(voc):,}",
                     "convert to YOLO/COCO before training"],
                ],
                title="8. DETECTION LABELS",
            ))
    else:
        print("\n== 6. ANNOTATION FILES ==\n   None found (.csv/.json/.jsonl/.xml/.txt/.yaml). "
              "Dataset appears to be classification-by-folder only.")

    # ---- problems
    if scan.zero_byte:
        print(render_table(["Zero-byte image (first %d)" % min(top, len(scan.zero_byte))],
                           [[p] for p in scan.zero_byte[:top]], title="9. ZERO-BYTE IMAGES"))
    if scan.corrupted:
        print(render_table(["Corrupted image (first %d)" % min(top, len(scan.corrupted))],
                           [[p] for p in scan.corrupted[:top]], title="10. CORRUPTED IMAGES"))
    if not scan.zero_byte and not scan.corrupted:
        print("\n== 9. INTEGRITY ==\n   No zero-byte or corrupted images detected"
              + ("" if _HAVE_PIL else " (Pillow missing -- only zero-byte check ran)."))

    print("\n" + "#" * 78)
    print("# END OF AUDIT")
    print("#" * 78)


def build_json(scan: Scan) -> dict[str, Any]:
    return {
        "root": str(scan.root),
        "totals": {
            "files": scan.total_files, "bytes": scan.total_bytes,
            "images": scan.image_count, "image_bytes": scan.image_bytes,
            "max_depth": scan.max_depth, "image_folders": len(scan.leaf_class_dirs),
        },
        "extensions": dict(scan.ext_counter),
        "class_distribution": [
            {"group": g, "class": c, "split": s, "images": n}
            for (g, c, s, n) in class_distribution(scan.dir_image_counts)
        ],
        "head_coverage": {h: guess_heads(scan.leaf_class_dirs).get(h, []) for h in HEAD_KEYWORDS},
        "annotations": scan.annotations,
        "zero_byte_images": scan.zero_byte,
        "corrupted_images": scan.corrupted,
        "pillow_available": _HAVE_PIL,
    }


# ============================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Standalone dataset audit for the SkinTypeAI training migration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("path", type=Path, help="dataset root folder to scan")
    ap.add_argument("--ext", nargs="+", default=None,
                    help="restrict to these image extensions (e.g. --ext .jpg .png)")
    ap.add_argument("--quick", action="store_true",
                    help="skip full pixel decode; only zero-byte + image-header check")
    ap.add_argument("--top", type=int, default=30, help="max rows per table")
    ap.add_argument("--json", type=Path, default=None, help="also write a full JSON report here")
    args = ap.parse_args(argv)

    root = args.path.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext} if args.ext else set(IMAGE_EXTS)

    if not _HAVE_PIL:
        print("NOTE: Pillow is not installed -- corruption check limited to zero-byte files.\n"
              "      Install it for a real decode test:  pip install Pillow\n", file=sys.stderr)

    print(f"Scanning {root} ...", file=sys.stderr)
    scan = Scan(root, exts, args.quick)
    scan.run()
    report(scan, args.top)

    if args.json:
        args.json.write_text(json.dumps(build_json(scan), indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    # exit non-zero if integrity problems were found (handy for CI)
    return 1 if (scan.zero_byte or scan.corrupted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
