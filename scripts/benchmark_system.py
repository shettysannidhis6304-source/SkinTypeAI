#!/usr/bin/env python3
"""
scripts/benchmark_system.py

Baseline hardware benchmark for the SkinTypeAI inference stack, for the paper's
"efficiency on commodity CPU" table.

For every model component it reports:
  * inference latency  -- mean +/- std over N timed runs (default 20), after 3 warmups
  * peak RAM           -- peak process RSS while that component is loaded + running
                          (each component runs in its OWN subprocess, so the numbers
                          do not accumulate across rows)
  * model size on disk -- MB (single file, or summed dir, or .xml + .bin for the IR)

Components:
  skin_analysis_model  models/skin_analysis_model.py       (weights/skin_analysis_model.pt)
  yolo_acne            models/yolo_detector.py             (glowlytics YOLOv8s)
  wrinkle_pytorch      models/skin_analysis/wrinkle_segmenter.py   (fp32 U-Net)
  wrinkle_openvino     weights/openvino/wrinkle_unet.xml    (fp16 IR - run convert_wrinkle_openvino.py first)
  vit_skin_type        dima806/skin_types_image_detection   (fallback ViT)
  vit_acne_severity    imfarzanansari/skintelligent-acne    (fallback ViT)

Usage (project root):
    .venv-new\\Scripts\\python.exe scripts\\benchmark_system.py
    .venv-new\\Scripts\\python.exe scripts\\benchmark_system.py --runs 30
    .venv-new\\Scripts\\python.exe scripts\\benchmark_system.py --only wrinkle_pytorch,wrinkle_openvino
    .venv-new\\Scripts\\python.exe scripts\\benchmark_system.py --out runs/benchmark.md
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS = PROJECT_ROOT / "weights"
COMPONENTS = [
    "skin_analysis_model",
    "yolo_acne",
    "wrinkle_pytorch",
    "wrinkle_openvino",          # OpenVINO fp16 on CPU
    "wrinkle_openvino_igpu",     # OpenVINO fp16 on the Intel Iris Xe iGPU (skips if no GPU device)
    "vit_skin_type",
    "vit_acne_severity",
]
RESULT_PREFIX = "RESULT_JSON:"


class _BenchSkip(Exception):
    """Raised by _build_callable when a component can't run in THIS environment
    (e.g. no Intel GPU) -- reported as 'skipped', not 'failed'."""


# ======================================================================= helpers
def _dir_or_file_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _disk_mb(component: str) -> float:
    if component == "skin_analysis_model":
        return _dir_or_file_mb(WEIGHTS / "skin_analysis_model.pt")
    if component == "yolo_acne":
        return _dir_or_file_mb(WEIGHTS / "hf_models" / "acne_detector_yolo" / "acne_detector_best.pt")
    if component == "wrinkle_pytorch":
        return _dir_or_file_mb(WEIGHTS / "hf_models" / "wrinkle_unet" / "wrinkle_model.safetensors")
    if component in ("wrinkle_openvino", "wrinkle_openvino_igpu"):
        d = WEIGHTS / "openvino"
        return _dir_or_file_mb(d / "wrinkle_unet.xml") + _dir_or_file_mb(d / "wrinkle_unet.bin")
    if component == "vit_skin_type":
        return _dir_or_file_mb(WEIGHTS / "hf_models" / "skin_type_vit")
    if component == "vit_acne_severity":
        return _dir_or_file_mb(WEIGHTS / "hf_models" / "acne_severity_vit")
    return 0.0


def _test_image_path() -> Path:
    """Prefer a real uploaded photo; otherwise synthesise a deterministic one."""
    for p in sorted((PROJECT_ROOT / "backend" / "uploads").glob("*.jp*g")):
        return p
    import numpy as np
    from PIL import Image

    scratch = PROJECT_ROOT / "runs" / "_bench_input.jpg"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    arr = (rng.random((640, 640, 3)) * 60 + 150).clip(0, 255).astype("uint8")  # skin-ish mid-tone
    Image.fromarray(arr).save(scratch, quality=92)
    return scratch


# ======================================================================= child mode
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def _time_runs(fn, runs: int, warmup: int = 3):
    """Run fn() warmup times, then `runs` timed times. Returns (latencies_ms, peak_rss_mb)."""
    for _ in range(warmup):
        fn()
    peak = _rss_mb()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
        cur = _rss_mb()
        if cur > peak:
            peak = cur
    return lat, peak


def _build_callable(component: str, image_path: Path):
    """Return (callable, backend_label). Raises if the component can't be loaded."""
    import numpy as np
    from PIL import Image

    if component == "skin_analysis_model":
        import cv2

        from models.skin_analysis_model import load_skin_analysis_model, predict

        ckpt_path = WEIGHTS / "skin_analysis_model.pt"
        model, ckpt = load_skin_analysis_model(str(ckpt_path), device="cpu")
        bgr = cv2.imread(str(image_path))
        cfg = ckpt.get("config")
        return (lambda: predict(model, bgr, cfg)), "PyTorch/CPU (fp32, from-scratch 403,656-param CNN)"

    if component == "yolo_acne":
        from models.yolo_detector import YoloDetector

        det = YoloDetector(
            WEIGHTS / "hf_models" / "acne_detector_yolo" / "acne_detector_best.pt",
            conf=0.25, iou=0.45, imgsz=640, device="cpu",
        )
        ip = str(image_path)
        return (lambda: det.detect(ip)), "Ultralytics/CPU (fp32, imgsz=640)"

    if component == "wrinkle_pytorch":
        from models.skin_analysis.wrinkle_segmenter import get_wrinkle_segmenter

        seg = get_wrinkle_segmenter(WEIGHTS / "hf_models" / "wrinkle_unet" / "wrinkle_model.safetensors")
        pil = Image.open(image_path).convert("RGB")
        return (lambda: seg.predict(pil, threshold=0.5)), "PyTorch/CPU (fp32, 512x512)"

    if component in ("wrinkle_openvino", "wrinkle_openvino_igpu"):
        try:
            import openvino as ov
        except ImportError as e:
            raise _BenchSkip(f"openvino not installed ({e})")

        xml = WEIGHTS / "openvino" / "wrinkle_unet.xml"
        if not xml.exists():
            raise _BenchSkip(f"{xml.name} missing - run scripts/convert_wrinkle_openvino.py first")

        ov_device = "GPU" if component == "wrinkle_openvino_igpu" else "CPU"
        core = ov.Core()
        if ov_device == "GPU" and "GPU" not in core.available_devices:
            raise _BenchSkip(
                f"no OpenVINO GPU device (Intel iGPU drivers). available={core.available_devices}"
            )
        try:
            compiled = core.compile_model(str(xml), ov_device)
        except Exception as e:  # driver present but compile fails
            raise _BenchSkip(f"compile_model({ov_device!r}) failed: {e}")
        out_port = compiled.output(0)
        pil = Image.open(image_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        x = np.ascontiguousarray((np.asarray(pil, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...])
        label = f"OpenVINO/{ov_device} (fp16, 512x512)"
        if ov_device == "GPU":
            try:
                label += f"  [{core.get_property('GPU', 'FULL_DEVICE_NAME')}]"
            except Exception:
                pass
        return (lambda: compiled(x)[out_port]), label

    if component in ("vit_skin_type", "vit_acne_severity"):
        from models.skin_analysis.classifier import get_classifier

        sub = "skin_type_vit" if component == "vit_skin_type" else "acne_severity_vit"
        clf = get_classifier(WEIGHTS / "hf_models" / sub, component)
        pil = Image.open(image_path).convert("RGB")
        return (lambda: clf.predict(pil)), "transformers/CPU (fp32, ViT-B/16 224)"

    raise ValueError(f"unknown component: {component}")


def run_child(component: str, runs: int, image_path: Path) -> dict:
    import numpy as np

    rss_start = _rss_mb()
    t0 = time.perf_counter()
    fn, backend = _build_callable(component, image_path)
    load_s = time.perf_counter() - t0
    rss_loaded = _rss_mb()

    lat, peak = _time_runs(fn, runs=runs)
    a = np.asarray(lat, dtype=np.float64)
    return {
        "component": component,
        "backend": backend,
        "ok": True,
        "runs": runs,
        "latency_ms_mean": round(float(a.mean()), 2),
        "latency_ms_std": round(float(a.std()), 2),
        "latency_ms_min": round(float(a.min()), 2),
        "latency_ms_p50": round(float(np.median(a)), 2),
        "fps": round(1000.0 / float(a.mean()), 2),
        "load_seconds": round(load_s, 2),
        "rss_start_mb": round(rss_start, 1),
        "rss_after_load_mb": round(rss_loaded, 1),
        "peak_rss_mb": round(peak, 1),
        "load_delta_mb": round(rss_loaded - rss_start, 1),
        "disk_mb": round(_disk_mb(component), 2),
    }


# ======================================================================= parent mode
def run_parent(components: list[str], runs: int, out_path: Path | None) -> int:
    image_path = _test_image_path()
    print(f"# SkinTypeAI - System Latency & Hardware Benchmark\n")
    print(f"- Machine    : {platform.processor() or platform.machine()}  |  {platform.system()} {platform.release()}")
    try:
        import psutil

        print(f"- CPU        : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical cores"
              f"  |  RAM {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except Exception:
        pass
    print(f"- Python     : {platform.python_version()}  |  torch {_torch_ver()}  |  device: CPU")
    print(f"- Test image : {image_path.name}  ({image_path.stat().st_size / 1024:.0f} KB)")
    print(f"- Protocol   : 3 warmup + {runs} timed runs per component, each in an isolated subprocess\n")

    results: list[dict] = []
    for c in components:
        print(f"  running {c} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--component", c, "--runs", str(runs), "--image", str(image_path)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        res = _parse_child(proc, c)
        results.append(res)
        if res.get("ok"):
            print(f"    {res['latency_ms_mean']:.1f} +/- {res['latency_ms_std']:.1f} ms"
                  f"  |  peak RSS {res['peak_rss_mb']:.0f} MB  |  disk {res['disk_mb']:.0f} MB", flush=True)
        elif res.get("skipped"):
            print(f"    skipped: {res.get('reason', 'unavailable in this environment')}", flush=True)
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}", flush=True)

    table = _markdown(results, runs)
    print("\n" + table)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = _report_header(image_path, runs)
        out_path.write_text(header + "\n" + table + "\n", encoding="utf-8")
        print(f"\nwritten -> {out_path}")
    # a 'skipped' component (e.g. no Intel iGPU) is not a failure
    return 0 if all(r.get("ok") or r.get("skipped") for r in results) else 1


def _torch_ver() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "?"


def _parse_child(proc: subprocess.CompletedProcess, component: str) -> dict:
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                break
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = " / ".join(err[-3:]) if err else f"exit {proc.returncode}"
    return {"component": component, "ok": False, "error": tail, "disk_mb": round(_disk_mb(component), 2)}


def _report_header(image_path: Path, runs: int) -> str:
    lines = ["# SkinTypeAI - System Latency & Hardware Benchmark", ""]
    lines.append(f"- Machine: {platform.processor() or platform.machine()} | {platform.system()} {platform.release()}")
    try:
        import psutil

        lines.append(f"- CPU: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical"
                     f" | RAM {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except Exception:
        pass
    lines.append(f"- Python {platform.python_version()} | torch {_torch_ver()} | device: CPU")
    lines.append(f"- Protocol: 3 warmup + {runs} timed runs, isolated subprocess per component")
    lines.append(f"- Test image: {image_path.name}")
    lines.append("")
    return "\n".join(lines)


def _markdown(results: list[dict], runs: int) -> str:
    h1 = "| Component | Backend | Latency mean ± std (ms) | p50 (ms) | Throughput (img/s) | Peak RAM (MB) | Load (s) | Disk (MB) |"
    h2 = "|---|---|---|---|---|---|---|---|"
    rows = [h1, h2]
    for r in results:
        if r.get("ok"):
            rows.append(
                f"| `{r['component']}` | {r['backend']} "
                f"| {r['latency_ms_mean']:.1f} ± {r['latency_ms_std']:.1f} "
                f"| {r['latency_ms_p50']:.1f} | {r['fps']:.2f} "
                f"| {r['peak_rss_mb']:.0f} | {r['load_seconds']:.1f} | {r['disk_mb']:.1f} |"
            )
        elif r.get("skipped"):
            rows.append(
                f"| `{r['component']}` | _skipped_ | {r.get('reason', 'unavailable')} | - | - | - | - | {r.get('disk_mb', 0):.1f} |"
            )
        else:
            rows.append(
                f"| `{r['component']}` | - | _failed_ | - | - | - | - | {r.get('disk_mb', 0):.1f} |"
            )

    # wrinkle speedup callout
    pt = next((r for r in results if r["component"] == "wrinkle_pytorch" and r.get("ok")), None)
    ov = next((r for r in results if r["component"] == "wrinkle_openvino" and r.get("ok")), None)
    igpu = next((r for r in results if r["component"] == "wrinkle_openvino_igpu" and r.get("ok")), None)
    igpu_skip = next((r for r in results if r["component"] == "wrinkle_openvino_igpu" and r.get("skipped")), None)
    note = ""
    if pt and ov:
        sx = pt["latency_ms_mean"] / max(ov["latency_ms_mean"], 1e-6)
        rx = pt["peak_rss_mb"] / max(ov["peak_rss_mb"], 1e-6)
        dx = pt["disk_mb"] / max(ov["disk_mb"], 1e-6)
        note = (f"\n**Wrinkle U-Net, PyTorch fp32 -> OpenVINO fp16 (CPU):** "
                f"{sx:.2f}x faster, {rx:.2f}x less peak RAM, {dx:.2f}x smaller on disk.")
    if ov and igpu:
        gx = ov["latency_ms_mean"] / max(igpu["latency_ms_mean"], 1e-6)
        note += (f"\n**Wrinkle U-Net, OpenVINO CPU -> Intel iGPU:** "
                 f"{gx:.2f}x {'faster' if gx >= 1 else 'slower'} "
                 f"({ov['latency_ms_mean']:.0f} ms CPU vs {igpu['latency_ms_mean']:.0f} ms iGPU).")
    elif igpu_skip:
        note += f"\n**Intel iGPU target skipped:** {igpu_skip.get('reason')}"

    # end-to-end estimate (active path: custom + yolo + wrinkle)
    active = {r["component"]: r for r in results if r.get("ok")}
    e2e_pt = _sum_active(active, "wrinkle_pytorch")
    e2e_ov = _sum_active(active, "wrinkle_openvino")
    e2e = ""
    if e2e_pt:
        e2e = f"\n**Active-path latency (custom + yolo_acne + wrinkle), sum of means:** {e2e_pt:.0f} ms with PyTorch U-Net"
        if e2e_ov:
            e2e += f", {e2e_ov:.0f} ms with OpenVINO U-Net"
        e2e += ". Excludes MediaPipe face crop + CV heuristics."

    return "\n".join(rows) + "\n" + note + e2e


def _sum_active(active: dict, wrinkle_key: str) -> float | None:
    need = ["skin_analysis_model", "yolo_acne", wrinkle_key]
    if not all(k in active for k in need):
        return None
    return sum(active[k]["latency_ms_mean"] for k in need)


# ======================================================================= entry
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=20, help="timed runs per component (default 20)")
    ap.add_argument("--only", type=str, default="", help="comma-separated subset of components")
    ap.add_argument("--out", type=Path, default=None, help="also write the Markdown table here")
    ap.add_argument("--component", type=str, default="", help=argparse.SUPPRESS)  # internal child mode
    ap.add_argument("--image", type=str, default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # keep the +/- glyph readable on Windows consoles
    except Exception:
        pass

    if args.component:  # ---- child ----
        try:
            res = run_child(args.component, args.runs, Path(args.image))
        except _BenchSkip as e:
            print(RESULT_PREFIX + json.dumps(
                {"component": args.component, "ok": False, "skipped": True, "reason": str(e),
                 "disk_mb": round(_disk_mb(args.component), 2)}))
            return 0
        except Exception as e:
            import traceback

            print(RESULT_PREFIX + json.dumps(
                {"component": args.component, "ok": False, "error": repr(e)}))
            traceback.print_exc()
            return 1
        print(RESULT_PREFIX + json.dumps(res))
        return 0

    # ---- parent ----
    comps = [c.strip() for c in args.only.split(",") if c.strip()] or COMPONENTS
    bad = [c for c in comps if c not in COMPONENTS]
    if bad:
        ap.error(f"unknown component(s): {bad}. choose from {COMPONENTS}")
    return run_parent(comps, args.runs, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
