"""
backend/app/routers/mobile.py

"Use Phone Camera" bridge -- Method A (zero-app QR capture).

  GET  /api/session/create-qr-session           (auth)  -> {session_id, capture_url, local_ip,
                                                            ip_confident, port, warnings[], qr_png, qr_svg}
  GET  /mobile-capture?session=<id>             (open)  -> the mobile capture page
  POST /api/session/{session_id}/mobile-upload  (open)  <- phone posts the high-res selfie
  GET  /api/session/{session_id}/status         (auth)  -> {status: "pending"|"ready", temp_image_id, ...}

Local-network routing:
  * detect_local_ip() ranks every IPv4 the host knows (UDP-probe default route +
    hostname A records), preferring RFC1918 Wi-Fi addresses and de-ranking
    Docker/WSL/VirtualBox virtual adapters. It reports (ip, confident, note).
  * create-qr-session then self-connects to ip:port; if that fails the backend is
    bound to localhost only and `warnings` says so (the phone would get a
    connection-refused). run-backend.bat starts uvicorn with --host 0.0.0.0.

Method B (phone as a live wireless webcam via DroidCam / Iriun / Phone Link) is
purely client-side instructions -- no backend involved.

The session store is in-memory (the dev server is a single process). Sessions
expire after SESSION_TTL_SECONDS.
"""
from __future__ import annotations

import ipaddress
import logging
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user
from app.models_db.base import get_db
from app.models_db.image import Image
from app.models_db.user import User

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mobile"])

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
_MOBILE_PAGE = _FRONTEND_DIR / "mobile_capture.html"
_TEMP_DIR = settings.upload_dir / "temp"          # phone frames land here as temp_image_id
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

SESSION_TTL_SECONDS = 600
_ALLOWED = {"image/jpeg", "image/png", "image/webp"}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # phone selfies are a few MB; 25 MB is generous

_lock = threading.Lock()
_sessions: dict[str, dict] = {}   # session_id -> {status, user_id, created_at, image_id}


def _prune() -> None:
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v["created_at"] > SESSION_TTL_SECONDS]:
        _sessions.pop(sid, None)


# Adapters we never want to hand a phone: Docker's default bridge (172.17-20.x),
# VirtualBox host-only (192.168.56.x), typical Hyper-V/WSL vEthernet ranges.
_VIRTUAL_HINT_PREFIXES = ("172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
                          "192.168.56.", "192.168.99.", "169.254.")


def _is_private_lan(ip: str) -> bool:
    """RFC1918 (home / office Wi-Fi) address?"""
    if ip.startswith(("192.168.", "10.")):
        return True
    if ip.startswith("172."):
        try:
            return 16 <= int(ip.split(".")[1]) <= 31
        except (IndexError, ValueError):
            return False
    return False


def _candidate_ips() -> list[str]:
    """Every IPv4 this host knows about, most-trustworthy first."""
    ips: list[str] = []
    # 1. UDP probe -- no packet is actually sent; connect() makes the OS choose
    #    the interface that owns the default route (the real Wi-Fi/Ethernet one).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:  # pragma: no cover - env dependent
        pass
    # 2. all A records for the hostname (catches the LAN IP when there is no
    #    default route, e.g. an isolated hotspot).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(info[4][0])
    except OSError:  # pragma: no cover
        pass
    seen: set[str] = set()
    return [ip for ip in ips if ip and not (ip in seen or seen.add(ip))]


def detect_local_ip() -> tuple[str, bool, str]:
    """Best LAN IP for a phone to reach this laptop.

    Returns (ip, confident, note). `confident` is False when only loopback was
    found (the phone cannot connect at all); `note` is a human hint when the
    chosen IP looks like a virtual adapter or an unusual range."""
    cands = _candidate_ips()

    def rank(ip: str) -> tuple:
        return (
            0 if ip.startswith("127.") else 1,                    # loopback last
            0 if ip.startswith("169.254.") else 1,                # APIPA (no DHCP) last
            0 if ip.startswith(_VIRTUAL_HINT_PREFIXES) else 1,    # virtual adapters lower
            1 if _is_private_lan(ip) else 0,                      # prefer RFC1918
        )

    ranked = sorted(cands, key=rank, reverse=True)
    if not ranked or ranked[0].startswith("127."):
        return ("127.0.0.1", False,
                "Could not detect a Wi-Fi / LAN address for this machine -- the phone will not be "
                "able to reach it. Connect the laptop to Wi-Fi and regenerate the code, or use "
                "Method B (phone as a wireless webcam).")
    ip = ranked[0]
    if ip.startswith(_VIRTUAL_HINT_PREFIXES):
        return (ip, True,
                f"Detected {ip}, which looks like a virtual adapter (Docker / WSL / VirtualBox). "
                "If the phone cannot connect, set your Wi-Fi IP manually below and regenerate.")
    if not _is_private_lan(ip):
        return (ip, True,
                f"Detected {ip}, which is not a typical home-Wi-Fi address -- the phone may not "
                "reach it if you are on a restricted or public network.")
    return ip, True, ""


def get_local_ip() -> str:
    """The active Wi-Fi / LAN IPv4 of this machine (UDP-probes the default route).
    Returns '127.0.0.1' only when nothing better can be found."""
    return detect_local_ip()[0]


def _valid_ipv4(value: str) -> str | None:
    try:
        addr = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    return str(addr) if addr.version == 4 else None


def _server_reachable_on(ip: str, port: int) -> bool:
    """True if THIS process is actually accepting TCP on ip:port. A False here
    almost always means uvicorn was started bound to 127.0.0.1 only."""
    try:
        with socket.create_connection((ip, int(port)), timeout=0.6):
            return True
    except OSError:
        return False


@router.get("/api/session/create-qr-session")
def create_qr_session(
    request: Request,
    ip: str | None = Query(None, description="Override the detected LAN IP (multi-adapter machines)."),
    user: User = Depends(get_current_user),
) -> dict:
    with _lock:
        _prune()
        session_id = uuid.uuid4().hex
        _sessions[session_id] = {"status": "pending", "user_id": user.id,
                                 "created_at": time.time(), "image_id": None}

    port = request.url.port or 8000
    warnings: list[str] = []
    override = _valid_ipv4(ip) if ip else None
    if ip and override is None:
        warnings.append(f"'{ip}' is not a valid IPv4 address -- using the auto-detected IP instead.")
    if override:
        chosen_ip, ip_confident, ip_note = override, True, ""
    else:
        chosen_ip, ip_confident, ip_note = detect_local_ip()
    ip = chosen_ip
    capture_url = f"http://{ip}:{port}/mobile-capture?session={session_id}"

    if ip_note:
        warnings.append(ip_note)
    # Binding check: if we detected a real LAN IP but cannot connect to ourselves
    # on it, uvicorn is bound to localhost only -> the phone will get "connection
    # refused" no matter what.
    if ip_confident and not _server_reachable_on(ip, port):
        warnings.append(
            f"The backend is not reachable at {ip}:{port} from the network -- it is bound to "
            f"localhost only. Restart it with  uvicorn app.main:app --host 0.0.0.0 --port {port}  "
            f"(run-backend.bat already does this), then regenerate the code."
        )
        logger.warning("create-qr-session: server not reachable on %s:%s (bound to localhost?)", ip, port)

    # Render the QR as a PNG data URI: a raster with a 4-module quiet zone and
    # high (Q) error correction scans far more reliably off a screen than an
    # inline <path>-stroke SVG (which needs external CSS for its stroke width).
    qr_png = qr_svg = None
    try:
        import segno

        qr = segno.make(capture_url, error="q")
        qr_png = qr.png_data_uri(scale=6, border=4, dark="#10281d", light="#ffffff")
        qr_svg = qr.svg_data_uri(scale=6, border=4)          # crisp vector fallback
    except Exception:  # pragma: no cover - segno is a pinned dependency
        logger.warning("segno unavailable; returning capture_url without a QR image")

    return {"session_id": session_id, "capture_url": capture_url,
            "local_ip": ip, "ip_confident": ip_confident, "port": port,
            "warnings": warnings, "qr_png": qr_png, "qr_svg": qr_svg}


@router.get("/mobile-capture", include_in_schema=False)
def mobile_capture_page() -> FileResponse:
    if not _MOBILE_PAGE.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "mobile capture page not found")
    return FileResponse(str(_MOBILE_PAGE), media_type="text/html")


@router.post("/api/session/{session_id}/mobile-upload", status_code=status.HTTP_201_CREATED)
def mobile_upload(session_id: str, file: UploadFile, db: Session = Depends(get_db)) -> dict:
    """Open endpoint (no JWT) -- the phone has no account. The 32-hex session id
    from the QR is the capability; sessions expire after 10 minutes."""
    with _lock:
        _prune()
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "This capture link has expired or is invalid. Generate a new QR code on the laptop.")
    if (file.content_type or "").split(";")[0].strip() not in _ALLOWED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported file type ({file.content_type or 'unknown'}). Use a JPEG or PNG photo.")

    ext = Path(file.filename or "phone.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    stored_name = f"{sess['user_id']}_phone_{session_id}{ext}"
    stored = f"temp/{stored_name}"                    # relative -> resolve_upload_path() prepends upload_dir
    dest = _TEMP_DIR / stored_name
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                        "Photo is too large (max 25 MB). Try again with a normal-resolution selfie.")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:  # pragma: no cover - disk full etc.
        dest.unlink(missing_ok=True)
        logger.exception("mobile-upload: failed to save %s", dest)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Could not save the photo on the laptop. Try again.") from exc
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded photo was empty.")

    image = Image(user_id=sess["user_id"], file_path=stored,
                  original_filename=file.filename or stored)
    db.add(image)
    db.commit()
    db.refresh(image)

    with _lock:
        s = _sessions.get(session_id)
        if s is not None:
            s["status"] = "ready"
            s["image_id"] = image.id
    logger.info("mobile-upload: session %s -> image %s", session_id, image.id)
    return {"status": "ready", "received": True}


@router.get("/api/session/{session_id}/status")
def session_status(session_id: str, user: User = Depends(get_current_user)) -> dict:
    with _lock:
        _prune()
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or expired.")
    if sess["user_id"] != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your session.")
    if sess["status"] == "ready" and sess["image_id"] is not None:
        return {"status": "ready", "temp_image_id": str(sess["image_id"]),
                "image_id": sess["image_id"], "image_url": f"/images/{sess['image_id']}"}
    return {"status": "pending", "temp_image_id": None, "image_id": None}
