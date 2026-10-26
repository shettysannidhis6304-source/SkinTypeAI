"""
backend/app/main.py

FastAPI entrypoint. Run from SkinTypeAI/backend with the .venv-new environment:

    ..\\.venv-new\\Scripts\\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

--host 0.0.0.0 is required for the "Use Phone Camera" QR bridge (a phone on the
same Wi-Fi must be able to reach this server); run-backend.bat does this. Running
`python -m app.main` uses the __main__ block below, which also binds 0.0.0.0.

Then hit http://127.0.0.1:8000/docs on this machine for interactive Swagger UI.
(The old .venv is broken -- Python 3.12 interpreter over cp311 packages. Use .venv-new.)
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, dashboard, mobile, overlay, upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SkinType AI", version="0.2.0")

# Wide open for local dev: the frontend is served from a different origin
# (static server on :5500) and the phone-capture page hits this API cross-device.
# allow_methods=["*"] covers the OPTIONS preflight. Tighten before any real deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(dashboard.router)
app.include_router(mobile.router)
app.include_router(overlay.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def _log_lan_url() -> None:
    try:
        from app.routers.mobile import get_local_ip

        ip = get_local_ip()
        if not ip.startswith("127."):
            logger.info("Phone-camera bridge: this machine is reachable at http://%s:8000 "
                        "(phone must be on the same Wi-Fi; allow Python through the firewall on Private networks)", ip)
        else:
            logger.warning("Phone-camera bridge: no LAN IP detected -- the QR capture will not work "
                           "until this machine is on Wi-Fi.")
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
