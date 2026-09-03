"""
backend/app/main.py

FastAPI entrypoint. Run from SkinTypeAI/backend with the .venv-new environment:

    ..\\.venv-new\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000

Then hit http://127.0.0.1:8000/docs for interactive Swagger UI.
(The old .venv is broken -- Python 3.12 interpreter over cp311 packages. Use .venv-new.)
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, dashboard, upload

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SkinType AI", version="0.2.0")

# Wide open for local dev since the frontend is served from a different
# origin (file:// or a static server) than the API. Tighten before any real
# deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(dashboard.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
