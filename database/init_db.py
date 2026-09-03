"""
database/init_db.py

Creates the SQLite schema (users, images, predictions tables) from the
SQLAlchemy models. Safe to re-run -- create_all() only creates tables that
don't already exist yet; it never drops or alters existing ones.

Usage (from the SkinTypeAI project root):
    .venv\\Scripts\\python.exe database\\init_db.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))  # so `import app...` resolves, same as running uvicorn from backend/

from app.models_db import Base  # noqa: E402
from app.models_db.base import engine  # noqa: E402


def main() -> None:
    db_path = PROJECT_ROOT / "database" / "skintype.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print(f"Database ready: {db_path}")
    print("Tables:", list(Base.metadata.tables.keys()))


if __name__ == "__main__":
    main()
