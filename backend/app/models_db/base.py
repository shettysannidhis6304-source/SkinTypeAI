"""
backend/app/models_db/base.py

SQLAlchemy engine/session/Base shared by every ORM model. SQLite for v1 (per
CLAUDE.md) -- swapping to Postgres later just means changing DATABASE_URL in
.env, no code changes needed elsewhere.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

# check_same_thread=False: FastAPI can serve a request on a different thread
# than the one that created the session; safe here because we open/close a
# fresh session per-request (see get_db below), never share one across threads.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
