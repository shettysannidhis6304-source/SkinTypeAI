"""
backend/app/models_db/__init__.py

Imports every ORM model so Base.metadata is fully populated for
database/init_db.py's create_all() call -- without this, tables for models
that nothing else imports yet would silently never get created.
"""
from app.models_db.base import Base  # noqa: F401
from app.models_db.image import Image  # noqa: F401
from app.models_db.prediction import Prediction  # noqa: F401
from app.models_db.user import User  # noqa: F401
