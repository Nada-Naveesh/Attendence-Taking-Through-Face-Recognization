"""
Database engine and session management.
Supports SQLite for rapid local development and testing,
while maintaining full compatibility with PostgreSQL + pgvector for production scale.
"""

from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from backend.core.config import settings

# Ensure data directory exists
if settings.DATABASE_URL.startswith("sqlite"):
    db_path = settings.DATA_PATH
    db_path.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a scoped database session
    and guarantees closure upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database schema by creating all registered tables.
    Safe to call repeatedly (idempotent).
    """
    # Import models here to ensure they are registered with Base metadata
    from backend.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
