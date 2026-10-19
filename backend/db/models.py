"""
SQLAlchemy ORM models for the Smart Attendance AI Platform.

Design Philosophy:
- Provides native SQLite support for rapid local development and testing.
- Schema is fully normalized and forward-compatible with PostgreSQL + pgvector
  (where `embedding BLOB` maps to `Vector(512)`).
"""

from __future__ import annotations
import datetime
from typing import Optional, List, Union
import numpy as np
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    LargeBinary,
    ForeignKey,
    Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.session import Base


class Student(Base):
    """
    Student entity representing an enrolled individual.
    Stores biometric facial representation as a serialized float32 vector embedding.
    """
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    roll_no = Column(String(50), unique=True, nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=True)  # Serialized float32 numpy vector (e.g., 512-D FaceNet)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationship to Attendance records
    attendances = relationship(
        "Attendance",
        back_populates="student",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    def set_embedding(self, vector: Union[np.ndarray, List[float]]) -> None:
        """
        Serializes a 1D facial embedding vector to binary format.

        Args:
            vector: Numpy array or list of floats (typically 128-d or 512-d).
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()
        self.embedding = arr.tobytes()

    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Deserializes binary embedding back into a 1D float32 numpy array.

        Returns:
            np.ndarray if embedding exists, else None.
        """
        if self.embedding is None:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize student attributes for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "roll_no": self.roll_no,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<Student(id={self.id}, name='{self.name}', roll_no='{self.roll_no}')>"


class Attendance(Base):
    """
    Attendance log record linking a student to a recognized verification event.
    Stores verification confidence score and anti-spoofing liveness verification status.
    """
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(
        Integer,
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    confidence = Column(Float, nullable=False)  # Cosine similarity metric [0.0 - 1.0]
    liveness_passed = Column(Boolean, default=False, nullable=False)

    # Relationship to Student entity
    student = relationship("Student", back_populates="attendances")

    # Compound index for fast date/student queries
    __table_args__ = (
        Index("idx_student_timestamp", "student_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        """Serialize attendance attributes for API responses."""
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "roll_no": self.student.roll_no if self.student else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": round(float(self.confidence), 4),
            "liveness_passed": bool(self.liveness_passed)
        }

    def __repr__(self) -> str:
        return (
            f"<Attendance(id={self.id}, student_id={self.student_id}, "
            f"confidence={self.confidence:.2f}, liveness={self.liveness_passed})>"
        )
