"""
Unit tests for database models, embedding serialization, and attendance service logic.
"""

from datetime import date
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.session import Base
from backend.db.models import Student, Attendance
from backend.services.attendance import AttendanceService


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_student_embedding_serialization(db_session):
    """
    Verify high-dimensional float32 vector serialization and deserialization.
    Simulates a 512-dimensional FaceNet biometric embedding vector.
    """
    # 1. Generate synthetic 512-D unit vector
    rng = np.random.RandomState(42)
    original_vector = rng.randn(512).astype(np.float32)
    original_vector /= np.linalg.norm(original_vector)

    # 2. Persist in database
    student = Student(name="Grace Hopper", roll_no="CS-1944-001")
    student.set_embedding(original_vector)
    db_session.add(student)
    db_session.commit()
    db_session.refresh(student)

    # 3. Retrieve and deserialize
    retrieved_student = db_session.query(Student).filter(Student.roll_no == "CS-1944-001").first()
    assert retrieved_student is not None
    assert retrieved_student.embedding is not None

    restored_vector = retrieved_student.get_embedding()
    assert restored_vector is not None
    assert restored_vector.shape == (512,)
    assert restored_vector.dtype == np.float32
    assert np.allclose(original_vector, restored_vector, atol=1e-7)


def test_attendance_model_relationship(db_session):
    """Verify bidirectional relationship and cascading deletes between Student and Attendance."""
    student = Student(name="Claude Shannon", roll_no="EE-1948-001")
    db_session.add(student)
    db_session.commit()

    att1 = Attendance(student_id=student.id, confidence=0.88, liveness_passed=True)
    att2 = Attendance(student_id=student.id, confidence=0.92, liveness_passed=True)
    db_session.add_all([att1, att2])
    db_session.commit()

    # Query student attendances
    assert len(student.attendances) == 2
    assert student.attendances[0].student.name == "Claude Shannon"

    # Verify cascade deletion
    db_session.delete(student)
    db_session.commit()
    records_count = db_session.query(Attendance).count()
    assert records_count == 0


def test_attendance_deduplication(db_session):
    """Verify AttendanceService prevents duplicate attendance logs on the same calendar day."""
    student = Student(name="Geoffrey Hinton", roll_no="AI-1986-001")
    db_session.add(student)
    db_session.commit()

    # First mark should succeed and create record
    rec1 = AttendanceService.mark_attendance(
        db=db_session,
        student_id=student.id,
        confidence=0.91,
        liveness_passed=True,
        allow_duplicate_same_day=False
    )
    assert rec1.id is not None

    # Second mark on same day should return the existing record without inserting duplicate
    rec2 = AttendanceService.mark_attendance(
        db=db_session,
        student_id=student.id,
        confidence=0.95,
        liveness_passed=True,
        allow_duplicate_same_day=False
    )
    assert rec2.id == rec1.id

    total_records = db_session.query(Attendance).filter(Attendance.student_id == student.id).count()
    assert total_records == 1
