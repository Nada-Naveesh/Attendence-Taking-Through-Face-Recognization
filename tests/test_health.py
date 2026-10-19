"""
Integration tests for FastAPI endpoints: root, health check, students, and attendance.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.db.session import Base, get_db

# In-memory SQLite for fast, isolated test suite execution
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Create test tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_root_endpoint():
    """Verify root endpoint returns API metadata and 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Smart Attendance" in data["title"]
    assert data["version"] == "1.0.0"
    assert "/api/v1/health" in data["health_check"]


def test_health_check_endpoint():
    """Verify health check returns healthy status and active database connection."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "components" in data
    assert data["uptime_seconds"] >= 0


def test_student_enrollment_api():
    """Verify student registration and retrieval via REST API."""
    # Enroll student
    payload = {"name": "Alan Turing", "roll_no": "CS-1936-001"}
    response = client.post("/api/v1/students", json=payload)
    assert response.status_code == 201
    student = response.json()
    assert student["name"] == "Alan Turing"
    assert student["roll_no"] == "CS-1936-001"
    assert student["has_embedding"] is False

    # Prevent duplicate roll_no
    dup_response = client.post("/api/v1/students", json=payload)
    assert dup_response.status_code == 409

    # List students
    list_response = client.get("/api/v1/students")
    assert list_response.status_code == 200
    students = list_response.json()
    assert len(students) == 1
    assert students[0]["name"] == "Alan Turing"


def test_attendance_logging_api():
    """Verify attendance logging and query filtering via REST API."""
    # 1. Create student
    res = client.post("/api/v1/students", json={"name": "Ada Lovelace", "roll_no": "MATH-1843-001"})
    student_id = res.json()["id"]

    # 2. Log attendance
    att_payload = {
        "student_id": student_id,
        "confidence": 0.94,
        "liveness_passed": True
    }
    att_res = client.post("/api/v1/attendance", json=att_payload)
    assert att_res.status_code == 201
    record = att_res.json()
    assert record["student_id"] == student_id
    assert record["student_name"] == "Ada Lovelace"
    assert record["confidence"] == 0.94
    assert record["liveness_passed"] is True

    # 3. Retrieve attendance list
    list_att = client.get("/api/v1/attendance")
    assert list_att.status_code == 200
    assert len(list_att.json()) == 1


def test_register_student_validation():
    """Verify biometric portrait registration validates face quality and rejects blank images."""
    import cv2
    import numpy as np

    # Generate a blank image with no faces
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", blank_img)
    blank_bytes = buffer.tobytes()

    response = client.post(
        "/api/v1/register",
        data={"name": "Nikola Tesla", "roll_no": "EE-1888-001"},
        files=[("files", ("portrait.jpg", blank_bytes, "image/jpeg"))]
    )
    # Expect 422 Unprocessable Entity because no face was detected in blank image
    assert response.status_code == 422
    assert "No valid face detected" in response.json()["detail"]
