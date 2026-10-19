"""
FastAPI route definitions for the Smart Attendance AI Platform.
"""

import time
from datetime import date
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.core.config import settings
from backend.db.session import get_db
from backend.db.models import Student, Attendance
from backend.api.schemas import (
    HealthResponse,
    StudentCreate,
    StudentResponse,
    AttendanceCreate,
    AttendanceResponse,
    RecognitionResult
)
from backend.services.attendance import AttendanceService
from backend.services.baseline_haar import BaselineHaarService

router = APIRouter()
START_TIME = time.time()
_baseline_service: Optional[BaselineHaarService] = None


def get_baseline_service() -> BaselineHaarService:
    """Lazy loader for Haar Cascade baseline service."""
    global _baseline_service
    if _baseline_service is None:
        _baseline_service = BaselineHaarService()
    return _baseline_service


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """
    Production-grade system health check.
    Verifies API status, runtime uptime, and active database connectivity.
    """
    db_status = "disconnected"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    uptime = time.time() - START_TIME

    # Check baseline model availability
    haar_available = settings.HAARCASCADE_PATH.exists()

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database=db_status,
        uptime_seconds=round(uptime, 2),
        components={
            "database_url": settings.DATABASE_URL.split("///")[-1],
            "detector_backend": settings.DETECTOR_BACKEND,
            "recognition_model": settings.RECOGNITION_MODEL,
            "haar_baseline_ready": haar_available,
            "anti_spoofing_ear_threshold": settings.EAR_THRESHOLD
        }
    )


# ------------------------------------------------------------------------------
# Student Management Endpoints
# ------------------------------------------------------------------------------

@router.get("/students", response_model=List[StudentResponse], tags=["Students"])
def list_students(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve all enrolled students with pagination."""
    students = db.query(Student).offset(skip).limit(limit).all()
    results = []
    for s in students:
        results.append(
            StudentResponse(
                id=s.id,
                name=s.name,
                roll_no=s.roll_no,
                has_embedding=s.embedding is not None,
                created_at=s.created_at
            )
        )
    return results


@router.post("/students", response_model=StudentResponse, status_code=status.HTTP_201_CREATED, tags=["Students"])
def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_db)
):
    """Enroll a new student metadata record."""
    existing = db.query(Student).filter(Student.roll_no == payload.roll_no).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Student with roll number '{payload.roll_no}' already exists."
        )

    student = Student(name=payload.name, roll_no=payload.roll_no)
    db.add(student)
    db.commit()
    db.refresh(student)

    return StudentResponse(
        id=student.id,
        name=student.name,
        roll_no=student.roll_no,
        has_embedding=False,
        created_at=student.created_at
    )


@router.post("/register", response_model=StudentResponse, status_code=status.HTTP_201_CREATED, tags=["Students"])
async def register_student_with_face(
    name: str = Form(..., description="Student full name"),
    roll_no: str = Form(..., description="Unique student matriculation/roll number"),
    file: UploadFile = File(..., description="Biometric portrait image (uploaded or webcam captured)"),
    db: Session = Depends(get_db),
    baseline_service: BaselineHaarService = Depends(get_baseline_service)
):
    """
    Biometric student registration with facial verification.
    Accepts reference portrait via file upload or camera snapshot.
    Validates face presence, saves reference photo to data/known_faces,
    extracts facial feature embedding, and persists the student record.
    """
    clean_name = name.strip()
    clean_roll = roll_no.strip()

    if not clean_name or not clean_roll:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student name and roll number cannot be empty."
        )

    existing = db.query(Student).filter(Student.roll_no == clean_roll).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Student with roll number '{clean_roll}' already exists."
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Could not decode."
        )

    # Detect face presence
    boxes = baseline_service.detect_faces(image)
    if not boxes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in portrait. Ensure adequate lighting and look directly at camera."
        )

    # Pick largest detected face
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    face_chip = baseline_service.extract_face(image, boxes[0])
    feature_vec = baseline_service.extract_spatial_lbp_histogram(face_chip)

    # Persist reference photo
    settings.KNOWN_FACES_PATH.mkdir(parents=True, exist_ok=True)
    save_filename = f"{clean_roll}.jpg"
    save_path = settings.KNOWN_FACES_PATH / save_filename
    cv2.imwrite(str(save_path), image)

    # Persist student in database
    student = Student(name=clean_name, roll_no=clean_roll)
    student.set_embedding(feature_vec)
    db.add(student)
    db.commit()
    db.refresh(student)

    return StudentResponse(
        id=student.id,
        name=student.name,
        roll_no=student.roll_no,
        has_embedding=True,
        created_at=student.created_at
    )


# ------------------------------------------------------------------------------
# Attendance Records Endpoints
# ------------------------------------------------------------------------------

@router.get("/attendance", response_model=List[AttendanceResponse], tags=["Attendance"])
def get_attendance(
    target_date: Optional[date] = Query(None, description="Filter attendance by date (YYYY-MM-DD)"),
    student_id: Optional[int] = Query(None, description="Filter attendance by student ID"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Retrieve attendance records with optional date and student filters."""
    records = AttendanceService.get_attendance_records(
        db, target_date=target_date, student_id=student_id, limit=limit
    )
    results = []
    for r in records:
        results.append(
            AttendanceResponse(
                id=r.id,
                student_id=r.student_id,
                student_name=r.student.name if r.student else None,
                roll_no=r.student.roll_no if r.student else None,
                timestamp=r.timestamp,
                confidence=r.confidence,
                liveness_passed=r.liveness_passed
            )
        )
    return results


@router.post("/attendance", response_model=AttendanceResponse, status_code=status.HTTP_201_CREATED, tags=["Attendance"])
def mark_attendance(
    payload: AttendanceCreate,
    db: Session = Depends(get_db)
):
    """Manually log an attendance verification record."""
    try:
        record = AttendanceService.mark_attendance(
            db=db,
            student_id=payload.student_id,
            confidence=payload.confidence,
            liveness_passed=payload.liveness_passed
        )
        return AttendanceResponse(
            id=record.id,
            student_id=record.student_id,
            student_name=record.student.name if record.student else None,
            roll_no=record.student.roll_no if record.student else None,
            timestamp=record.timestamp,
            confidence=record.confidence,
            liveness_passed=record.liveness_passed
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ------------------------------------------------------------------------------
# Baseline Haar Cascade Diagnostics (Phase 1 Baseline Verification)
# ------------------------------------------------------------------------------

@router.post("/baseline/detect", tags=["Baseline Evaluation"])
async def baseline_detect_faces(
    file: UploadFile = File(...),
    baseline_service: BaselineHaarService = Depends(get_baseline_service)
):
    """
    Test Haar Cascade face detector on an uploaded image.
    Returns detected bounding boxes for comparison with deep learning detectors.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Could not decode."
        )

    boxes = baseline_service.detect_faces(image)
    return {
        "filename": file.filename,
        "faces_detected": len(boxes),
        "bounding_boxes": [{"x": b[0], "y": b[1], "w": b[2], "h": b[3]} for b in boxes]
    }
