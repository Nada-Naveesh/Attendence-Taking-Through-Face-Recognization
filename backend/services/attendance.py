"""
Attendance Service managing attendance records, deduplication rules,
and analytical query aggregation.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.db.models import Attendance, Student


class AttendanceService:
    """Service handling attendance persistence, queries, and analytics."""

    @staticmethod
    def mark_attendance(
        db: Session,
        student_id: int,
        confidence: float,
        liveness_passed: bool = True,
        allow_duplicate_same_day: bool = False
    ) -> Attendance:
        """
        Record attendance for a verified student.

        Args:
            db: Active SQLAlchemy database session.
            student_id: ID of the recognized student.
            confidence: Biometric match confidence [0.0 - 1.0].
            liveness_passed: Whether anti-spoofing challenge passed.
            allow_duplicate_same_day: If False, prevents multiple records on same calendar day.

        Returns:
            Created Attendance database record.
        """
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found.")

        today = date.today()

        if not allow_duplicate_same_day:
            existing = (
                db.query(Attendance)
                .filter(
                    Attendance.student_id == student_id,
                    func.date(Attendance.timestamp) == today
                )
                .first()
            )
            if existing:
                return existing

        record = Attendance(
            student_id=student_id,
            confidence=confidence,
            liveness_passed=liveness_passed
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_attendance_records(
        db: Session,
        target_date: Optional[date] = None,
        student_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Attendance]:
        """
        Query attendance records with optional filtering.

        Args:
            db: Database session.
            target_date: Optional filter by calendar date.
            student_id: Optional filter by specific student.
            limit: Maximum records to return.

        Returns:
            List of Attendance records.
        """
        query = db.query(Attendance).join(Student)

        if target_date:
            query = query.filter(func.date(Attendance.timestamp) == target_date)
        if student_id:
            query = query.filter(Attendance.student_id == student_id)

        return query.order_by(Attendance.timestamp.desc()).limit(limit).all()

    @staticmethod
    def get_student_summary(db: Session, student_id: int) -> Dict[str, Any]:
        """Compute attendance summary for a student."""
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            raise ValueError(f"Student {student_id} not found.")

        total_days = db.query(func.count(func.distinct(func.date(Attendance.timestamp)))).scalar() or 1
        attended_days = (
            db.query(func.count(func.distinct(func.date(Attendance.timestamp))))
            .filter(Attendance.student_id == student_id)
            .scalar() or 0
        )

        attendance_rate = (attended_days / total_days) * 100.0 if total_days > 0 else 0.0

        return {
            "student_id": student.id,
            "name": student.name,
            "roll_no": student.roll_no,
            "attended_days": attended_days,
            "total_days": total_days,
            "attendance_rate_pct": round(attendance_rate, 2),
            "at_risk": attendance_rate < 75.0
        }
