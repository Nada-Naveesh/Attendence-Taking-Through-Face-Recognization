"""
Pydantic schemas for request and response serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """Health check payload."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    environment: str = Field(..., json_schema_extra={"example": "development"})
    database: str = Field(..., json_schema_extra={"example": "connected"})
    uptime_seconds: float = Field(..., json_schema_extra={"example": 120.5})
    components: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class StudentBase(BaseModel):
    """Base student properties."""
    name: str = Field(..., min_length=1, max_length=100, json_schema_extra={"example": "John Doe"})
    roll_no: str = Field(..., min_length=1, max_length=50, json_schema_extra={"example": "CS-2024-001"})


class StudentCreate(StudentBase):
    """Payload to create a new student."""
    pass


class StudentResponse(StudentBase):
    """Student response payload."""
    id: int
    has_embedding: bool = False
    samples_processed: int = 1
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AttendanceBase(BaseModel):
    """Base attendance record."""
    student_id: int
    confidence: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.92})
    liveness_passed: bool = Field(default=False)


class AttendanceCreate(AttendanceBase):
    """Payload to log attendance."""
    pass


class AttendanceResponse(AttendanceBase):
    """Attendance log response."""
    id: int
    student_name: Optional[str] = None
    roll_no: Optional[str] = None
    timestamp: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RecognitionResult(BaseModel):
    """Single facial recognition inference result."""
    matched: bool
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    roll_no: Optional[str] = None
    confidence: float = 0.0
    liveness_passed: bool = False
    box: Optional[List[int]] = Field(None, description="[x, y, w, h] or [x1, y1, x2, y2]")


class GroupRecognitionResponse(BaseModel):
    """Inference response for multi-face group image processing."""
    total_faces_detected: int
    recognized_count: int
    results: List[RecognitionResult]
    processed_at: datetime = Field(default_factory=datetime.utcnow)
