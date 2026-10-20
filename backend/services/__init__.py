"""
Services package containing computer vision, biometric recognition,
anti-spoofing liveness, and attendance services.
"""
from backend.services.baseline_haar import BaselineHaarService
from backend.services.face_recognition import DeepFaceService
from backend.services.liveness import LivenessService
from backend.services.attendance import AttendanceService
from backend.services.analytics import AttendanceAnalyticsService

__all__ = [
    "BaselineHaarService",
    "DeepFaceService",
    "LivenessService",
    "AttendanceService",
    "AttendanceAnalyticsService"
]
