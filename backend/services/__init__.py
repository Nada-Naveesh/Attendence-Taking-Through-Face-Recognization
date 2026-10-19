"""
Services package containing computer vision, biometric recognition,
anti-spoofing liveness, and attendance services.
"""
from backend.services.baseline_haar import BaselineHaarService
from backend.services.face_recognition import DeepFaceService

__all__ = ["BaselineHaarService", "DeepFaceService"]
