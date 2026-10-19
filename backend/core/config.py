"""
Application configuration module using Pydantic Settings.
Loads configuration from environment variables or .env file.
"""

from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project Root Directory calculation
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings schema and defaults."""

    # Project Information
    APP_NAME: str = "Smart Attendance & Analytics AI Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_PATH: Path = BASE_DIR
    DATA_PATH: Path = BASE_DIR / "data"
    MODELS_PATH: Path = BASE_DIR / "models"
    KNOWN_FACES_PATH: Path = BASE_DIR / "data" / "known_faces"
    TEST_SET_PATH: Path = BASE_DIR / "data" / "test_set"

    # Database Configuration
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'attendance.db'}"

    # Computer Vision & Deep Learning
    DETECTOR_BACKEND: str = "yolov8"  # "yolov8" | "mtcnn" | "haar"
    DETECTION_CONFIDENCE: float = 0.50
    RECOGNITION_MODEL: str = "Facenet512"  # "Facenet512" | "ArcFace" | "VGG-Face"
    DISTANCE_METRIC: str = "cosine"  # "cosine" | "euclidean"
    SIMILARITY_THRESHOLD: float = 0.68  # Minimum cosine similarity for a match (or 1 - cosine distance <= threshold)

    # Liveness & Anti-Spoofing Detection
    EAR_THRESHOLD: float = 0.25  # Eye Aspect Ratio threshold
    EAR_CONSECUTIVE_FRAMES: int = 3  # Min consecutive frames for blink confirmation
    LIVENESS_WINDOW_SECONDS: float = 3.0  # Challenge verification timeout

    # Baseline Haar Classifier Path
    HAARCASCADE_PATH: Path = BASE_DIR / "backend" / "services" / "assets" / "haarcascade_frontalface_default.xml"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"
    ]

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
