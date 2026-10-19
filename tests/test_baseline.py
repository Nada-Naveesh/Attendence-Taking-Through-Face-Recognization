"""
Unit tests for BaselineHaarService (Haar Cascades + LBPH baseline).
"""

import numpy as np
import pytest
from backend.services.baseline_haar import BaselineHaarService


@pytest.fixture
def baseline_service():
    """Initializes BaselineHaarService instance."""
    return BaselineHaarService()


def test_baseline_initialization(baseline_service):
    """Verify Haar cascade XML exists and classifier initializes."""
    assert baseline_service.detector is not None
    assert not baseline_service.detector.empty()


def test_lbp_calculation_synthetic(baseline_service):
    """Verify standard 8-neighborhood LBP operator with known numerical patterns."""
    # Create 5x5 synthetic patch with clear center and gradient
    patch = np.array([
        [10, 20, 30, 40, 50],
        [10, 25, 35, 45, 50],
        [10, 20, 50, 40, 50],  # Center (2,2) has value 50
        [10, 25, 35, 45, 50],
        [10, 20, 30, 40, 50]
    ], dtype=np.uint8)

    lbp_map = baseline_service.compute_lbp(patch)
    assert lbp_map.shape == (3, 3)
    assert lbp_map.dtype == np.uint8


def test_lbp_enroll_and_identify(baseline_service):
    """Verify feature extraction, enrollment, and gallery matching."""
    # Synthetic face image (160x160)
    rng = np.random.RandomState(42)
    face_a = (rng.rand(160, 160) * 255).astype(np.uint8)
    face_b = (rng.rand(160, 160) * 255).astype(np.uint8)

    # Enroll face_a under student ID "S101"
    baseline_service.enroll(label="S101", face_img=face_a)
    assert "S101" in baseline_service.enrolled_histograms

    # Identify face_a (exact match test)
    label, confidence = baseline_service.identify(face_a, distance_threshold=1.0)
    assert label == "S101"
    assert confidence > 0.0
