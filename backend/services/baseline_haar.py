"""
Baseline Computer Vision Service: Haar Cascades + LBPH (Local Binary Patterns Histogram).

Ported directly from the legacy implementation (old main.py / attendence.py)
to serve as an empirical scientific baseline for benchmarking against modern
deep-learning pipelines (YOLOv8 + FaceNet/ArcFace).

Theoretical Reference:
- Viola & Jones (2001): "Rapid Object Detection using a Boosted Cascade of Simple Features"
- Ahonen, Hadid, & Pietikäinen (2006): "Face Description with Local Binary Patterns"
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from backend.core.config import settings


class BaselineHaarService:
    """
    Encapsulates the classical Computer Vision pipeline:
    1. Viola-Jones Haar Feature-based Cascade Classifier for face detection.
    2. Local Binary Patterns Histogram (LBPH) for texture-based face recognition.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize detector with Haar Cascade XML file.

        Args:
            cascade_path: Path to haarcascade_frontalface_default.xml.
                          Defaults to settings.HAARCASCADE_PATH.
        """
        self.cascade_path = Path(cascade_path or settings.HAARCASCADE_PATH)
        if not self.cascade_path.exists():
            raise FileNotFoundError(
                f"Haar cascade XML not found at {self.cascade_path}. "
                "Ensure haarcascade_frontalface_default.xml is present."
            )
        self.detector = cv2.CascadeClassifier(str(self.cascade_path))

        # Check for OpenCV Contrib LBPH Recognizer
        self.has_cv2_face = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
        if self.has_cv2_face:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            self.recognizer = None

        # Memory store for enrolled LBP histograms: {label: [histogram_vectors]}
        self.enrolled_histograms: Dict[str, List[np.ndarray]] = {}

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade. Matches legacy scaleFactor and minNeighbors.

        Args:
            image: BGR or Grayscale image (numpy array).
            scale_factor: Image pyramid downscale factor (default: 1.2).
            min_neighbors: Minimum neighboring rectangles to retain candidate (default: 5).
            min_size: Minimum face bounding box size (default: 30x30).

        Returns:
            List of bounding boxes [(x, y, w, h), ...].
        """
        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_face(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (160, 160)
    ) -> np.ndarray:
        """
        Crop and resize detected face to normalized dimensions.

        Args:
            image: Input image.
            box: (x, y, w, h) bounding box.
            target_size: Output dimensions (width, height).

        Returns:
            Normalized grayscale face chip.
        """
        x, y, w, h = box
        h_img, w_img = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face_crop = image[y1:y2, x1:x2]
        if len(face_crop.shape) == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Histogram equalization for illumination normalization
        face_crop = cv2.equalizeHist(face_crop)
        return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def compute_lbp(image: np.ndarray) -> np.ndarray:
        """
        Pure NumPy implementation of standard 8-neighborhood Local Binary Pattern.
        Guarantees zero-dependency reproducibility regardless of cv2.face availability.

        Formula:
            LBP(x_c, y_c) = sum_{p=0}^7 s(g_p - g_c) * 2^p
            where s(x) = 1 if x >= 0 else 0
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = image.astype(np.int32)
        h, w = img.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        center = img[1:-1, 1:-1]
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),
            (1, 1), (1, 0), (1, -1),
            (0, -1)
        ]

        for power, (dy, dx) in enumerate(offsets):
            neighbor = img[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << power)

        return lbp

    def extract_spatial_lbp_histogram(
        self,
        face_gray: np.ndarray,
        grid_x: int = 8,
        grid_y: int = 8
    ) -> np.ndarray:
        """
        Extract spatially enhanced LBPH feature vector by partitioning into grid cells.

        Args:
            face_gray: Grayscale normalized face chip (e.g. 160x160).
            grid_x: Number of horizontal partitions.
            grid_y: Number of vertical partitions.

        Returns:
            Normalized concatenated 1D histogram vector.
        """
        lbp = self.compute_lbp(face_gray)
        h, w = lbp.shape
        cell_h = h // grid_y
        cell_w = w // grid_x

        histograms = []
        for r in range(grid_y):
            for c in range(grid_x):
                cell = lbp[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                # Normalize cell histogram
                hist = hist.astype(np.float32)
                norm = np.linalg.norm(hist)
                if norm > 0:
                    hist /= norm
                histograms.append(hist)

        full_vector = np.concatenate(histograms)
        full_norm = np.linalg.norm(full_vector)
        if full_norm > 0:
            full_vector /= full_norm
        return full_vector

    @staticmethod
    def chi_square_distance(hist_a: np.ndarray, hist_b: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute Chi-Square Distance between two LBPH distributions.
        Distance = 0.5 * sum((a_i - b_i)^2 / (a_i + b_i + eps))
        """
        return float(0.5 * np.sum(((hist_a - hist_b) ** 2) / (hist_a + hist_b + eps)))

    def enroll(self, label: str, face_img: np.ndarray) -> None:
        """Enroll a student face into the baseline gallery."""
        norm_face = self.extract_face(face_img, (0, 0, face_img.shape[1], face_img.shape[0]))
        feature = self.extract_spatial_lbp_histogram(norm_face)
        if label not in self.enrolled_histograms:
            self.enrolled_histograms[label] = []
        self.enrolled_histograms[label].append(feature)

    def identify(
        self,
        face_img: np.ndarray,
        distance_threshold: float = 0.45
    ) -> Tuple[Optional[str], float]:
        """
        Identify face against enrolled baseline gallery.

        Returns:
            (predicted_label, confidence_score) where confidence is in [0.0, 1.0].
        """
        if not self.enrolled_histograms:
            return None, 0.0

        norm_face = self.extract_face(face_img, (0, 0, face_img.shape[1], face_img.shape[0]))
        query_feature = self.extract_spatial_lbp_histogram(norm_face)

        best_label = None
        min_dist = float("inf")

        for label, templates in self.enrolled_histograms.items():
            for tmpl in templates:
                dist = self.chi_square_distance(query_feature, tmpl)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label

        # Convert Chi-Square distance to a pseudo-confidence score [0, 1]
        confidence = max(0.0, 1.0 - (min_dist / distance_threshold))

        if min_dist <= distance_threshold:
            return best_label, confidence
        return None, confidence
