"""
Deep Learning Facial Detection & Recognition Service.
Utilizes OpenCV Zoo official neural network models:
- YuNet (CVPR): Real-time deep convolutional face detection with 5 facial landmarks.
- SFace: Deep metric learning face recognizer producing 128-D aligned facial embeddings.

Features:
- Automatic 5-point facial landmark alignment (handles head tilt and pose variance).
- High-accuracy Cosine Similarity matching (>99.4% on LFW benchmark).
- Classical Haar Cascade fallback if ONNX weights are not present.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import cv2
import numpy as np
from backend.core.config import settings


class DeepFaceService:
    """Production-grade Deep Learning Biometric Recognition Engine."""

    def __init__(
        self,
        detector_path: Optional[str] = None,
        recognizer_path: Optional[str] = None
    ):
        self.detector_path = Path(detector_path or (settings.MODELS_PATH / "face_detection_yunet.onnx"))
        self.recognizer_path = Path(recognizer_path or (settings.MODELS_PATH / "face_recognition_sface.onnx"))

        self.has_deep_models = self.detector_path.exists() and self.recognizer_path.exists()

        if self.has_deep_models:
            # SFace deep metric recognizer
            self.recognizer = cv2.FaceRecognizerSF.create(str(self.recognizer_path), "")
            self.detector_cache: Dict[Tuple[int, int], cv2.FaceDetectorYN] = {}
        else:
            self.recognizer = None
            self.detector_cache = {}

        # Haar cascade fallback
        self.haar_path = Path(settings.HAARCASCADE_PATH)
        if self.haar_path.exists():
            self.haar_detector = cv2.CascadeClassifier(str(self.haar_path))
        else:
            self.haar_detector = None

    def _get_yunet_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        """Cache detector instance per image resolution for performance."""
        key = (width, height)
        if key not in self.detector_cache:
            detector = cv2.FaceDetectorYN.create(
                str(self.detector_path),
                "",
                (width, height),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            self.detector_cache[key] = detector
        return self.detector_cache[key]

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        Returns list of dicts with:
        - "box": [x, y, w, h]
        - "confidence": float
        - "raw_face": full YuNet detection array (with landmarks) if deep learning active
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if self.has_deep_models:
            detector = self._get_yunet_detector(w, h)
            _, faces = detector.detect(image)
            if faces is not None and len(faces) > 0:
                results = []
                for f in faces:
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    conf = float(f[14])
                    results.append({
                        "box": [max(0, x), max(0, y), fw, fh],
                        "confidence": conf,
                        "raw_face": f
                    })
                return results

        # Fallback to Haar Cascade
        if self.haar_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
            return [{"box": [int(x), int(y), int(fw), int(fh)], "confidence": 0.85, "raw_face": None} for (x, y, fw, fh) in faces]

        return []

    def extract_embedding(self, image: np.ndarray, face_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract normalized 128-dimensional deep metric embedding.
        Performs 5-point facial landmark geometric alignment.
        """
        if self.has_deep_models and face_info.get("raw_face") is not None:
            raw_face = face_info["raw_face"]
            aligned = self.recognizer.alignCrop(image, raw_face)
            feat = self.recognizer.feature(aligned)
            if feat is not None:
                arr = feat.flatten().astype(np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr

        # Fallback: Normalized spatial feature
        box = face_info["box"]
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (112, 112))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr = gray.flatten().astype(np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    def match_embeddings(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two unit embeddings.
        Returns score in [0.0, 1.0].
        """
        if feat1 is None or feat2 is None:
            return 0.0

        if feat1.shape != feat2.shape:
            # Incompatible dimensions
            return 0.0

        if self.has_deep_models and self.recognizer is not None and feat1.shape == (128,):
            score = self.recognizer.match(
                feat1.reshape(1, -1),
                feat2.reshape(1, -1),
                cv2.FaceRecognizerSF_FR_COSINE
            )
            # Map [-1, 1] to [0, 1]
            return float(max(0.0, (score + 1.0) / 2.0))

        # Standard dot product of unit vectors
        dot = float(np.dot(feat1, feat2))
        return max(0.0, min(1.0, dot))

    def match_group_faces(
        self,
        image: np.ndarray,
        enrolled_students: List[Tuple[int, str, str, np.ndarray]],
        similarity_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Detects all faces in a group image and matches them against enrolled students.
        Applies greedy maximum-margin assignment to ensure 1-to-1 matching.

        Args:
            image: Full BGR classroom/group frame.
            enrolled_students: List of tuples (student_id, student_name, roll_no, embedding_vector).
            similarity_threshold: Minimum cosine similarity score for positive identification.

        Returns:
            List of recognition result dictionaries for each detected face.
        """
        faces = self.detect_faces(image)
        if not faces:
            return []

        # If no enrolled students, return all detected faces as Unknown
        if not enrolled_students:
            return [{
                "matched": False,
                "student_id": None,
                "student_name": "Unknown",
                "roll_no": None,
                "confidence": round(face["confidence"], 4),
                "box": face["box"]
            } for face in faces]

        # Extract embeddings for all detected faces
        extracted_probes = []
        for face in faces:
            embedding = self.extract_embedding(image, face)
            extracted_probes.append((face, embedding))

        # Compute pairwise similarity candidate matches: (similarity, face_idx, student_idx)
        candidate_matches = []
        for face_idx, (face, probe_vec) in enumerate(extracted_probes):
            if probe_vec is None:
                continue
            for s_idx, (s_id, s_name, s_roll, s_emb) in enumerate(enrolled_students):
                if s_emb is None:
                    continue
                sim = self.match_embeddings(probe_vec, s_emb)
                if sim >= similarity_threshold:
                    candidate_matches.append((sim, face_idx, s_idx))

        # Sort candidate matches descending by similarity for greedy 1-to-1 matching
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        assigned_faces = set()
        assigned_students = set()
        face_matches = {}

        for sim, f_idx, s_idx in candidate_matches:
            if f_idx not in assigned_faces and s_idx not in assigned_students:
                assigned_faces.add(f_idx)
                assigned_students.add(s_idx)
                s_id, s_name, s_roll, _ = enrolled_students[s_idx]
                face_matches[f_idx] = {
                    "matched": True,
                    "student_id": s_id,
                    "student_name": s_name,
                    "roll_no": s_roll,
                    "confidence": round(sim, 4),
                    "box": extracted_probes[f_idx][0]["box"]
                }

        # Build final results for all detected faces (including unmatched)
        final_results = []
        for f_idx, (face, _) in enumerate(extracted_probes):
            if f_idx in face_matches:
                final_results.append(face_matches[f_idx])
            else:
                final_results.append({
                    "matched": False,
                    "student_id": None,
                    "student_name": "Unknown",
                    "roll_no": None,
                    "confidence": round(face["confidence"], 4),
                    "box": face["box"]
                })

        return final_results
