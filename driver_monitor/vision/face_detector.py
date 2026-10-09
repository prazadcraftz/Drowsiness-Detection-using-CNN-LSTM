"""Wrap the existing IMAGE detector, including the legacy BGR-as-SRGB input."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class FaceDetector:
    def __init__(self, model_path: str | Path):
        # Lazy imports allow feature/sequence-only engine use without vision packages.
        import mediapipe as mp
        from dataset import get_detector

        self._mp = mp
        self._closed = False
        try:
            # Reuse the original options: one face, blendshapes and matrices enabled,
            # no VIDEO mode, timestamps, tracking changes, or RGB conversion.
            self._detector = get_detector(str(model_path))
        except Exception as error:
            raise RuntimeError(f"Unable to initialize MediaPipe detector asset {model_path}: {error}") from error

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        if self._closed:
            raise RuntimeError("FaceDetector is closed")
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        detection_result = self._detector.detect(image)
        try:
            return np.array([(landmark.x, landmark.y)
                             for landmark in detection_result.face_landmarks[0]])
        except (IndexError, AttributeError):
            return None

    def close(self):
        if not self._closed:
            self._closed = True
            self._detector.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FaceDetector is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
