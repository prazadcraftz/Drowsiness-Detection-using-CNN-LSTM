"""Exact legacy ratios and pose calculations, in the frozen five-feature order."""

import numpy as np

from draw_landmarks import calculate_ear, calculate_mar, calculate_head_pose
from driver_monitor.config import FEATURE_ORDER

# Reusing the existing implementation deliberately preserves all pose quirks.
calculate_eye_ratio = calculate_ear
calculate_mouth_ratio = calculate_mar


def extract_features(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    right = calculate_eye_ratio(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
    left = calculate_eye_ratio(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
    mouth = calculate_mouth_ratio(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
    phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))
    return np.array([right, left, mouth, phi[0], theta[0]])
