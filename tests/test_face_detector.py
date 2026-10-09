from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.vision
def test_wrapper_preserves_input_coordinates_factory_and_cleanup():
    import mediapipe as mp
    from driver_monitor.config import PROJECT_ROOT
    from driver_monitor.vision.face_detector import FaceDetector

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 13
    frame[:, :, 1] = 27
    frame[:, :, 2] = 91  # distinctive channels catch an accidental BGR/RGB swap
    result = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=0.1, y=0.2, z=5),
                                            SimpleNamespace(x=0.3, y=0.4, z=6)]])
    native = SimpleNamespace(detect=lambda image: result, close=lambda: None)
    with patch("dataset.get_detector", return_value=native) as factory, \
            patch.object(native, "detect", wraps=native.detect) as detect, \
            patch.object(native, "close", wraps=native.close) as close:
        with FaceDetector(PROJECT_ROOT / "face_landmarker.task") as detector:
            landmarks = detector.detect(frame)
            np.testing.assert_array_equal(landmarks, [[0.1, 0.2], [0.3, 0.4]])
            image = detect.call_args.args[0]
            assert image.image_format == mp.ImageFormat.SRGB
            np.testing.assert_array_equal(image.numpy_view(), frame)
            assert landmarks.shape == (2, 2) and landmarks.dtype == np.float64
            native.detect = lambda image: SimpleNamespace(face_landmarks=[])
            assert detector.detect(frame) is None
            factory.assert_called_once_with(str(PROJECT_ROOT / "face_landmarker.task"))
        detector.close()
        close.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            detector.detect(frame)


@pytest.mark.vision
def test_detector_load_failure_is_useful():
    from driver_monitor.vision.face_detector import FaceDetector
    with patch("dataset.get_detector", side_effect=ValueError("bad task")):
        with pytest.raises(RuntimeError, match="Unable to initialize MediaPipe detector asset"):
            FaceDetector("bad.task")
