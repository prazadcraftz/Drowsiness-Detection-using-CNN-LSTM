import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from driver_monitor.config import CLASS_MAPPING, FEATURE_ORDER, PROJECT_ROOT, RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from driver_monitor.inference.normalizer import Normalizer
from driver_monitor.results import InferenceResult

ATOL, RTOL = 1e-6, 1e-5


class MissingFaceDetector:
    def __init__(self, *args):
        self.close_count = 0

    def detect(self, frame):
        return None

    def close(self):
        self.close_count += 1


def test_configuration_resolves_assets_outside_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RuntimeConfig.from_ini()
    assert config.model_path == PROJECT_ROOT / "saved_model.pt"
    assert config.normalization_path == PROJECT_ROOT / "norm.npy"
    assert config.face_model_path == PROJECT_ROOT / "face_landmarker.task"
    assert config.sequence_length == 16 and config.feature_count == 5 and config.device == "cpu"
    assert FEATURE_ORDER == ("EAR-R", "EAR-L", "MAR", "phi", "theta")
    mapping = dict(CLASS_MAPPING)
    configured = RuntimeConfig(class_mapping=mapping)
    mapping[0] = "changed"
    assert configured.class_mapping[0] == "Normal"
    with pytest.raises(TypeError):
        configured.class_mapping[0] = "changed"


@pytest.mark.parametrize("change, message", [
    ({"feature_count": 6}, "feature_count=5"),
    ({"sequence_length": 8}, "sequence_length=16"),
    ({"device": "cuda"}, "device='cpu'"),
    ({"class_mapping": {0: "Normal", 1: "Yawning"}}, "class_mapping"),
    ({"class_mapping": {0: "Yawning", 1: "Normal", 2: "Eye Close"}}, "class_mapping"),
    ({"cnn_dropout_rate": 0.5}, "cnn_dropout_rate=0.25"),
])
def test_configuration_rejects_contract_changes(change, message):
    with pytest.raises(ValueError, match=message):
        RuntimeConfig(**change)


@pytest.mark.parametrize("setting", ["model_path", "normalization_path", "face_model_path"])
def test_configuration_reports_missing_assets(tmp_path, setting):
    missing = tmp_path / "missing.file"
    with pytest.raises(FileNotFoundError, match=setting):
        RuntimeConfig(**{setting: missing})


def test_missing_or_invalid_ini_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration does not exist"):
        RuntimeConfig.from_ini(tmp_path / "missing.ini")
    path = tmp_path / "invalid.ini"
    path.write_text("[General]\nnum_classes=2\n")
    with pytest.raises(ValueError, match="num_classes=3"):
        RuntimeConfig.from_ini(path)


@pytest.mark.parametrize("ranges, message", [
    (np.ones((5,)), "shape"),
    (np.full((5, 2), np.nan), "finite"),
    (np.full((5, 2), np.inf), "finite"),
    (np.zeros((5, 2)), "maximum - minimum"),
    (np.tile([1.0, 0.0], (5, 1)), "maximum - minimum"),
    (np.tile(np.array([2, 1], dtype=np.uint8), (5, 1)), "maximum - minimum"),
])
def test_normalizer_rejects_invalid_ranges(tmp_path, ranges, message):
    path = tmp_path / "invalid.npy"
    np.save(path, ranges)
    with pytest.raises(ValueError, match=message):
        Normalizer(path)


def test_normalizer_load_failures_are_useful(tmp_path):
    with pytest.raises(FileNotFoundError, match="Normalization asset"):
        Normalizer(tmp_path / "missing.npy")
    path = tmp_path / "invalid.npy"
    path.write_text("not an npy archive")
    with pytest.raises(ValueError, match="Unable to load normalization"):
        Normalizer(path)
    archive = tmp_path / "ranges.npz"
    np.savez(archive, ranges=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="one NumPy array"):
        Normalizer(archive)


def test_normalizer_matches_all_frozen_observations_and_does_not_clip(frozen_phase2, record_property):
    _, frozen = frozen_phase2
    normalizer = Normalizer(PROJECT_ROOT / "norm.npy")
    rows = np.flatnonzero(frozen["face_detected"])
    actual = np.stack([normalizer.normalize(frozen["raw_features"][row]) for row in rows])
    expected = frozen["normalized_features"][rows]
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    record_property("phase3_normalization_max_difference", float(np.max(np.abs(actual - expected))))
    np.testing.assert_array_equal(normalizer.normalize(2 * normalizer.ranges[:, 1] - normalizer.ranges[:, 0]), 2)
    copy = normalizer.ranges
    copy[:] = 0
    np.testing.assert_array_equal(normalizer.ranges, frozen["normalization_ranges"])
    for malformed in (np.zeros(6), np.full(5, np.nan), np.full(5, np.inf)):
        with pytest.raises(ValueError, match="shape|finite"):
            normalizer.normalize(malformed)


@pytest.mark.vision
def test_new_feature_layer_matches_all_frozen_landmarks(frozen_phase2, record_property):
    from driver_monitor.vision.features import extract_features
    manifest, frozen = frozen_phase2
    rows = np.flatnonzero(frozen["face_detected"])
    actual = []
    for row in rows:
        source = manifest["sources"][int(frozen["source_ids"][row])]
        image = np.zeros((source["height"], source["width"], 3), dtype=np.uint8)
        actual.append(extract_features(image, frozen["landmarks_xy"][row]))
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, frozen["raw_features"][rows], atol=ATOL, rtol=RTOL)
    record_property("phase3_feature_max_difference", float(np.max(np.abs(actual - frozen["raw_features"][rows]))))


def test_engine_model_parity_and_inference_mode_all_699_tensors(frozen_phase2, record_property):
    _, frozen = frozen_phase2
    maximum_logits = maximum_probabilities = 0.0
    with DrowsinessInferenceEngine(initialize_detector=False) as engine:
        def check_mode(module, args):
            assert not module.training
            assert torch.is_inference_mode_enabled()
            assert args[0].device.type == "cpu"
        hook = engine.model.register_forward_pre_hook(check_mode)
        try:
            for index, tensor in enumerate(frozen["input_tensors"]):
                result = engine.infer_sequence(tensor[0])
                assert result.observation is None and len(engine.buffer) == 0
                assert result.input_tensor_shape == (1, 5, 16)
                assert result.logits.shape == result.probabilities.shape == (1, 3)
                np.testing.assert_array_equal(result.input_tensor, tensor)
                np.testing.assert_allclose(result.logits, frozen["logits"][index], atol=ATOL, rtol=RTOL)
                np.testing.assert_allclose(result.probabilities, frozen["probabilities"][index], atol=ATOL, rtol=RTOL)
                np.testing.assert_allclose(result.probabilities.sum(axis=1), [1.0], atol=ATOL, rtol=RTOL)
                assert result.raw_class_id == frozen["predicted_class_ids"][index]
                assert result.raw_class_name == frozen["predicted_class_names"][index]
                assert result.confidence == result.probabilities[0, result.raw_class_id]
                maximum_logits = max(maximum_logits, float(np.max(np.abs(result.logits - frozen["logits"][index]))))
                maximum_probabilities = max(maximum_probabilities, float(np.max(np.abs(result.probabilities - frozen["probabilities"][index]))))
        finally:
            hook.remove()
    record_property("phase3_model_logit_max_difference", maximum_logits)
    record_property("phase3_model_probability_max_difference", maximum_probabilities)
    record_property("phase3_model_class_matches", len(frozen["input_tensors"]))


def test_engine_recreates_all_windows_from_frozen_observations(frozen_phase2, record_property):
    manifest, frozen = frozen_phase2
    maxima = dict.fromkeys(["raw_features", "normalized_features", "sequences",
                           "input_tensors", "logits", "probabilities"], 0.0)
    matches = 0
    for source_id in range(len(manifest["sources"])):
        with DrowsinessInferenceEngine(initialize_detector=False) as engine:
            engine.detector = MissingFaceDetector()
            for row in np.flatnonzero(frozen["source_ids"] == source_id):
                if not frozen["face_detected"][row]:
                    before = engine.buffer.window() if engine.buffer.ready else None
                    result = engine.process_frame(None)
                    assert not result.observation.face_detected and not result.observation.ready_for_inference
                    assert result.observation.raw_features is result.observation.normalized_features is None
                    assert result.raw_class_id is None and result.logits is None
                    if before is not None:
                        np.testing.assert_array_equal(engine.buffer.window(), before)
                else:
                    result = engine.process_features(frozen["raw_features"][row])
                    for name in ("raw_features", "normalized_features"):
                        actual = getattr(result.observation, name)
                        np.testing.assert_allclose(actual, frozen[name][row], atol=ATOL, rtol=RTOL)
                        maxima[name] = max(maxima[name], float(np.max(np.abs(actual - frozen[name][row]))))
                    index = frozen["frame_window_indices"][row]
                    assert result.observation.ready_for_inference == (index >= 0)
                    if index >= 0:
                        for name, actual in [("sequences", result.sequence), ("input_tensors", result.input_tensor),
                                             ("logits", result.logits), ("probabilities", result.probabilities)]:
                            np.testing.assert_allclose(actual, frozen[name][index], atol=ATOL, rtol=RTOL)
                            maxima[name] = max(maxima[name], float(np.max(np.abs(actual - frozen[name][index]))))
                        assert result.input_tensor_shape == (1, 5, 16)
                        assert result.raw_class_id == frozen["predicted_class_ids"][index]
                        matches += 1
                    else:
                        assert result.logits is result.probabilities is result.raw_class_id is None
                assert result.observation.sequence_fill == frozen["sequence_fill"][row]
                assert result.observation.warming_up == frozen["warming_up"][row]
                assert result.observation.reset_occurred is False
    assert matches == manifest["window_count"] == 699
    for name, value in maxima.items():
        record_property("phase3_observations_" + name + "_max_difference", value)
    record_property("phase3_observations_class_matches", matches)


def test_model_normalization_detector_load_once_and_cleanup():
    from driver_monitor.inference import engine as module
    original_load = torch.load
    with patch.object(module, "CNN_LSTM", wraps=module.CNN_LSTM) as construct_model, \
            patch.object(module, "Normalizer", wraps=module.Normalizer) as construct_norm, \
            patch.object(module, "FaceDetector", side_effect=MissingFaceDetector) as construct_detector, \
            patch.object(torch, "load", wraps=original_load) as load, \
            patch.object(np, "load", wraps=np.load) as load_norm:
        engine = DrowsinessInferenceEngine()
        for _ in range(20):
            engine.process_frame(None)
            engine.process_features(np.zeros(5))
        assert construct_model.call_count == construct_norm.call_count == construct_detector.call_count == 1
        assert load.call_count == load_norm.call_count == 1
        assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
        assert not engine.model.training
        engine.close()
        engine.close()
        assert engine.detector.close_count == 1
        for operation in [lambda: engine.process_features(np.zeros(5)),
                          lambda: engine.infer_sequence(np.zeros((5, 16))),
                          lambda: engine.process_frame(None)]:
            with pytest.raises(RuntimeError, match="closed"):
                operation()


def test_context_manager_closes_after_exception():
    detector = MissingFaceDetector()
    with patch("driver_monitor.inference.engine.FaceDetector", return_value=detector):
        with pytest.raises(ValueError, match="intentional"):
            with DrowsinessInferenceEngine() as engine:
                raise ValueError("intentional caller error")
    assert detector.close_count == 1 and engine._closed


def test_corrupt_checkpoint_fails_before_detector_initialization(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_text("not a checkpoint")
    with patch("driver_monitor.inference.engine.FaceDetector") as detector:
        with pytest.raises(RuntimeError, match="Unable to load CNN-LSTM checkpoint"):
            DrowsinessInferenceEngine(dataclasses.replace(RuntimeConfig(), model_path=path))
        detector.assert_not_called()


def test_input_validation_and_result_snapshots(frozen_phase2):
    _, frozen = frozen_phase2
    with DrowsinessInferenceEngine(initialize_detector=False) as engine:
        for values in [np.zeros((5, 1)), np.zeros(6), np.full(5, np.nan), np.full(5, 1 + 2j)]:
            with pytest.raises(ValueError, match="shape|finite"):
                engine.process_features(values)
            assert len(engine.buffer) == 0
        for sequence in [np.zeros((16, 5)), np.zeros((1, 5, 16)), np.full((5, 16), np.inf),
                         np.full((5, 16), 1 + 2j)]:
            with pytest.raises(ValueError, match="shape|finite"):
                engine.infer_sequence(sequence)
        with pytest.raises(RuntimeError, match="requires detector initialization"):
            engine.process_frame(None)
        for row in range(16):
            result = engine.process_features(frozen["raw_features"][row])
        expected = engine.buffer.window()
        result.observation.normalized_features[:] = 999
        result.sequence[:] = 999
        result.input_tensor[:] = 999
        np.testing.assert_array_equal(engine.buffer.window(), expected)


def test_one_hundred_thousand_observations_have_bounded_storage(record_property):
    with DrowsinessInferenceEngine(initialize_detector=False) as engine:
        # Exercise actual normalization/buffering without spending 100k model forwards.
        # The replacement is a plain callable, so the test itself does not retain calls.
        with patch.object(engine, "infer_sequence", new=lambda sequence, *, observation=None:
                          InferenceResult(observation=observation)):
            maximum_length = maximum_bytes = 0
            for index in range(100_000):
                engine.process_features(np.arange(5, dtype=float) + index)
                maximum_length = max(maximum_length, len(engine.buffer))
                maximum_bytes = max(maximum_bytes, engine.buffer.feature_storage_bytes)
            assert maximum_length == 16 and maximum_bytes == 16 * 5 * 8 == 640
            assert engine.buffer._observations.maxlen == 16
            assert all(item.base is None and item.shape == (5,) for item in engine.buffer._observations)
            assert not any(isinstance(value, (list, dict)) for value in engine.__dict__.values())
            assert not any("history" in name for name in engine.__dict__)
            expected = np.stack([engine.normalizer.normalize(np.arange(5) + index)
                                 for index in range(100_000 - 16, 100_000)], axis=1)
            np.testing.assert_array_equal(engine.buffer.window(), expected)
    record_property("phase3_memory_observations", 100_000)
    record_property("phase3_memory_max_buffer_length", maximum_length)
    record_property("phase3_memory_max_feature_payload_bytes", maximum_bytes)


def test_new_engine_model_path_has_no_vision_import_dependency():
    code = r'''
import importlib.abc, sys
class NoVision(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cv2','mediapipe'}:
            raise AssertionError('Unexpected vision import: '+fullname)
sys.meta_path.insert(0,NoVision())
from driver_monitor import DrowsinessInferenceEngine
import numpy as np
with DrowsinessInferenceEngine(initialize_detector=False) as engine:
    for i in range(16):result=engine.process_features(np.zeros(5))
    assert result.input_tensor_shape==(1,5,16)
    assert result.logits.shape==(1,3)
assert 'cv2' not in sys.modules and 'mediapipe' not in sys.modules
print('model/feature-only engine imports no vision packages')
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.vision
def test_real_video_legacy_vs_engine_all_six_sources(tmp_path, pytestconfig, record_property):
    output = pytestconfig.getoption("--phase3-replay-output")
    output = Path(output).resolve() if output else tmp_path / "phase3_comparison"
    command = [sys.executable, "-B", str(PROJECT_ROOT / "tools" / "compare_phase3.py"),
               "--output", str(output)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["status"] == "PASS"
    for name in ["legacy_vs_frozen", "engine_vs_frozen", "engine_vs_fresh_legacy"]:
        comparison = summary[name]
        assert comparison["frame_count"] == 800
        assert comparison["exact_raw_class_matches"] == comparison["window_count"] == 699
        for stage, difference in comparison["max_absolute_difference"].items():
            record_property("phase3_video_" + name + "_" + stage + "_max_difference", difference)
        record_property("phase3_video_" + name + "_class_matches", comparison["exact_raw_class_matches"])


def test_comparison_tool_cannot_write_to_frozen_fixtures():
    from tools.compare_phase3 import run
    with pytest.raises(ValueError, match="never be written inside the frozen"):
        run(PROJECT_ROOT / "tests" / "fixtures")
