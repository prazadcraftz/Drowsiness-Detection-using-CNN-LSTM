"""Timestamped shared application path, diagnostics and real 20 Hz sources."""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

import main
from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from driver_monitor.results import InferenceResult, ObservationResult
from driver_monitor.vision.video_source import FramePacket

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source_type", ["webcam", "file"])
@pytest.mark.parametrize("fps", [10, 20, 30, 60])
def test_shared_timestamped_source_rates(monkeypatch, fps, source_type):
    class Source:
        def __init__(self, source):
            self.source_name = source_type
            self.file_path = None
            self.nominal_fps = fps
            self.index = 0
            self.closed = False
        def read(self):
            index = self.index
            self.index += 1
            return FramePacket(np.zeros((24, 32, 3), np.uint8), index, index / fps, source_type, index >= 2 * fps)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
    class Detector:
        calls = 0
        def detect(self, frame):
            self.calls += 1
            return np.ones((478, 2))
        def close(self):
            pass
    source = Source(None)
    engine = DrowsinessInferenceEngine(initialize_detector=False)
    engine.detector = Detector()
    from driver_monitor.vision import features
    monkeypatch.setattr(features, "extract_features", lambda frame, landmarks: np.ones(5))
    # Test policy/loop rather than repeat the separate 699-window model gate.
    monkeypatch.setattr(engine, "infer_sequence", lambda sequence, *, observation:
                        InferenceResult(observation, raw_class_id=0, raw_class_name="Normal", confidence=.5))
    monkeypatch.setattr(main, "VideoSource", lambda _: source)
    monkeypatch.setattr(main, "_create_engine", lambda: engine)
    for name in ["imshow", "waitKey", "destroyAllWindows"]:
        monkeypatch.setattr(main.cv2, name, lambda *args: pytest.fail("Headless GUI call"))
    summary = main.run(0 if source_type == "webcam" else "file.avi", no_display=True)
    assert summary.frames_decoded == summary.valid_observations == 2 * fps
    assert summary.sampled_valid_observations == min(fps, 20) * 2
    assert summary.not_sampled_valid_frames == summary.valid_observations - summary.sampled_valid_observations
    assert summary.predictions == summary.sampled_valid_observations - 15
    assert summary.capture_decode_source_fps == pytest.approx(fps)
    assert summary.effective_model_observation_fps == pytest.approx(min(fps, 20), rel=.02)
    assert summary.temporal_runtime_enabled and summary.gap_resets == 0
    assert engine.detector.calls == 2 * fps  # Detector is never skipped for timing numbers.
    assert source.closed and engine._closed


def test_display_not_sampled_latest_prediction_reset_and_no_face(monkeypatch):
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: texts.append(text))
    frame = np.zeros((480, 640, 3), np.uint8)
    packet = FramePacket(frame, 1, .01, "file")
    result = InferenceResult(ObservationResult(True, None, None, 16, False, sampled=False, sample_due=False))
    main.render_result(frame, packet, result, "file", latest_prediction=("Yawning", .8), target_fps=20, source_fps=60)
    assert "NOT SAMPLED" in texts and "Latest prediction: Yawning" in texts
    assert "Prediction: Yawning" not in texts
    texts.clear()
    result = InferenceResult(ObservationResult(False, None, None, 0, False, reset_occurred=True,
                                              gap_reset=True, sampled=False))
    main.render_result(frame, packet, result, "file")
    assert "NO FACE" in texts and "TEMPORAL RESET" in texts
    assert not any("Prediction" in text or "prediction" in text for text in texts)


@pytest.mark.vision
@pytest.mark.parametrize("filename", ["Normal/Normal_1.avi", "Yawning/Yawning_1.avi",
                                     "Eye Close/Eye Close_60.avi", "Yawning/Yawning_60.avi"])
def test_real_sampling_enabled_application(filename, frozen_phase2, monkeypatch, record_property):
    manifest, arrays = frozen_phase2
    source_index = next(i for i, source in enumerate(manifest["sources"]) if source["filename"] == filename)
    source = manifest["sources"][source_index]
    rows = np.arange(source["first_frame_row"], source["last_frame_row_exclusive"])
    with DrowsinessInferenceEngine() as engine:
        original = engine.process_frame
        original_policy = engine._temporal_decision
        policy_times = []
        seen = []
        def timed_policy(*args):
            start = time.perf_counter_ns()
            result = original_policy(*args)
            policy_times.append((time.perf_counter_ns() - start) / 1000)
            return result
        def compare(frame, *, timestamp_seconds):
            result = original(frame, timestamp_seconds=timestamp_seconds)
            row = rows[len(seen)]
            seen.append(row)
            obs = result.observation
            assert obs.timestamp_seconds == arrays["timestamps_seconds"][row]
            assert obs.face_detected == bool(arrays["face_detected"][row])
            assert obs.sample_due and not obs.gap_reset
            assert obs.sampled == obs.face_detected
            assert obs.sequence_fill == arrays["sequence_fill"][row]
            window = arrays["frame_window_indices"][row]
            assert (result.raw_class_id is not None) == (window >= 0)
            if obs.face_detected:
                np.testing.assert_array_equal(obs.raw_features, arrays["raw_features"][row])
                np.testing.assert_array_equal(obs.normalized_features, arrays["normalized_features"][row])
            if window >= 0:
                for key, actual in [("sequences", result.sequence), ("input_tensors", result.input_tensor),
                                    ("logits", result.logits), ("probabilities", result.probabilities)]:
                    np.testing.assert_array_equal(actual, arrays[key][window])
                assert result.raw_class_id == arrays["predicted_class_ids"][window]
            return result
        monkeypatch.setattr(engine, "_temporal_decision", timed_policy)
        monkeypatch.setattr(engine, "process_frame", compare)
        monkeypatch.setattr(main, "_create_engine", lambda: engine)
        summary = main.run(str(ROOT / filename), no_display=True)
    assert summary.temporal_runtime_enabled and summary.frames_decoded == source["decoded_frame_count"]
    assert summary.sampled_valid_observations == summary.valid_observations
    assert summary.not_sampled_valid_frames == summary.gap_resets == 0
    assert summary.warming_up_observations == 15
    record_property("phase5_real_summary", json.dumps(summary.__dict__))
    record_property("phase5_policy_microseconds_p50", float(np.percentile(policy_times, 50)))
    record_property("phase5_policy_microseconds_p95", float(np.percentile(policy_times, 95)))
    record_property("phase5_max_numeric_difference", 0.0)


def test_invalid_timestamp_cli_reports_error_and_cleans_up(monkeypatch, capsys):
    class Source:
        source_name = "invalid time"
        file_path = None
        nominal_fps = 20
        closed = False
        def read(self):
            return FramePacket(None, 0, None, "file")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
    class Detector:
        def detect(self, frame):
            return None
        def close(self):
            pass
    source = Source()
    engine = DrowsinessInferenceEngine(initialize_detector=False)
    engine.detector = Detector()
    monkeypatch.setattr(main, "VideoSource", lambda _: source)
    monkeypatch.setattr(main, "_create_engine", lambda: engine)
    assert main.main(["--source", "bad.avi", "--no-display"]) == 1
    assert "Source timestamp" in capsys.readouterr().err
    assert source.closed and engine._closed
