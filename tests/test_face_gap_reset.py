"""Elapsed valid-observation gaps, including gaps without missing-frame calls."""

import numpy as np
import pytest

from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from driver_monitor.inference.sampling import TemporalTimestampError


class MissingFace:
    def detect(self, frame):
        return None

    def close(self):
        pass


@pytest.fixture
def engine():
    with DrowsinessInferenceEngine(initialize_detector=False) as engine:
        engine.detector = MissingFace()
        yield engine


def observe(engine, timestamp):
    return engine.process_features(np.ones(5), timestamp_seconds=timestamp)


@pytest.mark.parametrize("gap,reset", [(.49, False), (.5, False), (.500001, True), (1., True)])
def test_valid_to_valid_boundary_and_no_missing_calls(engine, gap, reset):
    observe(engine, 1)
    result = observe(engine, 1 + gap)
    assert result.observation.gap_reset == reset
    assert result.observation.reset_occurred == reset
    assert result.observation.gap_seconds == pytest.approx(gap)
    assert result.observation.sequence_fill == (1 if reset else 2)
    assert result.raw_class_id is None


def test_short_no_face_run_retains_buffer_and_never_classifies_gap(engine):
    observe(engine, 0)
    for timestamp in [.05, .1, .3, .5]:
        result = engine.process_frame(None, timestamp_seconds=timestamp)
        assert not result.observation.face_detected and not result.observation.sampled
        assert not result.observation.gap_reset and len(engine.buffer) == 1
        assert result.raw_class_id is result.logits is result.probabilities is None
        assert result.observation.raw_features is result.observation.normalized_features is None


def test_long_no_face_resets_immediately_once_even_off_deadline(engine):
    observe(engine, 0)
    result = engine.process_frame(None, timestamp_seconds=.500001)
    assert result.observation.gap_reset and len(engine.buffer) == 0
    assert result.raw_class_id is None
    assert engine.temporal_runtime.last_valid_observation_timestamp is None
    assert not engine.process_frame(None, timestamp_seconds=1).observation.gap_reset
    result = observe(engine, 1.05)
    assert result.observation.sequence_fill == 1 and result.raw_class_id is None


def test_fifteen_observations_long_gap_then_sixteen_new_observations(engine, record_property):
    for index in range(15):
        result = observe(engine, index / 20)
        assert result.raw_class_id is None
    assert len(engine.buffer) == 15
    for index in range(16):
        result = observe(engine, 1.7 + index / 20)
        assert result.observation.sequence_fill == index + 1
        assert result.observation.gap_reset == (index == 0)
        assert (result.raw_class_id is not None) == (index == 15)
    assert result.input_tensor_shape == (1, 5, 16)
    record_property("synthetic_long_gap", "15 old discarded; fill restarted at 1; prediction only on new observation 16")


def test_unsampled_valid_frames_do_not_append_and_no_face_is_immediate(engine, monkeypatch):
    for index in range(16):
        observe(engine, index / 20)
    before = engine.buffer.window()
    result = observe(engine, .76)
    assert result.observation.face_detected and not result.observation.sample_due
    assert not result.observation.sampled and not result.observation.ready_for_inference
    assert result.raw_class_id is result.logits is None
    np.testing.assert_array_equal(engine.buffer.window(), before)
    result = engine.process_frame(None, timestamp_seconds=.77)
    assert not result.observation.face_detected and not result.observation.sample_due
    np.testing.assert_array_equal(engine.buffer.window(), before)


def test_invalid_timestamps_cannot_change_buffer_or_timing(engine):
    observe(engine, 0)
    state = vars(engine.temporal_runtime.sampler).copy()
    for invalid in [None, float("nan"), float("inf"), -1, 0]:
        with pytest.raises(TemporalTimestampError):
            observe(engine, invalid)
        assert len(engine.buffer) == 1 and vars(engine.temporal_runtime.sampler) == state


def test_untimed_compatibility_and_mode_transition(engine):
    for _ in range(15):
        engine.process_features(np.ones(5))
    result = observe(engine, 0)
    assert result.observation.gap_reset and result.observation.sequence_fill == 1
    with pytest.raises(TemporalTimestampError, match="Cannot mix"):
        engine.process_features(np.ones(5))


def test_disabled_sampling_keeps_gap_protection():
    config = RuntimeConfig(enable_temporal_sampling=False)
    with DrowsinessInferenceEngine(config, initialize_detector=False) as engine:
        for index in range(16):
            result = observe(engine, index / 60)
        assert result.raw_class_id is not None
        result = observe(engine, 1)
        assert result.observation.gap_reset and result.observation.sequence_fill == 1


def test_identical_sequence_in_timed_runtime_still_matches_frozen(engine, frozen_phase2):
    _, frozen = frozen_phase2
    observe(engine, 0)
    observe(engine, 1)
    before = len(engine.buffer)
    result = engine.infer_sequence(frozen["sequences"][0])
    np.testing.assert_array_equal(result.logits, frozen["logits"][0])
    np.testing.assert_array_equal(result.probabilities, frozen["probabilities"][0])
    assert result.raw_class_id == frozen["predicted_class_ids"][0] and len(engine.buffer) == before
