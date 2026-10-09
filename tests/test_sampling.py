"""Deadline selection at real source timestamps; no model/vision dependencies."""

import configparser
import math

import numpy as np
import pytest

from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.sampling import TemporalSampler, TemporalTimestampError


def test_first_exact_before_after_deadline():
    sampler = TemporalSampler()
    first = sampler.consider(0)
    assert first.sample_due and first.next_sample_time == .05
    assert not sampler.consider(.05 - 1e-7).sample_due
    assert sampler.consider(.05).sample_due
    assert not sampler.consider(.1 - 1e-7).sample_due
    assert sampler.consider(.1 + 1e-7).sample_due
    assert sampler.next_sample_time == pytest.approx(.15)


def test_missed_deadlines_and_jump_never_duplicate():
    sampler = TemporalSampler()
    assert sampler.consider(0).sample_due
    delayed = sampler.consider(.11)
    assert delayed.sample_due and delayed.missed_deadlines == 1
    assert delayed.next_sample_time == pytest.approx(.15)
    jump = sampler.consider(10)
    assert jump.sample_due and jump.missed_deadlines == 197
    assert jump.next_sample_time == pytest.approx(10.05)
    assert not sampler.consider(10.01).sample_due


@pytest.mark.parametrize("timestamp", [None, float("nan"), float("inf"), -1, True, "0"])
def test_invalid_timestamp_is_controlled_and_atomic(timestamp):
    sampler = TemporalSampler()
    sampler.consider(0)
    state = vars(sampler).copy()
    with pytest.raises(TemporalTimestampError):
        sampler.consider(timestamp)
    assert vars(sampler) == state


@pytest.mark.parametrize("timestamp", [1, .9, 0])
def test_repeated_and_backwards_timestamp(timestamp):
    sampler = TemporalSampler()
    sampler.consider(1)
    with pytest.raises(TemporalTimestampError, match="strictly increase"):
        sampler.consider(timestamp)
    assert sampler.last_timestamp == 1


@pytest.mark.parametrize("source_fps", [10, 20, 30, 60])
def test_source_cadence_and_no_duplication(source_fps, record_property):
    sampler = TemporalSampler()
    timestamps = np.arange(source_fps * 10) / source_fps
    accepted = [float(t) for t in timestamps if sampler.consider(t).sample_due]
    expected = min(source_fps, 20) * 10
    assert abs(len(accepted) - expected) <= 1
    assert len(accepted) == len(set(accepted)) and len(accepted) <= len(timestamps)
    effective = (len(accepted) - 1) / (accepted[-1] - accepted[0])
    assert effective == pytest.approx(min(source_fps, 20), rel=.01)
    record_property("synthetic_cadence", f"source={source_fps}, decoded={len(timestamps)}, sampled={len(accepted)}, effective={effective}")


@pytest.mark.parametrize("fps", [5, 12.5, 25, 40])
def test_other_target_rates(fps):
    sampler = TemporalSampler(fps)
    accepted = sum(sampler.consider(index / 100).sample_due for index in range(1000))
    assert abs(accepted - fps * 10) <= 1


def test_long_timeline_has_no_cumulative_drift_and_bounded_state():
    sampler = TemporalSampler()
    count = 0
    for index in range(100000):
        count += sampler.consider(12345 + index / 20).sample_due
    assert count == 100000
    assert sampler.next_sample_time == pytest.approx(12345 + 100000 / 20)
    assert len(vars(sampler)) == 5
    assert not any(isinstance(value, (list, dict, np.ndarray)) for value in vars(sampler).values())


def test_disabled_sampler_and_explicit_reset():
    sampler = TemporalSampler(enabled=False)
    assert all(sampler.consider(t).sample_due for t in [0, .001, .002])
    assert sampler.next_sample_time is None
    with pytest.raises(TemporalTimestampError):
        sampler.consider(.002)
    sampler.reset()
    assert sampler.consider(0).sample_due


def test_unrepresentable_timestamp_span_is_controlled_without_state_change():
    sampler = TemporalSampler(1e308)
    sampler.consider(0)
    state = vars(sampler).copy()
    with pytest.raises(TemporalTimestampError, match="span is too large"):
        sampler.consider(10)
    assert vars(sampler) == state


def test_disabled_sampler_does_not_require_a_target_timeline():
    sampler = TemporalSampler(1e308, enabled=False)
    assert sampler.consider(0).sample_due and sampler.consider(10).sample_due


@pytest.mark.parametrize("field,value", [("target_model_fps", 0), ("target_model_fps", -20),
                                        ("target_model_fps", math.inf), ("target_model_fps", math.nan),
                                        ("target_model_fps", True), ("max_face_gap_seconds", 0),
                                        ("max_face_gap_seconds", -.5), ("max_face_gap_seconds", math.nan),
                                        ("max_face_gap_seconds", math.inf), ("enable_temporal_sampling", 1)])
def test_config_rejects_invalid_temporal_settings(field, value):
    with pytest.raises(ValueError):
        RuntimeConfig(**{field: value})


def test_configuration_ini_and_defaults(tmp_path):
    original = RuntimeConfig.from_ini()
    assert (original.target_model_fps, original.max_face_gap_seconds, original.enable_temporal_sampling) == (20, .5, True)
    source = original.asset_root / "config.ini"
    # Keep asset root intact: an optional runtime section can be parsed from a file
    # in the repo, or explicit absolute asset paths from a copied INI.
    parser = configparser.ConfigParser()
    parser.read(source)
    parser.set("output", "model_path", original.model_path.as_posix())
    parser.set("data", "normalize_data_file", original.normalization_path.as_posix())
    parser.set("data", "mediapipe_model_path", original.face_model_path.as_posix())
    parser.add_section("runtime")
    parser.set("runtime", "target_model_fps", "12.5")
    parser.set("runtime", "max_face_gap_seconds", "0.7")
    parser.set("runtime", "enable_temporal_sampling", "false")
    path = tmp_path / "runtime.ini"
    with path.open("w") as handle:
        parser.write(handle)
    configured = RuntimeConfig.from_ini(path)
    assert (configured.target_model_fps, configured.max_face_gap_seconds, configured.enable_temporal_sampling) == (12.5, .7, False)
