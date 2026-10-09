"""Explicit timestamp event segmentation, observed duration and bounded storage."""

from dataclasses import replace
import json
import tracemalloc

import numpy as np
import pytest

from driver_monitor.config import CLASS_MAPPING
from driver_monitor.inference.decision import DecisionResult
from driver_monitor.inference.state_machine import DriverState, DriverStateResult
from driver_monitor.metrics import SessionMetrics, TimedEventTracker
from driver_monitor.results import InferenceResult, ObservationResult


def decision(t, state="ATTENTIVE", *, reset=False, prediction=True, sampled=True):
    state = DriverState(state)
    face = state != DriverState.NO_FACE
    prediction = prediction and face and state != DriverState.WARMING_UP and sampled
    class_id = 1 if state == DriverState.FATIGUE else 2 if state in (DriverState.SUSPECTED_DROWSY, DriverState.DROWSY, DriverState.ALERT) else 0
    obs = ObservationResult(face, None, None, 0 if reset or state == DriverState.WARMING_UP else 16,
                            prediction, timestamp_seconds=t, sampled=sampled and face, gap_reset=reset)
    raw = InferenceResult(obs, raw_class_id=class_id if prediction else None,
                          raw_class_name=CLASS_MAPPING[class_id] if prediction else None)
    driver = DriverStateResult(state, DriverState.ATTENTIVE, state != DriverState.ATTENTIVE, "test",
                               class_id, CLASS_MAPPING[class_id], .9, np.eye(3)[class_id],
                               .9 if class_id == 2 else .05, .9 if class_id == 1 else .05,
                               0, 0, 0, state == DriverState.ALERT, t)
    return DecisionResult(raw, driver)


def run_states(states):
    metrics = SessionMetrics()
    for t, state in states:
        metrics.update(decision(t, state))
    return metrics


def test_no_events_and_prediction_counts():
    metrics = run_states([(0,"WARMING_UP"),(1,"ATTENTIVE"),(2,"ATTENTIVE")])
    assert all(e.count == 0 for e in metrics.events.values())
    assert metrics.decoded_frames == metrics.sampled_observations == 3
    assert metrics.raw_counts == {"Normal":2,"Yawning":0,"Eye Close":0}
    assert metrics.smoothed_counts == metrics.raw_counts


def test_prolonged_yawn_is_one_event_and_recovery_ends_it():
    metrics = run_states([(0,"ATTENTIVE"),(2,"FATIGUE"),(2.05,"FATIGUE"),(2.1,"FATIGUE"),(2.15,"FATIGUE"),(2.2,"FATIGUE"),(3,"ATTENTIVE")])
    event = metrics.events["yawn"]
    assert event.count == 1 and not event.active
    assert event.start_timestamp == 2 and event.end_timestamp == 2.2
    assert event.duration_seconds == pytest.approx(.2)
    assert metrics.raw_counts["Yawning"] == 5


def test_separate_yawns():
    metrics = run_states([(0,"FATIGUE"),(1,"FATIGUE"),(2,"ATTENTIVE"),(3,"FATIGUE"),(5,"FATIGUE")])
    assert metrics.events["yawn"].count == 2
    assert metrics.events["yawn"].total_duration_seconds == 3
    assert metrics.events["yawn"].longest_duration_seconds == 2


def test_eye_drowsiness_and_alert_are_distinct():
    metrics = run_states([(0,"SUSPECTED_DROWSY"),(1,"DROWSY"),(2.5,"ALERT"),(3,"ALERT"),(4,"ATTENTIVE")])
    eye, drowsy, alert = [metrics.events[k] for k in ("eye_closure","drowsiness","alert")]
    assert eye.count == drowsy.count == alert.count == 1
    assert (eye.start_timestamp,drowsy.start_timestamp,alert.start_timestamp) == (0,1,2.5)
    assert (eye.duration_seconds,drowsy.duration_seconds,alert.duration_seconds) == (3,2,.5)
    assert not any(e.active for e in (eye,drowsy,alert))
    metrics.update(decision(5,"SUSPECTED_DROWSY"))
    assert eye.count == 2 and drowsy.count == alert.count == 1


@pytest.mark.parametrize("state,event", [("FATIGUE","yawn"),("SUSPECTED_DROWSY","eye_closure"),("DROWSY","drowsiness")])
def test_short_missing_face_preserves_identity_excludes_bridge(state,event):
    metrics = run_states([(0,state),(.1,state),(.15,"NO_FACE"),(.3,"NO_FACE"),(.4,state),(.5,state)])
    tracker = metrics.events[event]
    assert tracker.count == 1 and tracker.active
    assert tracker.duration_seconds == pytest.approx(.2)
    assert metrics.events["no_face"].count == 1
    assert metrics.events["no_face"].duration_seconds == pytest.approx(.25)
    assert metrics.state_seconds["NO_FACE"] == pytest.approx(.25)


def test_alert_episode_ends_on_no_face_then_counts_reentry():
    metrics = run_states([(0,"ALERT"),(.1,"ALERT"),(.2,"NO_FACE"),(.4,"ALERT"),(.5,"ALERT")])
    assert metrics.events["alert"].count == 2
    assert metrics.events["eye_closure"].count == metrics.events["drowsiness"].count == 1
    assert metrics.events["eye_closure"].duration_seconds == pytest.approx(.2)


@pytest.mark.parametrize("reset_state", ["NO_FACE","WARMING_UP"])
def test_long_reset_ends_at_last_evidence(reset_state):
    metrics = run_states([(0,"ALERT"),(.1,"ALERT"),(.2,"NO_FACE")])
    metrics.update(decision(10,reset_state,reset=True,prediction=False))
    for name in ("eye_closure","drowsiness"):
        e = metrics.events[name]
        assert not e.active and e.end_timestamp == .1 and e.duration_seconds == .1
    metrics.update(decision(11,"SUSPECTED_DROWSY"))
    assert metrics.events["eye_closure"].count == 2


def test_no_face_intervals_and_final_interval_not_invented():
    metrics = run_states([(0,"NO_FACE"),(.1,"NO_FACE"),(.3,"ATTENTIVE"),(.5,"NO_FACE"),(.7,"NO_FACE")])
    metrics.finalize()
    e = metrics.events["no_face"]
    assert e.count == 2 and e.total_duration_seconds == pytest.approx(.5)
    assert e.longest_duration_seconds == pytest.approx(.3) and e.end_timestamp == .7


def test_occupancy_previous_state_held_and_no_final_extension():
    metrics = run_states([(0,"WARMING_UP"),(.5,"ATTENTIVE"),(1,"FATIGUE"),(2,"DROWSY"),(3,"ALERT")])
    summary = metrics.finalize()
    assert summary["state_occupancy_seconds"] == {"WARMING_UP":.5,"NO_FACE":0,"ATTENTIVE":.5,"FATIGUE":1,"SUSPECTED_DROWSY":0,"DROWSY":1,"ALERT":0}
    assert sum(metrics.state_seconds.values()) == 3
    assert metrics.finalize() == summary
    with pytest.raises(RuntimeError,match="finalized"):
        metrics.update(decision(4))


def test_reset_accepts_new_epoch_and_clears_all_counts():
    metrics = run_states([(10,"ALERT"),(11,"ALERT")])
    metrics.finalize()
    metrics.reset()
    metrics.update(decision(0))
    assert metrics.decoded_frames == 1 and metrics.events["alert"].count == 0
    assert metrics.snapshot()["session_duration_seconds"] == 0


def test_held_unsampled_frames_do_not_duplicate_predictions_or_events():
    metrics = run_states([(0,"SUSPECTED_DROWSY")])
    metrics.update(decision(.01,"SUSPECTED_DROWSY",sampled=False))
    metrics.update(decision(.05,"SUSPECTED_DROWSY"))
    assert metrics.model_predictions == metrics.sampled_observations == 2
    assert metrics.events["eye_closure"].count == 1
    assert metrics.events["eye_closure"].duration_seconds == .05


@pytest.mark.parametrize("t", [0,-1,float("nan"),float("inf"),None,True])
def test_invalid_timestamps_do_not_mutate_aggregates(t):
    metrics = run_states([(0,"ATTENTIVE")])
    before = metrics.snapshot()
    with pytest.raises(ValueError):
        metrics.update(decision(t))
    assert metrics.snapshot() == before


def test_compatibility_mode_no_derived_fabrication():
    metrics = SessionMetrics()
    d = decision(0)
    metrics.update(replace(d,driver=None))
    assert metrics.model_predictions == 1 and sum(metrics.state_seconds.values()) == 0
    assert all(e.count == 0 for e in metrics.events.values())


def test_snapshots_are_owned_and_audio_distinct_from_logical_events():
    metrics = run_states([(0,"ALERT"),(1,"ALERT")])
    summary = metrics.snapshot({"audible_trigger_count":0,"suppressed_trigger_count":1})
    assert summary["events"]["alert"]["count"] == 1 and summary["audio"]["audible_trigger_count"] == 0
    summary["events"]["alert"]["count"] = 999
    summary["raw_prediction_counts"]["Normal"] = 999
    assert metrics.events["alert"].count == 1 and metrics.raw_counts["Normal"] == 0
    assert summary["observed_eye_evidence_fraction"] == 1


def test_100000_updates_have_fixed_storage(record_property):
    metrics = SessionMetrics()
    identities = [id(e) for e in metrics.events.values()]
    d = decision(0,"ALERT")
    tracemalloc.start()
    for i in range(100000):
        metrics.update(replace(d,driver=replace(d.driver,timestamp_seconds=i/20)))
        if i == 999:
            baseline = tracemalloc.get_traced_memory()[0]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert current-baseline < 16000
    assert identities == [id(e) for e in metrics.events.values()]
    assert len(metrics.events) == 5 and len(json.dumps(metrics.snapshot())) < 6000
    assert metrics.events["eye_closure"].count == 1
    record_property("metrics_updates",100000)
    record_property("retained_growth_bytes",current-baseline)
    record_property("peak_traced_bytes",peak)


def test_tracker_start_active_pause_end():
    e = TimedEventTracker()
    e.update(True,0)
    e.update(True,1)
    e.pause()
    e.update(True,10)
    e.update(True,11)
    e.end()
    assert e.count == 1 and e.duration_seconds == 2 and e.end_timestamp == 11


def test_reset_after_source_stall_does_not_invent_drowsy_occupancy():
    metrics = run_states([(0,"DROWSY"),(.1,"DROWSY")])
    metrics.update(decision(10,"WARMING_UP",reset=True,prediction=False))
    assert metrics.state_seconds["DROWSY"] == .1
    assert metrics.unobserved_interval_seconds == 9.9
    assert metrics.events["drowsiness"].end_timestamp == .1
