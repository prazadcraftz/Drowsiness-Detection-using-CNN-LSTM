"""Translate authoritative Phase 7 tracker transitions; no event business logic."""

from dataclasses import fields
import hashlib
import importlib.metadata
from pathlib import Path
import platform

from .sqlite_store import canonical_json

EVENT_NAMES = {"yawn": "YAWN", "eye_closure": "EYE_CLOSURE", "drowsiness": "DROWSINESS", "alert": "ALERT", "no_face": "NO_FACE"}


def session_provenance(engine, *, timed, audio_enabled, alert_cooldown_seconds):
    """Hash assets once at opt-in startup, exclude private absolute paths."""
    config = engine.config
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    def version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
    runtime = {field.name: getattr(config, field.name) for field in fields(config)
               if field.name not in ("asset_root", "model_path", "normalization_path", "face_model_path", "class_mapping")}
    runtime.update(class_mapping=dict(config.class_mapping), temporal_runtime_enabled=timed,
                   audio_enabled=audio_enabled, alert_cooldown_seconds=alert_cooldown_seconds)
    return {
        "checkpoint_sha256": sha256(config.model_path), "normalization_sha256": sha256(config.normalization_path),
        "face_task_sha256": sha256(config.face_model_path),
        "model_name": 'LSTMOnly32' if config.model_profile == 'lstm32' else type(engine.model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in engine.model.parameters()),
        **{name: getattr(config, name) for name in ("sequence_length", "target_model_fps", "max_face_gap_seconds", "smoothing_mode", "prediction_history_size")},
        "application_version": "phase10-sqlite-v1", "python_version": platform.python_version(),
        "torch_version": version("torch"), "opencv_version": version("opencv-contrib-python") or version("opencv-python"),
        "mediapipe_version": version("mediapipe"), "runtime_config_json": canonical_json(runtime),
    }


def metric_columns(aggregate, *, offline_processing_fps=None):
    output = {name: aggregate[name] for name in ("decoded_frames", "face_detected_frames", "no_face_frames", "sampled_observations", "model_predictions", "state_transitions", "valid_observed_decision_seconds", "unobserved_interval_seconds")}
    for mode in ("raw", "smoothed"):
        for name, label in (("normal", "Normal"), ("yawning", "Yawning"), ("eye_close", "Eye Close")):
            output[f"{name}_{mode}_count"] = aggregate[f"{mode}_prediction_counts"][label]
    for name, column in (("yawn", "yawn"), ("eye_closure", "eye_closure"), ("drowsiness", "drowsy"), ("alert", "alert"), ("no_face", "no_face")):
        output[f"{column}_events"] = aggregate["events"][name]["count"]
        output[f"longest_{column}_seconds"] = aggregate["events"][name]["longest_duration_seconds"]
    for name, seconds in aggregate["state_occupancy_seconds"].items():
        output[f"{name.lower()}_seconds"] = seconds
    output.update(eye_evidence_fraction=aggregate["observed_eye_evidence_fraction"],
                  effective_observation_fps=aggregate["effective_model_observation_fps"],
                  offline_processing_fps=offline_processing_fps)
    audio = aggregate.get("audio") or {}
    for column, key in (("audible_alert_requests", "audible_trigger_count"), ("audio_suppressed_requests", "suppressed_trigger_count"), ("audio_completed", "completed_playback_count"), ("audio_failed", "failed_playback_count")):
        output[column] = audio.get(key)
    return output


class SessionPersistence:
    """At most five open row IDs. ACTIVE observations never issue SQL.

    Call update immediately after each SessionMetrics.update. The tracker is
    the authority for identity, boundary and duration, including paused gaps.
    On an exception close the store without calling finalize: committed START
    and END rows survive, open durations remain NULL (unknown partial evidence).
    """

    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id
        self.open_ids = {}
        self.last_state = None
        self.finalized = False

    def update(self, metrics, decision):
        if self.finalized or metrics.finalized:
            raise RuntimeError("Cannot persist updates after finalization")
        state = None if decision.driver is None else decision.driver.state.value
        reset = decision.raw_result.observation.gap_reset or state == "WARMING_UP"
        changes = []
        for name, tracker in metrics.events.items():
            if tracker.active and name not in self.open_ids:
                changes.append(("start", name, tracker))
            elif not tracker.active and name in self.open_ids:
                changes.append(("end", name, tracker))
        # No transaction/SQL on ACTIVE or otherwise uneventful frames.
        if changes:
            next_ids = dict(self.open_ids)
            with self.store.transaction():
                for action, name, tracker in changes:
                    if action == "start":
                        next_ids[name] = self.store.start_event(self.session_id, EVENT_NAMES[name], tracker.start_timestamp, start_state=state)
                    else:
                        self.store.end_event(self.session_id, next_ids.pop(name), ended_at_source_seconds=tracker.end_timestamp,
                                             observed_duration_seconds=tracker.duration_seconds, end_state=state,
                                             closed_reason="temporal_reset" if reset and name in ("yawn", "eye_closure", "drowsiness") else "evidence_exit")
            self.open_ids = next_ids
        self.last_state = state

    def finalize(self, metrics, *, audio=None, offline_processing_fps=None):
        if self.finalized:
            return
        if not metrics.finalized:
            raise ValueError("Finalize SessionMetrics before persistence")
        aggregate = metrics.snapshot(audio)
        with self.store.transaction():
            for name, event_id in self.open_ids.items():
                tracker = metrics.events[name]
                self.store.end_event(self.session_id, event_id, ended_at_source_seconds=tracker.end_timestamp,
                                     observed_duration_seconds=tracker.duration_seconds, end_state=self.last_state, closed_reason="session_end")
            self.store.finalize_session(self.session_id, metrics=metric_columns(aggregate, offline_processing_fps=offline_processing_fps), aggregate=aggregate)
        self.open_ids.clear()
        self.finalized = True
