"""Engineering events; evidence durations exclude missing and reset bridges."""

from dataclasses import asdict, dataclass
import math
from numbers import Real

from driver_monitor.config import CLASS_MAPPING
from driver_monitor.inference.state_machine import DriverState


@dataclass(slots=True)
class TimedEventTracker:
    """One current/last episode and scalar aggregates; no episode history.

    update(True) starts or extends an episode. pause() preserves identity but
    breaks the observed-time bridge. end() ends at the last active observation.
    For NO_FACE only, end(boundary) includes the held interval until recovery.
    """

    active: bool = False
    start_timestamp: float | None = None
    last_active_timestamp: float | None = None
    end_timestamp: float | None = None
    duration_seconds: float = 0.0
    count: int = 0
    longest_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    _bridge_timestamp: float | None = None

    def update(self, active, timestamp):
        if not active:
            self.end()
            return
        if not self.active:
            self.active = True
            self.count += 1
            self.start_timestamp = timestamp
            self.end_timestamp = None
            self.duration_seconds = 0.0
        elif self._bridge_timestamp is not None:
            self._add(timestamp - self._bridge_timestamp)
        self.last_active_timestamp = self._bridge_timestamp = timestamp

    def _add(self, delta):
        self.duration_seconds += delta
        self.total_duration_seconds += delta
        self.longest_duration_seconds = max(self.longest_duration_seconds, self.duration_seconds)

    def pause(self):
        self._bridge_timestamp = None

    def end(self, boundary=None):
        if self.active:
            if boundary is not None and self._bridge_timestamp is not None:
                self._add(boundary - self._bridge_timestamp)
            self.end_timestamp = self.last_active_timestamp if boundary is None else boundary
            self.active = False
        self.pause()

    def snapshot(self):
        return {key: value for key, value in asdict(self).items() if not key.startswith("_")}


class SessionMetrics:
    """One update per decoded frame; no retained frames, vectors or snapshots."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_timestamp = self.end_timestamp = None
        self.decoded_frames = self.face_detected_frames = self.no_face_frames = 0
        self.sampled_observations = self.model_predictions = self.state_transitions = 0
        self.raw_counts = {name: 0 for name in CLASS_MAPPING.values()}
        self.smoothed_counts = dict(self.raw_counts)
        self.state_seconds = {state.value: 0.0 for state in DriverState}
        self.state_frame_counts = {state.value: 0 for state in DriverState}
        self.events = {name: TimedEventTracker() for name in ("yawn", "eye_closure", "drowsiness", "alert", "no_face")}
        self.valid_observed_decision_seconds = 0.0
        self.unobserved_interval_seconds = 0.0
        self._last_timestamp = self._last_state = self._last_valid_decision = None
        self._first_sample = self._last_sample = None
        self.finalized = False

    def update(self, decision, *, timestamp_seconds=None):
        if self.finalized:
            raise RuntimeError("Session finalized; reset before updating")
        raw, driver = decision.raw_result, decision.driver
        observation = raw.observation
        if observation is None:
            raise ValueError("SessionMetrics requires a decoded-frame observation")
        timestamp = (driver.timestamp_seconds if driver is not None else observation.timestamp_seconds)
        if timestamp_seconds is not None:
            timestamp = timestamp_seconds
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real)
                or not math.isfinite(timestamp) or timestamp < 0
                or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
            raise ValueError("Session timestamp must be finite, nonnegative and strictly increasing")
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        if self._last_timestamp is not None and self._last_state is not None:
            delta = timestamp - self._last_timestamp
            if observation.gap_reset and self._last_state != DriverState.NO_FACE.value:
                # A reset across a source stall cannot certify the previous face
                # state throughout the unseen bridge. Keep it explicitly unknown.
                self.unobserved_interval_seconds += delta
            else:
                self.state_seconds[self._last_state] += delta
        self.end_timestamp = self._last_timestamp = timestamp
        self.decoded_frames += 1
        self.face_detected_frames += int(observation.face_detected)
        self.no_face_frames += int(not observation.face_detected)
        if observation.face_detected and observation.sampled:
            self.sampled_observations += 1
            if self._first_sample is None:
                self._first_sample = timestamp
            self._last_sample = timestamp
        if raw.raw_class_id is not None:
            self.model_predictions += 1
            self.raw_counts[CLASS_MAPPING[raw.raw_class_id]] += 1
            if driver is not None and driver.smoothed_class_name is not None:
                self.smoothed_counts[driver.smoothed_class_name] += 1
        if driver is None:
            # Untimed legacy mode has no derived policy: do not invent states/events.
            self._last_state = None
            return
        state = driver.state
        self._last_state = state.value
        self.state_frame_counts[state.value] += 1
        self.state_transitions += int(driver.transition_occurred)
        no_face = self.events["no_face"]
        if state == DriverState.NO_FACE:
            no_face.update(True, timestamp)
        else:
            no_face.end(timestamp)
        evidence = [self.events[name] for name in ("yawn", "eye_closure", "drowsiness")]
        # Logical ALERT episodes follow external transitions, even across NO_FACE.
        if state == DriverState.ALERT:
            self.events["alert"].update(True, timestamp)
        else:
            self.events["alert"].end()
        if observation.gap_reset or state == DriverState.WARMING_UP:
            for event in evidence:
                event.end()
            self._last_valid_decision = None
        elif state == DriverState.NO_FACE:
            for event in evidence:
                event.pause()
            self._last_valid_decision = None
        elif raw.raw_class_id is not None and observation.sampled:
            if self._last_valid_decision is not None:
                self.valid_observed_decision_seconds += timestamp - self._last_valid_decision
            self._last_valid_decision = timestamp
            self.events["yawn"].update(state == DriverState.FATIGUE, timestamp)
            self.events["eye_closure"].update(state in (DriverState.SUSPECTED_DROWSY, DriverState.DROWSY, DriverState.ALERT), timestamp)
            self.events["drowsiness"].update(state in (DriverState.DROWSY, DriverState.ALERT), timestamp)

    def finalize(self):
        """Close at available observations, never add an interval after EOF."""
        if not self.finalized:
            for event in self.events.values():
                event.end()
            self.finalized = True
        return self.snapshot()

    def snapshot(self, audio=None):
        span = 0.0 if self.start_timestamp is None else self.end_timestamp - self.start_timestamp
        sample_span = None if self._first_sample is None else self._last_sample - self._first_sample
        return {
            "session_start_timestamp": self.start_timestamp, "session_end_timestamp": self.end_timestamp,
            "session_duration_seconds": span, "finalized": self.finalized,
            "decoded_frames": self.decoded_frames, "face_detected_frames": self.face_detected_frames,
            "no_face_frames": self.no_face_frames, "sampled_observations": self.sampled_observations,
            "model_predictions": self.model_predictions, "raw_prediction_counts": dict(self.raw_counts),
            "smoothed_prediction_counts": dict(self.smoothed_counts), "state_transitions": self.state_transitions,
            "state_occupancy_seconds": dict(self.state_seconds), "state_frame_counts": dict(self.state_frame_counts),
            "events": {name: event.snapshot() for name, event in self.events.items()},
            "valid_observed_decision_seconds": self.valid_observed_decision_seconds,
            "unobserved_interval_seconds": self.unobserved_interval_seconds,
            "face_observed_duration_seconds": sum(seconds for state, seconds in self.state_seconds.items() if state != "NO_FACE"),
            "observed_eye_evidence_fraction": (self.events["eye_closure"].total_duration_seconds / self.valid_observed_decision_seconds
                                               if self.valid_observed_decision_seconds > 0 else None),
            "effective_model_observation_fps": ((self.sampled_observations - 1) / sample_span
                                                if sample_span is not None and sample_span > 0 else None),
            "audio": None if audio is None else dict(audio),
        }
