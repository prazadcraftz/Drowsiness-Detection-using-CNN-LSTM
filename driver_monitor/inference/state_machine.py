"""Engineering interpretation of probability evidence; logical ALERT only."""

from dataclasses import dataclass
from enum import StrEnum
from copy import copy
import math
import numpy as np

from driver_monitor.config import RuntimeConfig
from .sampling import TemporalSampler
from .smoothing import SmoothedPrediction, probability_vector


class DriverState(StrEnum):
    WARMING_UP = "WARMING_UP"
    NO_FACE = "NO_FACE"
    ATTENTIVE = "ATTENTIVE"
    FATIGUE = "FATIGUE"
    SUSPECTED_DROWSY = "SUSPECTED_DROWSY"
    DROWSY = "DROWSY"
    ALERT = "ALERT"


@dataclass(frozen=True)
class DriverStateResult:
    state: DriverState
    previous_state: DriverState
    transition_occurred: bool
    transition_reason: str
    smoothed_class_id: int | None
    smoothed_class_name: str | None
    smoothed_confidence: float | None
    smoothed_probabilities: np.ndarray | None
    eye_close_probability: float | None
    yawning_probability: float | None
    eye_close_persistence_seconds: float
    yawning_persistence_seconds: float
    drowsy_duration_seconds: float
    alert_active: bool
    timestamp_seconds: float


class DriverStateMachine:
    def __init__(self, config=None):
        self.config = config or RuntimeConfig.from_ini()
        self.reset()

    def reset(self):
        # Reuse Phase 5's validation with selection disabled, not a new clock rule.
        self._clock = TemporalSampler(self.config.target_model_fps, enabled=False)
        self.state = DriverState.WARMING_UP
        self._active_state = DriverState.WARMING_UP
        self._last_evidence_time = None
        self._clear_persistence()

    def _clear_persistence(self):
        self._eye_active = self._yawn_active = False
        self._eye_seconds = self._yawn_seconds = 0.0
        self._last_evidence_time = None
        self._active_state = DriverState.WARMING_UP

    def validate_timestamp(self, timestamp):
        # Preview the exact Phase 5 validator without committing its scalar clock.
        copy(self._clock).consider(timestamp)

    def update(self, timestamp_seconds, *, face_detected, warming_up=False,
               gap_reset=False, prediction: SmoothedPrediction | None = None,
               new_prediction=True):
        probabilities = None if prediction is None else probability_vector(prediction.probabilities)
        timestamp = self._clock.consider(timestamp_seconds).timestamp_seconds
        previous = self.state
        if gap_reset:
            self._clear_persistence()
            prediction = probabilities = None
        if not face_detected:
            self._last_evidence_time = None  # Discard the bridge across missing time.
            state, reason = DriverState.NO_FACE, "gap_reset_no_face" if gap_reset else "no_face"
        elif warming_up or gap_reset:
            self._clear_persistence()
            state, reason = DriverState.WARMING_UP, "gap_reset" if gap_reset else "warming_up"
        elif prediction is None or not new_prediction:
            state, reason = self._active_state, "holding_without_new_prediction"
        else:
            delta = 0.0 if self._last_evidence_time is None else timestamp - self._last_evidence_time
            self._last_evidence_time = timestamp
            eye_was_active, yawn_was_active = self._eye_active, self._yawn_active
            eye_threshold = self.config.eye_close_exit_threshold if eye_was_active else self.config.eye_close_enter_threshold
            self._eye_active = probabilities[2] >= eye_threshold
            if self._eye_active:
                self._eye_seconds = self._eye_seconds + delta if eye_was_active else 0.0
                self._yawn_active, self._yawn_seconds = False, 0.0
                tolerance = max(1e-12, 8 * math.ulp(timestamp))
                if self._eye_seconds + tolerance >= self.config.drowsy_persistence_seconds:
                    duration = max(0.0, self._eye_seconds - self.config.drowsy_persistence_seconds)
                    if duration + tolerance >= self.config.alert_persistence_seconds:
                        state, reason = DriverState.ALERT, "alert_persistence_met"
                    else:
                        state, reason = DriverState.DROWSY, "eye_close_persistent"
                else:
                    state, reason = DriverState.SUSPECTED_DROWSY, "eye_close_elevated"
            else:
                self._eye_seconds = 0.0
                yawn_threshold = self.config.yawning_exit_threshold if yawn_was_active else self.config.yawning_enter_threshold
                self._yawn_active = probabilities[1] >= yawn_threshold
                if self._yawn_active:
                    self._yawn_seconds = self._yawn_seconds + delta if yawn_was_active else 0.0
                    if self._yawn_seconds + max(1e-12, 8 * math.ulp(timestamp)) >= self.config.yawning_persistence_seconds:
                        state, reason = DriverState.FATIGUE, "yawning_persistent"
                    else:
                        state, reason = DriverState.ATTENTIVE, "yawning_pending"
                else:
                    self._yawn_seconds = 0.0
                    state = DriverState.ATTENTIVE
                    reason = "normal_evidence" if probabilities[0] >= self.config.normal_enter_threshold else "insufficient_evidence"
            self._active_state = state
        self.state = state
        duration = max(0.0, self._eye_seconds - self.config.drowsy_persistence_seconds)
        return DriverStateResult(state, previous, state != previous, reason,
                                 None if prediction is None else prediction.class_id,
                                 None if prediction is None else prediction.class_name,
                                 None if prediction is None else prediction.confidence,
                                 None if probabilities is None else probabilities.copy(),
                                 None if probabilities is None else float(probabilities[2]),
                                 None if probabilities is None else float(probabilities[1]),
                                 self._eye_seconds, self._yawn_seconds, duration,
                                 state == DriverState.ALERT, timestamp)
