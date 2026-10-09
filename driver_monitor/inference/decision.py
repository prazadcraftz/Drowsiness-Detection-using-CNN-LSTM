"""One policy update per raw result; ML/sampling/buffer remain independent."""

from dataclasses import dataclass

from driver_monitor.config import RuntimeConfig
from driver_monitor.results import InferenceResult
from .smoothing import PredictionSmoother
from .state_machine import DriverStateMachine, DriverStateResult


@dataclass(frozen=True)
class DecisionResult:
    raw_result: InferenceResult
    driver: DriverStateResult | None


class DecisionLayer:
    def __init__(self, config=None, *, enabled=True):
        self.enabled = enabled
        self.config = config or RuntimeConfig.from_ini()
        self.smoother = PredictionSmoother(self.config.prediction_history_size, self.config.smoothing_mode)
        self.machine = DriverStateMachine(self.config)

    def reset(self):
        self.smoother.clear()
        self.machine.reset()

    def update(self, raw_result: InferenceResult, *, timestamp_seconds=None):
        if not self.enabled:
            return DecisionResult(raw_result, None)
        observation = raw_result.observation
        if observation is None:
            raise ValueError("DecisionLayer requires a stream observation, not a standalone sequence result")
        timestamp = observation.timestamp_seconds if timestamp_seconds is None else timestamp_seconds
        # Validate using the same Phase 5 validator before changing smoothing state.
        # The state machine will commit this clock after evidence validation.
        self.machine.validate_timestamp(timestamp)
        if observation.gap_reset or (observation.warming_up and len(self.smoother)):
            self.smoother.clear()
        new_prediction = (observation.face_detected and observation.sampled
                          and not observation.warming_up and not observation.gap_reset
                          and raw_result.raw_class_id is not None and raw_result.probabilities is not None)
        if new_prediction:
            if raw_result.probabilities.shape != (1, 3):
                raise ValueError("Raw engine probabilities must have shape (1, 3)")
            smooth = self.smoother.update(raw_result.probabilities[0])
        else:
            smooth = self.smoother.current
        driver = self.machine.update(timestamp, face_detected=observation.face_detected,
                                     warming_up=observation.warming_up, gap_reset=observation.gap_reset,
                                     prediction=smooth, new_prediction=new_prediction)
        return DecisionResult(raw_result, driver)
