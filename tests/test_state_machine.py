from dataclasses import replace

import numpy as np
import pytest

from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.smoothing import PredictionSmoother
from driver_monitor.inference.state_machine import DriverStateMachine, DriverState
from driver_monitor.inference.sampling import TemporalTimestampError

NORMAL=[.90,.05,.05]
YAWN=[.10,.80,.10]
EYE=[.05,.10,.85]


def update(machine,t,vector=None,**kwargs):
    prediction = None if vector is None else PredictionSmoother(1).update(vector)
    return machine.update(t,face_detected=kwargs.pop('face_detected',True),prediction=prediction,**kwargs)


def test_warming_no_face_and_normal_transition_reasons():
    machine=DriverStateMachine()
    assert update(machine,0,warming_up=True).state==DriverState.WARMING_UP
    result=update(machine,.1,face_detected=False)
    assert result.state==DriverState.NO_FACE and result.transition_reason=='no_face'
    result=update(machine,.2,NORMAL)
    assert result.state==DriverState.ATTENTIVE and result.previous_state==DriverState.NO_FACE
    assert result.transition_occurred and result.transition_reason=='normal_evidence'
    assert not result.alert_active


def test_eye_persistence_drowsy_alert_and_recovery():
    machine=DriverStateMachine()
    assert update(machine,0,EYE).state==DriverState.SUSPECTED_DROWSY
    assert update(machine,.999999,EYE).state==DriverState.SUSPECTED_DROWSY
    drowsy=update(machine,1,EYE)
    assert drowsy.state==DriverState.DROWSY and drowsy.transition_reason=='eye_close_persistent'
    assert update(machine,2.499999,EYE).state==DriverState.DROWSY
    alert=update(machine,2.5,EYE)
    assert alert.state==DriverState.ALERT and alert.alert_active
    assert alert.drowsy_duration_seconds==pytest.approx(1.5)
    result=update(machine,2.55,NORMAL)
    assert result.state==DriverState.ATTENTIVE and not result.alert_active
    assert result.eye_close_persistence_seconds==result.drowsy_duration_seconds==0


def test_yawning_separate_persistence_and_never_alerts():
    machine=DriverStateMachine()
    assert update(machine,0,YAWN).state==DriverState.ATTENTIVE
    assert update(machine,.299999,YAWN).state==DriverState.ATTENTIVE
    result=update(machine,.3,YAWN)
    assert result.state==DriverState.FATIGUE and result.transition_reason=='yawning_persistent'
    for t in [1,2,10,100]:
        result=update(machine,t,YAWN)
        assert result.state==DriverState.FATIGUE and not result.alert_active
        assert result.eye_close_persistence_seconds==0


@pytest.mark.parametrize("probability,expected", [(.699999,DriverState.ATTENTIVE),
                                                 (.7,DriverState.SUSPECTED_DROWSY),
                                                 (.700001,DriverState.SUSPECTED_DROWSY)])
def test_eye_enter_boundary(probability,expected):
    result=update(DriverStateMachine(),0,[1-probability,0,probability])
    assert result.state==expected


def test_eye_hysteresis_retains_band_and_exit_equality_then_resets():
    machine=DriverStateMachine()
    update(machine,0,[.29,0,.71])
    for index,p in enumerate([.69,.71,.69,.55],1):
        result=update(machine,index*.1,[1-p,0,p])
        assert result.state==DriverState.SUSPECTED_DROWSY
        assert not result.transition_occurred
    result=update(machine,.5,[.450001,0,.549999])
    assert result.state==DriverState.ATTENTIVE and result.eye_close_persistence_seconds==0


def test_short_missing_time_pauses_and_resumes_without_credit():
    machine=DriverStateMachine()
    update(machine,0,EYE)
    update(machine,.5,EYE)
    for t in [.55,.75,.9]:
        result=update(machine,t,face_detected=False,new_prediction=False)
        assert result.state==DriverState.NO_FACE and result.eye_close_persistence_seconds==.5
    result=update(machine,.95,EYE)
    assert result.eye_close_persistence_seconds==.5
    assert result.state==DriverState.SUSPECTED_DROWSY
    result=update(machine,1,EYE)
    assert result.eye_close_persistence_seconds==pytest.approx(.55)


def test_alert_is_suspended_on_no_face_and_reset_clears():
    machine=DriverStateMachine()
    update(machine,0,EYE)
    update(machine,2.5,EYE)
    result=update(machine,2.6,face_detected=False,new_prediction=False)
    assert result.state==DriverState.NO_FACE and not result.alert_active
    assert result.eye_close_persistence_seconds==2.5
    result=update(machine,3,face_detected=False,gap_reset=True,new_prediction=False)
    assert result.eye_close_persistence_seconds==0 and result.state==DriverState.NO_FACE
    assert result.transition_reason=='gap_reset_no_face'
    assert update(machine,3.1,warming_up=True).state==DriverState.WARMING_UP
    assert update(machine,4,EYE).state==DriverState.SUSPECTED_DROWSY


def test_unsampled_face_frames_hold_state_without_advancing_timer():
    machine=DriverStateMachine()
    update(machine,0,EYE)
    result=update(machine,.03,EYE,new_prediction=False)
    assert result.eye_close_persistence_seconds==0 and not result.transition_occurred
    result=update(machine,.05,EYE)
    assert result.eye_close_persistence_seconds==pytest.approx(.05)


@pytest.mark.parametrize("timestamp", [None,float('nan'),float('inf'),-1,0])
def test_timestamp_errors_use_phase5_contract_and_leave_state(timestamp):
    machine=DriverStateMachine()
    update(machine,0,EYE)
    with pytest.raises(TemporalTimestampError):
        update(machine,timestamp,EYE)
    assert machine.state==DriverState.SUSPECTED_DROWSY and machine._eye_seconds==0


def test_zero_durations_are_valid_and_logical_alert_can_start_immediately():
    config=RuntimeConfig(drowsy_persistence_seconds=0,alert_persistence_seconds=0)
    result=update(DriverStateMachine(config),0,EYE)
    assert result.state==DriverState.ALERT and result.alert_active


@pytest.mark.parametrize("fps",[10,20,30,60])
def test_persistence_depends_on_elapsed_time_not_prediction_count(fps):
    machine=DriverStateMachine()
    first_drowsy=first_alert=None
    for index in range(3*fps+1):
        timestamp=index/fps
        result=update(machine,timestamp,EYE)
        if result.state==DriverState.DROWSY and first_drowsy is None:
            first_drowsy=timestamp
        if result.state==DriverState.ALERT and first_alert is None:
            first_alert=timestamp
    assert first_drowsy==1 and first_alert==2.5


@pytest.mark.parametrize("change", [dict(prediction_history_size=0),dict(prediction_history_size=True),
                                    dict(smoothing_mode='unknown'),dict(normal_enter_threshold=np.nan),
                                    dict(yawning_enter_threshold=1.1),dict(eye_close_enter_threshold=-.1),
                                    dict(eye_close_exit_threshold=.8),dict(yawning_exit_threshold=.9),
                                    dict(drowsy_persistence_seconds=-1),dict(alert_persistence_seconds=np.inf),
                                    dict(yawning_persistence_seconds=True)])
def test_policy_configuration_validation(change):
    with pytest.raises(ValueError):
        RuntimeConfig(**change)
