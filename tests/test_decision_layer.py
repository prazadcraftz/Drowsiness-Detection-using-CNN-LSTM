from dataclasses import replace

import numpy as np
import pytest

import main
from driver_monitor.config import CLASS_MAPPING, RuntimeConfig
from driver_monitor.results import InferenceResult, ObservationResult
from driver_monitor.inference.decision import DecisionLayer
from driver_monitor.inference.state_machine import DriverState
from driver_monitor.inference.sampling import TemporalTimestampError
from driver_monitor.vision.video_source import FramePacket

NORMAL=[.9,.05,.05]
EYE=[.05,.1,.85]


def raw(timestamp, probabilities=None, *, face=True, fill=16, sampled=True, reset=False):
    observation=ObservationResult(face,None,None,fill,face and sampled and fill==16,
                                  timestamp_seconds=timestamp,sampled=sampled,gap_reset=reset,
                                  reset_occurred=reset)
    if probabilities is None:
        return InferenceResult(observation)
    vector=np.array([probabilities],dtype=np.float32)
    class_id=int(np.argmax(vector))
    return InferenceResult(observation,probabilities=vector,raw_class_id=class_id,
                           raw_class_name=CLASS_MAPPING[class_id],confidence=float(vector[0,class_id]))


def test_raw_data_preserved_separate_driver_result_and_snapshot_ownership():
    layer=DecisionLayer()
    result=raw(0,EYE)
    before=result.probabilities.copy()
    decision=layer.update(result)
    assert decision.raw_result is result and result.raw_class_name=='Eye Close'
    np.testing.assert_array_equal(result.probabilities,before)
    assert decision.driver.state==DriverState.SUSPECTED_DROWSY
    decision.driver.smoothed_probabilities[:]=0
    np.testing.assert_allclose(layer.smoother.current.probabilities,before[0])


def test_missing_face_preserves_smoother_and_pauses_without_fake_predictions():
    layer=DecisionLayer(RuntimeConfig(prediction_history_size=1))
    layer.update(raw(0,EYE))
    layer.update(raw(.5,EYE))
    for timestamp in [.55,.7,.9]:
        result=layer.update(raw(timestamp,face=False,sampled=False))
        assert result.driver.state==DriverState.NO_FACE
        assert result.driver.eye_close_persistence_seconds==.5
        assert len(layer.smoother)==1 and result.raw_result.raw_class_id is None
    assert layer.update(raw(.95,EYE)).driver.eye_close_persistence_seconds==.5
    assert layer.update(raw(1,EYE)).driver.eye_close_persistence_seconds==pytest.approx(.55)


@pytest.mark.parametrize("face",[True,False])
def test_gap_reset_clears_history_persistence_and_rewarms(face):
    layer=DecisionLayer()
    layer.update(raw(0,EYE))
    layer.update(raw(2.5,EYE))
    assert layer.machine.state==DriverState.ALERT
    result=layer.update(raw(3,face=face,fill=1 if face else 0,sampled=face,reset=True))
    assert len(layer.smoother)==0 and result.driver.eye_close_persistence_seconds==0
    assert result.driver.state==(DriverState.WARMING_UP if face else DriverState.NO_FACE)
    assert not result.driver.alert_active and result.driver.smoothed_probabilities is None
    result=layer.update(raw(3.05,fill=2))
    assert result.driver.state==DriverState.WARMING_UP
    assert layer.update(raw(4,EYE)).driver.state==DriverState.SUSPECTED_DROWSY


def test_unsampled_result_does_not_reuse_raw_as_new_evidence():
    layer=DecisionLayer()
    layer.update(raw(0,EYE))
    result=layer.update(raw(.01,sampled=False))
    assert len(layer.smoother)==1 and result.driver.eye_close_persistence_seconds==0
    assert result.raw_result.raw_class_id is None and not result.driver.transition_occurred


def test_restart_clears_history_and_accepts_new_timestamp_epoch():
    layer=DecisionLayer()
    layer.update(raw(10,EYE))
    layer.reset()
    assert len(layer.smoother)==0 and layer.machine.state==DriverState.WARMING_UP
    assert layer.update(raw(0,NORMAL)).driver.state==DriverState.ATTENTIVE


def test_invalid_time_and_probabilities_do_not_corrupt_smoother():
    layer=DecisionLayer()
    layer.update(raw(0,NORMAL))
    with pytest.raises(TemporalTimestampError):
        layer.update(raw(0,EYE))
    with pytest.raises(ValueError):
        layer.update(raw(.05,[np.nan,0,1]))
    assert len(layer.smoother)==1 and layer.machine._clock.last_timestamp==0
    assert layer.update(raw(.05,NORMAL)).driver.state==DriverState.ATTENTIVE


def test_disabled_policy_keeps_raw_compatibility_without_requiring_clock():
    layer=DecisionLayer(enabled=False)
    result=raw(None,EYE)
    decision=layer.update(result)
    assert decision.raw_result is result and decision.driver is None and len(layer.smoother)==0


def test_standalone_raw_sequence_remains_independent():
    with pytest.raises(ValueError,match='stream observation'):
        DecisionLayer().update(InferenceResult(None))


def test_frozen_all_699_predictions_remain_untouched_by_policy(frozen_phase2,record_property):
    manifest,arrays=frozen_phase2
    matched=0
    for source_id in range(len(manifest['sources'])):
        layer=DecisionLayer()
        for row in np.flatnonzero(arrays['source_ids']==source_id):
            timestamp=float(arrays['timestamps_seconds'][row])
            window=int(arrays['frame_window_indices'][row])
            observation=ObservationResult(bool(arrays['face_detected'][row]),None,None,
                                          int(arrays['sequence_fill'][row]),window>=0,
                                          timestamp_seconds=timestamp,sampled=bool(arrays['face_detected'][row]))
            result=InferenceResult(observation)
            if window>=0:
                result=replace(result,input_tensor_shape=(1,5,16),logits=arrays['logits'][window].copy(),
                               probabilities=arrays['probabilities'][window].copy(),
                               raw_class_id=int(arrays['predicted_class_ids'][window]),
                               raw_class_name=str(arrays['predicted_class_names'][window]),
                               confidence=float(arrays['probabilities'][window].max()))
            decision=layer.update(result)
            assert decision.raw_result is result
            if window>=0:
                np.testing.assert_array_equal(result.logits,arrays['logits'][window])
                np.testing.assert_array_equal(result.probabilities,arrays['probabilities'][window])
                assert result.raw_class_id==arrays['predicted_class_ids'][window]
                matched+=1
    assert matched==699
    record_property('phase6_policy_raw_matches',matched)


def test_application_one_raw_call_and_one_policy_update_per_frame(monkeypatch):
    class Source:
        source_name='fake'
        file_path=None
        nominal_fps=20
        index=0
        def read(self):
            index=self.index
            self.index+=1
            return FramePacket(np.zeros((480,640,3),np.uint8),index,index/20,'file',index>=4)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Engine:
        config=RuntimeConfig()
        calls=0
        def process_frame(self,frame,*,timestamp_seconds):
            self.calls+=1
            return raw(timestamp_seconds,EYE)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Layer(DecisionLayer):
        calls=0
        def update(self,*args,**kwargs):
            self.calls+=1
            return super().update(*args,**kwargs)
    engine=Engine()
    layer=Layer()
    monkeypatch.setattr(main,'VideoSource',lambda _:Source())
    monkeypatch.setattr(main,'_create_engine',lambda:engine)
    monkeypatch.setattr(main,'_create_decision_layer',lambda *args,**kwargs:layer)
    summary=main.run('fake',no_display=True)
    assert summary.frames_decoded==engine.calls==layer.calls==4


def test_display_separates_raw_smooth_state_and_timer(monkeypatch):
    texts=[]
    monkeypatch.setattr(main.cv2,'putText',lambda image,text,*args:texts.append(text))
    result=raw(0,EYE)
    driver=DecisionLayer().update(result).driver
    frame=np.zeros((480,640,3),np.uint8)
    main.render_result(frame,FramePacket(frame,0,0,'file'),result,'fake',policy=driver)
    assert any(text.startswith('Raw: Eye Close') for text in texts)
    assert any(text.startswith('Smooth: Eye Close') for text in texts)
    assert 'State: SUSPECTED_DROWSY' in texts and 'Eye-close timer: 0.00 s' in texts


def test_cli_policy_overrides_are_validated_without_loading_model(monkeypatch,capsys):
    monkeypatch.setattr(main,'_create_engine',lambda *args:pytest.fail('Invalid config loaded model'))
    assert main.main(['--source','0','--prediction-history-size','0','--no-display'])==1
    assert 'prediction_history_size' in capsys.readouterr().err
