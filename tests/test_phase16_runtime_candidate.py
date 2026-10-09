"""Profile integration gates without held-out candidate access."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from driver_monitor.inference.model_profile import PROFILES, build_model, get_profile
from driver_monitor.inference.temporal_buffer import TemporalBuffer
from driver_monitor.inference.smoothing import PredictionSmoother
from driver_monitor.inference.state_machine import DriverState
from driver_monitor.results import ObservationResult
from driver_monitor.ui.overlay import dashboard_text
from driver_monitor.vision.video_source import FramePacket
from evaluation.manifest import sha256

ROOT=Path(__file__).resolve().parents[1]


def test_profiles_and_hashes():
    assert set(PROFILES)=={'legacy','lstm32'}
    assert PROFILES['legacy'].sequence_length==16 and PROFILES['lstm32'].sequence_length==32
    assert PROFILES['lstm32'].parameters==209667
    for profile in PROFILES.values():
        checkpoint,norm=profile.validate_assets()
        assert sha256(checkpoint)==profile.checkpoint_sha256
        assert sha256(norm)==profile.normalizer_sha256
        assert profile.class_mapping=={0:'Normal',1:'Yawning',2:'Eye Close'}
    with pytest.raises(ValueError,match='Unknown model profile'):
        get_profile('unknown')


def test_candidate_config_and_frozen_policy():
    legacy=RuntimeConfig.from_ini();candidate=RuntimeConfig.from_ini(model_profile='lstm32')
    assert legacy.model_profile=='legacy' and legacy.sequence_length==16
    assert candidate.model_profile=='lstm32' and candidate.sequence_length==32
    assert candidate.target_model_fps==legacy.target_model_fps==20
    assert candidate.max_face_gap_seconds==legacy.max_face_gap_seconds==.5
    for key in ('smoothing_mode','prediction_history_size','normal_enter_threshold',
        'yawning_enter_threshold','yawning_exit_threshold','eye_close_enter_threshold',
        'eye_close_exit_threshold','yawning_persistence_seconds','drowsy_persistence_seconds',
        'alert_persistence_seconds'):
        assert getattr(candidate,key)==getattr(legacy,key)


def test_candidate_frozen_architecture_and_state():
    profile=get_profile('lstm32');model=build_model(profile).eval()
    state=torch.load(profile.paths()[0],map_location='cpu',weights_only=True)
    assert model.state_dict().keys()==state.keys()
    assert all(model.state_dict()[key].shape==state[key].shape for key in state)
    model.load_state_dict(state)
    assert sum(p.numel() for p in model.parameters())==209667
    captured=[];hook=model.lstm.register_forward_pre_hook(lambda module,args:captured.append(tuple(args[0].shape)))
    with torch.inference_mode():logits=model(torch.zeros(2,5,32))
    hook.remove()
    assert captured==[(2,32,5)] and logits.shape==(2,3) and torch.isfinite(logits).all()


@pytest.fixture
def engine():
    with DrowsinessInferenceEngine(RuntimeConfig.from_ini(model_profile='lstm32'),initialize_detector=False) as instance:
        yield instance


def test_31_32_33_warmup_and_stride_one(engine):
    assert engine.buffer.capacity==32
    raw=np.array([.1,.2,.3,.4,.5])
    for i in range(31):
        r=engine.process_features(raw)
        assert r.raw_class_id is None and r.observation.warming_up
        assert r.observation.sequence_fill==i+1
    first=engine.process_features(raw)
    second=engine.process_features(raw)
    assert first.raw_class_id in (0,1,2) and second.raw_class_id in (0,1,2)
    assert first.input_tensor_shape==second.input_tensor_shape==(1,5,32)
    assert first.observation.sequence_fill==second.observation.sequence_fill==32
    assert not first.observation.warming_up and not second.observation.warming_up
    assert len(engine.buffer)==32 and engine.buffer.feature_storage_bytes==32*5*8


def test_candidate_gap_reset_needs_32_fresh_observations(engine):
    raw=np.array([.1,.2,.3,.4,.5])
    for i in range(32):engine.process_features(raw,timestamp_seconds=i*.05)
    assert engine.buffer.ready
    reset=engine.process_features(raw,timestamp_seconds=3.0)
    assert reset.observation.gap_reset and len(engine.buffer)==1 and reset.raw_class_id is None
    for i in range(30):
        result=engine.process_features(raw,timestamp_seconds=3.05+i*.05)
        assert result.raw_class_id is None
    first=engine.process_features(raw,timestamp_seconds=4.55)
    assert first.raw_class_id in (0,1,2) and first.observation.sequence_fill==32


def test_short_gap_keeps_existing_buffer(engine):
    raw=np.array([.1,.2,.3,.4,.5])
    for i in range(32):engine.process_features(raw,timestamp_seconds=i*.05)
    class NoFace:
        def detect(self,frame):return None
        def close(self):pass
    engine.detector=NoFace()
    absent=engine.process_frame(np.zeros((8,8,3),np.uint8),timestamp_seconds=1.60)
    assert not absent.observation.face_detected and len(engine.buffer)==32
    result=engine.process_features(raw,timestamp_seconds=1.75)
    assert not result.observation.gap_reset and result.raw_class_id is not None


def test_buffer_generalization_and_legacy_default():
    assert TemporalBuffer().capacity==16
    assert TemporalBuffer(32).capacity==32
    for bad in (8,0,33,16.0):
        with pytest.raises(ValueError):TemporalBuffer(bad)
    old=ObservationResult(True,None,None,16,True)
    assert not old.warming_up and old.sequence_capacity==16
    assert '_sequence_capacity' not in asdict(old)


def test_dashboard_readiness_candidate():
    frame=np.zeros((480,640,3),dtype=np.uint8)
    observation=ObservationResult(True,None,None,31,False)
    object.__setattr__(observation,'_sequence_capacity',32)
    from driver_monitor.results import InferenceResult
    packet=FramePacket(frame,0,0.,'file')
    _,items=dashboard_text(frame,packet,InferenceResult(observation),'clip.avi',model_profile='lstm32')
    labels=[item.text for item in items]
    assert 'Sequence: 31 / 32' in labels and 'Model: LSTM32' in labels


def test_smoothing_and_states_stay_frozen():
    assert PredictionSmoother().capacity==5
    assert {s.value for s in DriverState}=={'WARMING_UP','NO_FACE','ATTENTIVE','FATIGUE','SUSPECTED_DROWSY','DROWSY','ALERT'}


def test_candidate_direct_and_legacy_parity_artifacts():
    candidate=json.loads((ROOT/'results/phase16/candidate_parity.json').read_text())
    legacy=json.loads((ROOT/'results/phase16/legacy_parity_final/result.json').read_text())
    assert candidate['exact_parity'] and candidate['verification_windows']==candidate['class_ids_exact']==64
    assert all(v==0 for v in candidate['maximum_differences'].values())
    assert legacy['exact_parity'] and legacy['windows_exact']==legacy['classes_exact']==699
    assert all(s['policy_session_exact'] and s['pixels_exact'] and s['frozen_mismatch_frames']==0 for s in legacy['sources'])
    assert all(v==0 for s in legacy['sources'] for v in s['maximum_differences'].values())


def test_cli_profile_and_invalid_profile():
    text=subprocess.run([sys.executable,'-B','main.py','--help'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    assert '--model-profile {legacy,lstm32}' in text
    with pytest.raises(ValueError,match='Unknown model profile'):
        RuntimeConfig(model_profile='not-a-profile')


def test_no_candidate_heldout_access():
    spec=json.loads((ROOT/'results/phase15/selection_spec.json').read_text())
    parity=json.loads((ROOT/'results/phase16/candidate_parity.json').read_text())
    assert parity['test_cache_reads']==parity['test_video_reads']==parity['candidate_test_inference_passes']==0
    rows=json.loads((ROOT/'results/phase16/profile_characterization.json').read_text())['rows']
    from evaluation.manifest import read_manifest
    manifest={r['relative_path']:r for r in read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv')}
    assert all(manifest[r['source']]['source_id'] in spec['full_development_source_ids'] for r in rows)


def test_saved_video_and_sqlite_candidate_round_trip():
    video=json.loads((ROOT/'results/phase16/saved_video_validation.json').read_text())
    database=json.loads((ROOT/'results/phase16/sqlite_validation.json').read_text())
    assert video['all_frames_readable'] and video['source_properties']==video['saved_properties']==[141,640,480,20.0]
    assert video['no_render_exceptions'] and video['output_sha256']==sha256(ROOT/'results/phase16/eye_close_60_lstm32.avi')
    assert database['session_finalized'] and database['aggregate_equals_session_metrics']
    assert database['events_equal_analytics_counts'] and database['no_frame_or_prediction_tables']
    assert database['model_name']=='LSTMOnly32' and database['parameter_count']==209667 and database['sequence_length']==32
    assert database['checkpoint_sha256']==PROFILES['lstm32'].checkpoint_sha256
    assert database['normalization_sha256']==PROFILES['lstm32'].normalizer_sha256


def test_characterization_is_descriptive_and_benchmark_matched():
    rows=json.loads((ROOT/'results/phase16/profile_characterization.json').read_text())['rows']
    assert len(rows)==12 and {r['profile'] for r in rows}=={'legacy','lstm32'}
    assert all(r['audio_disabled'] and r['warmup_observations_required']==PROFILES[r['profile']].sequence_length for r in rows)
    bench=json.loads((ROOT/'results/phase16/benchmark_matched.json').read_text())
    assert bench['repetitions_per_source']==2 and len(bench['jobs'])==16
    assert {j['profile'] for j in bench['jobs']}=={'legacy','lstm32'}
    assert bench['warmup_separate'] and bench['candidate_heldout_inference_passes']==0
    assert bench['latency_samples_sha256']==sha256(ROOT/'results/phase16/latency_samples.npz')


def test_candidate_10000_frame_boundedness():
    run=json.loads((ROOT/'results/phase16/long_run.json').read_text())
    assert run['decoded_frames_processed']==10000
    assert run['feature_buffer_capacity']==run['feature_buffer_final']==32
    assert run['smoother_capacity']==run['smoother_final']==5
    assert run['event_tracker_count']==5
    assert run['dashboard_cache_entries']<=run['dashboard_cache_capacity']==4
    assert not run['retained_per_frame_model_history']
    assert run['candidate_test_inference_passes']==0
