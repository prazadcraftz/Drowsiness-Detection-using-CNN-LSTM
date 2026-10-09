"""Exact visual behavior, bounded ownership and the selected detector contract."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2
import pytest

from driver_monitor.config import PROJECT_ROOT
from driver_monitor.ui.overlay import _dashboard_background, render_result
from driver_monitor.vision.video_source import FramePacket
from test_session_metrics import decision


@pytest.fixture(scope='module')
def reference_renderer():
    path=PROJECT_ROOT/'results/phase9_reference/overlay.py'
    spec=importlib.util.spec_from_file_location('phase9_visual_reference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.render_result


@pytest.mark.parametrize('state',['WARMING_UP','NO_FACE','ATTENTIVE','FATIGUE','SUSPECTED_DROWSY','DROWSY','ALERT'])
@pytest.mark.parametrize('shape',[(480,640),(720,1280),(160,320),(24,32),(3,5),(1,7)])
def test_cached_renderer_matches_all_pixels_and_owns_output(reference_renderer,state,shape):
    frame=np.random.default_rng(20260926).integers(0,256,(*shape,3),dtype=np.uint8)
    before=frame.copy(); policy=decision(0,state)
    raw=replace(policy.raw_result,confidence=.91)
    packet=FramePacket(frame,314,0,'file')
    options=dict(policy=policy.driver,source_fps=60,target_fps=20)
    expected=reference_renderer(frame,packet,raw,'long-name-'*15+'.avi',**options)
    actual=render_result(frame,packet,raw,'long-name-'*15+'.avi',**options)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(frame,before)
    assert not np.shares_memory(actual,frame)
    actual[:]=0
    np.testing.assert_array_equal(render_result(frame,packet,raw,'long-name-'*15+'.avi',**options),expected)


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32])
def test_resolution_dtype_eviction_and_next_frame_refresh(reference_renderer,dtype):
    _dashboard_background.cache_clear()
    policy=decision(0,'ATTENTIVE')
    policy=replace(policy,raw_result=replace(policy.raw_result,confidence=.91))
    for i in range(10):
        frame=np.full((40+i,64+i,3),17+i,dtype=dtype)
        packet=FramePacket(frame,i,i/20,'file')
        if dtype==np.uint8:
            expected=reference_renderer(frame,packet,policy.raw_result,'source.avi',policy=policy.driver)
            actual=render_result(frame,packet,policy.raw_result,'source.avi',policy=policy.driver)
            np.testing.assert_array_equal(actual,expected)
        else:
            # Installed OpenCV 5 putText accepts only CV_8U. Preserve the legacy
            # rejection rather than inventing support outside VideoSource's API.
            with pytest.raises(cv2.error,match='CV_8U'):
                reference_renderer(frame,packet,policy.raw_result,'source.avi',policy=policy.driver)
            with pytest.raises(cv2.error,match='CV_8U'):
                render_result(frame,packet,policy.raw_result,'source.avi',policy=policy.driver)
        assert _dashboard_background.cache_info().currsize<=4
        assert not _dashboard_background(frame.shape,frame.dtype.str).flags.writeable
    assert _dashboard_background.cache_info().currsize==4


def test_runtime_detector_retains_original_outputs_after_rejected_candidate():
    from mediapipe.tasks.python import vision
    from driver_monitor.vision.face_detector import FaceDetector
    native=SimpleNamespace(detect=lambda image:SimpleNamespace(face_landmarks=[]),close=lambda:None)
    with patch.object(vision.FaceLandmarker,'create_from_options',return_value=native) as create,patch.object(native,'close',wraps=native.close) as close:
        with FaceDetector(PROJECT_ROOT/'face_landmarker.task') as detector:
            assert detector.detect(np.zeros((24,32,3),np.uint8)) is None
            options=create.call_args.args[0]
            assert options.output_face_blendshapes is True
            assert options.output_facial_transformation_matrixes is True
            assert options.running_mode==vision.RunningMode.IMAGE and options.num_faces==1
            assert options.min_face_detection_confidence==options.min_face_presence_confidence==options.min_tracking_confidence==.5
            assert options.base_options.model_asset_path==str(PROJECT_ROOT/'face_landmarker.task')
        detector.close();close.assert_called_once()


@pytest.mark.parametrize('failure',['raw','overlay','pairs','cpu'])
def test_cache_only_gate_rejects_regression_or_inconsistent_target(failure):
    from copy import deepcopy
    from test_optimization import evidence
    from tools.optimize_phase9 import assess_cache_only
    from driver_monitor.metrics import benchmark as bench
    result=evidence();name=result['candidate'];result['methodology']['acceptance_protocol']='cache_only'
    for role,rows in result['aggregate'].items():
        rows.extend(dict(deepcopy(row),stage='overlay') for row in list(rows) if row['stage']=='application_processing')
    # The Phase9 gate correctly rejects the changed Phase16 inference source.
    assert assess_cache_only(result,'render')['decision']=='REJECTED'
    # Exercise its original performance assertions under the archived Phase9
    # source identity; no production hash or numerical parity is overridden.
    original_sha256=bench.sha256
    def phase9_source_sha256(path):
        if Path(path).resolve()==(PROJECT_ROOT/'driver_monitor/inference/engine.py').resolve():
            return original_sha256(PROJECT_ROOT/'results/phase9_reference/engine.py')
        return original_sha256(path)
    with patch.object(bench,'sha256',side_effect=phase9_source_sha256):
        assert assess_cache_only(result,'render')['decision']=='ELIGIBLE'
        if failure=='cpu':
            for run in result['runs']:
                if run['job']['candidate']==name:run['throughput']['processing_average_cpu_percent']=300
        else:
            for row in result['aggregate'][name]:
                if failure=='raw' and row['stage']=='inference_pipeline':row['distribution']['p95_ms']=40
                if failure=='overlay' and row['stage']=='overlay':row['distribution']['p50_ms']=31
                if failure=='pairs' and row['stage']=='application_processing':row['per_run_median_ms']=[20,31,31,31,31]
        assert assess_cache_only(result,'render')['decision']=='INCONCLUSIVE'


def test_application_engine_selects_the_verified_runtime_factory():
    from driver_monitor.inference import engine as module
    fake=SimpleNamespace(close=lambda:None)
    with patch.object(module,'FaceDetector',return_value=fake) as factory:
        with module.DrowsinessInferenceEngine() as engine:
            factory.assert_called_once_with(engine.config.face_model_path)
