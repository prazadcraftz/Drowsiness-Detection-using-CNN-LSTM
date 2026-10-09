"""Layout, diagnostic text and pixel ownership without a physical display."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from driver_monitor.metrics import SessionMetrics
from driver_monitor.ui.overlay import dashboard_text, render_result
from driver_monitor.vision.video_source import FramePacket
from test_session_metrics import decision


@pytest.mark.parametrize("state", ["WARMING_UP","NO_FACE","ATTENTIVE","FATIGUE","SUSPECTED_DROWSY","DROWSY","ALERT"])
def test_each_state_input_ownership_features_counters_and_fitted_text(state):
    frame = np.full((480,640,3),100,np.uint8)
    before = frame.copy()
    d = decision(0,state)
    d = replace(d,raw_result=replace(d.raw_result,confidence=.9,
                observation=replace(d.raw_result.observation,raw_features=np.array([.22,.24,.3,12.5,-4]))))
    metrics = SessionMetrics()
    metrics.update(d)
    packet = FramePacket(frame,0,0,"file")
    options = dict(policy=d.driver,metrics=metrics,source_fps=60,target_fps=20,
                   audio={"audible_trigger_count":0})
    actual_state, items = dashboard_text(frame,packet,d.raw_result,"source.avi",**options)
    output = render_result(frame,packet,d.raw_result,"source.avi",**options)
    assert output.shape == frame.shape and output is not frame
    np.testing.assert_array_equal(frame,before)
    texts = [item.text for item in items]
    assert actual_state == state and f"State: {state}" in texts
    assert any(t.startswith("Smooth: ") for t in texts)
    assert any(t.startswith("Raw: ") for t in texts)
    assert any(t.startswith("Yawns: ") for t in texts)
    assert any("Session: " in t and "Eye-Evidence Fraction" in t for t in texts)
    assert any("Pose phi:" in t for t in texts) == (state != "NO_FACE")
    for item in items:
        (w,h), baseline = cv2.getTextSize(item.text,cv2.FONT_HERSHEY_SIMPLEX,item.scale,1)
        assert item.x+w <= 640 and item.y+baseline < 480 and item.y-h >= 0


@pytest.mark.parametrize("shape", [(480,640),(720,1280),(160,320),(24,32)])
def test_dimensions_and_long_labels_fit(shape):
    frame = np.zeros((*shape,3),np.uint8)
    d = decision(0,"SUSPECTED_DROWSY")
    d = replace(d,raw_result=replace(d.raw_result,confidence=.9))
    packet = FramePacket(frame,1000000000,0,"file")
    name = "long-source-name-"*30 + ".avi"
    _, items = dashboard_text(frame,packet,d.raw_result,name,policy=d.driver)
    out = render_result(frame,packet,d.raw_result,name,policy=d.driver)
    assert out.shape == frame.shape
    for item in items:
        (w,h), baseline = cv2.getTextSize(item.text,cv2.FONT_HERSHEY_SIMPLEX,item.scale,1)
        assert item.x+w <= shape[1]+1 and item.y+baseline <= shape[0]


def test_reset_and_no_face_are_both_visible():
    frame = np.zeros((480,640,3),np.uint8)
    d = decision(1,"NO_FACE",reset=True)
    _, items = dashboard_text(frame,FramePacket(frame,0,1,"file"),d.raw_result,"fake",policy=d.driver)
    texts = [item.text for item in items]
    assert "TEMPORAL RESET" in texts and "NO FACE" in texts


def test_raw_smooth_and_derived_can_disagree_without_relabelling():
    frame = np.zeros((480,640,3),np.uint8)
    d = decision(0,"SUSPECTED_DROWSY")
    d = replace(d,raw_result=replace(d.raw_result,raw_class_name="Normal",confidence=.82))
    _, items = dashboard_text(frame,FramePacket(frame,0,0,"file"),d.raw_result,"fake",policy=d.driver)
    texts = [i.text for i in items]
    assert "Raw: Normal 82%" in texts and "Smooth: Eye Close 90%" in texts
    assert "State: SUSPECTED_DROWSY" in texts
