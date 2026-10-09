"""Application orchestration, side-effect isolation and frozen raw ownership."""

from dataclasses import asdict
import json

import numpy as np
import pytest

import main
from driver_monitor.alerts import AlertManager
from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.decision import DecisionLayer
from driver_monitor.metrics import SessionMetrics
from driver_monitor.results import InferenceResult, ObservationResult
from driver_monitor.vision.video_source import FramePacket
from test_session_metrics import decision


def test_frozen_699_raw_predictions_unchanged_above_policy(frozen_phase2,record_property):
    manifest, arrays = frozen_phase2
    matched = 0
    for source_id,source in enumerate(manifest["sources"]):
        layer, metrics = DecisionLayer(), SessionMetrics()
        with AlertManager(enabled=False,backend=lambda: pytest.fail("real audio")) as alerts:
            for row in range(source["first_frame_row"],source["last_frame_row_exclusive"]):
                t = float(arrays["timestamps_seconds"][row])
                face = bool(arrays["face_detected"][row])
                window = int(arrays["frame_window_indices"][row])
                obs = ObservationResult(face,arrays["raw_features"][row].copy() if face else None,
                                        arrays["normalized_features"][row].copy() if face else None,
                                        int(arrays["sequence_fill"][row]),window>=0,timestamp_seconds=t,sampled=face)
                raw = InferenceResult(obs)
                if window >= 0:
                    raw = InferenceResult(obs, (1,5,16), arrays["sequences"][window].copy(),
                                          arrays["input_tensors"][window].copy(),arrays["logits"][window].copy(),
                                          arrays["probabilities"][window].copy(),int(arrays["predicted_class_ids"][window]),
                                          str(arrays["predicted_class_names"][window]),float(arrays["probabilities"][window].max()))
                d = layer.update(raw)
                metrics.update(d)
                alerts.update(d)
                assert d.raw_result is raw
                if window >= 0:
                    for key,actual in [("sequences",raw.sequence),("input_tensors",raw.input_tensor),("logits",raw.logits),("probabilities",raw.probabilities)]:
                        np.testing.assert_array_equal(actual,arrays[key][window])
                    assert raw.raw_class_id == arrays["predicted_class_ids"][window]
                    matched += 1
            metrics.finalize()
        assert metrics.decoded_frames == source["decoded_frame_count"]
    assert matched == 699
    record_property("phase7_raw_windows_matched",matched)
    record_property("max_raw_numeric_difference",0.0)


@pytest.mark.parametrize("headless,no_audio", [(True,False),(False,True)])
def test_integration_order_one_update_per_frame_and_disabled_audio(monkeypatch,headless,no_audio):
    order = []
    class Source:
        source_name = "fake"
        file_path = None
        nominal_fps = 20
        index = 0
        def read(self):
            i = self.index
            self.index += 1
            order.append("read")
            return FramePacket(np.zeros((480,640,3),np.uint8),i,i/20,"file",i>=3)
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Engine:
        config = RuntimeConfig()
        def process_frame(self,frame,*,timestamp_seconds):
            assert not frame.any()
            order.append("raw")
            return decision(timestamp_seconds,"ALERT").raw_result
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Layer:
        def update(self,raw,**kwargs):
            order.append("policy")
            return decision(kwargs["timestamp_seconds"],"ALERT")
    class Metrics(SessionMetrics):
        def update(self,*args,**kwargs):
            order.append("metrics")
            return super().update(*args,**kwargs)
    class Alerts(AlertManager):
        def __init__(self,**kwargs):
            super().__init__(**kwargs,backend=lambda: pytest.fail("Disabled audio played"))
        def update(self,*args):
            order.append("audio")
            return super().update(*args)
    monkeypatch.setattr(main,"VideoSource",lambda _:Source())
    monkeypatch.setattr(main,"_create_engine",lambda:Engine())
    monkeypatch.setattr(main,"_create_decision_layer",lambda *args,**kwargs:Layer())
    monkeypatch.setattr(main,"SessionMetrics",Metrics)
    monkeypatch.setattr(main,"AlertManager",Alerts)
    original = main.render_result
    def render(*args,**kwargs):
        order.append("render")
        # Synthetic test raw result has no confidence; supply it for display.
        from dataclasses import replace
        args = (*args[:2],replace(args[2],confidence=.9),*args[3:])
        return original(*args,**kwargs)
    monkeypatch.setattr(main,"render_result",render)
    monkeypatch.setattr(main.cv2,"imshow",lambda *args:order.append("display"))
    monkeypatch.setattr(main.cv2,"waitKey",lambda _: -1)
    monkeypatch.setattr(main.cv2,"destroyAllWindows",lambda:None)
    summary = main.run(0,no_display=headless,no_audio=no_audio)
    expected = ["raw","policy","metrics","audio"]+([] if headless else ["render","display"])
    assert order == ["read"]+(expected+["read"])*3
    aggregate = summary.session_metrics
    assert aggregate["finalized"] and aggregate["events"]["alert"]["count"] == 1
    assert aggregate["audio"]["audible_trigger_count"] == 0 and aggregate["audio"]["disabled_trigger_count"] == 1
    assert aggregate["audio"]["closed"]
    json.dumps(asdict(summary))


def test_cli_no_audio_and_cooldown_forwarding(monkeypatch):
    seen = {}
    def run(**kwargs):
        seen.update(kwargs)
        from dataclasses import make_dataclass
        return make_dataclass("Summary",[("ok",bool)])(True)
    monkeypatch.setattr(main,"run",run)
    assert main.main(["--source","0","--no-display","--no-audio","--alert-cooldown-seconds","2.5"]) == 0
    assert seen["no_audio"] and seen["alert_cooldown_seconds"] == 2.5


def test_cli_bad_cooldown_fails_before_source_or_model(monkeypatch,capsys):
    monkeypatch.setattr(main,"VideoSource",lambda _:pytest.fail("Invalid audio config opened source"))
    assert main.main(["--source","0","--no-audio","--alert-cooldown-seconds","-1"]) == 1
    assert "alert_cooldown_seconds" in capsys.readouterr().err
