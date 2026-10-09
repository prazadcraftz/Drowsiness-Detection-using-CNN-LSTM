"""10,000 decoded development frames through one LSTM32 engine and policy."""

import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from driver_monitor.inference.decision import DecisionLayer
from driver_monitor.metrics import SessionMetrics
from driver_monitor.metrics.benchmark import ProcessTelemetry,memory_trend
from driver_monitor.vision.video_source import VideoSource
from driver_monitor.ui import overlay
from evaluation.manifest import write_json


def main():
    source=ROOT/'Normal/Normal_1.avi'
    count=10000;frames=predictions=0
    config=RuntimeConfig.from_ini(model_profile='lstm32')
    telemetry=ProcessTelemetry();telemetry.sample('startup',0)
    started=time.perf_counter()
    with DrowsinessInferenceEngine(config) as engine:
        decision=DecisionLayer(config);metrics=SessionMetrics()
        telemetry.sample('after_initialization',0)
        while frames<count:
            with VideoSource(source) as video:
                while frames<count:
                    packet=video.read()
                    if packet.eof:break
                    result=engine.process_frame(packet.frame,timestamp_seconds=frames/20)
                    policy=decision.update(result,timestamp_seconds=frames/20)
                    metrics.update(policy,timestamp_seconds=frames/20)
                    frames+=1;predictions+=int(result.raw_class_id is not None)
                    if frames%1000==0:
                        assert len(engine.buffer)<=32 and decision.smoother.capacity==5
                        assert len(metrics.events)==5 and overlay._dashboard_background.cache_info().currsize<=4
                        overlay.render_result(packet.frame,packet,result,video.source_name,
                            source_fps=20,target_fps=20,policy=policy.driver,metrics=metrics,
                            model_profile='lstm32')
                        telemetry.sample(f'frame_{frames}',frames)
                        print(frames,'frames',flush=True)
        metrics.finalize()
        buffer_count=len(engine.buffer);buffer_capacity=engine.buffer.capacity
        smoother_count=len(decision.smoother);event_trackers=len(metrics.events)
    total=time.perf_counter()-started
    samples=telemetry.summary()['samples']
    write_json(ROOT/'results/phase16/long_run.json',{'source':'Normal/Normal_1.avi','profile':'lstm32',
        'decoded_frames_processed':frames,'model_predictions':predictions,'wall_seconds':total,
        'feature_buffer_capacity':buffer_capacity,'feature_buffer_final':buffer_count,
        'smoother_capacity':5,'smoother_final':smoother_count,'event_tracker_count':event_trackers,
        'dashboard_cache_capacity':overlay._dashboard_background.cache_info().maxsize,
        'dashboard_cache_entries':overlay._dashboard_background.cache_info().currsize,
        'retained_per_frame_model_history':False,'rss_samples':samples,'rss_trend':memory_trend(samples),
        'scope':'one engineering long-run; RSS descriptive, no proof of absence of memory leak',
        'candidate_test_inference_passes':0})


if __name__=='__main__':main()
