"""Fresh-process paired CPU benchmark for two fixed integrated model profiles."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifest import read_manifest,sha256,write_json

SOURCES=('Normal/Normal_1.avi','Yawning/Yawning_1.avi','Eye Close/Eye Close_60.avi','Yawning/Yawning_60.avi')
STAGES=('model_forward','raw_inference_pipeline','application_processing','frame_loop')


def distribution(values):
    if not len(values):return None
    return {'sample_count':len(values),'mean_ms':float(np.mean(values)),
        'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),
        'p99_ms':float(np.percentile(values,99)),'minimum_ms':float(np.min(values)),
        'maximum_ms':float(np.max(values))}


def guard_source(source):
    folds=json.loads((ROOT/'results/phase12/cv_folds.json').read_text())
    rows={r['relative_path']:r for r in read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv')}
    if source not in SOURCES or rows[source]['source_id'] not in folds['development_source_ids']:
        raise ValueError('Benchmark may use only predeclared development videos')


def worker(job):
    import cv2
    import torch
    from driver_monitor.config import RuntimeConfig
    from driver_monitor.inference.engine import DrowsinessInferenceEngine
    from driver_monitor.inference.decision import DecisionLayer
    from driver_monitor.metrics import SessionMetrics
    from driver_monitor.metrics.benchmark import ProcessTelemetry
    from driver_monitor.alerts import AlertManager
    from driver_monitor.vision.video_source import VideoSource
    profile,source=job['profile'],job['source']
    guard_source(source)
    torch.set_num_threads(8)
    if torch.get_num_interop_threads()!=8:torch.set_num_interop_threads(8)
    telemetry=ProcessTelemetry();telemetry.sample('startup',0)
    config=RuntimeConfig.from_ini(model_profile=profile)
    samples={name:[] for name in STAGES};frames=predictions=0;first=None
    with DrowsinessInferenceEngine(config) as engine,VideoSource(ROOT/source) as video,AlertManager(enabled=False) as alerts:
        decision=DecisionLayer(config);metrics=SessionMetrics()
        telemetry.sample('after_initialization',0)
        original=engine.model.forward
        def timed_forward(*args,**kwargs):
            start=time.perf_counter_ns();value=original(*args,**kwargs)
            samples['model_forward'].append((time.perf_counter_ns()-start)/1e6)
            return value
        engine.model.forward=timed_forward
        wall_start=time.perf_counter();cpu_start=time.process_time()
        while True:
            loop_start=time.perf_counter_ns();packet=video.read()
            if packet.eof:break
            app_start=time.perf_counter_ns();raw=engine.process_frame(packet.frame,timestamp_seconds=packet.timestamp_seconds)
            infer_end=time.perf_counter_ns();policy=decision.update(raw)
            metrics.update(policy,timestamp_seconds=packet.timestamp_seconds);alerts.update(policy)
            end=time.perf_counter_ns();frames+=1
            predictions+=int(raw.raw_class_id is not None)
            if raw.raw_class_id is not None and first is None:first=packet.timestamp_seconds
            if first is not None:
                samples['raw_inference_pipeline'].append((infer_end-app_start)/1e6)
                samples['application_processing'].append((end-app_start)/1e6)
                samples['frame_loop'].append((end-loop_start)/1e6)
        wall=time.perf_counter()-wall_start;cpu=time.process_time()-cpu_start
        engine.model.forward=original
        metrics.finalize();telemetry.sample('end',frames)
    result={'profile':profile,'source':source,'source_sha256':sha256(ROOT/source),'repetition':job['repetition'],
        'frames':frames,'predictions':predictions,'first_prediction_timestamp_seconds':first,
        'required_accepted_observations':config.sequence_length,'warmup_nominal_seconds':config.sequence_length/20,
        'samples_ms':samples,'distributions':{name:distribution(samples[name]) for name in STAGES},
        'offline_fps':frames/wall,'loop_wall_seconds':wall,'loop_process_cpu_seconds':cpu,
        'weighted_process_cpu_percent':100*cpu/wall,'rss':telemetry.summary()['samples'],
        'thread_settings':{'torch_intra':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads(),
                           'opencv':cv2.getNumThreads()},
        'settings':{'audio':False,'sqlite':False,'render':False,'target_model_fps':config.target_model_fps,
                    'max_face_gap_seconds':config.max_face_gap_seconds,'detector':'unchanged FaceDetector'}}
    write_json(ROOT/job['output'],result)


def orchestrate():
    out=ROOT/'results/phase16';directory=out/'benchmark_jobs';directory.mkdir(exist_ok=False)
    jobs=[{'profile':profile,'source':source,'repetition':rep,
           'output':f'results/phase16/benchmark_jobs/{profile}_{source.replace("/","_").replace(" ","_")}_{rep}.json'}
          for rep in range(2) for source in SOURCES for profile in ('legacy','lstm32')]
    random.Random(1601).shuffle(jobs)
    write_json(out/'benchmark_order.json',{'seed':1601,'jobs':jobs,'fresh_process_per_job':True,
        'same_sources_settings_except_profile':True,'no_heldout_sources':True})
    completed=[]
    for job in jobs:
        path=directory/(Path(job['output']).stem+'.job.json')
        write_json(path,job)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(path)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120,
                              env=dict(os.environ,MPLCONFIGDIR=str(ROOT/'.cache/matplotlib')),
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            (directory/(path.stem+'.error.txt')).write_text(result.stdout+'\n'+result.stderr)
            raise RuntimeError(f'Benchmark job failed: {job}')
        completed.append(json.loads((ROOT/job['output']).read_text()))
        print(job['profile'],job['source'],job['repetition'],'FPS',round(completed[-1]['offline_fps'],2),flush=True)
    aggregate={};raw={}
    for profile in ('legacy','lstm32'):
        selected=[r for r in completed if r['profile']==profile]
        aggregate[profile]={'stages':{},'offline_fps_per_source':{},
            'weighted_process_cpu_percent':100*sum(r['loop_process_cpu_seconds'] for r in selected)/sum(r['loop_wall_seconds'] for r in selected),
            'rss_steady_bytes':int(statistics.median(r['rss'][-1]['rss_bytes'] for r in selected)),
            'rss_peak_bytes':max(r['rss'][-1]['peak_rss_bytes'] for r in selected)}
        for stage in STAGES:
            values=[v for r in selected for v in r['samples_ms'][stage]]
            aggregate[profile]['stages'][stage]=distribution(values)
            raw[profile+'_'+stage+'_ms']=np.asarray(values)
        for source in SOURCES:
            aggregate[profile]['offline_fps_per_source'][source]=statistics.median(r['offline_fps'] for r in selected if r['source']==source)
    np.savez_compressed(out/'latency_samples.npz',**raw)
    write_json(out/'benchmark_matched.json',{'scope':'paired fresh-process integrated CPU engineering benchmark; no candidate accuracy',
        'profiles':['legacy','lstm32'],'sources':list(SOURCES),'repetitions_per_source':2,
        'fixed_order_seed':1601,'jobs':completed,'aggregate':aggregate,
        'latency_samples_sha256':sha256(out/'latency_samples.npz'),
        'stage_definitions':{'model_forward':'integrated model.forward only, steady predictions',
            'raw_inference_pipeline':'engine.process_frame after first prediction',
            'application_processing':'engine, decision, SessionMetrics, disabled AlertManager after first prediction; decode excluded',
            'frame_loop':'decode plus application processing after first prediction'},
        'warmup_separate':True,'test_video_reads':0,'candidate_heldout_inference_passes':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    worker(json.loads(Path(args.worker).read_text())) if args.worker else orchestrate()
