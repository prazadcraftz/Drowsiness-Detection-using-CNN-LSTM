"""Benchmark-only timing/telemetry. Production algorithms are never edited.

Raw samples belong to finite benchmark experiments, not SessionMetrics.
Runtime imports are lazy so their startup cost can be measured separately.
"""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import wraps
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = ("Normal/Normal_1.avi", "Yawning/Yawning_1.avi",
                   "Eye Close/Eye Close_60.avi", "Yawning/Yawning_60.avi")
STAGES = (
    "frame_decode", "mediapipe_detect", "feature_extraction", "ear_right", "ear_left",
    "mar", "pose_phi_theta", "normalization", "temporal_policy", "buffer_append",
    "buffer_window", "tensor_creation", "cnn_lstm_forward", "softmax", "result_constructor",
    "sequence_inference", "sequence_other_residual", "inference_pipeline", "smoothing",
    "state_machine", "decision_layer", "session_metrics", "audio_request", "overlay",
    "video_write", "application_processing", "frame_loop", "model_only_forward",
    "model_forward_and_result", "mediapipe_only",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_ms(values):
    """Linear interpolated quantiles; population standard deviation; no filtering."""
    values = list(values)
    if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered)-1)*q
        low = int(position)
        high = min(low+1,len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)
    return {"sample_count":len(values),"mean_ms":statistics.fmean(values),
            "p50_ms":percentile(.5),"p90_ms":percentile(.9),"p95_ms":percentile(.95),
            "p99_ms":percentile(.99),"min_ms":ordered[0],"max_ms":ordered[-1],
            "std_ms":statistics.pstdev(values)}


def budget_50ms(values):
    stats = distribution_ms(values)
    return {"sample_count":len(values),"percent_at_or_below_50ms":100*sum(v<=50 for v in values)/len(values),
            "p95_to_50ms_ratio":stats["p95_ms"]/50,"p99_to_50ms_ratio":stats["p99_ms"]/50}


def write_new_json(path, payload):
    path = Path(path).resolve()
    frozen = ROOT/"tests"/"fixtures"
    if path == frozen or frozen in path.parents:
        raise ValueError("Benchmark output cannot overwrite frozen fixtures")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
        handle.write("\n")


def ordered_jobs(sources, runs=5, modes=("headless","render"), seed=20260926):
    if type(runs) is not int or runs < 1 or not sources or not modes:
        raise ValueError("At least one source, mode and run are required")
    if any(mode not in ("headless","render","encode") for mode in modes):
        raise ValueError("Unknown benchmark mode")
    jobs = [{"kind":"pipeline","source":str(source),"run":run,"mode":mode}
            for run in range(runs) for source in sources for mode in modes]
    random.Random(seed).shuffle(jobs)
    return jobs


class SampleBook:
    def __init__(self,warmup_observations=20,clock=time.perf_counter_ns):
        if type(warmup_observations) is not int or warmup_observations < 0:
            raise ValueError("warmup_frames must be a nonnegative integer")
        self.warmup_observations = warmup_observations
        self.clock = clock
        self.valid_observations = 0
        self.frame_index = 0
        self.frames = []
        self.samples = {name:[] for name in STAGES}

    def add(self,stage,elapsed_ns):
        if stage not in self.samples:
            raise ValueError(f"Unknown stage: {stage}")
        if type(elapsed_ns) is not int or elapsed_ns < 0:
            raise ValueError("Elapsed time must be a nonnegative integer in nanoseconds")
        self.samples[stage].append((self.frame_index,elapsed_ns))

    @contextmanager
    def measure(self,stage):
        start = self.clock()
        try:
            yield
        finally:
            self.add(stage,self.clock()-start)

    def classify(self,raw):
        observation = raw.observation
        steady = self.valid_observations >= self.warmup_observations
        if observation.face_detected and observation.sampled:
            self.valid_observations += 1
        self.frames.append({"frame_index":self.frame_index,"timestamp_seconds":observation.timestamp_seconds,
                            "face_detected":observation.face_detected,"sampled":observation.sampled,
                            "prediction":raw.raw_class_id is not None,"steady":steady})

    def export(self):
        steady = {frame["frame_index"] for frame in self.frames if frame["steady"]}
        exported = {}
        for stage,samples in self.samples.items():
            if not samples:
                continue
            all_ms = [ns/1_000_000 for _,ns in samples]
            selected = [ns/1_000_000 for frame,ns in samples if frame in steady]
            exported[stage] = {"raw_elapsed_ns":[ns for _,ns in samples],
                               "frame_indices":[frame for frame,_ in samples],
                               "all_frames":distribution_ms(all_ms),
                               "steady_state":distribution_ms(selected) if selected else None,
                               "warmup_sample_count":len(samples)-len(selected)}
            if stage in ("inference_pipeline","application_processing","frame_loop") and selected:
                exported[stage]["steady_state_50ms"] = budget_50ms(selected)
        return exported


class ProcessTelemetry:
    """Windows working-set RSS and process CPU time; no psutil dependency."""
    def __init__(self):
        self.start_wall,self.start_cpu = time.perf_counter(),time.process_time()
        self.last_wall,self.last_cpu = self.start_wall,self.start_cpu
        self.samples = []

    @staticmethod
    def memory():
        if os.name != "nt":
            return {"rss_bytes":None,"peak_rss_bytes":None,"limitation":"Working-set telemetry implemented for Windows only"}
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        psapi = ctypes.WinDLL("psapi",use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError(ctypes.get_last_error(),"GetProcessMemoryInfo failed")
        return {"rss_bytes":int(counters.WorkingSetSize),"peak_rss_bytes":int(counters.PeakWorkingSetSize)}

    def sample(self,label,frame_count):
        wall,cpu = time.perf_counter(),time.process_time()
        interval = wall-self.last_wall
        percent = 100*(cpu-self.last_cpu)/interval if interval > 0 else None
        row = {"label":label,"processed_frames":frame_count,"elapsed_seconds":wall-self.start_wall,
               "process_cpu_percent":percent,**self.memory()}
        self.samples.append(row)
        self.last_wall,self.last_cpu = wall,cpu
        return row

    def summary(self):
        wall,cpu = time.perf_counter()-self.start_wall,time.process_time()-self.start_cpu
        values = [r["process_cpu_percent"] for r in self.samples if r["process_cpu_percent"] is not None and r["label"] not in ("startup","after_initialization")]
        return {"average_process_cpu_percent":100*cpu/wall if wall>0 else None,
                "representative_peak_interval_cpu_percent":max(values) if values else None,
                "cpu_percent_convention":"100 * summed process CPU seconds / wall seconds; may exceed 100% across cores",
                "samples":self.samples,"wall_seconds":wall,"process_cpu_seconds":cpu}


def memory_trend(samples):
    rows = [row for row in samples if row.get("rss_bytes") is not None]
    if len(rows)<2:
        return {"sample_count":len(rows),"assessment":"insufficient memory samples"}
    x=[r["processed_frames"] for r in rows]; y=[r["rss_bytes"] for r in rows]
    xmean,ymean=statistics.fmean(x),statistics.fmean(y)
    denominator=sum((v-xmean)**2 for v in x)
    slope=sum((a-xmean)*(b-ymean) for a,b in zip(x,y))/denominator if denominator else 0.0
    changes=[b-a for a,b in zip(y,y[1:])]
    return {"sample_count":len(rows),"first_rss_bytes":y[0],"last_rss_bytes":y[-1],
            "net_change_bytes":y[-1]-y[0],"min_rss_bytes":min(y),"max_rss_bytes":max(y),
            "linear_slope_bytes_per_frame":slope,"strictly_monotonic_growth":all(c>0 for c in changes),
            "positive_interval_fraction":sum(c>0 for c in changes)/len(changes),
            "assessment":"Descriptive RSS trend only; not proof of absence or presence of a memory leak"}


def runtime_imports():
    start=time.perf_counter_ns()
    import numpy as np
    import torch
    import cv2
    import mediapipe as mp
    import main
    from driver_monitor.config import RuntimeConfig
    from driver_monitor.inference.engine import DrowsinessInferenceEngine
    from driver_monitor.inference.decision import DecisionLayer
    from driver_monitor.metrics import SessionMetrics
    from driver_monitor.alerts import AlertManager
    from driver_monitor.ui.overlay import render_result
    from driver_monitor.vision.video_source import VideoSource
    return locals() | {"runtime_import_ns":time.perf_counter_ns()-start}


def thread_settings(runtime):
    torch,cv2=runtime["torch"],runtime["cv2"]
    return {"torch_num_threads":torch.get_num_threads(),"torch_num_interop_threads":torch.get_num_interop_threads(),
            "opencv_num_threads":cv2.getNumThreads(),"logical_cpu_count":os.cpu_count(),
            "mkldnn_enabled":torch.backends.mkldnn.enabled,"cuda_available":torch.cuda.is_available()}


def environment(runtime):
    metadata={"cpu_model":platform.processor(),"os":platform.platform(),"python":platform.python_version(),
              "python_full":sys.version,"python_executable":sys.executable,"device":"cpu",
              "software":{name:importlib.metadata.version(name) for name in ("torch","opencv-contrib-python","mediapipe","numpy")},
              "thread_settings":thread_settings(runtime),"ram_bytes":None,"physical_cpu_count":None,
              "power_conditions":{},"process_affinity_mask":None,
              "background_conditions":"Other processes were not stopped or controlled. Frequency, thermals and background load can affect results."}
    if os.name == "nt":
        import winreg
        from ctypes import wintypes
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            metadata["cpu_model"]=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
        class MemoryStatus(ctypes.Structure):
            _fields_=[("length",wintypes.DWORD),("load",wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in
                ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
        class PowerStatus(ctypes.Structure):
            _fields_=[(name,wintypes.BYTE) for name in ("ac","flag","percent","reserved")]+[(name,wintypes.DWORD) for name in ("life","full_life")]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
        if kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
            metadata["ram_bytes"]=int(memory.total_phys)
        power=PowerStatus()
        if kernel.GetSystemPowerStatus(ctypes.byref(power)):
            metadata["power_conditions"]={"ac_line_status":int(power.ac),"battery_flag":int(power.flag),"battery_percent":int(power.percent)}
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        process_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process_mask),ctypes.byref(system_mask)):
            metadata["process_affinity_mask"]=int(process_mask.value)
        commands={"physical_cpu_count":["powershell","-NoProfile","-Command","(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                  "active_power_scheme":["powercfg","/getactivescheme"]}
        for key,command in commands.items():
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
                value=result.stdout.strip()
                metadata[key]=int(value) if key=="physical_cpu_count" and value.isdigit() else value or "unavailable"
            except (OSError,subprocess.TimeoutExpired):
                metadata[key]="unavailable"
    return metadata


class FrozenVerifier:
    def __init__(self,np,source):
        directory=ROOT/"tests"/"fixtures"
        self.manifest=json.loads((directory/"manifest.json").read_text())
        assert sha256(directory/"legacy_baseline.npz")==self.manifest["fixture_sha256"]
        with np.load(directory/"legacy_baseline.npz",allow_pickle=False) as archive:
            self.arrays={key:archive[key] for key in archive.files}
        self.source=next((r for r in self.manifest["sources"] if (ROOT/r["filename"]).resolve()==Path(source).resolve()),None)
        self.np=np
        self.matched=0
        if self.source is not None:
            assert sha256(source)==self.source["sha256"]

    def check(self,index,raw):
        if self.source is None:
            return
        row=self.source["first_frame_row"]+index
        arrays=self.arrays; obs=raw.observation
        assert obs.timestamp_seconds==arrays["timestamps_seconds"][row]
        assert obs.face_detected==bool(arrays["face_detected"][row])
        assert obs.sequence_fill==arrays["sequence_fill"][row]
        if obs.face_detected:
            self.np.testing.assert_array_equal(obs.raw_features,arrays["raw_features"][row])
            self.np.testing.assert_array_equal(obs.normalized_features,arrays["normalized_features"][row])
        window=int(arrays["frame_window_indices"][row])
        assert (raw.raw_class_id is not None)==(window>=0)
        if window>=0:
            for name,actual in (("sequences",raw.sequence),("input_tensors",raw.input_tensor),("logits",raw.logits),("probabilities",raw.probabilities)):
                self.np.testing.assert_array_equal(actual,arrays[name][window])
            assert raw.raw_class_id==arrays["predicted_class_ids"][window]
            self.matched+=1


@contextmanager
def instrumentation(engine,decision,book):
    """Scoped forwarding wrappers restore methods/functions even on errors."""
    import torch
    import driver_monitor.inference.engine as engine_module
    from driver_monitor.vision import features
    with ExitStack() as stack:
        def hook(target,name,stage):
            original=getattr(target,name)
            @wraps(original)
            def measured(*args,**kwargs):
                label=stage(*args,**kwargs) if callable(stage) else stage
                with book.measure(label):
                    return original(*args,**kwargs)
            stack.enter_context(patch.object(target,name,measured))
        for target,name,stage in (
            (engine.detector,"detect","mediapipe_detect"),(engine.normalizer,"normalize","normalization"),
            (engine.temporal_runtime,"observe","temporal_policy"),(engine.buffer,"append","buffer_append"),
            (engine.buffer,"window","buffer_window"),(engine.model,"forward","cnn_lstm_forward"),
            (engine,"process_frame","inference_pipeline"),(engine,"infer_sequence","sequence_inference"),
            (torch,"tensor","tensor_creation"),(torch.nn.Softmax,"forward","softmax"),
            (engine_module,"InferenceResult","result_constructor"),
            (features,"extract_features","feature_extraction"),
            (features,"calculate_eye_ratio",lambda landmarks,indices:"ear_right" if indices[0]==160 else "ear_left"),
            (features,"calculate_mouth_ratio","mar"),(features,"calculate_head_pose","pose_phi_theta"),
            (decision.smoother,"update","smoothing"),(decision.machine,"update","state_machine"),
            (decision,"update","decision_layer")):
            hook(target,name,stage)
        yield


def _startup_engine(runtime):
    """Time existing constructor calls, without changing their implementation."""
    import driver_monitor.inference.engine as module
    stages={}
    with ExitStack() as stack:
        for name in ("Normalizer","CNN_LSTM","FaceDetector"):
            original=getattr(module,name)
            def measured(*args,_name=name,_original=original,**kwargs):
                started=time.perf_counter_ns()
                result=_original(*args,**kwargs)
                stages[_name+"_construction_ns"]=time.perf_counter_ns()-started
                return result
            stack.enter_context(patch.object(module,name,measured))
        started=time.perf_counter_ns()
        engine=runtime["DrowsinessInferenceEngine"]()
        stages["engine_initialization_ns"]=time.perf_counter_ns()-started
    # Residual includes checkpoint load, eval/cpu setup, config and buffer setup.
    stages["engine_initialization_other_ns"]=stages["engine_initialization_ns"]-sum(
        value for name,value in stages.items() if name.endswith("_construction_ns"))
    return engine,stages


def pipeline_run(job,warmup=20):
    telemetry=ProcessTelemetry()
    telemetry.sample("startup",0)
    runtime=runtime_imports()
    telemetry.sample("after_runtime_imports",0)
    threads=thread_settings(runtime)
    source=Path(job["source"]).resolve()
    verifier=FrozenVerifier(runtime["np"],source)
    book=SampleBook(warmup)
    started=time.perf_counter_ns()
    engine,startup=_startup_engine(runtime)
    startup["runtime_import_ns"]=runtime["runtime_import_ns"]
    decision=runtime["DecisionLayer"](engine.config)
    metrics=runtime["SessionMetrics"]()
    frames=0
    steady_loop_ns=[]
    with engine, runtime["AlertManager"](enabled=False) as alerts, runtime["VideoSource"](source) as video:
        startup["components_initialization_ns"]=time.perf_counter_ns()-started
        telemetry.sample("after_initialization",0)
        writer=None
        latest=None
        loop_start,cpu_start=time.perf_counter(),time.process_time()
        with ExitStack() as resources:
            resources.enter_context(instrumentation(engine,decision,book))
            while True:
                book.frame_index=frames
                frame_start=time.perf_counter_ns()
                decode_start=time.perf_counter_ns()
                packet=video.read()
                decode_ns=time.perf_counter_ns()-decode_start
                if packet.eof:
                    startup["eof_read_ns"]=decode_ns
                    break
                book.add("frame_decode",decode_ns)
                app_start=time.perf_counter_ns()
                raw=engine.process_frame(packet.frame,timestamp_seconds=packet.timestamp_seconds)
                book.classify(raw)
                policy=decision.update(raw)
                with book.measure("session_metrics"):
                    metrics.update(policy)
                with book.measure("audio_request"):
                    alerts.update(policy)
                observation=raw.observation
                if observation.gap_reset or not observation.face_detected:
                    latest=None
                if raw.raw_class_id is not None:
                    latest=(raw.raw_class_name,raw.confidence)
                if job["mode"] in ("render","encode"):
                    with book.measure("overlay"):
                        display=runtime["render_result"](packet.frame,packet,raw,video.source_name,
                            latest_prediction=latest,source_fps=video.nominal_fps,target_fps=engine.config.target_model_fps,
                            policy=policy.driver,metrics=metrics,audio=alerts.snapshot(packet.timestamp_seconds))
                book.add("application_processing",time.perf_counter_ns()-app_start)
                if job["mode"]=="encode":
                    if writer is None:
                        path=ROOT/".cache"/"phase8_encoding"/job["encoding_filename"]
                        writer=runtime["main"]._open_writer(path,video.nominal_fps or 20.0,packet.frame)
                        resources.callback(writer.release)
                    with book.measure("video_write"):
                        writer.write(display)
                book.add("frame_loop",time.perf_counter_ns()-frame_start)
                if book.frames[-1]["steady"]:
                    steady_loop_ns.append(book.samples["frame_loop"][-1][1])
                # Verification/telemetry occur AFTER timing endpoints; whole-loop
                # throughput includes their overhead and is labeled instrumented.
                verifier.check(frames,raw)
                if raw.raw_class_id is not None:
                    parts=("tensor_creation","cnn_lstm_forward","softmax","result_constructor")
                    parent=book.samples["sequence_inference"][-1][1]
                    child=sum(book.samples[name][-1][1] for name in parts)
                    book.add("sequence_other_residual",parent-child)
                frames+=1
                if frames%50==0:
                    telemetry.sample("processing",frames)
            wall=time.perf_counter()-loop_start
            cpu=time.process_time()-cpu_start
        telemetry.sample("end_of_processing",frames)
        metrics.finalize()
        session=metrics.snapshot(alerts.snapshot(metrics.end_timestamp))
    assert thread_settings(runtime)==threads
    if verifier.source is not None:
        assert frames==verifier.source["decoded_frame_count"]
    telemetry.sample("after_cleanup",frames)
    session["audio"]=alerts.snapshot(metrics.end_timestamp)
    return {"job":job,"source_sha256":sha256(source),"thread_settings":threads,"startup":startup,
            "frames":book.frames,"stages":book.export(),"session_metrics":session,
            "raw_parity":{"source_is_frozen":verifier.source is not None,"windows_matched":verifier.matched,
                          "max_numeric_difference":0.0 if verifier.source is not None else None},
            "throughput":{"instrumented_offline_processing_fps":frames/wall,"loop_wall_seconds":wall,
                "processing_average_cpu_percent":100*cpu/wall,
                "steady_measured_frame_work_fps":len(steady_loop_ns)/(sum(steady_loop_ns)/1e9) if steady_loop_ns else None,
                "effective_model_observation_fps":session["effective_model_observation_fps"]},
            "resources":telemetry.summary()}


def model_only_run(job,warmup=20):
    telemetry=ProcessTelemetry();telemetry.sample("startup",0)
    runtime=runtime_imports()
    np,torch=runtime["np"],runtime["torch"]
    threads=thread_settings(runtime)
    frozen=FrozenVerifier(np,ROOT/DEFAULT_SOURCES[0])
    arrays=frozen.arrays
    started=time.perf_counter_ns()
    with runtime["DrowsinessInferenceEngine"](initialize_detector=False) as engine:
        init_ns=time.perf_counter_ns()-started
        telemetry.sample("after_initialization",0)
        # Tensor preparation is outside the model-only timing endpoints.
        prepared=[torch.tensor(t,dtype=torch.float32) for t in arrays["input_tensors"]]
        forward,combined=[],[]
        from driver_monitor.results import InferenceResult
        count=len(prepared)
        for i in range(warmup+count):
            index=i%count
            tensor=prepared[index]
            with torch.inference_mode():
                start=time.perf_counter_ns()
                logits=engine.model(tensor)
                elapsed=time.perf_counter_ns()-start
            if i>=warmup:
                forward.append(elapsed)
            np.testing.assert_array_equal(logits.numpy(),arrays["logits"][index])
        for i in range(warmup+count):
            index=i%count
            tensor=prepared[index]
            start=time.perf_counter_ns()
            with torch.inference_mode():
                logits=engine.model(tensor)
                if logits.shape!=(1,3) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid model logits")
                probabilities=torch.nn.Softmax(dim=1)(logits)
                class_id=torch.argmax(probabilities,1).item()
                confidence=probabilities[0,class_id].item()
            result=InferenceResult(None,tuple(tensor.shape),arrays["sequences"][index].copy(),
                tensor.cpu().numpy().copy(),logits.cpu().numpy().copy(),probabilities.cpu().numpy().copy(),
                class_id,engine.config.class_mapping[class_id],confidence)
            elapsed=time.perf_counter_ns()-start
            if i>=warmup:
                combined.append(elapsed)
            np.testing.assert_array_equal(result.probabilities,arrays["probabilities"][index])
            assert result.raw_class_id==arrays["predicted_class_ids"][index]
        telemetry.sample("end_of_processing",count*2)
    assert thread_settings(runtime)==threads
    return {"job":job,"thread_settings":threads,"startup":{"runtime_import_ns":runtime["runtime_import_ns"],"engine_initialization_ns":init_ns},
            "warmup_calls_per_workload":warmup,"frozen_tensor_shape":[1,5,16],"raw_windows_verified":count,
            "stages":{name:{"raw_elapsed_ns":samples,"steady_state":distribution_ms([v/1e6 for v in samples])}
                      for name,samples in (("model_only_forward",forward),("model_forward_and_result",combined))},
            "resources":telemetry.summary()}


def mediapipe_only_run(job,warmup=20):
    telemetry=ProcessTelemetry();telemetry.sample("startup",0)
    runtime=runtime_imports()
    from driver_monitor.vision.face_detector import FaceDetector
    config=runtime["RuntimeConfig"].from_ini()
    threads=thread_settings(runtime)
    source=Path(job["source"]).resolve()
    verifier=FrozenVerifier(runtime["np"],source)
    samples=[]
    faces=frames=0
    started=time.perf_counter_ns()
    with FaceDetector(config.face_model_path) as detector, runtime["VideoSource"](source) as video:
        init_ns=time.perf_counter_ns()-started
        telemetry.sample("after_initialization",0)
        while True:
            packet=video.read()
            if packet.eof: break
            start=time.perf_counter_ns()
            landmarks=detector.detect(packet.frame)
            elapsed=time.perf_counter_ns()-start
            face=landmarks is not None
            steady=faces>=warmup
            faces+=int(face)
            samples.append({"frame_index":frames,"elapsed_ns":elapsed,"face_detected":face,"steady":steady})
            if verifier.source is not None:
                row=verifier.source["first_frame_row"]+frames
                assert face==bool(verifier.arrays["face_detected"][row])
            frames+=1
        telemetry.sample("end_of_processing",frames)
    assert thread_settings(runtime)==threads
    selected=[r for r in samples if r["steady"]]
    distributions={}
    for label,predicate in (("combined",lambda r:True),("face_present",lambda r:r["face_detected"]),("no_face",lambda r:not r["face_detected"])):
        values=[r["elapsed_ns"]/1e6 for r in selected if predicate(r)]
        distributions[label]=distribution_ms(values) if values else None
    return {"job":job,"source_sha256":sha256(source),"thread_settings":threads,
            "startup":{"runtime_import_ns":runtime["runtime_import_ns"],"detector_and_source_initialization_ns":init_ns},
            "samples":samples,"steady_state":distributions,"resources":telemetry.summary()}


def control_run(job):
    """Time the actual Phase 7 main.run with all stage wrappers removed."""
    runtime=runtime_imports()
    threads=thread_settings(runtime)
    main=runtime["main"]
    with ExitStack() as stack:
        if job["mode"]=="render":
            # Keep production rendering branch; omit GUI refresh/wait timings.
            for name in ("imshow","waitKey","destroyAllWindows"):
                stack.enter_context(patch.object(main.cv2,name,lambda *args: -1))
        summary=main.run(str(Path(job["source"]).resolve()),no_display=job["mode"]=="headless",no_audio=True)
    assert thread_settings(runtime)==threads
    return {"job":job,"application":asdict(summary),"thread_settings":threads,
            "offline_processing_fps":summary.processing_throughput_fps,
            "effective_model_observation_fps":summary.effective_model_observation_fps,
            "runtime_import_ns":runtime["runtime_import_ns"],"stage_timing_hooks":False}


def long_memory_run(job,frames=10000):
    if type(frames) is not int or frames<1:
        raise ValueError("long_frames must be a positive integer")
    telemetry=ProcessTelemetry();telemetry.sample("startup",0)
    runtime=runtime_imports()
    threads=thread_settings(runtime)
    engine,startup=_startup_engine(runtime)
    decision=runtime["DecisionLayer"](engine.config)
    metrics=runtime["SessionMetrics"]()
    count=passes=0
    with engine, runtime["AlertManager"](enabled=False) as alerts:
        telemetry.sample("after_initialization",0)
        loop_start=time.perf_counter()
        while count<frames:
            with runtime["VideoSource"](job["source"]) as video:
                while count<frames:
                    packet=video.read()
                    if packet.eof:break
                    # Continuous 20-Hz observation clock across real video passes.
                    timestamp=count/(video.nominal_fps or 20)
                    raw=engine.process_frame(packet.frame,timestamp_seconds=timestamp)
                    policy=decision.update(raw)
                    metrics.update(policy)
                    alerts.update(policy)
                    if job["mode"]=="render":
                        runtime["render_result"](packet.frame,packet,raw,video.source_name,policy=policy.driver,
                            source_fps=video.nominal_fps,target_fps=engine.config.target_model_fps,
                            metrics=metrics,audio=alerts.snapshot(timestamp))
                    count+=1
                    if count%100==0:
                        telemetry.sample("processing",count)
            passes+=1
        elapsed=time.perf_counter()-loop_start
        telemetry.sample("end_of_processing",count)
        metrics.finalize()
        session=metrics.snapshot(alerts.snapshot(metrics.end_timestamp))
        payload={"feature_buffer_entries":len(engine.buffer),"feature_buffer_bytes":engine.buffer.feature_storage_bytes,
                 "smoother_entries":len(decision.smoother),"smoother_bytes":decision.smoother.storage_bytes,
                 "event_tracker_count":len(metrics.events)}
    assert thread_settings(runtime)==threads
    telemetry.sample("after_cleanup",count)
    session["audio"]=alerts.snapshot(metrics.end_timestamp)
    rows=[r for r in telemetry.samples if r["label"]=="processing"]
    return {"job":job,"processed_frames":count,"video_passes":passes,"loop_wall_seconds":elapsed,
            "offline_processing_fps":count/elapsed,"thread_settings":threads,"startup":startup,
            "timing_histories_disabled":True,"session_metrics":session,"bounded_payload":payload,
            "resources":telemetry.summary(),"rss_trend_after_2000_frames":memory_trend([r for r in rows if r["processed_frames"]>=2000]),
            "rss_trend_second_half":memory_trend([r for r in rows if r["processed_frames"]>=frames/2])}


def aggregate_runs(runs):
    grouped=defaultdict(list)
    for run in runs:
        job=run["job"]
        if job["kind"] not in ("pipeline","model_only"):
            continue
        for stage,stats in run["stages"].items():
            if job["kind"]=="pipeline":
                steady={r["frame_index"] for r in run["frames"] if r["steady"]}
                values=[ns/1e6 for frame,ns in zip(stats["frame_indices"],stats["raw_elapsed_ns"]) if frame in steady]
            else:
                values=[ns/1e6 for ns in stats["raw_elapsed_ns"]]
            if values:
                key=(job["kind"],job.get("mode","isolated"),job.get("source","frozen_tensors"),stage)
                grouped[key].append(values)
                if job["kind"]=="pipeline":
                    grouped[(job["kind"],job["mode"],"ALL_SOURCES",stage)].append(values)
    result=[]
    for (kind,mode,source,stage),samples in grouped.items():
        all_values=[v for values in samples for v in values]
        medians=[statistics.median(values) for values in samples]
        result.append({"kind":kind,"mode":mode,"source":source,"stage":stage,"distribution":distribution_ms(all_values),
                       "per_run_median_ms":medians,"between_run_median_std_ms":statistics.pstdev(medians),
                       "run_count":len(samples),"budget_50ms":budget_50ms(all_values) if stage in ("inference_pipeline","application_processing","frame_loop") else None})
    return result


def worker(job,warmup=20,long_frames=10000):
    kind=job["kind"]
    if kind=="pipeline":return pipeline_run(job,warmup)
    if kind=="model_only":return model_only_run(job,warmup)
    if kind=="mediapipe_only":return mediapipe_only_run(job,warmup)
    if kind=="control":return control_run(job)
    if kind=="long_memory":return long_memory_run(job,long_frames)
    raise ValueError(f"Unknown job kind {kind}")
