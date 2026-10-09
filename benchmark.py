"""Independent Phase 8 CPU baseline runner. No optimization switches."""

from __future__ import annotations

import time
BOOTSTRAP_START_NS=time.perf_counter_ns()

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from driver_monitor.metrics.benchmark import (
    DEFAULT_SOURCES, ROOT, aggregate_runs, environment, ordered_jobs, runtime_imports,
    sha256, worker, write_new_json,
)

BOOTSTRAP_IMPORT_NS=time.perf_counter_ns()-BOOTSTRAP_START_NS


def run_benchmarks(sources,*,runs=5,warmup_frames=20,seed=20260926,modes=("headless","render"),
                   output="results/benchmark_baseline.json",baseline_suite=False,long_frames=0):
    output=Path(output).resolve()
    frozen=ROOT/"tests"/"fixtures"
    if output==frozen or frozen in output.parents:
        raise ValueError("Benchmark evidence cannot be written into frozen fixtures")
    if output.exists():
        raise FileExistsError(f"Benchmark evidence already exists: {output}")
    if type(warmup_frames) is not int or warmup_frames<0:
        raise ValueError("warmup_frames must be a nonnegative integer")
    if type(long_frames) is not int or long_frames<0:
        raise ValueError("long_frames must be a nonnegative integer")
    sources=[str(Path(source).resolve()) for source in sources]
    if any(not Path(source).is_file() for source in sources):
        raise FileNotFoundError("All benchmark sources must be existing local video files")
    if len(set(sources))!=len(sources):
        raise ValueError("Benchmark sources must be unique")
    jobs=ordered_jobs(sources,runs,modes,seed)
    jobs += [{"kind":"model_only","run":run} for run in range(runs)]
    jobs += [{"kind":"mediapipe_only","source":source,"run":run} for run in range(runs) for source in sources]
    # Repeated actual Phase 7 runs have no stage instrumentation/verification.
    jobs += [{"kind":"control","source":source,"run":run,"mode":mode}
             for run in range(runs) for source in sources for mode in modes if mode!="encode"]
    if baseline_suite:
        for extra in ("Normal/Normal_60.avi","Eye Close/Eye Close_1.avi"):
            resolved=str((ROOT/extra).resolve())
            if resolved not in sources:
                jobs.append({"kind":"pipeline","source":resolved,"run":0,"mode":"parity_check"})
        jobs.append({"kind":"pipeline","source":sources[0],"run":0,"mode":"encode_probe"})
        long_frames=long_frames or 10000
    random.Random(seed).shuffle(jobs)
    if long_frames:
        jobs.append({"kind":"long_memory","source":str((ROOT/"Yawning/Yawning_60.avi").resolve()),"run":0,
                     "mode":"render","frames":long_frames})
    run_dir=output.parent/(output.stem+"_runs")
    run_dir.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
    runtime=runtime_imports()
    metadata=environment(runtime)
    payloads=[]
    launch_records=[]
    for index,job in enumerate(jobs):
        job=dict(job)
        reported_mode=job.get("mode")
        if reported_mode=="parity_check":job["mode"]="headless"
        if reported_mode=="encode_probe":job["mode"]="encode"
        if job.get("mode")=="encode":
            job["encoding_filename"]=output.stem+f"_job_{index:03d}.avi"
        input_path=run_dir/f"job_{index:03d}_input.json"
        result_path=run_dir/f"job_{index:03d}.json"
        write_new_json(input_path,job)
        command=[sys.executable,"-B",str(ROOT/"benchmark.py"),"--worker-job",str(input_path),
                 "--worker-output",str(result_path),"--warmup-frames",str(warmup_frames),"--long-frames",str(long_frames)]
        started=time.perf_counter_ns()
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        process_wall_ns=time.perf_counter_ns()-started
        (run_dir/f"job_{index:03d}_stderr.log").write_text(completed.stderr,encoding="utf-8")
        if completed.returncode!=0:
            raise RuntimeError(f"Benchmark job {index} failed; stopped immediately. See {run_dir/f'job_{index:03d}_stderr.log'}")
        payload=json.loads(result_path.read_text(encoding="utf-8"))
        payload["experiment_role"]="parity_check" if reported_mode=="parity_check" else "encoding_probe" if reported_mode=="encode_probe" else "baseline"
        payloads.append(payload)
        launch_records.append({"job_index":index,"job":job,"role":payload["experiment_role"],
                               "process_launch_to_exit_ns":process_wall_ns,"result_path":str(result_path)})
        print(f"[{index+1}/{len(jobs)}] {job['kind']} {job.get('mode','')} {Path(job.get('source','frozen tensors')).name} run {job['run']+1}",flush=True)
    source_paths={job["source"] for job in jobs if "source" in job}
    unique_matches={}
    for run in payloads:
        if run["job"]["kind"]=="pipeline" and run["raw_parity"]["source_is_frozen"]:
            source=run["job"]["source"]
            unique_matches[source]=max(unique_matches.get(source,0),run["raw_parity"]["windows_matched"])
    if baseline_suite:
        assert sum(unique_matches.values())==699,"Incomplete frozen raw coverage"
    config=runtime["RuntimeConfig"].from_ini()
    config_fields={name:str(value) if isinstance(value,Path) else dict(value) if name=="class_mapping" else value
                   for name,value in vars(config).items()}
    code_files=[ROOT/'main.py',ROOT/'benchmark.py']+[p for p in (ROOT/'driver_monitor').rglob('*.py')]
    result={"status":"BASELINE_MEASUREMENTS_COMPLETE","phase":8,"optimization_performed":False,
            "environment":metadata,"configuration":config_fields,
            "methodology":{"warmup_sampled_valid_observations":warmup_frames,"repetitions":runs,
                "seed":seed,"fresh_process_per_job":True,"modes":list(modes),"audio_enabled":False,
                "display_refresh_measured":False,"percentile_method":"linear interpolation; population std",
                "timing_clock":"perf_counter_ns; integer nanoseconds retained without rounding",
                "inclusive_parent_stages":"Nested stage times overlap. Do not sum parent and child timings.",
                "instrumentation_overhead":"Forwarding wrappers and bookkeeping affect parent timings. Repeated actual main.run controls have no stage hooks, raw verification or telemetry inside the frame loop.",
                "standalone_workloads":"Model uses all 699 prebuilt frozen tensors; warmup calls discarded per workload. MediaPipe uses identically decoded source frames and excludes features.",
                "feature_pose_boundary":"phi and theta come from one shared calculate_head_pose call; measured jointly, never arbitrarily split.",
                "sequence_residual":"sequence_inference minus tensor_creation, forward, softmax and result_constructor; includes validation, unsqueeze, argmax/confidence, NumPy copies, inference-mode context and wrapper/bookkeeping overhead.",
                "throughput":"Instrumented whole-loop includes verification and telemetry. Actual main.run controls report offline_processing_fps separately. Steady measured frame-work FPS excludes untimed bookkeeping; neither is webcam or model FPS."},
            "source_sha256":{str(Path(source).relative_to(ROOT)) if Path(source).is_relative_to(ROOT) else source:sha256(source) for source in source_paths},
            "source_code_sha256":{str(path.relative_to(ROOT)) if path.is_absolute() else str(path):sha256(path) for path in code_files},
            "run_order":launch_records,"runs":payloads,
            "aggregate":aggregate_runs([r for r in payloads if r["experiment_role"]=="baseline"]),
            "unique_frozen_raw_windows_matched":sum(unique_matches.values()),"raw_matches_by_source":unique_matches}
    write_new_json(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",nargs="+",help="One or more local videos; defaults to four representative frozen sources")
    parser.add_argument("--runs",type=int,default=5)
    parser.add_argument("--warmup-frames",type=int,default=20,help="Exclude first N sampled valid observations in each pipeline run")
    parser.add_argument("--seed",type=int,default=20260926)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--no-render",action="store_true")
    group.add_argument("--include-render",action="store_true",help="Headless and render-only modes (default)")
    parser.add_argument("--include-encoding",action="store_true",help="Also repeat annotated AVI encoding as a separate mode")
    parser.add_argument("--long-frames",type=int,default=0)
    parser.add_argument("--baseline-suite",action="store_true",help="Include six-source parity coverage, one encoding probe and 10,000-frame RSS test")
    parser.add_argument("--output",default="results/benchmark_baseline.json")
    parser.add_argument("--worker-job",help=argparse.SUPPRESS)
    parser.add_argument("--worker-output",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
    try:
        if args.worker_job:
            job=json.loads(Path(args.worker_job).read_text(encoding="utf-8"))
            result=worker(job,args.warmup_frames,args.long_frames)
            result.setdefault("startup",{})["benchmark_module_import_ns"]=BOOTSTRAP_IMPORT_NS
            write_new_json(args.worker_output,result)
        else:
            modes=("headless",) if args.no_render else ("headless","render")
            if args.include_encoding:modes+=('encode',)
            run_benchmarks(args.source or DEFAULT_SOURCES,runs=args.runs,warmup_frames=args.warmup_frames,
                seed=args.seed,modes=modes,output=args.output,baseline_suite=args.baseline_suite,long_frames=args.long_frames)
    except (OSError,ValueError,RuntimeError,AssertionError) as error:
        print(f"Benchmark stopped: {error}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
