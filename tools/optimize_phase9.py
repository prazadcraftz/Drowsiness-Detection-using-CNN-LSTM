"""Fresh-process, single-change Phase 9 experiments; never overwrite evidence."""
from __future__ import annotations

import time
BOOTSTRAP_START_NS = time.perf_counter_ns()

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from driver_monitor.metrics import benchmark as bench

CANDIDATES = {
    "baseline": {},
    "no_blendshapes": {"output_face_blendshapes": False},
    "no_matrices": {"output_facial_transformation_matrixes": False},
    "no_unused_outputs": {"output_face_blendshapes": False, "output_facial_transformation_matrixes": False},
    "video": {"video": True},
    "intra1": {"intra": 1}, "intra2": {"intra": 2}, "intra4": {"intra": 4},
    "interop1": {"interop": 1}, "interop2": {"interop": 2},
    "cv1": {"cv": 1}, "cv2": {"cv": 2}, "cv4": {"cv": 4},
    "pose_cache": {"pose_cache": True},
    "dashboard_cache": {"dashboard_cache": True},
    "contiguous_tensor": {"contiguous_tensor": True},
    "production": {},
}


def canonical(value):
    if is_dataclass(value): return canonical(asdict(value))
    if isinstance(value, Enum): return value.value
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    if hasattr(value, "tolist"): return value.tolist()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def reference_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'results/phase9_reference'/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def reference_engine(module):
    """Use saved source in the same globals so Phase 8 constructor hooks work."""
    original = vars(module).copy()
    path = ROOT/'results/phase9_reference/engine.py'
    try:
        exec(compile(path.read_text(),str(path),'exec'),vars(module))
        yield
    finally:
        vars(module).clear()
        vars(module).update(original)


@contextmanager
def configuration(name):
    """All experimental overrides are scoped; each worker is a fresh process."""
    settings = CANDIDATES[name]
    runtime = bench.runtime_imports()
    if name == "production":
        yield runtime
        return
    torch, cv2, np = runtime["torch"], runtime["cv2"], runtime["np"]
    torch.set_num_threads(settings.get("intra", 8))
    torch.set_num_interop_threads(settings.get("interop", 8))
    cv2.setNumThreads(settings.get("cv", 12))
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    reference_baseline = name == 'baseline' and (ROOT/'results/phase9_selection.json').exists()
    if not reference_baseline:
        import dataset
    import driver_monitor.inference.engine as engine_module
    import driver_monitor.ui.overlay as overlay
    import driver_monitor.vision.features as features
    options = dict(output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    options.update({k: v for k, v in settings.items() if k.startswith("output_")})
    if settings.get("video"):
        options["running_mode"] = vision.RunningMode.VIDEO

    def detector(path):
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)), **options))

    with ExitStack() as stack:
        import driver_monitor.vision.face_detector as detector_module
        if reference_baseline:
            reference_face = reference_module('phase9_reference_face','face_detector.py')
            stack.enter_context(patch.object(detector_module,'FaceDetector',reference_face.FaceDetector))
            stack.enter_context(reference_engine(engine_module))
        else:
            stack.enter_context(patch.object(dataset, "get_detector", detector))
            if hasattr(detector_module, 'create_runtime_detector'):
                stack.enter_context(patch.object(detector_module, 'create_runtime_detector', detector))
        if settings.get('contiguous_tensor'):
            original_tensor = torch.tensor
            def tensor(data, *args, **kwargs):
                value = original_tensor(data, *args, **kwargs)
                if isinstance(data, np.ndarray) and data.shape == (5,16) and data.dtype == np.float64:
                    value = value.contiguous()
                return value
            stack.enter_context(patch.object(torch, 'tensor', tensor))
        # The saved Phase 8 renderer remains the A/B reference after promotion.
        reference = reference_module('phase9_reference_overlay','overlay.py')
        stack.enter_context(patch.object(overlay, "render_result", reference.render_result))
        stack.enter_context(patch.object(runtime["main"], "render_result", reference.render_result))
        if settings.get("video"):
            from driver_monitor.vision.face_detector import FaceDetector
            def detect(self, frame):
                image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
                result = self._detector.detect_for_video(image, self._phase9_timestamp_ms)
                return np.array([(p.x, p.y) for p in result.face_landmarks[0]]) if result.face_landmarks else None
            original = engine_module.DrowsinessInferenceEngine.process_frame
            def process(self, frame, *, timestamp_seconds):
                self.detector._phase9_timestamp_ms = round(timestamp_seconds * 1000)
                return original(self, frame, timestamp_seconds=timestamp_seconds)
            stack.enter_context(patch.object(FaceDetector, "detect", detect))
            stack.enter_context(patch.object(engine_module.DrowsinessInferenceEngine, "process_frame", process))
        if settings.get("pose_cache"):
            import draw_landmarks
            @lru_cache(maxsize=4)
            def constants(height, width):
                camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
                distortion = np.zeros((4, 1))
                return camera, distortion
            code = inspect.getsource(draw_landmarks.calculate_head_pose)
            start = code.index("    # Image size")
            end = code.index("    # Solve the PnP problem")
            code = code[:start] + "    camera_matrix, dist_coeffs = constants(*image.shape[:2])\n\n" + code[end:]
            namespace = vars(draw_landmarks) | {"constants": constants}
            exec(code, namespace)
            stack.enter_context(patch.object(features, "calculate_head_pose", namespace["calculate_head_pose"]))
        if settings.get("dashboard_cache"):
            @lru_cache(maxsize=4)
            def background(shape, dtype):
                image = np.empty(shape, dtype=dtype)
                image[:] = (24, 20, 17)
                h, w = shape[:2]; sx, sy = w / 640, h / 480
                cv2.line(image, (round(14*sx), round(357*sy)), (round(626*sx), round(357*sy)), (70, 61, 55), 1)
                return image
            code = inspect.getsource(overlay.render_result)
            code = code.replace("image = frame.copy()", "image = background(frame.shape, frame.dtype.str).copy()")
            code = code.replace("    image[:] = (24, 20, 17)\n", "")
            code = code.replace("    cv2.line(image, (round(14*sx), round(357*sy)), (round(626*sx), round(357*sy)), (70, 61, 55), 1)\n", "")
            namespace = vars(overlay) | {"background": background}
            exec(code, namespace)
            render = namespace["render_result"]
            stack.enter_context(patch.object(overlay, "render_result", render))
            stack.enter_context(patch.object(runtime["main"], "render_result", render))
        runtime['DrowsinessInferenceEngine'] = engine_module.DrowsinessInferenceEngine
        runtime['render_result'] = overlay.render_result
        yield runtime


def parity(name, directory):
    """Complete six-source landmarks, frozen numeric fields and policy/event trace."""
    with configuration(name) as runtime:
        np = runtime["np"]
        manifest = json.loads((ROOT / "tests/fixtures/manifest.json").read_text())
        results = []
        for source in manifest["sources"]:
            filename = source["filename"]
            verifier = bench.FrozenVerifier(np, ROOT / filename)
            landmarks, policies, pixels = [], [], []
            differences = {key: 0.0 for key in ("raw_features", "normalized_features", "sequences", "input_tensors", "logits", "probabilities")}
            errors, classes, frames = 0, 0, 0
            with runtime["DrowsinessInferenceEngine"]() as engine, runtime["VideoSource"](ROOT / filename) as video:
                decision = runtime["DecisionLayer"](engine.config)
                metrics = runtime["SessionMetrics"]()
                detect = engine.detector.detect
                def capture(frame):
                    value = detect(frame)
                    landmarks.append(np.full((478, 2), np.nan) if value is None else value.copy())
                    return value
                engine.detector.detect = capture
                while True:
                    packet = video.read()
                    if packet.eof: break
                    raw = engine.process_frame(packet.frame, timestamp_seconds=packet.timestamp_seconds)
                    try: verifier.check(frames, raw)
                    except AssertionError: errors += 1
                    row = source["first_frame_row"] + frames
                    win = int(verifier.arrays["frame_window_indices"][row])
                    for key, actual in (("raw_features", raw.observation.raw_features), ("normalized_features", raw.observation.normalized_features),
                                        ("sequences", raw.sequence), ("input_tensors", raw.input_tensor), ("logits", raw.logits), ("probabilities", raw.probabilities)):
                        idx = row if key.endswith("features") else win
                        if actual is not None and idx >= 0:
                            differences[key] = max(differences[key], float(np.max(np.abs(actual - verifier.arrays[key][idx]))))
                    if win >= 0 and raw.raw_class_id == int(verifier.arrays["predicted_class_ids"][win]): classes += 1
                    policy = decision.update(raw); metrics.update(policy)
                    policies.append(digest(policy))
                    if name in ("baseline", "dashboard_cache", "production"):
                        rendered = runtime["main"].render_result(packet.frame, packet, raw, video.source_name,
                            source_fps=video.nominal_fps, target_fps=engine.config.target_model_fps, policy=policy.driver, metrics=metrics)
                        pixels.append(hashlib.sha256(rendered.tobytes()).hexdigest())
                    frames += 1
                metrics.finalize()
                session = digest(metrics.snapshot())
            array = np.array(landmarks)
            path = directory / (filename.replace("/", "_").replace(" ", "_") + ".npz")
            np.savez_compressed(path, landmarks=array)
            reference = ROOT / "results/phase9_parity/baseline" / path.name
            landmark_diff, landmark_equal = 0.0, True
            if name != "baseline":
                with np.load(reference) as archive: expected = archive["landmarks"]
                landmark_equal = np.array_equal(expected, array, equal_nan=True)
                finite = np.isfinite(expected) & np.isfinite(array)
                landmark_diff = float(np.max(np.abs(array[finite]-expected[finite]))) if finite.any() else 0.0
            results.append({"source": filename, "source_sha256": bench.sha256(ROOT/filename), "frames": frames,
                "frozen_mismatch_frames": errors, "windows_exact": verifier.matched, "classes_exact": classes,
                "maximum_differences": differences, "landmarks_exact": landmark_equal, "landmark_max_difference": landmark_diff,
                "landmark_archive": str(path), "policy_trace_sha256": digest(policies), "session_sha256": session,
                "pixel_trace_sha256": digest(pixels) if pixels else None})
        baseline_path = ROOT / "results/phase9_parity/baseline/result.json"
        baseline = results if name == "baseline" else json.loads(baseline_path.read_text())["sources"]
        for value, reference in zip(results, baseline):
            value["policy_session_exact"] = all(value[k] == reference[k] for k in ("policy_trace_sha256", "session_sha256"))
            value["pixels_exact"] = value["pixel_trace_sha256"] == reference["pixel_trace_sha256"] if name in ("baseline", "dashboard_cache", "production") else None
        passed = all(r["frozen_mismatch_frames"] == 0 and r["landmarks_exact"] and r["policy_session_exact"] and r["pixels_exact"] is not False for r in results)
        return {"candidate": name, "settings": CANDIDATES[name], "exact_parity": passed, "windows_exact": sum(r["windows_exact"] for r in results),
                "classes_exact": sum(r["classes_exact"] for r in results), "sources": results, "threads": bench.thread_settings(runtime)}


def timing_worker(job):
    # Initialize overrides at the benchmark's own lazy-import boundary, so RSS
    # startup and import timings retain the Phase 8 scopes in final evidence.
    detector_options=[]
    def record_options(resources):
        from mediapipe.tasks.python import vision
        original_create=vision.FaceLandmarker.create_from_options
        def create(options):
            detector_options.append({'running_mode':options.running_mode.value,'num_faces':options.num_faces,
                'output_face_blendshapes':options.output_face_blendshapes,
                'output_facial_transformation_matrixes':options.output_facial_transformation_matrixes,
                'min_face_detection_confidence':options.min_face_detection_confidence,
                'min_face_presence_confidence':options.min_face_presence_confidence,
                'min_tracking_confidence':options.min_tracking_confidence,
                'model_asset_path':options.base_options.model_asset_path,'delegate':str(options.base_options.delegate)})
            return original_create(options)
        resources.enter_context(patch.object(vision.FaceLandmarker,'create_from_options',create))
    if job['candidate'] == 'production':
        with ExitStack() as resources:
            original_imports=bench.runtime_imports
            initialized=False
            def production_imports():
                nonlocal initialized
                runtime=original_imports()
                if not initialized:
                    record_options(resources)
                    initialized=True
                return runtime
            resources.enter_context(patch.object(bench,'runtime_imports',production_imports))
            result=bench.worker(job,20)
            result['detector_options']=detector_options
            if job['kind']=='long_memory':
                import driver_monitor.ui.overlay as overlay
                info=overlay._dashboard_background.cache_info()
                result['presentation_cache']={'entries':info.currsize,'capacity':info.maxsize,
                    'source_frames_retained':False,'steady_cached_background_bytes':info.currsize*640*480*3,
                    'byte_count_scope':'This fixed 640x480 uint8 long-run source only'}
            return result
    original_imports = bench.runtime_imports
    initialized = {}
    with ExitStack() as resources:
        def lazy_imports():
            if not initialized:
                started = time.perf_counter_ns()
                with patch.object(bench, 'runtime_imports', original_imports):
                    resources.enter_context(configuration(job['candidate']))
                refreshed = original_imports()
                record_options(resources)
                refreshed['runtime_import_ns'] = time.perf_counter_ns()-started
                initialized.update(refreshed)
            return initialized.copy()
        resources.enter_context(patch.object(bench, 'runtime_imports', lazy_imports))
        if job.get("failed_parity"):
            # Timing a rejected experiment must not masquerade as frozen parity.
            bench.FrozenVerifier.check = lambda self, index, raw: None
        result = bench.worker(job, 20)
        result['detector_options']=detector_options
        if job.get("failed_parity"):
            result["raw_parity"] = {"verified": False, "reason": "Rejected candidate; see independent complete parity evidence"}
        return result


def final_suite(label='combined'):
    """Phase 8 method, with fresh contemporaneous paired baseline workers."""
    output = ROOT/"results/benchmark_optimized.json"
    matched_output = ROOT/"results/phase9_final_matched.json"
    if output.exists() or matched_output.exists(): raise FileExistsError("Final evidence exists")
    cache_only=label=='cache_only'
    if not cache_only:
        raise RuntimeError('Combined candidate rejected; current production restores inference. Use --label cache_only; historical framework is archived with failed evidence.')
    assert all(bench.sha256(ROOT/path)==bench.sha256(ROOT/'results/phase9_reference'/filename)
        for path,filename in [('driver_monitor/inference/engine.py','engine.py'),('driver_monitor/vision/face_detector.py','face_detector.py')])
    selection_path=ROOT/('results/phase9_selection_cache_only.json' if cache_only else 'results/phase9_selection.json')
    parity_path=ROOT/('results/phase9_parity/production_cache_only/result.json' if cache_only else 'results/phase9_parity/production/result.json')
    directory = ROOT/("results/phase9_final_cache_only_runs" if cache_only else "results/phase9_final_runs")
    directory.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((ROOT/"results/benchmark_baseline.json").read_text())
    baseline_hashes = {str(Path(k)): v for k, v in baseline['source_sha256'].items()}
    groups = []
    for rep in range(5):
        groups.append({"kind":"model_only", "run":rep})
        for source in bench.DEFAULT_SOURCES:
            assert bench.sha256(ROOT/source) == baseline_hashes[str(Path(source))]
            for mode in ("headless","render"):
                for kind in ("pipeline","control"):
                    groups.append({"kind":kind,"source":str((ROOT/source).resolve()),"mode":mode,"run":rep})
            groups.append({"kind":"mediapipe_only","source":str((ROOT/source).resolve()),"run":rep})
    for source in ("Normal/Normal_60.avi","Eye Close/Eye Close_1.avi"):
        groups.append({"kind":"pipeline","source":str((ROOT/source).resolve()),"mode":"headless","run":0,"experiment_role":"parity_check"})
    groups.append({"kind":"pipeline","source":str((ROOT/bench.DEFAULT_SOURCES[0]).resolve()),"mode":"encode","run":0,"experiment_role":"encoding_probe"})
    rng = random.Random(20260926); rng.shuffle(groups)
    jobs = []
    for group in groups:
        pair = ["baseline", "production"]; rng.shuffle(pair)
        jobs.extend(group | {"candidate":name} for name in pair)
    jobs.append({"kind":"long_memory","source":str((ROOT/'Yawning/Yawning_60.avi').resolve()),"mode":"render","run":0,"frames":10000,"candidate":"production"})
    payloads = []
    for index, job in enumerate(jobs):
        if job.get('mode') == 'encode': job['encoding_filename'] = f'phase9_final_{label}_job_{index:03d}.avi'
        payloads.append(dispatch(job, directory/f'job_{index:03d}.json'))
        print(f"FINAL [{index+1}/{len(jobs)}] {job['candidate']} {job['kind']} {job.get('mode','')} {Path(job.get('source','model')).name}",flush=True)
    selected = [r for r in payloads if r['job']['candidate']=='production']
    reference = [r for r in payloads if r['job']['candidate']=='baseline']
    unique = {}
    for run in selected:
        if run['job']['kind']=='pipeline':
            source=run['job']['source']; unique[source]=max(unique.get(source,0),run['raw_parity']['windows_matched'])
    assert sum(unique.values())==699
    runtime=bench.runtime_imports(); config=runtime['RuntimeConfig'].from_ini()
    configuration_fields={k:str(v) if isinstance(v,Path) else dict(v) if k=='class_mapping' else v for k,v in vars(config).items()}
    def aggregate(runs):
        values=bench.aggregate_runs([r for r in runs if 'experiment_role' not in r['job']])
        for row in values: row['mean_run_median_ms']=statistics.fmean(row['per_run_median_ms'])
        return values
    result={'status':'OPTIMIZED_MEASUREMENTS_COMPLETE','phase':9,'optimization_performed':True,
            'environment':bench.environment(runtime),'configuration':configuration_fields,
            'selection':json.loads(selection_path.read_text()),
            'methodology':baseline['methodology']|{'paired_contemporaneous_baseline':True,'seed':20260926,'fresh_process_per_job':True,'long_frames':10000,'acceptance_protocol':'cache_only','predeclared_protocol_sha256':bench.sha256(ROOT/'docs/phase9_cache_only_protocol.json')},
            'source_sha256':baseline['source_sha256'],'phase8_baseline_sha256':bench.sha256(ROOT/'results/benchmark_baseline.json'),
            'source_code_sha256':{str(p.relative_to(ROOT)):bench.sha256(p) for p in [ROOT/'main.py',ROOT/'dataset.py',ROOT/'tools/optimize_phase9.py']+list((ROOT/'driver_monitor').rglob('*.py'))},
            'runs':selected,'aggregate':aggregate(selected),'unique_frozen_raw_windows_matched':sum(unique.values()),'raw_matches_by_source':unique}
    reference_aggregate=aggregate(reference)
    parity_result=json.loads(parity_path.read_text())
    gates={}
    for mode in ('headless','render'):
        evidence={'candidate':'production','parity':parity_result,'methodology':{'sources':list(bench.DEFAULT_SOURCES),'acceptance_protocol':'cache_only'},
            'aggregate':{'baseline':[r for r in reference_aggregate if r['mode']==mode],
                         'production':[r for r in result['aggregate'] if r['mode']==mode]},
            'runs':[r for r in payloads if r['job']['kind']=='pipeline' and r['job'].get('mode')==mode and not r['job'].get('experiment_role')]}
        gates[mode]=assess_cache_only(evidence,mode) if cache_only else assess(evidence)
    result['final_paired_gates']=gates
    if any(gate['decision']!='ELIGIBLE' for gate in gates.values()):
        failure_path=ROOT/('results/phase9_final_cache_only_failed_gate.json' if cache_only else 'results/phase9_final_failed_gate.json')
        bench.write_new_json(failure_path,{'candidate':result,'matched_baseline_runs':reference,'matched_aggregate':reference_aggregate})
        raise RuntimeError(f'Final performance gate failed; complete evidence retained, no benchmark_optimized.json published: {gates}')
    bench.write_new_json(output,result)
    bench.write_new_json(matched_output,{'methodology':result['methodology'],'runs':reference,'aggregate':reference_aggregate,
        'production_result':str(output),'production_sha256':bench.sha256(output),'run_order':jobs})
    return result


def assess_cache_only(result,mode):
    """New cache-only protocol: target rendering; untouched raw/headless controls.

    Defined before cache-only timing. Do not apply to failed detector candidates.
    Exact restoration of the entire engine/detector source is mandatory. Raw
    and headless P50/P95 must not regress >5%, CPU >10%. In render mode require
    >=1% overlay/app P50 reduction, no P95 regression, >=80% paired app wins.
    """
    restored=all(bench.sha256(ROOT/path)==bench.sha256(ROOT/'results/phase9_reference'/filename)
        for path,filename in [('driver_monitor/inference/engine.py','engine.py'),('driver_monitor/vision/face_detector.py','face_detector.py')])
    if result['methodology'].get('acceptance_protocol')!='cache_only' or not restored or not result['parity']['exact_parity']:
        return {'decision':'REJECTED','reason':'Cache-only requires exact inference source restoration and parity'}
    rows=result['aggregate']; name=result['candidate']
    def row(role,stage):return next(r for r in rows[role] if r['source']=='ALL_SOURCES' and r['stage']==stage)
    improvements={stage:{q:100*(row('baseline',stage)['distribution'][q]-row(name,stage)['distribution'][q])/row('baseline',stage)['distribution'][q]
        for q in ('p50_ms','p95_ms')} for stage in (('inference_pipeline','application_processing','overlay') if mode=='render' else ('inference_pipeline','application_processing'))}
    cpu={role:statistics.fmean(r['throughput']['processing_average_cpu_percent'] for r in result['runs'] if r['job']['candidate']==role) for role in ('baseline',name)}
    passes=all(v>=-5 for v in improvements['inference_pipeline'].values()) and cpu[name]<=1.1*cpu['baseline']
    wins=[]
    if mode=='render':
        passes=passes and all(improvements[s]['p50_ms']>=1 and improvements[s]['p95_ms']>=0 for s in ('overlay','application_processing'))
        for source in result['methodology']['sources']:
            def medians(role):return next(r['per_run_median_ms'] for r in rows[role] if r['stage']=='application_processing' and Path(r['source']).resolve()==(ROOT/source).resolve())
            wins.extend(c<b for b,c in zip(medians('baseline'),medians(name)))
        passes=passes and sum(wins)/len(wins)>=.8
    else:
        passes=passes and all(v>=-5 for v in improvements['application_processing'].values())
    return {'decision':'ELIGIBLE' if passes else 'INCONCLUSIVE','reason':'Cache-only render benefit and unchanged-inference controls passed' if passes else 'Cache-only target/non-regression/CPU gate failed',
        'protocol':'cache_only','improvements_percent':improvements,'paired_application_median_wins':sum(wins),'pairs':len(wins),'mean_cpu_percent':cpu,
        'raw_headless_non_regression_tolerance_percent':5,'inference_sources_exactly_restored':restored,'headless_speedup_claim':False}


def dispatch(job, path):
    input_path = path.with_name(path.stem + "_input.json")
    bench.write_new_json(input_path, job)
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(input_path), "--output", str(path)],
        cwd=ROOT, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    path.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode: raise RuntimeError(f"Worker failed; evidence retained: {path.with_suffix('.stderr.log')}")
    return json.loads(path.read_text())


def assess(result):
    """Conservative pre-promotion gate, using paired per-source run medians."""
    if not result["parity"]["exact_parity"]:
        return {"decision": "REJECTED", "reason": "Exact landmark/numeric/policy/pixel parity failed"}
    name = result["candidate"]
    rows = result["aggregate"]
    def find(candidate, stage):
        return next(r for r in rows[candidate] if r["source"] == "ALL_SOURCES" and r["stage"] == stage)
    stages = ("overlay", "application_processing") if name == "dashboard_cache" else ("inference_pipeline", "application_processing")
    improvements = {}
    for stage in stages:
        base, candidate = find("baseline", stage), find(name, stage)
        improvements[stage] = {q: 100*(base["distribution"][q]-candidate["distribution"][q])/base["distribution"][q] for q in ("p50_ms", "p95_ms")}
    stage = stages[0]
    sources = result["methodology"]["sources"]
    wins = []
    for source in sources:
        def row(candidate):
            return next(r for r in rows[candidate] if Path(r["source"]).resolve() == (ROOT/source).resolve() and r["stage"] == stage)
        base, candidate = row("baseline"), row(name)
        wins.extend(c < b for b, c in zip(base["per_run_median_ms"], candidate["per_run_median_ms"]))
    cpu = {}
    for candidate in ("baseline", name):
        runs = [r for r in result["runs"] if r["job"]["candidate"] == candidate and r["job"]["kind"] == "pipeline"]
        cpu[candidate] = statistics.fmean(r["throughput"]["processing_average_cpu_percent"] for r in runs)
    passes = (all(v["p50_ms"] >= 1 and v["p95_ms"] >= 0 for v in improvements.values())
              and sum(wins)/len(wins) >= .8 and cpu[name] <= cpu["baseline"]*1.1)
    return {"decision": "ELIGIBLE" if passes else "INCONCLUSIVE", "reason": "Requires final combined and regression gates" if passes else "Repeated latency/variability/CPU benefit insufficient",
            "improvements_percent": improvements, "paired_median_wins": sum(wins), "pairs": len(wins), "mean_cpu_percent": cpu}


def experiment(name, runs, sources, mode):
    if type(runs) is not int or runs < 1 or not sources or len(set(sources)) != len(sources):
        raise ValueError("Positive repetitions and unique existing sources required")
    if any(not (ROOT/source).is_file() for source in sources):
        raise FileNotFoundError("Experiment source missing")
    out = ROOT / "results/phase9_runs" / name
    out.mkdir(parents=True, exist_ok=False)
    parity_result = json.loads((ROOT / f"results/phase9_parity/{name}/result.json").read_text())
    jobs = []
    for rep in range(runs):
        for source in sources:
            pair = ["baseline", name]
            random.Random(20260926 + rep * 101 + sources.index(source)).shuffle(pair)
            jobs += [{"kind": "pipeline", "source": str((ROOT/source).resolve()), "mode": mode, "run": rep,
                      "candidate": candidate, "failed_parity": candidate != "baseline" and not parity_result["exact_parity"]} for candidate in pair]
        if name.startswith(("intra", "interop")) and parity_result["exact_parity"]:
            jobs += [{"kind": "model_only", "run": rep, "candidate": candidate} for candidate in ("baseline", name)]
    results = []
    for index, job in enumerate(jobs):
        results.append(dispatch(job, out / f"job_{index:03d}.json"))
        print(f"{name} [{index+1}/{len(jobs)}] {job['candidate']} {Path(job.get('source','model')).name}", flush=True)
    grouped = {candidate: bench.aggregate_runs([r for r in results if r["job"]["candidate"] == candidate]) for candidate in ("baseline", name)}
    for rows in grouped.values():
        for row in rows: row["mean_run_median_ms"] = statistics.fmean(row["per_run_median_ms"])
    return {"candidate": name, "parity": parity_result, "methodology": {"seed": 20260926, "runs": runs, "sources": sources,
            "mode": mode, "warmup_valid_observations": 20, "fresh_process_per_job": True, "paired_order": True}, "aggregate": grouped, "runs": results}


def main():
    bootstrap_import_ns=time.perf_counter_ns()-BOOTSTRAP_START_NS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", choices=CANDIDATES)
    parser.add_argument("--experiment", choices=[n for n in CANDIDATES if n not in ("baseline", "production")])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sources", nargs="+", default=["Normal/Normal_1.avi", "Yawning/Yawning_60.avi"])
    parser.add_argument("--mode", choices=("headless", "render"), default="headless")
    parser.add_argument("--worker")
    parser.add_argument("--final-suite",action="store_true")
    parser.add_argument("--label",choices=('combined','cache_only'),default='combined')
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.final_suite:
        final_suite(args.label)
    elif args.worker:
        job = json.loads(Path(args.worker).read_text())
        framework_sha256=bench.sha256(Path(__file__))
        if job["kind"] == "parity":
            result = parity(job["candidate"], Path(args.output).parent)
        else: result = timing_worker(job)
        result['worker_framework_sha256']=framework_sha256
        result.setdefault('startup',{})['benchmark_module_import_ns']=bootstrap_import_ns
        bench.write_new_json(args.output, result)
    elif args.parity:
        directory = Path(args.output).resolve().parent if args.output else ROOT / "results/phase9_parity" / args.parity
        directory.mkdir(parents=True, exist_ok=False)
        result = dispatch({"kind": "parity", "candidate": args.parity}, directory / "result.json")
        print(json.dumps({k: result[k] for k in ("candidate", "exact_parity", "windows_exact", "classes_exact")}))
    elif args.experiment:
        result = experiment(args.experiment, args.runs, args.sources, args.mode)
        result["assessment"] = assess(result)
        bench.write_new_json(ROOT / f"results/phase9_{args.experiment}.json", result)
        print(json.dumps(result["assessment"]), flush=True)
    else: parser.error("Choose --parity or --experiment")


if __name__ == "__main__": main()
