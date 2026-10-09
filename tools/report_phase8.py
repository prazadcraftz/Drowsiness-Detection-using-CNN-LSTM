"""Generate Phase 8 reports from completed measurements and regression XML."""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))

from driver_monitor.metrics.benchmark import budget_50ms, distribution_ms, sha256, write_new_json


def steady_values(run,stage,prediction_only=False):
    stats=run['stages'].get(stage)
    if stats is None:return []
    frames={r['frame_index'] for r in run['frames'] if r['steady'] and (not prediction_only or r['prediction'])}
    return [ns/1e6 for i,ns in zip(stats['frame_indices'],stats['raw_elapsed_ns']) if i in frames]


def fmt(value):
    return f'{value:.4f}' if abs(value)<1 else f'{value:.3f}'


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+[
        '| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def create_reports(result_path=ROOT/'results/benchmark_baseline.json',xml_path=ROOT/'docs/phase8_tests.xml'):
    report_path=ROOT/'docs/BENCHMARK_RESULTS.md'
    phase_path=ROOT/'docs/PHASE8_BENCHMARK_BASELINE.md'
    validation_path=ROOT/'docs/phase8_validation.json'
    if any(p.exists() for p in (report_path,phase_path,validation_path)):
        raise FileExistsError('Choose new report files; never silently overwrite evidence')
    data=json.loads(Path(result_path).read_text(encoding='utf-8'))
    assert data['optimization_performed'] is False and data['unique_frozen_raw_windows_matched']==699
    runs=data['runs']
    baseline=[r for r in runs if r['experiment_role']=='baseline']
    pipeline=[r for r in baseline if r['job']['kind']=='pipeline']
    controls=[r for r in baseline if r['job']['kind']=='control']
    isolated_models=[r for r in baseline if r['job']['kind']=='model_only']
    isolated_detectors=[r for r in baseline if r['job']['kind']=='mediapipe_only']
    long=next(r for r in runs if r['job']['kind']=='long_memory')
    assert long['processed_frames']>=10000
    assert len(pipeline)==40 and len(controls)==40 and len(isolated_models)==5 and len(isolated_detectors)==20
    expected_threads=data['environment']['thread_settings']
    assert all(r['thread_settings']==expected_threads for r in runs)
    suite=ET.parse(xml_path).getroot().find('testsuite')
    assert suite is not None and all(int(suite.get(key,'0'))==0 for key in ('failures','errors','skipped'))
    counts=Counter(c.get('file','').replace('\\','/') for c in suite.findall('testcase'))
    groups={'Phase 2':['test_inference_parity'],'Phase 3':['test_engine','test_face_detector','test_temporal_buffer'],
            'Phase 4':['test_main','test_video_source'],'Phase 5':['test_sampling','test_face_gap_reset','test_temporal_application'],
            'Phase 6':['test_smoothing','test_state_machine','test_decision_layer'],
            'Phase 7':['test_session_metrics','test_alert_manager','test_overlay','test_local_monitor'],
            'Phase 8':['test_benchmark']}
    phase_counts={phase:sum(counts[f'tests/{name}.py'] for name in names) for phase,names in groups.items()}
    assert list(phase_counts.values())[:6]==[9,43,53,60,64,60]
    assert sum(phase_counts.values())==int(suite.get('tests'))
    before=json.loads((ROOT/'docs/phase8_before_hashes.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in before.items())
    assert all(sha256(ROOT/name)==digest for name,digest in data['source_sha256'].items())
    phase7=json.loads((ROOT/'docs/phase7_validation.json').read_text())
    for group in ('preserved_sha256','phase7_source_sha256'):
        assert all(sha256(ROOT/name)==digest for name,digest in phase7[group].items())
    phase_sources=['benchmark.py','driver_monitor/metrics/benchmark.py','tests/test_benchmark.py','tools/report_phase8.py']
    phase_hashes={name:sha256(ROOT/name) for name in phase_sources}
    for name,digest in data['source_code_sha256'].items():
        assert sha256(ROOT/name)==digest,name
    pooled={(r['mode'],r['stage']):r for r in data['aggregate'] if r['kind']=='pipeline' and r['source']=='ALL_SOURCES'}
    resources=[]
    for mode in ('headless','render'):
        selected=[r for r in pipeline if r['job']['mode']==mode]
        weighted_cpu=sum(r['throughput']['processing_average_cpu_percent']*r['throughput']['loop_wall_seconds'] for r in selected)/sum(r['throughput']['loop_wall_seconds'] for r in selected)
        peak=max(s['process_cpu_percent'] for r in selected for s in r['resources']['samples'] if s['label']=='processing')
        resources.append((mode,weighted_cpu,peak))
    startup={}
    startup_names=('benchmark_module_import_ns','runtime_import_ns','Normalizer_construction_ns','CNN_LSTM_construction_ns',
                   'FaceDetector_construction_ns','engine_initialization_other_ns','engine_initialization_ns','components_initialization_ns')
    for key in startup_names:
        values=[r['startup'][key]/1e6 for r in pipeline if key in r['startup']]
        startup[key]=distribution_ms(values)
    for stage in ('mediapipe_detect','cnn_lstm_forward','overlay'):
        values=[r['stages'][stage]['raw_elapsed_ns'][0]/1e6 for r in pipeline if stage in r['stages']]
        startup['first_call_'+stage]=distribution_ms(values)
    rss={}
    for label in ('startup','after_runtime_imports','after_initialization','processing','end_of_processing','after_cleanup'):
        values=[s['rss_bytes']/2**20 for r in pipeline for s in r['resources']['samples'] if s['label']==label and s['rss_bytes'] is not None]
        rss[label]={'sample_count':len(values),'median_mib':statistics.median(values),'min_mib':min(values),'max_mib':max(values)}
    peak_rss=max(s['peak_rss_bytes'] for r in pipeline for s in r['resources']['samples'])/2**20
    budgets=[]
    for mode in ('headless','render'):
        for stage in ('inference_pipeline','application_processing'):
            for predicted in (False,True):
                values=[v for r in pipeline if r['job']['mode']==mode for v in steady_values(r,stage,predicted)]
                budgets.append({'mode':mode,'stage':stage,'population':'prediction-bearing steady frames' if predicted else 'all steady decoded frames',**budget_50ms(values)})
    # Rank disjoint top-level application components by total observed time.
    contribution={}
    core_contribution=[]
    leaf_stages=('mediapipe_detect','feature_extraction','normalization','temporal_policy','buffer_append','buffer_window',
                 'tensor_creation','cnn_lstm_forward','softmax','result_constructor','sequence_other_residual')
    for mode in ('headless','render'):
        selected=[r for r in pipeline if r['job']['mode']==mode]
        total=sum(sum(steady_values(r,'application_processing')) for r in selected)
        names=['inference_pipeline','decision_layer','session_metrics','audio_request']+(['overlay'] if mode=='render' else [])
        durations={name:sum(sum(steady_values(r,name)) for r in selected) for name in names}
        durations['application_other']=total-sum(durations.values())
        contribution[mode]=sorted([{'stage':name,'total_ms':value,'percentage_of_application_processing':100*value/total}
                                   for name,value in durations.items()],key=lambda r:r['total_ms'],reverse=True)
        if mode=='headless':
            raw_total=durations['inference_pipeline']
            leaf={name:sum(sum(steady_values(r,name)) for r in selected) for name in leaf_stages}
            leaf['raw_other']=raw_total-sum(leaf.values())
            assert leaf['raw_other']>=0
            core_contribution=sorted([{'stage':name,'total_ms':value,'percentage_of_raw_processing':100*value/raw_total}
                                      for name,value in leaf.items()],key=lambda r:r['total_ms'],reverse=True)
    validation={'phase':8,'status':'PASS','tests':{'passed':int(suite.get('tests')),'failed':0,'errors':0,'skipped':0,'seconds':float(suite.get('time'))},
                'tests_by_phase':phase_counts,'prior_files_preserved':len(before),'preserved_sha256':before,
                'phase8_source_sha256':phase_hashes,'raw_windows_matched':699,'max_raw_numeric_difference':0.0,
                'benchmark_jobs':len(runs),'raw_baseline_json_sha256':sha256(result_path),'thread_settings':expected_threads,
                'optimization_performed':False,'phase7_baseline_verified':True,'processing_budget':budgets,'application_contribution':contribution,
                'raw_contribution':core_contribution,'startup_statistics':startup,'rss_summary':rss,'peak_rss_mib':peak_rss,
                'long_memory_trend':long['rss_trend_second_half']}
    env=data['environment'];threads=env['thread_settings']
    supplement_path=ROOT/'docs/phase8_hardware_supplement.json'
    supplement=json.loads(supplement_path.read_text()) if supplement_path.exists() else {}
    physical=supplement.get('physical_cpu_count',env['physical_cpu_count'])
    validation['hardware_supplement']=supplement
    summary=['# CPU benchmark baseline — Phase 8','',
             '**BASELINE measurements. No optimization was performed.** Results are specific to this machine and these offline video workloads.','',
             '## Machine and environment','',
             f"CPU: {env['cpu_model']}; {physical} physical / {threads['logical_cpu_count']} logical CPUs; OS-visible RAM {env['ram_bytes']/2**30:.2f} GiB. OS: {env['os']}. Python {env['python']} ({env['python_executable']}).",'',
             'The initial CIM physical-core query was unavailable under current permissions. A read-only post-benchmark GetLogicalProcessorInformationEx topology query recorded the physical count in phase8_hardware_supplement.json; original benchmark metadata remains unchanged.','',
             'Software: '+', '.join(f'{name} {value}' for name,value in env['software'].items())+'.','',
             f"Device CPU; CUDA available: {threads['cuda_available']}; MKLDNN enabled: {threads['mkldnn_enabled']}; Torch intra-op/inter-op {threads['torch_num_threads']}/{threads['torch_num_interop_threads']}; OpenCV threads {threads['opencv_num_threads']}; affinity mask {env['process_affinity_mask']}.",'',
             f"Active Windows power scheme: {env.get('active_power_scheme','unavailable')}. Power status: {env['power_conditions']} (AC line 1 = connected; battery flag 128 = no battery). This reads the active scheme, not the Windows UI power-slider setting. Frequency/temperature were not logged or controlled. {env['background_conditions']}",'',
             '## Methodology and repetitions','',
             f"{len(runs)} sequential jobs, each in a fresh interpreter: 40 instrumented pipeline runs (four sources × five repetitions × headless/render), 40 actual uninstrumented main.run controls with the same repetitions/modes, 20 detector-only runs, five model-only runs, two additional frozen-source parity runs, one encoding probe, and one 10,000-frame memory run. Randomized source/workload ordering uses fixed seed {data['methodology']['seed']}; full execution order is retained. The long memory diagnostic runs last. No slow run was discarded.",'',
             'Headless runs execute raw inference, decisions, metrics and disabled audio requests. Render runs add the unchanged dashboard but do not refresh a GUI. Encoding probe adds MJPG AVI writing. Uninstrumented render controls use the original main.run display branch with only GUI calls replaced by no-ops; rendering/policy/metrics remain intact. No playback occurs in any run.','',
             'Pipeline steady state excludes every stage sample from frames up to and including the first 20 sampled valid observations. No-face/unsampled frames do not advance this warm-up counter. Warm-up records remain in raw output but are excluded from steady statistics. First-call/model/allocator effects are retained separately rather than silently removed. Detector-only uses the first 20 valid detections (equivalent on these 20-FPS sources). Isolated model workloads each discard 20 warm-up calls, then cover all 699 frozen prebuilt (1,5,16) tensors.','',
             'Timers use perf_counter_ns; integer nanoseconds and frame IDs are preserved without rounding. Reported milliseconds use linear-interpolated P50/P90/P95/P99 and population standard deviation. Per-run medians, source-specific variability, pooled distributions, sample counts, raw records, CPU and RSS are stored in JSON. Pooled percentiles are sample-weighted, not averages of run percentiles.','',
             'Forwarding wrappers add observer cost to parent stages. Timers end before parity comparisons/RSS sampling; whole instrumented loop throughput still includes those operations. Actual main.run controls remove stage wrappers, raw comparisons and RSS sampling, and report full local-application throughput independently. These are different measurement populations, not an optimization comparison.','',
             'Stage boundaries: frame_decode includes VideoSource.read decode plus packet/timestamp creation; mediapipe_detect includes existing BGR-as-SRGB Image creation and landmark-result conversion; feature_extraction includes the unchanged five-feature function; phi/theta share one pose call. tensor_creation covers torch.tensor conversion; unsqueeze remains in sequence_other_residual. result_constructor times the dataclass call itself; validation, argmax/confidence, copies and inference-mode bookkeeping are in sequence_other_residual. Nested inclusive stages must not be summed with their children.','',
             '## Steady-state stage latencies','',
             'Values are per invocation, in milliseconds. Headless/render each have five runs per source. Parents/children overlap; differing counts reflect which stages run on missing-face frames.','']
    stage_order=('frame_decode','mediapipe_detect','feature_extraction','ear_right','ear_left','mar','pose_phi_theta','normalization',
                 'temporal_policy','buffer_append','buffer_window','tensor_creation','cnn_lstm_forward','softmax','result_constructor',
                 'sequence_other_residual','sequence_inference','inference_pipeline','smoothing','state_machine','decision_layer',
                 'session_metrics','audio_request','overlay','application_processing','frame_loop')
    rows=[]
    for stage in stage_order:
        stats=pooled.get(('headless',stage)) or pooled.get(('render',stage))
        d=stats['distribution'];rows.append([stage,d['sample_count'],fmt(d['mean_ms']),fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms']),fmt(d['std_ms'])])
    summary+=table(['Stage','Samples','Mean','P50','P95','P99','Std'],rows)
    summary+=['','The overlay row comes from render mode; other rows above come from headless mode. P90/min/max and separate rendering-mode values are in the raw JSON. A state-machine or smoothing row is a child of decision_layer; feature substeps are children of feature_extraction.','',
              '## Isolated model and detector','']
    rows=[]
    for stage in ('model_only_forward','model_forward_and_result'):
        stat=next(r for r in data['aggregate'] if r['kind']=='model_only' and r['stage']==stage)
        d=stat['distribution'];rows.append([stage,d['sample_count'],fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms']),fmt(d['mean_ms']),fmt(d['std_ms'])])
    summary+=table(['Prepared-tensor workload','Samples','P50','P95','P99','Mean','Std'],rows)
    summary+=['','Forward-only starts immediately before model(prebuilt_tensor) and stops at returned logits. Forward+result additionally performs the existing finite check, softmax, argmax/confidence and InferenceResult/array copies over prepared input; no vision or tensor creation is timed. Its small postprocessing harness mirrors the production result expressions and verifies every frozen output; full-pipeline runs still call the unchanged production infer_sequence. Prepared frozen tensors use the layout produced by torch.tensor on archived arrays; production tensor preparation is measured separately. Value parity does not establish identical strides/copy costs across these workloads.','']
    rows=[]
    for label,predicate in [('combined',lambda s:True),('face present',lambda s:s['face_detected']),('no face',lambda s:not s['face_detected'])]:
        values=[s['elapsed_ns']/1e6 for r in isolated_detectors for s in r['samples'] if s['steady'] and predicate(s)]
        d=distribution_ms(values);rows.append([label,len(values),fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms']),fmt(d['mean_ms']),fmt(d['std_ms'])])
    summary+=table(['Detector-only population','Samples','P50','P95','P99','Mean','Std'],rows)
    summary+=['','These detector-only runs decode the same source frames before starting their detect timer and never execute feature calculations/model/policy. No-face population is small and comes only from Yawning_60; tail estimates for those 55 samples are limited.','',
              '## Full processing latency and 50-ms reference','']
    rows=[]
    for mode in ('headless','render'):
        for stage in ('inference_pipeline','application_processing','frame_loop'):
            d=pooled[(mode,stage)]['distribution'];rows.append([mode,stage,d['sample_count'],fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms'])])
    summary+=table(['Mode','Boundary','Samples','P50','P95','P99'],rows)
    summary+=['','inference_pipeline spans already-decoded frame → raw process_frame result. application_processing spans already-decoded frame → raw/policy/metrics/audio request/dashboard ready; encode is excluded. frame_loop adds source read and optional write. GUI refresh, arbitrary sleeps and asynchronous beep duration are never included.','']
    summary+=table(['Mode','Stage','Relevant population','Samples','<=50 ms','P95 / 50','P99 / 50'],[
        [b['mode'],b['stage'],b['population'],b['sample_count'],f"{b['percent_at_or_below_50ms']:.2f}%",fmt(b['p95_to_50ms_ratio']),fmt(b['p99_to_50ms_ratio'])] for b in budgets])
    summary+=['','The 50-ms reference corresponds to the unchanged 20-Hz model cadence. These offline observations do not establish webcam capture-to-display latency or a universal real-time guarantee.','',
              '## Offline throughput and between-run variability','']
    rows=[]
    for source in data['raw_matches_by_source']:
        name=str(Path(source).relative_to(ROOT))
        for mode in ('headless','render'):
            selected=[r for r in controls if r['job']['source']==source and r['job']['mode']==mode]
            if not selected:continue
            fps=[r['offline_processing_fps'] for r in selected]
            rows.append([name,mode,len(fps),fmt(statistics.median(fps)),fmt(min(fps)),fmt(max(fps)),fmt(statistics.pstdev(fps)),fmt(statistics.median(r['effective_model_observation_fps'] for r in selected))])
    summary+=table(['Source','Mode','Runs','Offline FPS P50','Min','Max','Std','Observation FPS'],rows)
    summary+=['','offline_processing_fps is decoded frames / processing wall seconds from actual main.run, including all decoded/warm-up frames and source reads, but excluding initialization. It is neither webcam FPS nor model FPS. Effective observation FPS counts sampled valid-face observations over source timestamp span. The target remains 20 Hz; the three continuously visible sources average 20.0 observations/sec, while Yawning_60 averages 18.382 because of no-face gaps. Individual throughput/control summaries are retained.','',
              'Per-source run median variability for application_processing (ms):','']
    rows=[]
    for stat in data['aggregate']:
        if stat['kind']=='pipeline' and stat['stage']=='application_processing' and stat['source']!='ALL_SOURCES':
            rows.append([str(Path(stat['source']).relative_to(ROOT)),stat['mode'],stat['run_count'],', '.join(fmt(v) for v in stat['per_run_median_ms']),fmt(stat['between_run_median_std_ms'])])
    summary+=table(['Source','Mode','Runs','Per-run P50','Between-run std'],rows)
    summary+=['','## CPU, RSS and startup','']
    summary+=table(['Mode','Processing CPU average','Representative peak interval CPU'],[[mode,f'{average:.2f}%',f'{peak:.2f}%'] for mode,average,peak in resources])
    summary+=['','CPU = summed process CPU seconds / wall seconds × 100. Values above 100% reflect multiple cores; 1200% would represent all 12 logical CPUs fully occupied. Average above is weighted by instrumented processing wall time; interval peaks use periodic process-time deltas, not instantaneous system CPU peaks. Whole-worker averages (including imports/init), per-interval CPU and CPU time are separately retained. CPU figures apply to instrumented jobs; the reference main.run controls have no CPU-sampling observer.','']
    summary+=table(['RSS boundary','Samples','Median MiB','Min MiB','Max MiB'],[[label,v['sample_count'],fmt(v['median_mib']),fmt(v['min_mib']),fmt(v['max_mib'])] for label,v in rss.items()])
    summary += ['',f'Maximum reported pipeline peak working-set RSS: {peak_rss:.3f} MiB. Windows GetProcessMemoryInfo supplies WorkingSetSize (RSS) and PeakWorkingSetSize. Startup RSS is sampled after benchmark module imports but before heavy runtime imports; after-initialization RSS precedes frame processing. Steady periodic RSS is sampled every 50 frames. These whole-process figures include libraries, detector/model/native allocations, verification arrays and timing-record storage; they are not the 640-byte feature payload. No psutil or runtime dependency was added.','']
    summary+=table(['Startup component','Samples','P50 ms','P95 ms'],[[name,v['sample_count'],fmt(v['p50_ms']),fmt(v['p95_ms'])] for name,v in startup.items()])
    summary+=['','Engine initialization includes the component constructors above; its other residual includes checkpoint loading, CPU/eval setup and configuration/buffer setup. components_initialization includes engine, policy, metrics, audio manager and source construction. Those parent rows overlap. Runtime imports and benchmark module imports are measured separately; OS interpreter launch is not isolated, but full process-launch-to-exit wall time is retained for every job. Filesystem/OS caches are not flushed, so a fresh interpreter is not a cold machine/cache guarantee.','',
              '## 10,000-frame memory observation','',
              f"One continuous initialized Phase 7 stack decoded {long['processed_frames']} real frames over {long['video_passes']} repeated Yawning_60 passes, including short missing-face gaps; rendering enabled and real audio disabled. Run duration {long['loop_wall_seconds']:.3f} s; offline throughput {long['offline_processing_fps']:.3f} FPS. Source-clock timestamps are extended continuously across passes at nominal 20 Hz; no interpolation or extra observations are fabricated.",'',
              'Detailed stage histories are disabled in this diagnostic. Only periodic resource samples every 100 frames and bounded session aggregates are retained. The source is reopened at EOF; this is a stress workload, not a continuation of the frozen parity evaluation across video seams. The original detector/model/policy/UI algorithms remain intact. RSS includes measurement/ctypes metadata and native allocator caches.','']
    trend=long['rss_trend_second_half'];trend2=long['rss_trend_after_2000_frames']
    summary+=table(['RSS population','Samples','First MiB','Last MiB','Change MiB','Range MiB','Slope bytes/frame'],[
        [label,t['sample_count'],fmt(t['first_rss_bytes']/2**20),fmt(t['last_rss_bytes']/2**20),fmt(t['net_change_bytes']/2**20),
         f"{t['min_rss_bytes']/2**20:.3f}–{t['max_rss_bytes']/2**20:.3f}",fmt(t['linear_slope_bytes_per_frame'])]
        for label,t in [('after frame 2000',trend2),('second half',trend)]])
    summary += ['',f"Strictly monotonic RSS growth in second half: {trend['strictly_monotonic_growth']}; positive interval fraction {trend['positive_interval_fraction']:.3f}. These are descriptive observations over one long run. RSS declined substantially during this run; Windows working-set residency can change independently of committed allocations. Private commit/allocator histories were not collected, so the cause of the decline is not established. No claim of 'no memory leak' is made, and no change was applied to force a flatter curve.",'',
                f"Final bounded payload: {long['bounded_payload']}. Event/feature histories remain bounded. Whole-process/native RSS is a different measurement.",'',
                '![Long-run working-set RSS](phase8_rss.png)','',
                '## Measured bottleneck ranking','',
                'Disjoint raw components use summed steady invocation time / summed raw inference time. Per-call medians are shown separately; nested EAR/MAR/pose and sequence parents are omitted from the percentage sum. Residual rows account for unmeasured glue/observer overhead.','']
    summary+=table(['Raw stage','P50 ms per call','Core time contribution'],[[r['stage'],fmt(pooled[('headless',r['stage'])]['distribution']['p50_ms']) if r['stage']!='raw_other' else 'residual',f"{r['percentage_of_raw_processing']:.2f}%"] for r in core_contribution])
    summary+=['','Application-level contribution includes raw inference, decision, metrics, audio request, render and remaining glue without double counting:','']
    summary+=table(['Mode','Stage','Application time contribution'],[[mode,r['stage'],f"{r['percentage_of_application_processing']:.2f}%"] for mode,rows in contribution.items() for r in rows])
    summary+=['',f"Largest measured raw component: **{core_contribution[0]['stage']}**, {core_contribution[0]['percentage_of_raw_processing']:.2f}% of summed core time. This conclusion follows measurements rather than the earlier expectation. No optimization recommendation or speedup comparison is made.",'',
              '![Measured stage latencies](phase8_stages.png)','',
              '## Encoding probe and limitations','']
    encoding=next(r for r in runs if r['experiment_role']=='encoding_probe')
    write=encoding['stages']['video_write']['steady_state'];overlay=encoding['stages']['overlay']['steady_state']
    summary+=table(['Single-source encoding probe','Samples','P50 ms','P95 ms','P99 ms'],[[name,d['sample_count'],fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms'])] for name,d in [('overlay',overlay),('VideoWriter.write',write)]])
    summary+=['',
              'The encoding probe uses Normal_1, same 640×480/20-FPS MJPG output and safe writer as Phase 7. The probe is one diagnostic run, not a definitive repeated encoding baseline; it is excluded from pooled headless/render comparisons. write latency measures VideoWriter.write return time, not delayed encoding flush/release or durable-disk completion. The annotated AVI lives in ignored .cache/phase8_encoding/.','',
              'Limitations: short clips yield modest per-run tail sample counts; no-face tail has only 55 isolated samples; 20 warm-up observations may not remove every hardware thermal/cache effect; process CPU peaks are periodic approximations; working-set RSS varies with OS/native allocation behavior; benchmarks are offline and have no GUI-refresh/audio-device latency; observer overhead remains in instrumented parents; no external background-load/frequency/thermal control; no independently calibrated clinical/event accuracy is evaluated. None of these measurements imply medical validity.','',
              '## Regression and evidence','',
              f"Full Phase 2–8 suite: **{suite.get('tests')} passed, 0 failed/errors/skipped** in {float(suite.get('time')):.3f} s, run after benchmark collection. All {len(before)} pre-existing files remain byte-identical. Frozen features, normalized observations, tensors/sequences/logits/probabilities/classes remain exact across all six instrumented sources: **699/699 unique windows, numeric difference 0.0**. Model-only runs verify all 699 outputs on each repetition.",'',
              'Evidence: ../results/benchmark_baseline.json contains environment/configuration/hashes/order, complete per-run samples, distributions and resource telemetry. ../results/benchmark_baseline_runs/ preserves individual job inputs/results/logs. phase8_validation.json records regression counts, preservation hashes, derived budget/contribution/RSS summaries.','',
              '## Exact commands','',
              'Run from the repository root; choose NEW output paths for reruns:','',
              '```powershell',
              r'& .\.venv-baseline\Scripts\python.exe -B benchmark.py --baseline-suite --runs 5 --warmup-frames 20 --include-render --long-frames 10000 --seed 20260926 --output results/benchmark_baseline.json',
              r'& .\.venv-baseline\Scripts\python.exe -B benchmark.py --source "Normal/Normal_1.avi" "Yawning/Yawning_1.avi" --runs 5 --no-render --output results/another_baseline.json',
              r'& .\.venv-baseline\Scripts\python.exe -B benchmark.py --source "Normal/Normal_1.avi" --runs 5 --include-encoding --output results/encoding_baseline.json',
              r'& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase8_tests.xml',
              r'& .\.venv-baseline\Scripts\python.exe -B tools/report_phase8.py',
              '```','',
              '**PHASE 8 PASS.** Stopped after baseline measurement. No Phase 9 optimization was begun.','']
    # Standard plotting only after measurements finish; no extra runtime setting changes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in long['resources']['samples'] if r['label']=='processing']
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot([r['processed_frames'] for r in rows],[r['rss_bytes']/2**20 for r in rows],color='#347da4',linewidth=1.5)
    ax.axvline(2000,color='#999999',linestyle='--',linewidth=.8)
    ax.axvspan(5000,10000,color='#347da4',alpha=.07)
    ax.set(xlabel='Processed real frames',ylabel='Process working-set RSS (MiB)',title='Phase 8 baseline: repeated video, dashboard enabled')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'docs/phase8_rss.png',dpi=160);plt.close(fig)
    names=['mediapipe_detect','feature_extraction','cnn_lstm_forward','decision_layer','session_metrics','overlay']
    p50=[(pooled.get(('headless',n)) or pooled[('render',n)])['distribution']['p50_ms'] for n in names]
    p95=[(pooled.get(('headless',n)) or pooled[('render',n)])['distribution']['p95_ms'] for n in names]
    fig,ax=plt.subplots(figsize=(9,4.5))
    positions=list(range(len(names)))
    ax.barh([v+.17 for v in positions],p50,height=.32,label='P50',color='#347da4')
    ax.barh([v-.17 for v in positions],p95,height=.32,label='P95',color='#eaa34c')
    ax.set_yticks(positions,names);ax.invert_yaxis();ax.set_xscale('log')
    ax.set(xlabel='Latency per call (ms, logarithmic scale)',title='Measured baseline components; overlay from render mode')
    ax.legend();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'docs/phase8_stages.png',dpi=160);plt.close(fig)
    with report_path.open('x',encoding='utf-8') as handle:handle.write('\n'.join(summary))
    final=['# Phase 8 — CPU benchmark baseline','',
           f"**PHASE 8 PASS.** {len(runs)} measured jobs; {suite.get('tests')} tests passed; all 699 frozen raw windows remain exact. All {len(before)} prior files, including the entire Phase 7 runtime, are unchanged.",'',
           '## 1–2. Files created and modified','',
           'Created benchmark.py, driver_monitor/metrics/benchmark.py, tests/test_benchmark.py, tools/report_phase8.py; results/benchmark_baseline.json and individual job evidence; docs/BENCHMARK_RESULTS.md, this report, phase8_validation.json, preservation hashes/test XML and two scientific PNGs. Only ignored .cache smoke jobs and optional encoding AVI artifacts are additional development outputs. **No pre-existing file was modified. No dependency was added.**','',
           '## 3–10. Methodology, machine and run setup','',
           'The companion [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) records hardware/software versions, CPU/RAM/OS, affinity/power/background limitations, unchanged thread settings, representative source hashes, fixed seed/run order, warm-up policy, repetitions and independent workload definitions. Each of four representative sources has five instrumented and five uninstrumented runs in each of headless and render-only modes. Standalone model and detector jobs each have five repetitions. All raw timing samples/run results are retained.','',
           '## 11–23. Stage measurements','',
           'The companion stage table reports decode/packet acquisition, MediaPipe, whole features and EAR-R/EAR-L/MAR/joint phi-theta substeps, normalization, temporal sampling/gap policy, buffer append/window, tensor conversion, CNN-LSTM, softmax, constructor/residual result work, smoothing/state machine/decision, metrics, audio request, dashboard and the separate encoding probe. Every stage distribution has count/mean/P50/P90/P95/P99/min/max/population std and unrounded integer-nanosecond samples in JSON. Audio playback duration is excluded and real sound is disabled. Nested timing boundaries and residuals are defined explicitly.','',
           '## 24–27. Full-pipeline latency and throughput','']
    final+=table(['Mode','Raw P50/P95/P99 ms','Application P50/P95/P99 ms'],[[mode,
        '/'.join(fmt(pooled[(mode,'inference_pipeline')]['distribution'][key]) for key in ('p50_ms','p95_ms','p99_ms')),
        '/'.join(fmt(pooled[(mode,'application_processing')]['distribution'][key]) for key in ('p50_ms','p95_ms','p99_ms'))] for mode in ('headless','render')])
    final+=['','Offline FPS and per-run variability are reported per source/mode from actual unchanged main.run controls; instrumented loop rates are retained separately. Target cadence remains 20 Hz. Effective valid observation FPS is 20.0 on the three continuously visible sources and 18.382 on Yawning_60 with no-face gaps; it is not wall-clock throughput. See the companion throughput table for all five runs/ranges/standard deviations.','',
            '## 28–31. CPU, RSS, long run and startup','',
            f"CPU measurements use standard-library process time; RSS/peak working set use Windows APIs without psutil. The companion report separates startup, after imports/init, steady and end/cleanup RSS. Maximum measured pipeline peak RSS is {peak_rss:.3f} MiB. Process CPU percentages may exceed 100% because CPU seconds sum across cores. Imports, normalizer/model/detector construction and initialization residual are measured separately.",'',
            f"10,000 real frames were processed through inference, policy, analytics, disabled audio and rendering in {long['loop_wall_seconds']:.3f} s. Final feature/smoothing/event payload remains bounded: {long['bounded_payload']}. Second-half RSS change {trend['net_change_bytes']/2**20:.3f} MiB; fitted slope {trend['linear_slope_bytes_per_frame']:.3f} bytes/frame. This is a measured one-run RSS observation, not proof of absence of a leak; measurement metadata/native caches are included. The full curve and post-warm-up trend statistics are retained.",'',
            '## 32–33. Frame budget and bottleneck','',
            'The companion report gives both all-steady-frame and prediction-bearing-frame percentages within 50 ms, plus P95/P99-to-budget ratios for raw and application processing. No universal real-time guarantee is inferred from a mean.','',
            f"Measured largest raw component: **{core_contribution[0]['stage']}**, {core_contribution[0]['percentage_of_raw_processing']:.2f}% of summed core processing time. Full disjoint rankings are derived from observed timing contributions, with residuals and source-dependent invocation counts accounted for. No optimization was performed/recommended or speedup percentage claimed.",'',
            '## 34–35. Raw parity and full regression','']
    final+=table(['Phase','Passed tests'],phase_counts.items())
    final+=['',f"**{suite.get('tests')} passed, 0 failed/errors/skipped**, after benchmark collection. Existing 289 Phase 2–7 tests are byte-identical. Six-source instrumented checks cover **699/699 unique raw windows**, with unchanged features, sequences, tensors, logits, probabilities and classes (maximum numerical difference 0.0). Checkpoint, normalization, MediaPipe settings, BGR/RGB behavior, resolution, sampling, thread counts, policy/audio/UI algorithms and source videos are unchanged.",'',
            '## 36–37. Limitations and reproduction','',
            'Hardware-specific offline results; no thermal/frequency/background-load controls; modest clip/no-face tail sample counts; observer overhead in detailed timings; Windows RSS instrumentation/allocator metadata; one long-memory run; one nondefinitive encoding probe; no GUI-refresh or audible-device latency. These are engineering measurements without medical/event accuracy claims. Exact reproducible commands and every sample/reference are in BENCHMARK_RESULTS.md and results/benchmark_baseline.json. Existing outputs are never overwritten silently; interrupted/error jobs retain individual evidence and stop the experiment.','',
            '## 38. Decision','',
            '**PHASE 8 PASS.** CPU baseline established without production modifications or optimization. Stopped after Phase 8. Phase 9 and later phases, retraining, database persistence and alternative models have not begun.','']
    with phase_path.open('x',encoding='utf-8') as handle:handle.write('\n'.join(final))
    write_new_json(validation_path,validation)
    return validation


if __name__=='__main__':
    print(json.dumps({'status':create_reports()['status']},indent=2))
