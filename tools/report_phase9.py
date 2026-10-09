"""Create evidence-backed Phase 9 reports after all gates; preserve earlier reports."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from driver_monitor.metrics.benchmark import budget_50ms, distribution_ms, sha256, write_new_json
from tools.report_phase8 import steady_values, table, fmt
from tools.optimize_phase9 import assess_cache_only


def pooled(data):
    return {(r['mode'],r['stage']):r for r in data['aggregate'] if r['kind']=='pipeline' and r['source']=='ALL_SOURCES'}


def pipeline(data):
    return [r for r in data['runs'] if r['job']['kind']=='pipeline' and not r['job'].get('experiment_role') and r.get('experiment_role','baseline')=='baseline']


def resources(data):
    runs=pipeline(data); result={}
    for mode in ('headless','render'):
        selected=[r for r in runs if r['job']['mode']==mode]
        wall=sum(r['throughput']['loop_wall_seconds'] for r in selected)
        cpu=sum(r['throughput']['processing_average_cpu_percent']*r['throughput']['loop_wall_seconds'] for r in selected)/wall
        rows=[s for r in selected for s in r['resources']['samples']]
        steady=[s['rss_bytes']/2**20 for s in rows if s['label']=='processing']
        result[mode]={'processing_cpu_percent':cpu,'peak_interval_cpu_percent':max(s['process_cpu_percent'] for s in rows if s['label']=='processing'),
                      'median_processing_rss_mib':statistics.median(steady),'peak_rss_mib':max(s['peak_rss_bytes'] for s in rows)/2**20}
    return result


def contributions(data):
    runs=[r for r in pipeline(data) if r['job']['mode']=='headless']
    total=sum(sum(steady_values(r,'inference_pipeline')) for r in runs)
    names=('mediapipe_detect','feature_extraction','normalization','temporal_policy','buffer_append','buffer_window','tensor_creation','cnn_lstm_forward','softmax','result_constructor','sequence_other_residual')
    durations={name:sum(sum(steady_values(r,name)) for r in runs) for name in names}
    durations['raw_other']=total-sum(durations.values())
    assert durations['raw_other']>=0
    return sorted([{'stage':name,'total_ms':value,'percent_raw':100*value/total} for name,value in durations.items()],key=lambda r:r['total_ms'],reverse=True)


def report():
    output_paths=[ROOT/'docs/PHASE9_EXPERIMENTS.md',ROOT/'docs/PHASE9_CPU_OPTIMIZATION.md',ROOT/'docs/phase9_validation.json',ROOT/'results/phase9_experiments.json']
    if any(p.exists() for p in output_paths):raise FileExistsError('Final Phase 9 reports already exist')
    baseline=json.loads((ROOT/'results/benchmark_baseline.json').read_text())
    final=json.loads((ROOT/'results/benchmark_optimized.json').read_text())
    matched=json.loads((ROOT/'results/phase9_final_matched.json').read_text())
    selection=json.loads((ROOT/'results/phase9_selection_cache_only.json').read_text())
    parity=json.loads((ROOT/'results/phase9_parity/production_cache_only/result.json').read_text())
    failed_combined=json.loads((ROOT/'results/phase9_final_failed_gate.json').read_text())
    assert final['selection']['accepted_candidates']==selection['accepted_candidates']==['dashboard_cache']
    assert final['methodology']['acceptance_protocol']=='cache_only'
    assert parity['exact_parity'] and parity['windows_exact']==parity['classes_exact']==699
    assert final['unique_frozen_raw_windows_matched']==699
    assert len(final['runs'])==109 and len(matched['runs'])==108
    framework_hash=sha256(ROOT/'tools/optimize_phase9.py')
    protocol=json.loads((ROOT/'docs/phase9_cache_only_protocol.json').read_text())
    assert protocol['framework_sha256']==framework_hash==sha256(ROOT/'results/phase9_reference/optimization_framework_cache_only.py')
    assert final['methodology']['predeclared_protocol_sha256']==sha256(ROOT/'docs/phase9_cache_only_protocol.json')
    previous_framework_hash=sha256(ROOT/'results/phase9_reference/optimization_framework_combined.py')
    assert all(r['worker_framework_sha256']==previous_framework_hash for r in failed_combined['candidate']['runs']+failed_combined['matched_baseline_runs'])
    assert all(r['worker_framework_sha256']==framework_hash for r in final['runs']+matched['runs'])
    assert all(sha256(ROOT/name)==value for name,value in final['source_code_sha256'].items())
    for variant,runs in (('baseline',matched['runs']),('production',final['runs'])):
        for run in runs:
            expected=run['job']['kind']!='model_only'
            assert len(run['detector_options'])==int(expected)
            for options in run['detector_options']:
                assert options['running_mode']=='IMAGE' and options['num_faces']==1
                assert options['delegate']=='None'
                assert options['output_face_blendshapes'] is True and options['output_facial_transformation_matrixes'] is True
                assert options['min_face_detection_confidence']==options['min_face_presence_confidence']==options['min_tracking_confidence']==.5
    def session_key(run):
        job=run['job']
        return (job['kind'],job.get('source'),job.get('mode'),job['run'],job.get('experiment_role'))
    reference_sessions={session_key(r):r['session_metrics'] for r in matched['runs'] if 'session_metrics' in r}
    assert all(r['session_metrics']==reference_sessions[session_key(r)] for r in final['runs'] if r['job']['kind']=='pipeline')
    native=json.loads((ROOT/'results/phase9_native_landmarks.json').read_text())
    assert native['exact']
    before={Path(name).as_posix():value for name,value in json.loads((ROOT/'docs/phase9_before_hashes.json').read_text()).items()}
    changed={name:{'before':old,'after':sha256(ROOT/name)} for name,old in before.items() if sha256(ROOT/name)!=old}
    allowed={'driver_monitor/ui/overlay.py'}
    assert set(changed)<=allowed,changed
    preserved={name:old for name,old in before.items() if name not in changed}
    assert sha256(ROOT/'results/benchmark_baseline.json')==before['results/benchmark_baseline.json']
    suite=ET.parse(ROOT/'docs/phase9_tests_cache_only.xml').getroot().find('testsuite')
    assert suite is not None and all(int(suite.get(k,'0'))==0 for k in ('failures','errors','skipped'))
    counts=Counter(c.get('file','').replace('\\','/') for c in suite.findall('testcase'))
    groups={'Phase 2':['test_inference_parity'],'Phase 3':['test_engine','test_face_detector','test_temporal_buffer'],
            'Phase 4':['test_main','test_video_source'],'Phase 5':['test_sampling','test_face_gap_reset','test_temporal_application'],
            'Phase 6':['test_smoothing','test_state_machine','test_decision_layer'],'Phase 7':['test_session_metrics','test_alert_manager','test_overlay','test_local_monitor'],
            'Phase 8':['test_benchmark'],'Phase 9':['test_optimization','test_cpu_optimization']}
    phase_counts={phase:sum(counts[f'tests/{name}.py'] for name in names) for phase,names in groups.items()}
    assert list(phase_counts.values())[:7]==[9,43,53,60,64,60,27]
    assert sum(phase_counts.values())==int(suite.get('tests'))
    gates={}
    sources=list(__import__('driver_monitor.metrics.benchmark',fromlist=['DEFAULT_SOURCES']).DEFAULT_SOURCES)
    for mode in ('headless','render'):
        evidence={'candidate':'production','parity':parity,'methodology':{'sources':sources,'acceptance_protocol':'cache_only'},
                  'aggregate':{'baseline':[r for r in matched['aggregate'] if r['mode']==mode], 'production':[r for r in final['aggregate'] if r['mode']==mode]},
                  'runs':[r for r in matched['runs']+final['runs'] if r['job']['kind']=='pipeline' and r['job'].get('mode')==mode and not r['job'].get('experiment_role')]}
        gates[mode]=assess_cache_only(evidence,mode)
        assert gates[mode]['decision']=='ELIGIBLE',gates
    # Preserve the original provisional selection artifact; final reports record
    # promotion only after the independent final acceptance checks above/below.
    selection=dict(selection)
    selection['promotion_status']=f"ACCEPTED: fresh cache-only benchmark, {suite.get('tests')}-test regression and bounded-memory checks passed"
    old_stats,new_stats,match_stats=pooled(baseline),pooled(final),pooled(matched)
    latency=[]
    for mode in ('headless','render'):
        for stage in ('inference_pipeline','application_processing','mediapipe_detect','cnn_lstm_forward')+ (('overlay',) if mode=='render' else ()):
            old,new,match=[d[(mode,stage)]['distribution'] for d in (old_stats,new_stats,match_stats)]
            latency.append({'mode':mode,'stage':stage,'phase8':old,'matched_baseline':match,'optimized':new,
                'phase8_improvement_percent':{q:100*(old[q]-new[q])/old[q] for q in ('p50_ms','p95_ms','p99_ms')},
                'matched_improvement_percent':{q:100*(match[q]-new[q])/match[q] for q in ('p50_ms','p95_ms','p99_ms')}})
    old_resources,new_resources,match_resources=resources(baseline),resources(final),resources(matched)
    for mode in ('headless','render'):
        assert new_resources[mode]['median_processing_rss_mib']<=1.1*match_resources[mode]['median_processing_rss_mib']
        assert new_resources[mode]['peak_rss_mib']<=1.1*match_resources[mode]['peak_rss_mib']
    long=next(r for r in final['runs'] if r['job']['kind']=='long_memory')
    old_long=next(r for r in baseline['runs'] if r['job']['kind']=='long_memory')
    assert long['session_metrics']==old_long['session_metrics'],'Long-run policy/event session changed'
    assert long['processed_frames']==10000 and long['timing_histories_disabled']
    assert long['bounded_payload']['feature_buffer_entries']==16 and long['bounded_payload']['smoother_entries']==5 and long['bounded_payload']['event_tracker_count']==5
    assert long['presentation_cache']['entries']==1 and long['presentation_cache']['capacity']==4
    all_experiments=[]
    for path in sorted((ROOT/'results').glob('phase9_*.json')):
        value=json.loads(path.read_text())
        if 'assessment' not in value: continue
        decision='ACCEPTED' if value['candidate'] in selection['accepted_candidates'] else 'REJECTED' if value['candidate'] in selection['rejected_candidates'] else value['assessment']['decision']
        if decision=='ELIGIBLE':decision='INCONCLUSIVE'
        all_experiments.append({'candidate':value['candidate'],'decision':decision,'evidence':str(path.relative_to(ROOT)),
             'assessment':value['assessment'],'parity':value['parity'],'methodology':value['methodology'],'aggregate':value['aggregate']})
        memory={};cpu={}
        for role in ('baseline',value['candidate']):
            selected=[r for r in value['runs'] if r['job']['candidate']==role and r['job']['kind']=='pipeline']
            samples=[s for r in selected for s in r['resources']['samples']]
            memory[role]={'median_processing_rss_mib':statistics.median(s['rss_bytes']/2**20 for s in samples if s['label']=='processing'),
                          'peak_rss_mib':max(s['peak_rss_bytes'] for s in samples)/2**20}
            cpu[role]=statistics.fmean(r['throughput']['processing_average_cpu_percent'] for r in selected)
        all_experiments[-1]['memory']=memory;all_experiments[-1]['mean_processing_cpu_percent']=cpu
    probes={p.stem.removesuffix('_corrected'):json.loads(p.read_text()) for p in (ROOT/'results/phase9_probes').glob('*_corrected.json')}
    assert len(probes)==6 and all(p.get('archive_preloaded_outside_timers') for p in probes.values())
    invalid_probes={p.stem:json.loads(p.read_text()) for p in (ROOT/'results/phase9_probes').glob('*.json')
                    if not p.stem.endswith('_corrected') and not json.loads(p.read_text()).get('archive_preloaded_outside_timers')}
    all_experiments.append({'candidate':'initial_snapshot_probe','decision':'REJECTED','reason':'Invalid measurement: NpzFile access decompressed entire archive members inside the copy timer. Retained initial files; corrected probes preload once outside endpoints. No optimization selected from invalid evidence.',
                           'evidence':[f'results/phase9_probes/{name}.json' for name in invalid_probes]})
    for name in ('pose_cache','cv1','cv2','cv4'):
        if not any(r['candidate']==name for r in all_experiments):
            all_experiments.append({'candidate':name,'decision':'INCONCLUSIVE','reason':'Exact parity passed; allocation/OpenCV diagnostic did not establish a worthwhile full-pipeline benefit. No production change.',
                'parity':json.loads((ROOT/f'results/phase9_parity/{name}/result.json').read_text()),'evidence':f'results/phase9_probes/{name}.json'})
    all_experiments.extend([
        {'candidate':'combined_unused_outputs_dashboard','decision':'REJECTED','reason':'Final original raw/application gate failed in both modes; modest standalone detector benefit was not reproduced. No lowered gate was applied to this candidate. Detector/engine changes were reverted byte-for-byte before a separately declared cache-only validation.',
         'evidence':'results/phase9_final_failed_gate.json','original_final_gates':failed_combined['candidate']['final_paired_gates'],
         'source_archive':'results/phase9_reference/optimization_framework_combined.py'},
        {'candidate':'tensor_from_numpy_owned','decision':'INCONCLUSIVE','reason':'Equivalent tensor values in diagnostic probe, but <1% Phase 8 stage contribution and no full-window/app matched gate; retain simple owned torch.tensor path.'},
        {'candidate':'ring_buffer','decision':'INCONCLUSIVE','reason':'Evaluated existing append/window allocations: .013/.024 ms Phase 8 P50; bounded deque retained, no speculative rewrite.'},
        {'candidate':'remove_result_copies','decision':'REJECTED','reason':'Snapshot ownership is required; copy stage diagnostic is small and no safe app benefit established.'},
        {'candidate':'encoding','decision':'INCONCLUSIVE','reason':'Separate optional write cost retained; no encoding configuration change or inference speedup attribution.'}])
    experiment_payload={'phase':9,'phase8_baseline_sha256':sha256(ROOT/'results/benchmark_baseline.json'),'selection':selection,
        'experiments':all_experiments,'allocation_opencv_probes':probes,'invalid_initial_probes':invalid_probes,'final_paired_gates':gates,'final_latency':latency,
        'methodology':'Exact six-source parity before timing; one change per experiment; seeded paired fresh-process workers; raw samples retained; five repetitions per representative source/mode. Diagnostic microprobes are not production acceptance evidence.'}
    validation={'phase':9,'status':'PASS','tests':{'passed':int(suite.get('tests')),'failures':0,'errors':0,'skipped':0,'seconds':float(suite.get('time'))},
        'tests_by_phase':phase_counts,'changed_files':changed,'prior_files_preserved':len(preserved),'preserved_sha256':preserved,
        'phase8_baseline_sha256':sha256(ROOT/'results/benchmark_baseline.json'),'optimized_sha256':sha256(ROOT/'results/benchmark_optimized.json'),
        'raw_windows_exact':699,'classes_exact':699,'max_raw_numeric_difference':0.0,'landmarks_policy_session_exact':True,'rejected_detector_experiment_native_fields_exact':native['fields'],'production_engine_detector_source_restored':True,'paired_pipeline_sessions_exact':True,'long_run_session_exact':True,
        'final_paired_gates':gates,'final_selection':selection,'predeclared_cache_only_protocol':protocol,'rejected_combined_gates':failed_combined['candidate']['final_paired_gates'],'worker_framework_sha256':framework_hash,'latency':latency,'resources':{'phase8':old_resources,'matched':match_resources,'optimized':new_resources},
        'raw_contribution':contributions(final),'long_memory_trend':long['rss_trend_second_half'],'bounded_payload':long['bounded_payload']}
    experiments=['# Phase 9 CPU experiments','', 'Each serious experiment has independent six-source correctness evidence and five fresh paired timing repetitions. Rejected and inconclusive attempts remain recorded. No slow run was discarded.','',
        'Gate defined before latency measurement: exact landmark/raw/policy parity; pooled raw/application P50 improvement ≥1%; no pooled P95 regression; ≥80% paired source median wins; mean process CPU ≤110% of paired baseline. Dashboard uses overlay/application boundaries. ELIGIBLE is provisional until combined production, unchanged prior tests and boundedness pass. These are conservative engineering criteria, not statistical significance claims.','',
        'The first 217-job combined comparison failed the original gate: headless raw P50 reduction 0.27% with 14/20 median wins; render raw 0.52% with 13/20 wins and a 0.29% P95 regression. Render app improved 8.79% P50/8.22% P95, but this did not justify accepting the detector change. Complete measurements, source snapshots and original failed gates remain preserved.','',
        'After restoring detector and engine byte-for-byte, a separate cache-only protocol was declared before fresh timing: exact raw/policy/pixel parity, original engine/detector source hashes, render overlay/application P50 improvement >=1%, no target P95 regression, >=80% paired application median wins, CPU <=110% and processing/peak RSS <=110%. Untouched raw/headless controls allow at most 5% P50/P95 regression; no headless inference optimization is claimed. This protocol cannot accept the old detector candidate because it requires restoration and an explicit cache-only methodology marker.','',
        'Normal_1 and Yawning_60 are the two initial MediaPipe representatives. Thread sweeps use Normal_1 plus all 699 prebuilt tensors. Dashboard is a render-only representative experiment. Both failed combined and final cache-only comparisons use all four Phase 8 sources and both modes. All warm-up rules, hashes and raw samples are in each evidence JSON.','']
    rows=[]
    for entry in all_experiments:
        name=entry['candidate']; agg=entry.get('aggregate')
        b=c='—'; cpu='—';memory='—'
        if agg:
            def dist(candidate,stage):return next(r['distribution'] for r in agg[candidate] if r['source']=='ALL_SOURCES' and r['stage']==stage)
            stage='overlay' if name=='dashboard_cache' else 'inference_pipeline'
            base,cand=dist('baseline',stage),dist(name,stage)
            b=f"{base['p50_ms']:.3f}/{base['p95_ms']:.3f}";c=f"{cand['p50_ms']:.3f}/{cand['p95_ms']:.3f}"
            cp=entry.get('mean_processing_cpu_percent')
            if cp:cpu=f"{cp['baseline']:.1f}/{cp[name]:.1f}%"
            m=entry['memory'];memory=f"{m['baseline']['median_processing_rss_mib']:.1f}/{m[name]['median_processing_rss_mib']:.1f} (peak {m['baseline']['peak_rss_mib']:.1f}/{m[name]['peak_rss_mib']:.1f})"
        parity_word='exact 699/699' if entry.get('parity',{}).get('exact_parity') else 'FAILED' if 'parity' in entry else 'diagnostic/deferred'
        rows.append([name,b,c,parity_word,cpu,memory,entry['decision']])
    experiments+=table(['Candidate','Baseline P50/P95 ms','Candidate P50/P95 ms','Parity','Baseline/candidate CPU','RSS/peak MiB baseline/candidate','Decision'],rows)
    experiments+=['','## MediaPipe option experiments','']
    rows=[]
    for entry in all_experiments:
        name=entry['candidate']
        if name not in ('no_blendshapes','no_matrices','no_unused_outputs','video'):continue
        cells=[]
        for role in ('baseline',name):
            d=next(r['distribution'] for r in entry['aggregate'][role] if r['source']=='ALL_SOURCES' and r['stage']=='mediapipe_detect')
            cells.append(f"{d['p50_ms']:.3f}/{d['p95_ms']:.3f}")
        rows.append([name,*cells,entry['decision']])
    experiments+=table(['Candidate','Baseline detector P50/P95 ms','Candidate detector P50/P95 ms','Decision'],rows)
    experiments+=['','## Thread sweep','']
    thread_rows=[]
    for entry in all_experiments:
        name=entry['candidate']
        if not name.startswith(('intra','interop')):continue
        agg=entry['aggregate']; threads=entry['parity']['threads']
        def values(role,stage,model=False):
            row=next(r for r in agg[role] if r['stage']==stage and (r['kind']=='model_only' if model else r['source']=='ALL_SOURCES'))
            d=row['distribution'];return f"{d['p50_ms']:.3f}/{d['p95_ms']:.3f}/{d['p99_ms']:.3f}"
        thread_rows.append([name,f"{threads['torch_num_threads']}/{threads['torch_num_interop_threads']}",
            values('baseline','model_only_forward',True),values(name,'model_only_forward',True),values(name,'mediapipe_detect'),
            values(name,'inference_pipeline'),values(name,'application_processing'),entry['decision']])
    experiments+=table(['Candidate','Intra/inter','Baseline model P50/P95/P99','Candidate model','Candidate detector','Candidate raw','Candidate app','Decision'],thread_rows)
    experiments+=['','Rows use raw pipeline latency except dashboard_cache, which uses overlay latency. Detailed MediaPipe/model/application metrics, per-source/per-run medians, mean/std, pooled samples, startup and RSS remain in the JSON; the short table is not a substitute for them.','',
        '## Individual purposes and decisions','',
        'MediaPipe output switches disable only unused facial expression/effect outputs; IMAGE, one face, CPU asset, .5 confidence defaults and legacy BGR-as-SRGB input remain. no_unused_outputs measures the two previously isolated switches together; its benefit is not attributed independently to both switches. All experiments compare the runtime’s 478×2 x/y landmarks. The combined candidate also has a separate six-source native x/y/z/visibility/presence equality check with both full archives retained. Installed API signature/docstring and file hash are in docs/phase9_mediapipe_options.json. VIDEO receives rounded monotonic source milliseconds but changes tracking/landmarks and is rejected; timing runs explicitly disable frozen verification after recording the failure, and cannot claim parity. LIVE_STREAM callbacks, confidence changes, resizing/ROI and detector skipping were not introduced.','',
        'Intra-op 1/2/4 versus 8 and inter-op 1/2 versus 8 are independent fresh-process sweeps; inter-op tests keep intra-op 8. Model-only and raw/app distributions are measured, not inferred from core count. OpenCV 1/2/4 versus 12 have complete exact parity and separate five-repeat head-pose/render probes; probe results determine whether a full pipeline thread experiment is justified. They do not imply MediaPipe thread control.','',
        'Intra-op 1 and 2 are rejected for measured full-pipeline P95 regression despite lower process CPU and favorable standalone model timings. Intra-op 4 and inter-op 1/2 remain inconclusive because they did not meet the repeated full-path acceptance gate. No cross-library scheduling or cache cause was established from these measurements.','',
        'Head-pose cache reuses only camera/distortion constants for up to four resolutions and retains all solvePnP/Rodrigues/decomposition/arithmetic quirks. Its exact six-source feature gate passed, but it is retained only if worthwhile full-path benefit is established. The fixed 3-D model points were already global. EAR/MAR arithmetic and feature ordering remain untouched.','',
        'Tensor probe compares torch.tensor with a fresh owned float32 NumPy array plus torch.from_numpy, with equal tensor values; production ownership/copy guarantees remain. Deque/ring and result-copy changes are not promoted based on tiny diagnostic differences. Snapshot copies retain independent prior-result ownership.','',
        'Dashboard cache stores a bounded background and invariant separator for frame shape/dtype, copies it each frame, then resizes the current source and draws all dynamic state/text every frame. Six-source dashboard pixels were exact; resolution/dtype/input ownership tests supplement that check. Encoding remains separate and unchanged.','',
        '## Probe distributions','']
    rows=[]
    for name,value in probes.items():
        for stage,stats in value['aggregate'].items():
            d=stats['pooled'];rows.append([name,stage,fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(stats['mean_run_median_ms']),fmt(stats['std_run_median_ms'])])
    experiments+=table(['Probe','Workload','P50 ms','P95 ms','Mean run median','Std run median'],rows)
    experiments+=['','Probes use prepared inputs and finite histories; they include clock/call overhead and omit MediaPipe/decision/source acquisition. Counts, raw samples and per-run medians are retained. They cannot establish end-to-end speedup.','',
        '## Reproduction','', 'Use .venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --parity baseline, then --parity <candidate>, followed by --experiment <candidate> --runs 5. Thread experiments add --sources Normal/Normal_1.avi. Dashboard adds --mode render. Allocation probes use tools\\probe_phase9_allocations.py --candidate <name> --output <new-path>. Existing evidence is never overwritten; run in a separate checkout/output location for a new experiment. Final commands and production configuration are in PHASE9_CPU_OPTIMIZATION.md.','']
    report_lines=['# Phase 9 CPU optimization','', '**PHASE 9 PASS.** Exact raw, landmark, policy/event and dashboard contracts preserved. Hardware-specific offline measurements; no accuracy, medical efficacy or sustained webcam claim.','',
        '## Files and scope','',
        'Created tools/optimize_phase9.py, tools/probe_phase9_allocations.py, tools/verify_phase9_landmarks.py, tools/report_phase9.py, tests/test_optimization.py, tests/test_cpu_optimization.py, results/phase9_experiments.json, results/benchmark_optimized.json, results/phase9_final_matched.json, results/phase9_final_failed_gate.json, docs/phase9_cache_only_protocol.json, individual experiment/parity/probe/reference evidence and Phase 9 reports/validation/XML. Modified: '+', '.join(changed)+'. Earlier tests, Phase 8 reports/evidence, source videos, canonical dataset/draw_landmarks/model/inference source, assets, model architecture/weights, normalization, window, sampling, thresholds/history/cooldown and decision/event/audio algorithms remain byte-identical.','',
        '## Methodology and preserved baseline','',
        f"Phase 8 baseline SHA-256: {sha256(ROOT/'results/benchmark_baseline.json')}. {len(preserved)} prior files remain byte-identical; docs/phase9_validation.json records every preserved/changed hash. Runtime reference source copies are retained under results/phase9_reference for reproducing baseline behavior after promotion.",'',
        'The same Phase 8 timing code executes 109 final production jobs and 108 fresh paired baseline jobs: each variant has 40 instrumented four-source × five-repetition × headless/render runs, 40 actual main.run controls, 20 detector-only and five model-only jobs, two extra parity clips and one encoding probe. Production alone adds the matching 10,000-frame render memory run. Fixed seed 20260926 randomizes workload pairs and order within each pair. Jobs run sequentially in fresh interpreters; baseline restores the recorded 8/8/12 thread settings, legacy MediaPipe outputs and saved renderer. Final production workers call the production path without experimental overrides.','',
        'First 20 sampled valid observations are excluded using unchanged Phase 8 boundaries; raw warm-up timings remain. Integer nanoseconds/frame indices are retained. Pooled distributions use linear interpolated quantiles and population std; JSON includes per-run medians, their mean/std and all pooled samples. No bootstrap/significance claim. Instrumentation, verification outside stage endpoints, actual uninstrumented main.run controls, disabled audio and omitted GUI refresh match Phase 8. Nested stages overlap and must not be double-counted.','',
        'Source hashes/configuration/environment and installed options are archived. Machine: '+final['environment']['cpu_model']+'; Windows, CPU only, '+str(final['environment']['thread_settings'])+'. Software: '+str(final['environment']['software'])+'. Physical cores: eight (Phase 8 read-only topology supplement); twelve logical. Power scheme/AC/affinity recorded; frequency, thermal state, caches and unrelated background work were not controlled.','',
        '## Experiments and selected configuration','',
        'All attempts, hypotheses, isolated settings, parity, performance and decisions are in PHASE9_EXPERIMENTS.md and results/phase9_experiments.json. Accepted candidates: '+', '.join(selection['accepted_candidates'])+'. Selection rationale: '+selection['reason'].rstrip('.')+'. Final cache-only production passes the separately declared target/control gates; provisional standalone detector benefit was not sufficient.','',
        'results/phase9_selection.json retains the initial provisional combined decision. results/phase9_selection_cache_only.json retains the later provisional cache-only protocol, also copied into benchmark_optimized.json. Final promotion is recorded separately as ACCEPTED in docs/phase9_validation.json and results/phase9_experiments.json after regression, fresh matched-mode and memory checks. Failed combined source/worker evidence remains separately archived.','',
        'Final settings: '+str(selection['production_configuration'])+'. Only the dashboard renderer changes. Detector and engine source bytes are restored to the Phase 8 versions; unused outputs remain enabled because the final matched detector benefit was insufficient. Installed thread defaults are retained; no user tuning option was added. Detection and rendering still run on every decoded frame; IMAGE mode and legacy color/pose mathematics remain. Event/session behavior is verified through complete per-frame policy traces and finalized session hashes. The presentation cache is bounded; source/input/result ownership stays intact.','',
        '## Raw/application/detector/model/overlay before and after','',
        'Phase 8 is the requested immutable historical reference. Fresh matched baseline is the stronger comparison for attributing improvement; both are shown. Milliseconds and reductions use (baseline−optimized)/baseline ×100 independently at each boundary/percentile.','']
    report_lines+=['Raw, detector and model rows are unchanged-path controls: their before/after differences are descriptive variation. The accepted optimization targets render overlay and application processing; no headless inference speedup is claimed.','']
    rows=[]
    for row in latency:
        old,new,match=row['phase8'],row['optimized'],row['matched_baseline']
        rows.append([row['mode'],row['stage'],f"{old['p50_ms']:.3f}/{old['p95_ms']:.3f}/{old['p99_ms']:.3f}",
            f"{match['p50_ms']:.3f}/{match['p95_ms']:.3f}/{match['p99_ms']:.3f}",f"{new['p50_ms']:.3f}/{new['p95_ms']:.3f}/{new['p99_ms']:.3f}",
            f"{row['phase8_improvement_percent']['p50_ms']:.2f}/{row['phase8_improvement_percent']['p95_ms']:.2f}%",f"{row['matched_improvement_percent']['p50_ms']:.2f}/{row['matched_improvement_percent']['p95_ms']:.2f}%"])
    report_lines+=table(['Mode','Boundary','Phase 8 P50/P95/P99','Matched baseline','Optimized','vs Phase 8 P50/P95 reduction','Matched reduction'],rows)
    report_lines+=['','## Isolated workloads, startup, throughput and budget','']
    rows=[]
    for label,data in [('Phase 8',baseline),('Matched baseline',matched),('Optimized',final)]:
        for row in data['aggregate']:
            if row['kind']=='model_only':
                d=row['distribution'];rows.append([label,row['stage'],fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms'])])
        values=[s['elapsed_ns']/1e6 for r in data['runs'] if r['job']['kind']=='mediapipe_only' for s in r['samples'] if s['steady']]
        d=distribution_ms(values);rows.append([label,'detector_only',fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms'])])
    report_lines+=table(['Variant','Isolated workload','P50','P95','P99'],rows)
    report_lines+=['','Prepared model tensors have the Phase 8 archive layout; standalone and production tensor-layout/copy costs differ even when values match. Speedups are compared within matched workloads.','']
    rows=[]
    for stage in ('runtime_import_ns','engine_initialization_ns','components_initialization_ns','FaceDetector_construction_ns'):
        values=[]
        for data in (baseline,matched,final):
            sample=[r['startup'][stage]/1e6 for r in pipeline(data) if stage in r['startup']]
            values.append(fmt(statistics.median(sample)))
        rows.append([stage,*values])
    report_lines+=table(['Startup P50 ms','Phase 8','Matched','Optimized'],rows)
    rows=[]
    for stage in ('mediapipe_detect','cnn_lstm_forward','overlay'):
        cells=[]
        for data in (baseline,matched,final):
            first=[r['stages'][stage]['raw_elapsed_ns'][0]/1e6 for r in pipeline(data) if stage in r['stages'] and r['stages'][stage]['raw_elapsed_ns']]
            cells.append(fmt(statistics.median(first)))
        rows.append([stage,*cells])
    report_lines+=['']+table(['First call median ms (outside steady distribution)','Phase 8','Matched','Optimized'],rows)
    rows=[]
    for label,data in [('Phase 8',baseline),('Matched',matched),('Optimized',final)]:
        probe=next(r for r in data['runs'] if r['job'].get('experiment_role',r.get('experiment_role'))=='encoding_probe')
        for stage in ('overlay','video_write'):
            d=distribution_ms(steady_values(probe,stage))
            rows.append([label,stage,fmt(d['p50_ms']),fmt(d['p95_ms']),fmt(d['p99_ms'])])
    report_lines+=['']+table(['Separate single encoding probe','Boundary','P50 ms','P95 ms','P99 ms'],rows)
    report_lines+=['','Encoding is optional and remains unchanged. These one-probe distributions do not establish an encoding optimization or contribute to inference speedup claims.','']
    rows=[]
    for source in sources:
        for mode in ('headless','render'):
            cells=[]
            for data in (baseline,matched,final):
                rates=[r['offline_processing_fps'] for r in data['runs'] if r['job']['kind']=='control' and Path(r['job']['source']).name==Path(source).name and r['job']['mode']==mode]
                cells.append(f"{statistics.median(rates):.3f} ({min(rates):.3f}–{max(rates):.3f}; std {statistics.pstdev(rates):.3f})")
            rows.append([source,mode,*cells])
    report_lines+=['']+table(['Source','Mode','Phase 8 offline FPS P50/range/std','Matched baseline','Optimized'],rows)
    report_lines+=['','These are full actual main.run offline throughput figures, including source decode, policy, metrics and optional rendering; no verification/stage-hook/RSS overhead. They are not model cadence or measured webcam FPS. Target valid observation cadence remains 20 Hz; no-face gaps reduce average valid observations on Yawning_60. No GUI-refresh/audio-playback latency was measured.','']
    rows=[]
    for mode in ('headless','render'):
        for stage in ('inference_pipeline','application_processing'):
            for label,data in [('Phase 8',baseline),('Matched',matched),('Optimized',final)]:
                values=[v for r in pipeline(data) if r['job']['mode']==mode for v in steady_values(r,stage)]
                b=budget_50ms(values);rows.append([mode,stage,label,f"{b['percent_at_or_below_50ms']:.3f}%",fmt(b['p95_to_50ms_ratio']),fmt(b['p99_to_50ms_ratio'])])
    report_lines+=table(['Mode','Boundary','Variant','≤50 ms','P95/50','P99/50'],rows)
    report_lines+=['','## CPU, RSS and long-run boundedness','']
    rows=[]
    for label,data in [('Phase 8',old_resources),('Matched',match_resources),('Optimized',new_resources)]:
        for mode,v in data.items():rows.append([label,mode,f"{v['processing_cpu_percent']:.2f}%",f"{v['peak_interval_cpu_percent']:.2f}%",fmt(v['median_processing_rss_mib']),fmt(v['peak_rss_mib'])])
    report_lines+=table(['Variant','Mode','Weighted process CPU','Peak interval CPU','Median processing RSS MiB','Peak RSS MiB'],rows)
    report_lines+=['','CPU is summed process CPU seconds/wall seconds ×100 and can exceed 100% across cores. Periodic peaks are sample interval peaks, not instantaneous hardware utilization. RSS is Windows working-set residency, not committed/live allocation bytes.','',
        f"Long run: {long['processed_frames']} frames, {long['video_passes']} source passes, {long['loop_wall_seconds']:.3f} s, {long['offline_processing_fps']:.3f} offline FPS. Phase 8 comparison: {old_long['loop_wall_seconds']:.3f} s/{old_long['offline_processing_fps']:.3f} FPS. Final bounded payload: {long['bounded_payload']}; presentation cache: {long['presentation_cache']}. Raw timing histories disabled; resource samples every 100 frames. Persistent engine/policy and continuous 20-Hz timestamps match Phase 8. No new retained per-frame production history.",'',
        'After 2,000 frames: '+str(long['rss_trend_after_2000_frames'])+'. Second half: '+str(long['rss_trend_second_half'])+'. Descriptive residency fluctuations do not prove absence/presence of a leak; neither private commit nor allocator histories were collected. Phase 8 itself showed a large residency decline with an unestablished cause. Fixed feature/smoother/event structures provide direct boundedness evidence.','',
        '## Final measured bottleneck','']
    report_lines+=table(['Disjoint raw component','Share of summed headless raw processing'],[[r['stage'],f"{r['percent_raw']:.2f}%"] for r in validation['raw_contribution']])
    report_lines+=['','Shares use summed steady elapsed time with inclusive parents removed, not sums of percentile values. Features include the unchanged shared head-pose calculation. Rendering is outside raw inference and remains separately reported. Optional encoding probes retain raw video_write timings; no encoding speedup is claimed.','',
        '## Regression, limitations and reproduction','',
        f"{int(suite.get('tests'))} tests pass with zero failures/errors/skips; previous 316 tests and assertions are byte-identical. Counts: {phase_counts}. All 699 classes, landmarks, raw/normalized features, sequences, float32 tensors, logits/probabilities and policy/final-session traces remain exact; maximum raw numeric difference 0.0. Dashboard pixels remain exact on all six reference videos. Full XML: docs/phase9_tests_cache_only.xml. The first combined configuration also passed 380 tests (docs/phase9_tests_pass.xml), but failed the performance gate and was reverted. Initial new-test fixture failures are retained in docs/phase9_tests.xml and docs/phase9_tests_final.xml: absent prediction confidence, then an incorrect expectation that OpenCV 5 supports non-uint8 putText images. The tests now use valid confidence and verify identical legacy rejection for unsupported depths; runtime and prior assertions were unchanged. Hardware sound/webcam validation was not repeated or fabricated.",'',
        'Limitations: hardware-specific offline clips; finite five-repeat experiments; unlogged thermals/frequency/cache/background load; instrumented observer cost; source-specific/no-face tail counts; one long-memory run; working-set telemetry; single encoding probe; no live webcam, real GUI refresh or physical audio throughput. No retraining/resplits/new model/window/database/Phase 10 work, clinical claims, event accuracy or false-positive claims.','',
        'Commands executed (PowerShell, repo root):','',
        '```powershell',
        '$env:MPLCONFIGDIR = Join-Path (Get-Location) ".cache/matplotlib"',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --parity baseline',
        'foreach ($phase9Candidate in @("no_blendshapes", "no_matrices", "no_unused_outputs", "video", "intra1", "intra2", "intra4", "interop1", "interop2", "cv1", "cv2", "cv4", "pose_cache", "dashboard_cache", "contiguous_tensor")) {',
        '    & .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --parity $phase9Candidate',
        '}',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment no_blendshapes --runs 5',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment no_matrices --runs 5',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment no_unused_outputs --runs 5',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment video --runs 5 --sources Normal/Normal_1.avi',
        'foreach ($phase9Candidate in @("intra1", "intra2", "intra4", "interop1", "interop2")) {',
        '    & .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment $phase9Candidate --runs 5 --sources Normal/Normal_1.avi',
        '}',
        'foreach ($phase9Candidate in @("baseline", "cv1", "cv2", "cv4", "pose_cache", "dashboard_cache")) {',
        '    & .\\.venv-baseline\\Scripts\\python.exe -B tools\\probe_phase9_allocations.py --candidate $phase9Candidate --output "results/phase9_probes/${phase9Candidate}_corrected.json"',
        '}',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment dashboard_cache --runs 5 --mode render',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --experiment contiguous_tensor --runs 5',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\verify_phase9_landmarks.py',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --parity production --output results/phase9_parity/production_cache_only/result.json',
        '& .\\.venv-baseline\\Scripts\\python.exe -B -m pytest -q --junitxml=docs/phase9_tests_cache_only.xml',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\optimize_phase9.py --final-suite --label cache_only',
        '& .\\.venv-baseline\\Scripts\\python.exe -B tools\\report_phase9.py',
        '```','',
        'Existing evidence uses exclusive writes; reproduce in a separate checkout/output workspace rather than overwriting it. Copy the current code/environment, source clips, model assets and Phase 2–8 evidence, plus results/phase9_reference, results/phase9_selection.json, results/phase9_selection_cache_only.json, docs/phase9_before_hashes.json, docs/phase9_cache_only_protocol.json and the retained failed combined result. Keep the archived reference/runtime/framework sources; their hashes identify the failed combined and final cache-only configurations. Omit the generated Phase 9 run/parity/probe folders, native archives, experiment/master results and final reports from that fresh workspace. The retained reference sources supply the original runtime after production promotion; their hashes are recorded in the preservation evidence. Final A/B baseline overrides exist only in tools, and production workers use the selected clean runtime. Test results and parity gates must be rerun after any runtime edit.','',
        '## Resume-safe wording','']
    h=next(r for r in latency if r['mode']=='headless' and r['stage']=='inference_pipeline')
    rendered_app=next(r for r in latency if r['mode']=='render' and r['stage']=='application_processing')
    render_control_medians=[statistics.median(r['offline_processing_fps'] for r in final['runs'] if r['job']['kind']=='control' and r['job']['mode']=='render' and Path(r['job']['source']).name==Path(source).name) for source in sources]
    report_lines += [f"Reduced CPU-only dashboard application median processing latency from {rendered_app['matched_baseline']['p50_ms']:.2f} to {rendered_app['optimized']['p50_ms']:.2f} ms ({rendered_app['matched_improvement_percent']['p50_ms']:.1f}% reduction in randomized paired local video benchmarks), preserving 699/699 exact frozen outputs and unchanged policy/event behavior. Full application offline processing with the dashboard achieved {min(render_control_medians):.2f}–{max(render_control_medians):.2f} FPS (range of four per-source five-run medians), without GPU acceleration. This measures local video processing, not sustained physical webcam throughput or accuracy.",'',
        f"The unchanged raw headless control separately measured {h['matched_baseline']['p50_ms']:.2f} to {h['optimized']['p50_ms']:.2f} ms P50 ({h['matched_improvement_percent']['p50_ms']:.1f}% descriptive paired difference). This is not a claimed headless inference optimization. Historical Phase 8 values remain in the before/after table; the resume wording uses the contemporaneous matched render comparison to reduce attribution to machine-wide drift.",'',
        '**Stopped after Phase 9.**','']
    # Build every table before publishing any final report, so malformed evidence
    # cannot leave a partially generated PASS report.
    write_new_json(ROOT/'results/phase9_experiments.json',experiment_payload)
    write_new_json(ROOT/'docs/phase9_validation.json',validation)
    with (ROOT/'docs/PHASE9_EXPERIMENTS.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(experiments))
    with (ROOT/'docs/PHASE9_CPU_OPTIMIZATION.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(report_lines))
    print(json.dumps({'phase9':'PASS','tests':validation['tests'],'raw_parity':699,'accepted':selection['accepted_candidates'],'latency':latency[:2]},indent=2))


if __name__=='__main__': report()
