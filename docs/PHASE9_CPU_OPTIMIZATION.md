# Phase 9 CPU optimization

**PHASE 9 PASS.** Exact raw, landmark, policy/event and dashboard contracts preserved. Hardware-specific offline measurements; no accuracy, medical efficacy or sustained webcam claim.

## Files and scope

Created tools/optimize_phase9.py, tools/probe_phase9_allocations.py, tools/verify_phase9_landmarks.py, tools/report_phase9.py, tests/test_optimization.py, tests/test_cpu_optimization.py, results/phase9_experiments.json, results/benchmark_optimized.json, results/phase9_final_matched.json, results/phase9_final_failed_gate.json, docs/phase9_cache_only_protocol.json, individual experiment/parity/probe/reference evidence and Phase 9 reports/validation/XML. Modified: driver_monitor/ui/overlay.py. Earlier tests, Phase 8 reports/evidence, source videos, canonical dataset/draw_landmarks/model/inference source, assets, model architecture/weights, normalization, window, sampling, thresholds/history/cooldown and decision/event/audio algorithms remain byte-identical.

## Methodology and preserved baseline

Phase 8 baseline SHA-256: 8e9b6f01f85488c91df322df1e2bb091f6d645408e0fb372915238989dc5e9f7. 505 prior files remain byte-identical; docs/phase9_validation.json records every preserved/changed hash. Runtime reference source copies are retained under results/phase9_reference for reproducing baseline behavior after promotion.

The same Phase 8 timing code executes 109 final production jobs and 108 fresh paired baseline jobs: each variant has 40 instrumented four-source × five-repetition × headless/render runs, 40 actual main.run controls, 20 detector-only and five model-only jobs, two extra parity clips and one encoding probe. Production alone adds the matching 10,000-frame render memory run. Fixed seed 20260926 randomizes workload pairs and order within each pair. Jobs run sequentially in fresh interpreters; baseline restores the recorded 8/8/12 thread settings, legacy MediaPipe outputs and saved renderer. Final production workers call the production path without experimental overrides.

First 20 sampled valid observations are excluded using unchanged Phase 8 boundaries; raw warm-up timings remain. Integer nanoseconds/frame indices are retained. Pooled distributions use linear interpolated quantiles and population std; JSON includes per-run medians, their mean/std and all pooled samples. No bootstrap/significance claim. Instrumentation, verification outside stage endpoints, actual uninstrumented main.run controls, disabled audio and omitted GUI refresh match Phase 8. Nested stages overlap and must not be double-counted.

Source hashes/configuration/environment and installed options are archived. Machine: 12th Gen Intel(R) Core(TM) i5-12450H; Windows, CPU only, {'torch_num_threads': 8, 'torch_num_interop_threads': 8, 'opencv_num_threads': 12, 'logical_cpu_count': 12, 'mkldnn_enabled': True, 'cuda_available': False}. Software: {'torch': '2.14.0+cpu', 'opencv-contrib-python': '5.0.0.93', 'mediapipe': '1.0.1', 'numpy': '2.5.3'}. Physical cores: eight (Phase 8 read-only topology supplement); twelve logical. Power scheme/AC/affinity recorded; frequency, thermal state, caches and unrelated background work were not controlled.

## Experiments and selected configuration

All attempts, hypotheses, isolated settings, parity, performance and decisions are in PHASE9_EXPERIMENTS.md and results/phase9_experiments.json. Accepted candidates: dashboard_cache. Selection rationale: Final four-source matched evidence did not reproduce the unused-output raw P50 benefit (0.27% headless/0.52% render; 14/20 and 13/20 median wins). Reject that production change. Validate only the separately eligible pixel-exact dashboard cache, with raw/headless behavior unchanged. Final cache-only production passes the separately declared target/control gates; provisional standalone detector benefit was not sufficient.

results/phase9_selection.json retains the initial provisional combined decision. results/phase9_selection_cache_only.json retains the later provisional cache-only protocol, also copied into benchmark_optimized.json. Final promotion is recorded separately as ACCEPTED in docs/phase9_validation.json and results/phase9_experiments.json after regression, fresh matched-mode and memory checks. Failed combined source/worker evidence remains separately archived.

Final settings: {'device': 'cpu', 'mediapipe_running_mode': 'IMAGE', 'num_faces': 1, 'output_face_blendshapes': True, 'output_facial_transformation_matrixes': True, 'color_contract': 'legacy BGR passed as SRGB', 'thread_policy': 'retain installed defaults; measured Torch intra/inter 8/8, OpenCV 12', 'torch_intra': 8, 'torch_interop': 8, 'opencv_threads': 12, 'sequence_length': 16, 'smoother_entries': 5, 'target_observations_per_second': 20, 'detector_on_every_decoded_frame': True, 'dashboard_update_every_frame': True, 'dashboard_cache_capacity': 4}. Only the dashboard renderer changes. Detector and engine source bytes are restored to the Phase 8 versions; unused outputs remain enabled because the final matched detector benefit was insufficient. Installed thread defaults are retained; no user tuning option was added. Detection and rendering still run on every decoded frame; IMAGE mode and legacy color/pose mathematics remain. Event/session behavior is verified through complete per-frame policy traces and finalized session hashes. The presentation cache is bounded; source/input/result ownership stays intact.

## Raw/application/detector/model/overlay before and after

Phase 8 is the requested immutable historical reference. Fresh matched baseline is the stronger comparison for attributing improvement; both are shown. Milliseconds and reductions use (baseline−optimized)/baseline ×100 independently at each boundary/percentile.

Raw, detector and model rows are unchanged-path controls: their before/after differences are descriptive variation. The accepted optimization targets render overlay and application processing; no headless inference speedup is claimed.

**Interpretation:** PASS applies to the separately declared matched rendering target, not a blanket improvement over every Phase 8 measurement. Historical render application P95 increased from 35.403 to 39.058 ms; its fresh baseline was already 42.666 ms. Historical headless raw P95 increased from 31.475 to 39.893 ms, versus a fresh baseline of 38.876 ms. The detector/engine code is byte-identical, and the source, thread settings and timing scopes match; the cause of the broader measurement shift was not established. The defensible optimization claim is the 8.58% P50/8.46% P95 improvement against the fresh paired render baseline. Headless app P95 was 2.44% slower against its fresh control, within the predeclared 5% control threshold.

| Mode | Boundary | Phase 8 P50/P95/P99 | Matched baseline | Optimized | vs Phase 8 P50/P95 reduction | Matched reduction |
|---|---|---|---|---|---|---|
| headless | inference_pipeline | 28.242/31.475/40.869 | 28.385/38.876/42.603 | 28.499/39.893/44.072 | -0.91/-26.74% | -0.40/-2.61% |
| headless | application_processing | 28.611/31.884/41.463 | 28.726/39.447/43.137 | 28.856/40.408/44.620 | -0.86/-26.73% | -0.45/-2.44% |
| headless | mediapipe_detect | 23.500/26.005/33.525 | 23.603/32.833/36.189 | 23.747/33.428/36.673 | -1.05/-28.54% | -0.61/-1.81% |
| headless | cnn_lstm_forward | 2.169/2.882/4.593 | 1.842/2.691/3.544 | 1.925/2.722/5.769 | 11.23/5.53% | -4.56/-1.15% |
| render | inference_pipeline | 27.698/30.328/32.535 | 29.232/36.008/43.638 | 29.150/36.088/43.392 | -5.24/-18.99% | 0.28/-0.22% |
| render | application_processing | 32.311/35.403/38.723 | 34.069/42.666/50.621 | 31.146/39.058/46.353 | 3.61/-10.32% | 8.58/8.46% |
| render | mediapipe_detect | 23.124/25.442/26.839 | 24.441/29.419/35.448 | 24.329/29.835/36.727 | -5.21/-17.27% | 0.46/-1.41% |
| render | cnn_lstm_forward | 2.135/2.555/3.314 | 2.106/2.681/6.157 | 2.091/2.547/3.310 | 2.10/0.34% | 0.74/5.01% |
| render | overlay | 4.356/4.847/6.497 | 4.412/6.457/7.213 | 1.613/2.395/2.788 | 62.96/50.59% | 63.43/62.91% |

## Isolated workloads, startup, throughput and budget

| Variant | Isolated workload | P50 | P95 | P99 |
|---|---|---|---|---|
| Phase 8 | model_only_forward | 1.635 | 2.141 | 2.796 |
| Phase 8 | model_forward_and_result | 1.962 | 2.406 | 2.768 |
| Phase 8 | detector_only | 26.480 | 38.074 | 47.600 |
| Matched baseline | model_only_forward | 1.633 | 2.372 | 2.851 |
| Matched baseline | model_forward_and_result | 2.014 | 2.962 | 4.277 |
| Matched baseline | detector_only | 25.565 | 35.760 | 43.333 |
| Optimized | model_only_forward | 1.364 | 2.256 | 2.959 |
| Optimized | model_forward_and_result | 2.011 | 2.667 | 3.297 |
| Optimized | detector_only | 25.681 | 38.450 | 46.445 |

Prepared model tensors have the Phase 8 archive layout; standalone and production tensor-layout/copy costs differ even when values match. Speedups are compared within matched workloads.

| Startup P50 ms | Phase 8 | Matched | Optimized |
|---|---|---|---|
| runtime_import_ns | 2978.860 | 3464.070 | 2820.648 |
| engine_initialization_ns | 190.343 | 186.446 | 164.890 |
| components_initialization_ns | 199.165 | 192.875 | 173.864 |
| FaceDetector_construction_ns | 121.600 | 139.115 | 123.797 |

| First call median ms (outside steady distribution) | Phase 8 | Matched | Optimized |
|---|---|---|---|
| mediapipe_detect | 30.593 | 22.352 | 21.650 |
| cnn_lstm_forward | 7.630 | 8.646 | 7.020 |
| overlay | 42.134 | 38.975 | 36.683 |

| Separate single encoding probe | Boundary | P50 ms | P95 ms | P99 ms |
|---|---|---|---|---|
| Phase 8 | overlay | 4.110 | 6.159 | 6.455 |
| Phase 8 | video_write | 3.884 | 4.517 | 4.602 |
| Matched | overlay | 2.717 | 3.061 | 3.214 |
| Matched | video_write | 3.020 | 3.338 | 3.419 |
| Optimized | overlay | 1.104 | 1.366 | 1.413 |
| Optimized | video_write | 3.037 | 3.427 | 3.880 |

Encoding is optional and remains unchanged. These one-probe distributions do not establish an encoding optimization or contribute to inference speedup claims.


| Source | Mode | Phase 8 offline FPS P50/range/std | Matched baseline | Optimized |
|---|---|---|---|---|
| Normal/Normal_1.avi | headless | 33.122 (31.998–36.882; std 1.702) | 43.198 (25.911–58.980; std 11.566) | 43.505 (24.793–58.799; std 11.749) |
| Normal/Normal_1.avi | render | 29.412 (25.810–41.689; std 5.495) | 21.778 (21.496–50.323; std 11.762) | 23.567 (22.773–54.251; std 13.038) |
| Yawning/Yawning_1.avi | headless | 34.343 (32.889–40.794; std 2.863) | 32.847 (26.274–54.415; std 9.646) | 33.180 (25.882–49.814; std 7.978) |
| Yawning/Yawning_1.avi | render | 28.891 (26.843–39.701; std 5.625) | 30.394 (27.073–38.121; std 4.914) | 29.943 (23.684–41.673; std 6.002) |
| Eye Close/Eye Close_60.avi | headless | 32.988 (27.212–36.795; std 3.066) | 36.375 (30.709–51.967; std 8.113) | 32.954 (24.444–57.312; std 11.581) |
| Eye Close/Eye Close_60.avi | render | 29.089 (28.277–39.819; std 4.406) | 29.757 (21.638–46.507; std 8.139) | 31.921 (23.072–56.477; std 11.354) |
| Yawning/Yawning_60.avi | headless | 49.469 (32.407–50.856; std 8.187) | 34.673 (27.304–65.794; std 13.537) | 35.015 (26.134–57.553; std 10.625) |
| Yawning/Yawning_60.avi | render | 29.463 (27.742–38.735; std 4.002) | 31.220 (28.263–38.553; std 3.437) | 32.087 (31.096–42.987; std 4.372) |

These are full actual main.run offline throughput figures, including source decode, policy, metrics and optional rendering; no verification/stage-hook/RSS overhead. They are not model cadence or measured webcam FPS. Target valid observation cadence remains 20 Hz; no-face gaps reduce average valid observations on Yawning_60. No GUI-refresh/audio-playback latency was measured.

Uninstrumented throughput has substantial run-to-run variability. Render median throughput increased on three of four sources; Yawning_1 decreased from 30.394 to 29.943 FPS against the fresh baseline. This does not establish a uniform full-loop FPS improvement across all workloads. The instrumented render application median improved in all 20 matched source/run pairs.

| Mode | Boundary | Variant | ≤50 ms | P95/50 | P99/50 |
|---|---|---|---|---|---|
| headless | inference_pipeline | Phase 8 | 99.569% | 0.6295 | 0.8174 |
| headless | inference_pipeline | Matched | 99.784% | 0.7775 | 0.8521 |
| headless | inference_pipeline | Optimized | 99.655% | 0.7979 | 0.8814 |
| headless | application_processing | Phase 8 | 99.569% | 0.6377 | 0.8293 |
| headless | application_processing | Matched | 99.784% | 0.7889 | 0.8627 |
| headless | application_processing | Optimized | 99.526% | 0.8082 | 0.8924 |
| render | inference_pipeline | Phase 8 | 99.871% | 0.6066 | 0.6507 |
| render | inference_pipeline | Matched | 99.483% | 0.7202 | 0.8728 |
| render | inference_pipeline | Optimized | 99.871% | 0.7218 | 0.8678 |
| render | application_processing | Phase 8 | 99.784% | 0.7081 | 0.7745 |
| render | application_processing | Matched | 98.966% | 0.8533 | 1.012 |
| render | application_processing | Optimized | 99.741% | 0.7812 | 0.9271 |

## CPU, RSS and long-run boundedness

| Variant | Mode | Weighted process CPU | Peak interval CPU | Median processing RSS MiB | Peak RSS MiB |
|---|---|---|---|---|---|
| Phase 8 | headless | 245.20% | 352.20% | 312.699 | 314.141 |
| Phase 8 | render | 274.07% | 713.68% | 322.031 | 323.477 |
| Matched | headless | 560.28% | 800.68% | 315.174 | 316.508 |
| Matched | render | 430.18% | 784.70% | 324.445 | 326.527 |
| Optimized | headless | 560.01% | 796.44% | 314.619 | 315.727 |
| Optimized | render | 451.09% | 796.22% | 324.705 | 326.086 |

CPU is summed process CPU seconds/wall seconds ×100 and can exceed 100% across cores. Periodic peaks are sample interval peaks, not instantaneous hardware utilization. RSS is Windows working-set residency, not committed/live allocation bytes.

Current process CPU utilization is substantially higher than historical Phase 8 in both the fresh baseline and candidate; its cause was not established. Against the fresh baseline, weighted render CPU increased 4.86%, processing RSS increased about 0.08% and peak RSS decreased about 0.14%. These pass the declared 10% CPU/RSS limits. The cache is a latency optimization; no CPU utilization reduction is claimed.

Long run: 10000 frames, 73 source passes, 265.558 s, 37.657 offline FPS. Phase 8 comparison: 332.616 s/30.065 FPS. Final bounded payload: {'feature_buffer_entries': 16, 'feature_buffer_bytes': 640, 'smoother_entries': 5, 'smoother_bytes': 120, 'event_tracker_count': 5}; presentation cache: {'entries': 1, 'capacity': 4, 'source_frames_retained': False, 'steady_cached_background_bytes': 921600, 'byte_count_scope': 'This fixed 640x480 uint8 long-run source only'}. Raw timing histories disabled; resource samples every 100 frames. Persistent engine/policy and continuous 20-Hz timestamps match Phase 8. No new retained per-frame production history.

After 2,000 frames: {'sample_count': 81, 'first_rss_bytes': 332918784, 'last_rss_bytes': 336527360, 'net_change_bytes': 3608576, 'min_rss_bytes': 332918784, 'max_rss_bytes': 336592896, 'linear_slope_bytes_per_frame': 262.21707678410115, 'strictly_monotonic_growth': False, 'positive_interval_fraction': 0.5375, 'assessment': 'Descriptive RSS trend only; not proof of absence or presence of a memory leak'}. Second half: {'sample_count': 51, 'first_rss_bytes': 335798272, 'last_rss_bytes': 336527360, 'net_change_bytes': 729088, 'min_rss_bytes': 335798272, 'max_rss_bytes': 336592896, 'linear_slope_bytes_per_frame': 94.28584253393664, 'strictly_monotonic_growth': False, 'positive_interval_fraction': 0.52, 'assessment': 'Descriptive RSS trend only; not proof of absence or presence of a memory leak'}. Descriptive residency fluctuations do not prove absence/presence of a leak; neither private commit nor allocator histories were collected. Phase 8 itself showed a large residency decline with an unestablished cause. Fixed feature/smoother/event structures provide direct boundedness evidence.

Second-half RSS increased by 0.70 MiB (about 0.22% of the final working set); the net increase after frame 2,000 was 3.44 MiB. This is small positive residency drift, not a claimed plateau or leak-free result. The measured production structures and cache remained fixed in size.

## Final measured bottleneck

MediaPipe remains the largest raw stage at 83.10%, versus 83.12% in Phase 8. The raw bottleneck did not shift; dashboard work is outside raw inference.

| Disjoint raw component | Share of summed headless raw processing |
|---|---|
| mediapipe_detect | 83.10% |
| feature_extraction | 7.18% |
| cnn_lstm_forward | 6.97% |
| sequence_other_residual | 1.44% |
| raw_other | 0.50% |
| tensor_creation | 0.31% |
| temporal_policy | 0.16% |
| normalization | 0.10% |
| buffer_window | 0.09% |
| softmax | 0.07% |
| buffer_append | 0.05% |
| result_constructor | 0.04% |

Shares use summed steady elapsed time with inclusive parents removed, not sums of percentile values. Features include the unchanged shared head-pose calculation. Rendering is outside raw inference and remains separately reported. Optional encoding probes retain raw video_write timings; no encoding speedup is claimed.

## Regression, limitations and reproduction

380 tests pass with zero failures/errors/skips; previous 316 tests and assertions are byte-identical. Counts: {'Phase 2': 9, 'Phase 3': 43, 'Phase 4': 53, 'Phase 5': 60, 'Phase 6': 64, 'Phase 7': 60, 'Phase 8': 27, 'Phase 9': 64}. All 699 classes, landmarks, raw/normalized features, sequences, float32 tensors, logits/probabilities and policy/final-session traces remain exact; maximum raw numeric difference 0.0. Dashboard pixels remain exact on all six reference videos. Full XML: docs/phase9_tests_cache_only.xml. The first combined configuration also passed 380 tests (docs/phase9_tests_pass.xml), but failed the performance gate and was reverted. Initial new-test fixture failures are retained in docs/phase9_tests.xml and docs/phase9_tests_final.xml: absent prediction confidence, then an incorrect expectation that OpenCV 5 supports non-uint8 putText images. The tests now use valid confidence and verify identical legacy rejection for unsupported depths; runtime and prior assertions were unchanged. Hardware sound/webcam validation was not repeated or fabricated.

Limitations: hardware-specific offline clips; finite five-repeat experiments; unlogged thermals/frequency/cache/background load; instrumented observer cost; source-specific/no-face tail counts; one long-memory run; working-set telemetry; single encoding probe; no live webcam, real GUI refresh or physical audio throughput. No retraining/resplits/new model/window/database/Phase 10 work, clinical claims, event accuracy or false-positive claims.

Commands executed (PowerShell, repo root):

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".cache/matplotlib"
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --parity baseline
foreach ($phase9Candidate in @("no_blendshapes", "no_matrices", "no_unused_outputs", "video", "intra1", "intra2", "intra4", "interop1", "interop2", "cv1", "cv2", "cv4", "pose_cache", "dashboard_cache", "contiguous_tensor")) {
    & .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --parity $phase9Candidate
}
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment no_blendshapes --runs 5
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment no_matrices --runs 5
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment no_unused_outputs --runs 5
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment video --runs 5 --sources Normal/Normal_1.avi
foreach ($phase9Candidate in @("intra1", "intra2", "intra4", "interop1", "interop2")) {
    & .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment $phase9Candidate --runs 5 --sources Normal/Normal_1.avi
}
foreach ($phase9Candidate in @("baseline", "cv1", "cv2", "cv4", "pose_cache", "dashboard_cache")) {
    & .\.venv-baseline\Scripts\python.exe -B tools\probe_phase9_allocations.py --candidate $phase9Candidate --output "results/phase9_probes/${phase9Candidate}_corrected.json"
}
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment dashboard_cache --runs 5 --mode render
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --experiment contiguous_tensor --runs 5
& .\.venv-baseline\Scripts\python.exe -B tools\verify_phase9_landmarks.py
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --parity production --output results/phase9_parity/production_cache_only/result.json
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase9_tests_cache_only.xml
& .\.venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --final-suite --label cache_only
& .\.venv-baseline\Scripts\python.exe -B tools\report_phase9.py
```

Existing evidence uses exclusive writes; reproduce in a separate checkout/output workspace rather than overwriting it. Copy the current code/environment, source clips, model assets and Phase 2–8 evidence, plus results/phase9_reference, results/phase9_selection.json, results/phase9_selection_cache_only.json, docs/phase9_before_hashes.json, docs/phase9_cache_only_protocol.json and the retained failed combined result. Keep the archived reference/runtime/framework sources; their hashes identify the failed combined and final cache-only configurations. Omit the generated Phase 9 run/parity/probe folders, native archives, experiment/master results and final reports from that fresh workspace. The retained reference sources supply the original runtime after production promotion; their hashes are recorded in the preservation evidence. Final A/B baseline overrides exist only in tools, and production workers use the selected clean runtime. Test results and parity gates must be rerun after any runtime edit.

## Resume-safe wording

Reduced CPU-only dashboard application median processing latency from 34.07 to 31.15 ms (8.6% reduction in randomized paired local video benchmarks), preserving 699/699 exact frozen outputs and unchanged policy/event behavior. Full application offline processing with the dashboard achieved 23.57–32.09 FPS (range of four per-source five-run medians), without GPU acceleration. This measures local video processing, not sustained physical webcam throughput or accuracy.

The unchanged raw headless control separately measured 28.38 to 28.50 ms P50 (-0.4% descriptive paired difference). This is not a claimed headless inference optimization. Historical Phase 8 values remain in the before/after table; the resume wording uses the contemporaneous matched render comparison to reduce attribution to machine-wide drift.

**Stopped after Phase 9.**
