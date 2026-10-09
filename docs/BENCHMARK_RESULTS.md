# CPU benchmark baseline — Phase 8

**BASELINE measurements. No optimization was performed.** Results are specific to this machine and these offline video workloads.

## Machine and environment

CPU: 12th Gen Intel(R) Core(TM) i5-12450H; 8 physical / 12 logical CPUs; OS-visible RAM 15.73 GiB. OS: Windows-11-10.0.26220-SP0. Python 3.12.14 (baseline virtual environment).

The initial CIM physical-core query was unavailable under current permissions. A read-only post-benchmark GetLogicalProcessorInformationEx topology query recorded the physical count in phase8_hardware_supplement.json; original benchmark metadata remains unchanged.

Software: torch 2.14.0+cpu, opencv-contrib-python 5.0.0.93, mediapipe 1.0.1, numpy 2.5.3.

Device CPU; CUDA available: False; MKLDNN enabled: True; Torch intra-op/inter-op 8/8; OpenCV threads 12; affinity mask 4095.

Active Windows power scheme: Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced). Power status: {'ac_line_status': 1, 'battery_flag': 9, 'battery_percent': 100} (AC line 1 = connected; battery flag 128 = no battery). This reads the active scheme, not the Windows UI power-slider setting. Frequency/temperature were not logged or controlled. Other processes were not stopped or controlled. Frequency, thermals and background load can affect results.

## Methodology and repetitions

109 sequential jobs, each in a fresh interpreter: 40 instrumented pipeline runs (four sources × five repetitions × headless/render), 40 actual uninstrumented main.run controls with the same repetitions/modes, 20 detector-only runs, five model-only runs, two additional frozen-source parity runs, one encoding probe, and one 10,000-frame memory run. Randomized source/workload ordering uses fixed seed 20260926; full execution order is retained. The long memory diagnostic runs last. No slow run was discarded.

Headless runs execute raw inference, decisions, metrics and disabled audio requests. Render runs add the unchanged dashboard but do not refresh a GUI. Encoding probe adds MJPG AVI writing. Uninstrumented render controls use the original main.run display branch with only GUI calls replaced by no-ops; rendering/policy/metrics remain intact. No playback occurs in any run.

Pipeline steady state excludes every stage sample from frames up to and including the first 20 sampled valid observations. No-face/unsampled frames do not advance this warm-up counter. Warm-up records remain in raw output but are excluded from steady statistics. First-call/model/allocator effects are retained separately rather than silently removed. Detector-only uses the first 20 valid detections (equivalent on these 20-FPS sources). Isolated model workloads each discard 20 warm-up calls, then cover all 699 frozen prebuilt (1,5,16) tensors.

Timers use perf_counter_ns; integer nanoseconds and frame IDs are preserved without rounding. Reported milliseconds use linear-interpolated P50/P90/P95/P99 and population standard deviation. Per-run medians, source-specific variability, pooled distributions, sample counts, raw records, CPU and RSS are stored in JSON. Pooled percentiles are sample-weighted, not averages of run percentiles.

Forwarding wrappers add observer cost to parent stages. Timers end before parity comparisons/RSS sampling; whole instrumented loop throughput still includes those operations. Actual main.run controls remove stage wrappers, raw comparisons and RSS sampling, and report full local-application throughput independently. These are different measurement populations, not an optimization comparison.

Stage boundaries: frame_decode includes VideoSource.read decode plus packet/timestamp creation; mediapipe_detect includes existing BGR-as-SRGB Image creation and landmark-result conversion; feature_extraction includes the unchanged five-feature function; phi/theta share one pose call. tensor_creation covers torch.tensor conversion; unsqueeze remains in sequence_other_residual. result_constructor times the dataclass call itself; validation, argmax/confidence, copies and inference-mode bookkeeping are in sequence_other_residual. Nested inclusive stages must not be summed with their children.

## Steady-state stage latencies

Values are per invocation, in milliseconds. Headless/render each have five runs per source. Parents/children overlap; differing counts reflect which stages run on missing-face frames.

| Stage | Samples | Mean | P50 | P95 | P99 | Std |
|---|---|---|---|---|---|---|
| frame_decode | 2320 | 1.799 | 1.819 | 2.125 | 2.731 | 0.4806 |
| mediapipe_detect | 2320 | 23.548 | 23.500 | 26.005 | 33.525 | 2.898 |
| feature_extraction | 2265 | 1.866 | 1.910 | 2.148 | 2.617 | 0.3538 |
| ear_right | 2265 | 0.0792 | 0.0787 | 0.0961 | 0.1279 | 0.0212 |
| ear_left | 2265 | 0.0395 | 0.0405 | 0.0466 | 0.0666 | 0.0127 |
| mar | 2265 | 0.0378 | 0.0393 | 0.0437 | 0.0614 | 0.0081 |
| pose_phi_theta | 2265 | 1.661 | 1.704 | 1.918 | 2.306 | 0.3338 |
| normalization | 2265 | 0.0264 | 0.0258 | 0.0332 | 0.0627 | 0.0115 |
| temporal_policy | 2320 | 0.0372 | 0.0365 | 0.0460 | 0.0626 | 0.0218 |
| buffer_append | 2265 | 0.0132 | 0.0130 | 0.0162 | 0.0272 | 0.0044 |
| buffer_window | 2265 | 0.0241 | 0.0237 | 0.0304 | 0.0586 | 0.0065 |
| tensor_creation | 2265 | 0.0783 | 0.0748 | 0.1098 | 0.1492 | 0.0250 |
| cnn_lstm_forward | 2265 | 2.312 | 2.169 | 2.882 | 4.593 | 2.181 |
| softmax | 2265 | 0.0194 | 0.0193 | 0.0244 | 0.0356 | 0.0036 |
| result_constructor | 2320 | 0.0086 | 0.0084 | 0.0109 | 0.0161 | 0.0032 |
| sequence_other_residual | 2265 | 0.3834 | 0.3761 | 0.4684 | 0.5850 | 0.0658 |
| sequence_inference | 2265 | 2.801 | 2.648 | 3.458 | 5.244 | 2.206 |
| inference_pipeline | 2320 | 28.329 | 28.242 | 31.475 | 40.869 | 4.632 |
| smoothing | 2265 | 0.1541 | 0.1509 | 0.1938 | 0.2447 | 0.0260 |
| state_machine | 2320 | 0.0615 | 0.0615 | 0.0811 | 0.1053 | 0.0140 |
| decision_layer | 2320 | 0.2881 | 0.2850 | 0.3578 | 0.4541 | 0.0523 |
| session_metrics | 2320 | 0.0245 | 0.0246 | 0.0304 | 0.0433 | 0.0049 |
| audio_request | 2320 | 0.0091 | 0.0090 | 0.0113 | 0.0192 | 0.0022 |
| overlay | 2320 | 4.232 | 4.356 | 4.847 | 6.497 | 0.7724 |
| application_processing | 2320 | 28.691 | 28.611 | 31.884 | 41.463 | 4.662 |
| frame_loop | 2320 | 30.497 | 30.425 | 33.812 | 43.715 | 4.852 |

The overlay row comes from render mode; other rows above come from headless mode. P90/min/max and separate rendering-mode values are in the raw JSON. A state-machine or smoothing row is a child of decision_layer; feature substeps are children of feature_extraction.

## Isolated model and detector

| Prepared-tensor workload | Samples | P50 | P95 | P99 | Mean | Std |
|---|---|---|---|---|---|---|
| model_only_forward | 3495 | 1.635 | 2.141 | 2.796 | 1.516 | 0.6211 |
| model_forward_and_result | 3495 | 1.962 | 2.406 | 2.768 | 1.821 | 0.5056 |

Forward-only starts immediately before model(prebuilt_tensor) and stops at returned logits. Forward+result additionally performs the existing finite check, softmax, argmax/confidence and InferenceResult/array copies over prepared input; no vision or tensor creation is timed. Its small postprocessing harness mirrors the production result expressions and verifies every frozen output; full-pipeline runs still call the unchanged production infer_sequence. Prepared frozen tensors use the layout produced by torch.tensor on archived arrays; production tensor preparation is measured separately. Value parity does not establish identical strides/copy costs across these workloads.

| Detector-only population | Samples | P50 | P95 | P99 | Mean | Std |
|---|---|---|---|---|---|---|
| combined | 2320 | 26.480 | 38.074 | 47.600 | 27.897 | 7.604 |
| face present | 2265 | 26.519 | 38.224 | 47.661 | 27.993 | 7.637 |
| no face | 55 | 22.303 | 33.001 | 34.021 | 23.933 | 4.578 |

These detector-only runs decode the same source frames before starting their detect timer and never execute feature calculations/model/policy. No-face population is small and comes only from Yawning_60; tail estimates for those 55 samples are limited.

## Full processing latency and 50-ms reference

| Mode | Boundary | Samples | P50 | P95 | P99 |
|---|---|---|---|---|---|
| headless | inference_pipeline | 2320 | 28.242 | 31.475 | 40.869 |
| headless | application_processing | 2320 | 28.611 | 31.884 | 41.463 |
| headless | frame_loop | 2320 | 30.425 | 33.812 | 43.715 |
| render | inference_pipeline | 2320 | 27.698 | 30.328 | 32.535 |
| render | application_processing | 2320 | 32.311 | 35.403 | 38.723 |
| render | frame_loop | 2320 | 34.037 | 37.245 | 40.718 |

inference_pipeline spans already-decoded frame → raw process_frame result. application_processing spans already-decoded frame → raw/policy/metrics/audio request/dashboard ready; encode is excluded. frame_loop adds source read and optional write. GUI refresh, arbitrary sleeps and asynchronous beep duration are never included.

| Mode | Stage | Relevant population | Samples | <=50 ms | P95 / 50 | P99 / 50 |
|---|---|---|---|---|---|---|
| headless | inference_pipeline | all steady decoded frames | 2320 | 99.57% | 0.6295 | 0.8174 |
| headless | inference_pipeline | prediction-bearing steady frames | 2265 | 99.56% | 0.6303 | 0.8228 |
| headless | application_processing | all steady decoded frames | 2320 | 99.57% | 0.6377 | 0.8293 |
| headless | application_processing | prediction-bearing steady frames | 2265 | 99.56% | 0.6385 | 0.8329 |
| render | inference_pipeline | all steady decoded frames | 2320 | 99.87% | 0.6066 | 0.6507 |
| render | inference_pipeline | prediction-bearing steady frames | 2265 | 99.87% | 0.6067 | 0.6534 |
| render | application_processing | all steady decoded frames | 2320 | 99.78% | 0.7081 | 0.7745 |
| render | application_processing | prediction-bearing steady frames | 2265 | 99.78% | 0.7084 | 0.7759 |

The 50-ms reference corresponds to the unchanged 20-Hz model cadence. These offline observations do not establish webcam capture-to-display latency or a universal real-time guarantee.

## Offline throughput and between-run variability

| Source | Mode | Runs | Offline FPS P50 | Min | Max | Std | Observation FPS |
|---|---|---|---|---|---|---|---|
| Yawning\Yawning_1.avi | headless | 5 | 34.343 | 32.889 | 40.794 | 2.863 | 20.000 |
| Yawning\Yawning_1.avi | render | 5 | 28.891 | 26.843 | 39.701 | 5.625 | 20.000 |
| Yawning\Yawning_60.avi | headless | 5 | 49.469 | 32.407 | 50.856 | 8.187 | 18.382 |
| Yawning\Yawning_60.avi | render | 5 | 29.463 | 27.742 | 38.735 | 4.002 | 18.382 |
| Eye Close\Eye Close_60.avi | headless | 5 | 32.988 | 27.212 | 36.795 | 3.066 | 20.000 |
| Eye Close\Eye Close_60.avi | render | 5 | 29.089 | 28.277 | 39.819 | 4.406 | 20.000 |
| Normal\Normal_1.avi | headless | 5 | 33.122 | 31.998 | 36.882 | 1.702 | 20.000 |
| Normal\Normal_1.avi | render | 5 | 29.412 | 25.810 | 41.689 | 5.495 | 20.000 |

offline_processing_fps is decoded frames / processing wall seconds from actual main.run, including all decoded/warm-up frames and source reads, but excluding initialization. It is neither webcam FPS nor model FPS. Effective observation FPS counts sampled valid-face observations over source timestamp span. The target remains 20 Hz; the three continuously visible sources average 20.0 observations/sec, while Yawning_60 averages 18.382 because of no-face gaps. Individual throughput/control summaries are retained.

Per-source run median variability for application_processing (ms):

| Source | Mode | Runs | Per-run P50 | Between-run std |
|---|---|---|---|---|
| Yawning\Yawning_1.avi | render | 5 | 34.836, 34.005, 31.788, 31.629, 31.805 | 1.340 |
| Yawning\Yawning_60.avi | headless | 5 | 28.651, 27.542, 28.463, 28.082, 26.179 | 0.8873 |
| Eye Close\Eye Close_60.avi | headless | 5 | 29.119, 29.245, 29.404, 29.391, 28.413 | 0.3657 |
| Eye Close\Eye Close_60.avi | render | 5 | 33.404, 31.884, 33.675, 32.790, 32.243 | 0.6752 |
| Normal\Normal_1.avi | render | 5 | 20.640, 34.032, 31.714, 34.081, 34.063 | 5.213 |
| Yawning\Yawning_1.avi | headless | 5 | 29.968, 28.343, 26.405, 28.547, 28.163 | 1.136 |
| Normal\Normal_1.avi | headless | 5 | 28.298, 28.649, 29.612, 27.922, 30.147 | 0.8294 |
| Yawning\Yawning_60.avi | render | 5 | 30.218, 31.335, 30.929, 31.032, 31.770 | 0.5113 |

## CPU, RSS and startup

| Mode | Processing CPU average | Representative peak interval CPU |
|---|---|---|
| headless | 245.20% | 352.20% |
| render | 274.07% | 713.68% |

CPU = summed process CPU seconds / wall seconds × 100. Values above 100% reflect multiple cores; 1200% would represent all 12 logical CPUs fully occupied. Average above is weighted by instrumented processing wall time; interval peaks use periodic process-time deltas, not instantaneous system CPU peaks. Whole-worker averages (including imports/init), per-interval CPU and CPU time are separately retained. CPU figures apply to instrumented jobs; the reference main.run controls have no CPU-sampling observer.

| RSS boundary | Samples | Median MiB | Min MiB | Max MiB |
|---|---|---|---|---|
| startup | 40 | 39.363 | 38.867 | 40.230 |
| after_runtime_imports | 40 | 238.234 | 237.582 | 238.809 |
| after_initialization | 40 | 293.977 | 293.156 | 294.906 |
| processing | 80 | 317.438 | 311.242 | 323.312 |
| end_of_processing | 40 | 317.863 | 311.676 | 323.461 |
| after_cleanup | 40 | 290.789 | 282.395 | 300.805 |

Maximum reported pipeline peak working-set RSS: 323.477 MiB. Windows GetProcessMemoryInfo supplies WorkingSetSize (RSS) and PeakWorkingSetSize. Startup RSS is sampled after benchmark module imports but before heavy runtime imports; after-initialization RSS precedes frame processing. Steady periodic RSS is sampled every 50 frames. These whole-process figures include libraries, detector/model/native allocations, verification arrays and timing-record storage; they are not the 640-byte feature payload. No psutil or runtime dependency was added.

| Startup component | Samples | P50 ms | P95 ms |
|---|---|---|---|
| benchmark_module_import_ns | 40 | 483.132 | 725.351 |
| runtime_import_ns | 40 | 2978.860 | 6045.612 |
| Normalizer_construction_ns | 40 | 0.8267 | 1.278 |
| CNN_LSTM_construction_ns | 40 | 25.972 | 71.211 |
| FaceDetector_construction_ns | 40 | 121.600 | 214.583 |
| engine_initialization_other_ns | 40 | 20.473 | 38.645 |
| engine_initialization_ns | 40 | 190.343 | 317.398 |
| components_initialization_ns | 40 | 199.165 | 330.632 |
| first_call_mediapipe_detect | 40 | 30.593 | 49.654 |
| first_call_cnn_lstm_forward | 40 | 7.630 | 13.113 |
| first_call_overlay | 20 | 42.134 | 53.374 |

Engine initialization includes the component constructors above; its other residual includes checkpoint loading, CPU/eval setup and configuration/buffer setup. components_initialization includes engine, policy, metrics, audio manager and source construction. Those parent rows overlap. Runtime imports and benchmark module imports are measured separately; OS interpreter launch is not isolated, but full process-launch-to-exit wall time is retained for every job. Filesystem/OS caches are not flushed, so a fresh interpreter is not a cold machine/cache guarantee.

## 10,000-frame memory observation

One continuous initialized Phase 7 stack decoded 10000 real frames over 73 repeated Yawning_60 passes, including short missing-face gaps; rendering enabled and real audio disabled. Run duration 332.616 s; offline throughput 30.065 FPS. Source-clock timestamps are extended continuously across passes at nominal 20 Hz; no interpolation or extra observations are fabricated.

Detailed stage histories are disabled in this diagnostic. Only periodic resource samples every 100 frames and bounded session aggregates are retained. The source is reopened at EOF; this is a stress workload, not a continuation of the frozen parity evaluation across video seams. The original detector/model/policy/UI algorithms remain intact. RSS includes measurement/ctypes metadata and native allocator caches.

| RSS population | Samples | First MiB | Last MiB | Change MiB | Range MiB | Slope bytes/frame |
|---|---|---|---|---|---|---|
| after frame 2000 | 81 | 317.023 | 127.758 | -189.266 | 126.805–317.828 | -34843.075 |
| second half | 51 | 238.781 | 127.758 | -111.023 | 126.805–238.781 | -13879.246 |

Strictly monotonic RSS growth in second half: False; positive interval fraction 0.320. These are descriptive observations over one long run. RSS declined substantially during this run; Windows working-set residency can change independently of committed allocations. Private commit/allocator histories were not collected, so the cause of the decline is not established. No claim of 'no memory leak' is made, and no change was applied to force a flatter curve.

Final bounded payload: {'feature_buffer_entries': 16, 'feature_buffer_bytes': 640, 'smoother_entries': 5, 'smoother_bytes': 120, 'event_tracker_count': 5}. Event/feature histories remain bounded. Whole-process/native RSS is a different measurement.


## Measured bottleneck ranking

Disjoint raw components use summed steady invocation time / summed raw inference time. Per-call medians are shown separately; nested EAR/MAR/pose and sequence parents are omitted from the percentage sum. Residual rows account for unmeasured glue/observer overhead.

| Raw stage | P50 ms per call | Core time contribution |
|---|---|---|
| mediapipe_detect | 23.500 | 83.12% |
| cnn_lstm_forward | 2.169 | 7.97% |
| feature_extraction | 1.910 | 6.43% |
| sequence_other_residual | 0.3761 | 1.32% |
| raw_other | residual | 0.44% |
| tensor_creation | 0.0748 | 0.27% |
| temporal_policy | 0.0365 | 0.13% |
| normalization | 0.0258 | 0.09% |
| buffer_window | 0.0237 | 0.08% |
| softmax | 0.0193 | 0.07% |
| buffer_append | 0.0130 | 0.05% |
| result_constructor | 0.0084 | 0.03% |

Application-level contribution includes raw inference, decision, metrics, audio request, render and remaining glue without double counting:

| Mode | Stage | Application time contribution |
|---|---|---|
| headless | inference_pipeline | 98.74% |
| headless | decision_layer | 1.00% |
| headless | application_other | 0.14% |
| headless | session_metrics | 0.09% |
| headless | audio_request | 0.03% |
| render | inference_pipeline | 85.50% |
| render | overlay | 13.35% |
| render | decision_layer | 0.88% |
| render | application_other | 0.17% |
| render | session_metrics | 0.08% |
| render | audio_request | 0.03% |

Largest measured raw component: **mediapipe_detect**, 83.12% of summed core time. This conclusion follows measurements rather than the earlier expectation. No optimization recommendation or speedup comparison is made.


## Encoding probe and limitations

| Single-source encoding probe | Samples | P50 ms | P95 ms | P99 ms |
|---|---|---|---|---|
| overlay | 111 | 4.110 | 6.159 | 6.455 |
| VideoWriter.write | 111 | 3.884 | 4.517 | 4.602 |

The encoding probe uses Normal_1, same 640×480/20-FPS MJPG output and safe writer as Phase 7. The probe is one diagnostic run, not a definitive repeated encoding baseline; it is excluded from pooled headless/render comparisons. write latency measures VideoWriter.write return time, not delayed encoding flush/release or durable-disk completion. The annotated AVI lives in ignored .cache/phase8_encoding/.

Limitations: short clips yield modest per-run tail sample counts; no-face tail has only 55 isolated samples; 20 warm-up observations may not remove every hardware thermal/cache effect; process CPU peaks are periodic approximations; working-set RSS varies with OS/native allocation behavior; benchmarks are offline and have no GUI-refresh/audio-device latency; observer overhead remains in instrumented parents; no external background-load/frequency/thermal control; no independently calibrated clinical/event accuracy is evaluated. None of these measurements imply medical validity.

## Regression and evidence

Full Phase 2–8 suite: **316 passed, 0 failed/errors/skipped** in 194.211 s, run after benchmark collection. All 134 pre-existing files remain byte-identical. Frozen features, normalized observations, tensors/sequences/logits/probabilities/classes remain exact across all six instrumented sources: **699/699 unique windows, numeric difference 0.0**. Model-only runs verify all 699 outputs on each repetition.

Evidence: ../results/benchmark_baseline.json contains environment/configuration/hashes/order, complete per-run samples, distributions and resource telemetry. ../results/benchmark_baseline_runs/ preserves individual job inputs/results/logs. phase8_validation.json records regression counts, preservation hashes, derived budget/contribution/RSS summaries.

## Exact commands

Run from the repository root; choose NEW output paths for reruns:

```powershell
& .\.venv-baseline\Scripts\python.exe -B benchmark.py --baseline-suite --runs 5 --warmup-frames 20 --include-render --long-frames 10000 --seed 20260926 --output results/benchmark_baseline.json
& .\.venv-baseline\Scripts\python.exe -B benchmark.py --source "Normal/Normal_1.avi" "Yawning/Yawning_1.avi" --runs 5 --no-render --output results/another_baseline.json
& .\.venv-baseline\Scripts\python.exe -B benchmark.py --source "Normal/Normal_1.avi" --runs 5 --include-encoding --output results/encoding_baseline.json
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase8_tests.xml
& .\.venv-baseline\Scripts\python.exe -B tools/report_phase8.py
```

**PHASE 8 PASS.** Stopped after baseline measurement. No Phase 9 optimization was begun.
