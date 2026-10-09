# Phase 8 — CPU benchmark baseline

**PHASE 8 PASS.** 109 measured jobs; 316 tests passed; all 699 frozen raw windows remain exact. All 134 prior files, including the entire Phase 7 runtime, are unchanged.

## 1–2. Files created and modified

Created benchmark.py, driver_monitor/metrics/benchmark.py, tests/test_benchmark.py, tools/report_phase8.py; results/benchmark_baseline.json and individual job evidence; docs/BENCHMARK_RESULTS.md, this report, phase8_validation.json, preservation hashes/test XML and two scientific PNGs. Only ignored .cache smoke jobs and optional encoding AVI artifacts are additional development outputs. **No pre-existing file was modified. No dependency was added.**

## 3–10. Methodology, machine and run setup

The companion [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) records hardware/software versions, CPU/RAM/OS, affinity/power/background limitations, unchanged thread settings, representative source hashes, fixed seed/run order, warm-up policy, repetitions and independent workload definitions. Each of four representative sources has five instrumented and five uninstrumented runs in each of headless and render-only modes. Standalone model and detector jobs each have five repetitions. All raw timing samples/run results are retained.

## 11–23. Stage measurements

The companion stage table reports decode/packet acquisition, MediaPipe, whole features and EAR-R/EAR-L/MAR/joint phi-theta substeps, normalization, temporal sampling/gap policy, buffer append/window, tensor conversion, CNN-LSTM, softmax, constructor/residual result work, smoothing/state machine/decision, metrics, audio request, dashboard and the separate encoding probe. Every stage distribution has count/mean/P50/P90/P95/P99/min/max/population std and unrounded integer-nanosecond samples in JSON. Audio playback duration is excluded and real sound is disabled. Nested timing boundaries and residuals are defined explicitly.

## 24–27. Full-pipeline latency and throughput

| Mode | Raw P50/P95/P99 ms | Application P50/P95/P99 ms |
|---|---|---|
| headless | 28.242/31.475/40.869 | 28.611/31.884/41.463 |
| render | 27.698/30.328/32.535 | 32.311/35.403/38.723 |

Offline FPS and per-run variability are reported per source/mode from actual unchanged main.run controls; instrumented loop rates are retained separately. Target cadence remains 20 Hz. Effective valid observation FPS is 20.0 on the three continuously visible sources and 18.382 on Yawning_60 with no-face gaps; it is not wall-clock throughput. See the companion throughput table for all five runs/ranges/standard deviations.

## 28–31. CPU, RSS, long run and startup

CPU measurements use standard-library process time; RSS/peak working set use Windows APIs without psutil. The companion report separates startup, after imports/init, steady and end/cleanup RSS. Maximum measured pipeline peak RSS is 323.477 MiB. Process CPU percentages may exceed 100% because CPU seconds sum across cores. Imports, normalizer/model/detector construction and initialization residual are measured separately.

10,000 real frames were processed through inference, policy, analytics, disabled audio and rendering in 332.616 s. Final feature/smoothing/event payload remains bounded: {'feature_buffer_entries': 16, 'feature_buffer_bytes': 640, 'smoother_entries': 5, 'smoother_bytes': 120, 'event_tracker_count': 5}. Second-half RSS change -111.023 MiB; fitted slope -13879.246 bytes/frame. This is a measured one-run RSS observation, not proof of absence of a leak; measurement metadata/native caches are included. The full curve and post-warm-up trend statistics are retained.

## 32–33. Frame budget and bottleneck

The companion report gives both all-steady-frame and prediction-bearing-frame percentages within 50 ms, plus P95/P99-to-budget ratios for raw and application processing. No universal real-time guarantee is inferred from a mean.

Measured largest raw component: **mediapipe_detect**, 83.12% of summed core processing time. Full disjoint rankings are derived from observed timing contributions, with residuals and source-dependent invocation counts accounted for. No optimization was performed/recommended or speedup percentage claimed.

## 34–35. Raw parity and full regression

| Phase | Passed tests |
|---|---|
| Phase 2 | 9 |
| Phase 3 | 43 |
| Phase 4 | 53 |
| Phase 5 | 60 |
| Phase 6 | 64 |
| Phase 7 | 60 |
| Phase 8 | 27 |

**316 passed, 0 failed/errors/skipped**, after benchmark collection. Existing 289 Phase 2–7 tests are byte-identical. Six-source instrumented checks cover **699/699 unique raw windows**, with unchanged features, sequences, tensors, logits, probabilities and classes (maximum numerical difference 0.0). Checkpoint, normalization, MediaPipe settings, BGR/RGB behavior, resolution, sampling, thread counts, policy/audio/UI algorithms and source videos are unchanged.

## 36–37. Limitations and reproduction

Hardware-specific offline results; no thermal/frequency/background-load controls; modest clip/no-face tail sample counts; observer overhead in detailed timings; Windows RSS instrumentation/allocator metadata; one long-memory run; one nondefinitive encoding probe; no GUI-refresh or audible-device latency. These are engineering measurements without medical/event accuracy claims. Exact reproducible commands and every sample/reference are in BENCHMARK_RESULTS.md and results/benchmark_baseline.json. Existing outputs are never overwritten silently; interrupted/error jobs retain individual evidence and stop the experiment.

## 38. Decision

**PHASE 8 PASS.** CPU baseline established without production modifications or optimization. Stopped after Phase 8. Phase 9 and later phases, retraining, database persistence and alternative models have not begun.
