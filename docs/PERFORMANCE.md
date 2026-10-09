# CPU performance and measurement scope

These results come from local-video offline processing on one Windows CPU configuration. They measure software boundaries on that machine; they do **not** establish sustained physical-webcam FPS, latency on other devices, GUI refresh timing, or alert effectiveness. At the 20 Hz observation target, 50 ms is a useful per-frame reference, but it is not a universal real-time guarantee.

## Phase 8 baseline and Phase 9 dashboard change

Phase 8 measured the legacy pipeline with an unchanged runtime. Headless raw-pipeline P50 was **28.242 ms** and headless application P50 **28.611 ms**; render-mode raw P50 was **27.698 ms** and application P50 **32.311 ms**. The largest measured raw processing component was MediaPipe detection, **83.12%** of summed core processing time. These are separate workload modes and should not be subtracted as if they were a paired optimization experiment. [Phase 8 report](PHASE8_BENCHMARK_BASELINE.md) defines all boundaries and retained samples.

Phase 9 accepted a pixel-exact dashboard cache. In its **contemporaneous matched render-application comparison**, P50 fell from **34.069 to 31.146 ms**, an **8.58% improvement**. This is a dashboard/application rendering result, **not an 8.6% model-inference speedup**. Raw inference stayed substantially the same; an attempted raw change was rejected by its matched gate. See [Phase 9 report](PHASE9_CPU_OPTIMIZATION.md).

## Matched Phase 16 profile comparison

The two profiles were run through the same headless local-video path on four development clips, two repetitions per profile/source, **16 fresh processes** in seeded shuffled order. Detector, sampler, Torch 8/8 threads, source set, and timing instrumentation were matched; only profile changed. The measured steady-state P50 values are:

| Boundary, ms | Legacy CNN-LSTM16 | LSTM-only32 |
|---|---:|---:|
| Integrated model forward only | **2.142** | **3.749** |
| Raw inference pipeline | **28.460** | **30.088** |
| Headless application processing, decode excluded | **28.745** | **30.381** |
| Full frame loop including decode | **30.484** | **32.094** |

The raw pipeline includes detection, feature processing, and model path; application processing adds policy/metrics and other frame work. Full frame loop also includes source decode. These Phase 16 measurements were **headless, audio disabled, SQLite disabled, and rendering disabled**; they cannot quantify those optional paths. Warm-up and initialization were outside the steady-state latency distributions. LSTM-only32 has a larger model-only forward cost and a longer first-prediction warm-up: 32 accepted observations versus 16. Because MediaPipe dominates much of raw processing, the model-only difference is only part of full-pipeline cost. [Phase 16 benchmark evidence](../results/phase16/benchmark_matched.json) includes P95/P99 and raw samples; [integration report](PHASE16_RUNTIME_INTEGRATION.md) explains the protocol.

Do not combine the Phase 8, Phase 9, and Phase 16 numbers into one claimed speedup: they used different workloads, boundaries, or stages. Process CPU can exceed 100% when several cores contribute. The bounded 10,000-frame runs describe one finite memory trajectory and do not prove absence of a leak. [Limitations](LIMITATIONS.md) collects the practical caveats.
