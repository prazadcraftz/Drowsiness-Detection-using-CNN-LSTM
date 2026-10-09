# Four evidence-backed STAR stories

These are speaking outlines. Keep the metric's model, split, and measurement boundary attached when telling each story. The [claim registry](CLAIMS.md) and [limitations](LIMITATIONS.md) give the longer caveats.

## A. Discovering and correcting evaluation leakage

- **Situation:** The historical workflow made overlapping windows, randomly split those windows, and fitted normalization across the whole corpus. Shared observations and evaluation-set extrema could leak into training.
- **Task:** Produce a reproducible estimate for a clean CNN-LSTM16 baseline without using the old displayed result as a generalization claim.
- **Action:** Inventoried and hashed raw videos; fixed a source-level train/validation/test manifest before window creation; fitted the normalizer only on training videos; checked source/observation identity disjointness; selected the checkpoint on validation data and evaluated the held-out cohort once.
- **Result:** The **clean CNN-LSTM16 baseline** achieved **94.1909% window-level test accuracy and 0.941808 macro-F1** on 482 windows from **27 held-out videos**. The result is video-independent, with no subject-ID or event-onset guarantee. [Phase 11 evidence](PHASE11_LEAKAGE_SAFE_EVALUATION.md).

## B. Extracting a reusable inference engine without breaking legacy behavior

- **Situation:** The original pipeline combined decoding, feature extraction, normalization, sequence modeling, and presentation in a compatibility-sensitive application.
- **Task:** Make the runtime modular enough to support webcam/video input and later model-profile integration while retaining known production outputs.
- **Action:** Separated source, vision, inference, decision, metrics, alerts, UI, and storage boundaries; kept the five-feature and legacy preprocessing contracts; introduced bounded temporal state and explicit `legacy`/`lstm32` profiles; used frozen replay fixtures to gate changes.
- **Result:** `legacy` remains the CLI default with **699/699 frozen windows/classes exact**. The opt-in LSTM-only32 integration matched **64/64 development verification sequences** against direct model inference with maximum numerical difference **0.0**. [Phase 16 integration](PHASE16_RUNTIME_INTEGRATION.md) and [testing](TESTING.md).

## C. Measuring the CPU bottleneck and scoping an optimization claim

- **Situation:** End-to-end local-video latency included vision, temporal inference, policy, and dashboard work; a model-only timing would not explain the user-visible path.
- **Task:** Find the dominant stage and accept only an optimization supported by matched measurements and compatibility checks.
- **Action:** Instrumented CPU stages and identified MediaPipe detection at about **83% of summed raw processing time** in the tested Phase 8/9 workload. A raw-path idea failed its final matched gate, so it was not kept. A pixel-exact dashboard cache passed parity and matched render-application latency checks.
- **Result:** Matched render-application P50 changed from **34.069 to 31.146 ms**, an **8.58% dashboard/application improvement**. No neural-inference speedup was claimed; the result is machine-specific offline-video evidence. [Phase 9 evidence](PHASE9_CPU_OPTIMIZATION.md) and [performance](PERFORMANCE.md).

## D. Selecting a development candidate through controlled studies

- **Situation:** The project had a compatible CNN-LSTM16 runtime, but architecture and window-length choices had not been compared under one grouped development protocol.
- **Task:** Select one candidate without using the Phase 11 test cohort as a tuning set.
- **Action:** Studied class balancing, kept unweighted cross-entropy, compared CNN-only/LSTM-only/CNN-LSTM/CNN-GRU/CNN-BiLSTM architectures, and then evaluated 8/16/24/32-observation windows under grouped development cross-validation. After seeing the complete development matrix, froze a quality-first LSTM-only32 choice and refitted it on all 153 development videos.
- **Result:** LSTM-only32 reached **0.949829 mean window macro-F1 under development cross-validation**; its N=32 versus N=16 paired difference was **+0.017553** in mean F1 with **9/9 positive pairs**. It became an explicit runtime option, **not** an externally validated final-test model. [Phase 14 study](PHASE14_WINDOW_LENGTH.md), [Phase 15 candidate](PHASE15_FINAL_CANDIDATE.md), and [ML evaluation](ML_EVALUATION.md).
