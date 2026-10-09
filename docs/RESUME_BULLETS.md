# Resume-ready project material

**Project:** Driver Drowsiness Detection · Python, PyTorch, OpenCV, MediaPipe, NumPy, SQLite, pytest. Pick **one** version below for a resume; the variants intentionally reuse the same evidence. The clean CNN-LSTM16 test metrics and LSTM-only32 development cross-validation metrics describe different models and evaluation settings. See the [claim registry](CLAIMS.md).

## A. Recommended two-bullet version (one-page resume)

- Built a local CPU-only driver-monitoring pipeline with MediaPipe facial landmarks, PyTorch temporal models, 20 Hz timestamp-aware sampling, missing-face recovery, smoothed state tracking, dashboard alerts, and optional SQLite session records.
- Rebuilt evaluation with video-level splits and train-only normalization: the **clean CNN-LSTM16 baseline** reached **94.19% window-level test accuracy and 0.9418 macro-F1** on 27 held-out videos; architecture/window studies selected a **209,667-parameter LSTM-only32 candidate** with **0.9498 mean development cross-validation macro-F1**.

## B. Three-bullet version (balanced)

- Engineered a local CPU-based webcam/video pipeline in Python, OpenCV, MediaPipe, and PyTorch, using five landmark features, rolling temporal windows, probability smoothing, and a seven-state driver policy.
- Corrected overlapping-window evaluation leakage with raw-video splits and train-only normalization; the **clean CNN-LSTM16 baseline** achieved **94.19% window-level test accuracy and 0.9418 macro-F1** on 27 held-out videos.
- Compared temporal architectures and 8/16/24/32-observation windows under **development cross-validation**, selecting LSTM-only32 at **0.9498 mean window macro-F1**; preserved **699/699 legacy** and **64/64 candidate** exact numerical parity checks.

## C. Four-bullet version (project-heavy resume)

- Built a reusable local CPU inference engine for webcam and local video, with MediaPipe landmarks, five engineered facial features, two PyTorch model profiles, and bounded timestamp-aware temporal state.
- Added smoothed driver states, timed events, a dashboard, nonblocking local alert requests, and optional SQLite sessions/events/aggregate metrics with model provenance.
- Implemented video-independent evaluation with source-first splitting and train-only normalization; the **clean CNN-LSTM16 baseline** reached **94.19% window-level test accuracy** and **0.9418 macro-F1** across 27 held-out videos.
- Ran controlled **development cross-validation** across architectures and window lengths, selecting a **209,667-parameter LSTM-only32** candidate at **0.9498 mean macro-F1**; verified **699/699** frozen legacy outputs and **64/64** candidate integration outputs exactly.

## ATS-safe software/AI placement version

- Developed a Python computer-vision pipeline with OpenCV, MediaPipe, NumPy, PyTorch LSTM/CNN-LSTM models, temporal sampling, driver-state logic, and optional SQLite persistence for local webcam/video monitoring.
- Improved model evaluation using video-level data splits, train-only preprocessing, and cross-validation; the **clean CNN-LSTM16 baseline** achieved **94.19% window-level test accuracy (0.9418 macro-F1)** on 27 held-out videos.
- Benchmarked CPU latency and validated software with pytest unit/integration tests, **699/699** exact legacy regression windows, and **64/64** exact LSTM-only32 integration sequences; the candidate's **0.9498 macro-F1 is development cross-validation**, not a test result.

## AI/ML-role version

- Audited and corrected leakage from overlapping-window splitting and whole-corpus normalization by fixing a raw-video split manifest and fitting preprocessing only on training videos.
- Trained a clean CNN-LSTM16 baseline that achieved **94.19% window-level held-out test accuracy and 0.9418 macro-F1** on 27 videos; subject identity and event-onset labels were unavailable.
- Ran grouped **development cross-validation** for class balancing, CNN/LSTM/GRU/BiLSTM ablations, and window lengths; selected LSTM-only32 with **0.9498 mean window macro-F1**, then reproduced its fixed-seed development refit on the recorded CPU environment.

## Software-engineering-role version

- Extracted a reusable Python inference engine from a legacy CNN-LSTM application, retaining an explicit `legacy` default and adding an opt-in LSTM-only32 model profile.
- Designed bounded temporal buffering, face-gap reset, deterministic state/event tracking, CLI controls, nonblocking audio requests, and local SQLite records with checkpoint/normalizer provenance.
- Verified backward compatibility with **699/699** exact frozen legacy windows and candidate integration with **64/64** exact verification sequences; the full Phase 17 pytest suite passed **578 tests**.

These bullets are grounded in the [Phase 11 clean evaluation](ML_EVALUATION.md), [Phase 16 integration](PHASE16_RUNTIME_INTEGRATION.md), [testing record](TESTING.md), and [limitations](LIMITATIONS.md). The candidate has no pristine final-test estimate, and the Phase 16 latency results are local-video, machine-specific measurements.
