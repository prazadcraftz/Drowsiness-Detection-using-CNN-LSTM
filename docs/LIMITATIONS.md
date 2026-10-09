# Limitations and intended use

This is a **research/engineering prototype, not a medically or clinically validated drowsiness device**. It must not be treated as a guaranteed safety monitor.

- The dataset is small and custom. It has no reliable subject IDs; the evaluation is **video-independent**, and people or scenes may overlap across videos. New-driver generalization has not been established.
- Labels cover whole-video actions. There is no trustworthy event-onset, frame-level, or alert-outcome ground truth, so event precision/recall and detection delay were not validated.
- The clean Phase 11 CNN-LSTM16 baseline used only **27 held-out videos**. Its 100% video result is a finite-cohort observation, not a general performance guarantee.
- The Phase 15/16 LSTM-only32 candidate was selected on development cross-validation and fitted on 153 development videos. It has **no pristine external final generalization estimate**. Phase 11 test metrics must not be transferred to it.
- The feature pipeline intentionally retains legacy color and head-pose semantics, plus historical normalization behavior, to preserve saved-model compatibility. These quirks may limit transferability.
- State thresholds and timers are **engineering defaults** retained from a frozen policy; they were not tuned or clinically validated for the candidate.
- Face-detection failures remove observations; long gaps reset the temporal buffer. LSTM-only32 requires 32 accepted observations before its first prediction, a nominal **1.60 s accumulation** at 20 Hz, longer than the legacy 16-observation profile.
- Phase 8/9/16 performance measurements are CPU- and machine-specific offline-video results. Physical webcam performance, GUI refresh, audible playback latency, and broad hardware behavior were not established by those benchmarks.
- Optional SQLite stores session/event summaries and provenance without raw frames or per-frame prediction tables. The application is **designed for local processing** and requires no cloud inference, but this is not an absolute privacy or security guarantee. Local video, annotated output, database files, and operating-system access still need appropriate handling.
- Finite long-run memory checks show bounded in-process feature/smoothing/event structures under tested conditions; one run does not prove absence of all memory leaks.

See [ML evaluation](ML_EVALUATION.md), [performance](PERFORMANCE.md), and the [claim registry](CLAIMS.md) before quoting any project result.
