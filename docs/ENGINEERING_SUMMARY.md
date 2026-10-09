# Engineering summary: Phases 1–16

The project moved from an opaque historical CNN-LSTM script to a measured local monitoring pipeline, while keeping a frozen compatibility path. The Phase 2 audit established observable baseline behavior; Phase 3 extracted an engine behind exact replay checks. Phase 4 unified webcam and local-video input. Phase 5 added timestamp-aware 20 Hz accepted-observation sampling and bounded temporal windows. Phase 6 defined probability smoothing and state transitions; Phase 7 added event/session accounting, a dashboard, and local audio requests.

Phase 8 measured the CPU path and identified face detection as the largest raw cost. Phase 9 accepted a pixel-exact dashboard cache after a matched gate: render-application P50 improved **34.069 → 31.146 ms (8.58%)**. It did not establish a model-inference speedup. Phase 10 added optional local SQLite sessions, events, aggregates, and model provenance without raw-frame or per-frame prediction tables.

The ML work then addressed evaluation quality. Phase 11 replaced random overlapping-window evaluation and whole-corpus normalization with a video-independent split, train-only normalization, and validation-selected clean CNN-LSTM16 checkpoint. Its **482-window / 27-video held-out test** produced **94.1909% window accuracy** and **0.941808 macro-F1**. The 100% video result concerns only those 27 videos; no subject IDs or event-onset ground truth were available. The clean baseline checkpoint is distinct from the historical runtime `legacy` asset.

Phase 12 did not support class balancing. Phase 13 compared architectures; Phase 14 compared window lengths under grouped **development** cross-validation. After those results, Phase 15 froze LSTM-only32 as the quality-first development candidate, refitted it on all 153 development videos, and verified deterministic repeat behavior in its recorded CPU environment. Its **0.949829 mean window macro-F1 is development cross-validation**, not final test. Phase 16 integrated it as an explicit profile in the shared runtime, preserved `legacy` as CLI default, verified **699/699 legacy** and **64/64 candidate** exact parity, characterized six development videos, and measured profile latency. No pristine external candidate generalization estimate exists.

The design is local and CPU-only, with optional saved video and SQLite storage. Its thresholds are engineering defaults; it is not a clinically validated safety system. For detailed methods and boundaries, see [architecture](ARCHITECTURE.md), [ML evaluation](ML_EVALUATION.md), [performance](PERFORMANCE.md), [testing](TESTING.md), and [limitations](LIMITATIONS.md).

## Resume-safe candidate bullets

- Built a local CPU-only driver-monitoring pipeline using MediaPipe facial landmarks and temporal sequence models, with timestamp-aware sampling, missing-face recovery, smoothing, event tracking, and nonblocking alert requests.
- Developed a video-independent ML evaluation pipeline with train-only normalization; the **clean CNN-LSTM16 baseline** achieved **94.19% window-level test accuracy and 0.9418 macro-F1** on 27 held-out videos.
- Compared temporal architectures and window lengths under grouped **development cross-validation**, selecting a **209,667-parameter LSTM-only32 candidate** with **0.9498 mean development macro-F1**.
- Added optional SQLite session/event persistence and a dual-profile runtime; verified **699/699 legacy frozen outputs** and **64/64 integrated candidate development outputs** by exact numerical parity.

These are candidate explanations, not final resume edits. The [claim registry](CLAIMS.md) gives safe phrasing and forbidden attributions.
