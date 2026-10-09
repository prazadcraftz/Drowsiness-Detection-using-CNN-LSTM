# Interview guide: technically defensible explanations

**Why a temporal model instead of frame-only classification?** Eye closure and yawning evolve over time; a single frame gives less context for persistence and transient changes. The runtime collects 16 or 32 accepted observations and then predicts on every new accepted observation. This motivation does not prove superior performance against a frame-only model in an unrun comparison.

**Why handcrafted facial features?** Five landmark-derived channels (EAR-R, EAR-L, MAR, phi, theta) provide a compact, inspectable input and keep local CPU inference modest. Their legacy color/head-pose semantics were retained because the saved models learned that exact contract. This is a compatibility choice, not a claim that the features are universally robust.

**Why split videos before constructing windows?** Overlapping windows from one video share observations. Splitting windows at random can put near-duplicates in train and test; splitting whole source videos first prevents that specific leakage. Phase 11 also fits normalization on train videos only and checks source/observation identities.

**What was wrong with the historical evaluation?** It used a random overlapping-window split and whole-corpus normalization, making its generalization figure unreliable. The clean Phase 11 protocol used a fixed manifest, source-level split, train-only normalization, validation-selected checkpoint, and one held-out test. It is video-independent; subject independence cannot be claimed because subject IDs are unavailable.

**Why reject class weighting?** Phase 12's development study did not justify adopting it under the chosen quality checks. The final candidate kept unweighted cross-entropy; it was not changed after seeing a final test result. See the [balancing study](PHASE12_CLASS_BALANCING.md).

**Why LSTM-only rather than CNN-LSTM?** In the development architecture/window studies, the LSTM-only N=32 configuration had the highest observed mean window macro-F1 (**0.949829**). This is an empirical development result, not a universal architecture theorem or a candidate final-test result. The quality-first choice was frozen after the Phase 14 development matrix was seen.

**Why N=32?** The Phase 14 grouped development study favored it; training used stride 16, while runtime uses rolling stride 1. It costs a longer 1.60 s nominal warm-up and a slower model-only forward pass than legacy on the Phase 16 CPU benchmark. The chosen trade-off is documented in [ML evaluation](ML_EVALUATION.md) and [performance](PERFORMANCE.md).

**Why does legacy remain the default?** It preserves the historical CLI and saved-model contract with **699/699 exact frozen outputs**. `lstm32` is an explicit development-selected recommendation, not a silent replacement. Its candidate parity was **64/64 exact** against direct Phase 15 inference.

**Why is LSTM32 recommended but not externally validated?** Its support comes from development cross-validation and integration checks. The Phase 11 test cohort had already been evaluated for the baseline, so it is not a pristine untouched final cohort for the selected candidate. There were zero candidate passes over those videos/caches in Phases 15/16. No candidate final-test number is claimed.

**Why does MediaPipe dominate latency?** Phase 8/9 stage measurements attributed about **83%** of summed raw processing time to detector work on the tested local-video workload. Both profiles still detect on every decoded frame; the model-only forward pass is only one part of the frame cost. That share is workload/machine-specific.

**Why not tune alert thresholds now?** The whole-video labels lack event-onset truth and cannot support defensible threshold optimization. The runtime kept its engineering defaults frozen during candidate integration, so behavior can be characterized without turning development fixtures into an unreported tuning set. [State machine](STATE_MACHINE.md) lists the defaults.
