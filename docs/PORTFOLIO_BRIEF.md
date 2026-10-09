# Driver Drowsiness Detection

**One sentence:** A local CPU-based driver-monitoring prototype that turns facial landmarks into temporal predictions, interpretable driver states, and optional session records.

## The problem

A single video frame can be ambiguous: a blink, a prolonged eye closure, and a momentary face-detection failure need different treatment. This project combines measurements across time and keeps the model's class prediction separate from an engineering policy that decides when to display fatigue or alert states. It is a research/engineering prototype, **not validated as a road-safety device**.

## What runs locally

The application accepts a webcam or local video. MediaPipe detects a face on each decoded frame, and the feature extractor computes five values in a fixed order: right and left eye aspect ratios (EAR-R and EAR-L), mouth aspect ratio (MAR), and two head-pose angles (phi and theta). Timestamp-aware sampling targets 20 accepted face observations per second. A bounded rolling window feeds a PyTorch temporal classifier; a five-prediction probability smoother and seven-state policy then track attentive, fatigue, suspected-drowsy, drowsy, alert, warm-up, and no-face states. Session metrics record logical events, while a dashboard shows readiness, features, predictions, and state. Local audio requests are nonblocking. Optional SQLite storage records sessions, event episodes, final aggregates, and model provenance without raw-frame or per-frame prediction tables.

There are two runtime profiles. The historical **CNN-LSTM16** model has 246,595 parameters and remains the CLI default so saved-model behavior stays compatible. The **LSTM-only32** model has 209,667 parameters and is a development-selected, explicitly opted-in recommendation. It first predicts after 32 accepted observations: 1.60 seconds of nominal accumulation at 20 Hz, or 1.55 seconds from first to last observation. Thereafter it predicts on every newly accepted observation. A face gap longer than 0.5 seconds clears stale context and requires 32 fresh observations; missing observations are never fabricated. The models' legacy color and head-pose input semantics were retained to preserve their learned contracts.

## How the ML evidence was built

The original evaluation randomly split overlapping windows and fitted normalization over the full corpus, which could leak nearly identical source observations and evaluation-set extrema into training. The corrected Phase 11 study split **raw videos before window creation**, fitted normalization on train videos only, fixed a source manifest, selected the checkpoint on validation data, and evaluated a separate held-out video cohort. The **clean CNN-LSTM16 baseline** achieved **94.1909% window-level test accuracy** and **0.941808 macro-F1** across 482 windows. It classified all **27 held-out videos** in that cohort, but 27 is too small to generalize the 100% video result broadly. Subject IDs were unavailable, so the **video-independent** split cannot establish independence between people. Whole-video labels do not establish event onset or alert accuracy.

Separate development studies tested class balancing, architecture, and observation-window length. Unweighted cross-entropy remained the choice. The LSTM-only architecture and 32-observation window produced **0.949829 mean window macro-F1 under development cross-validation**. Phase 15 froze that choice and refitted it on 153 development videos. The LSTM32 candidate has **no pristine external final-test estimate**; the Phase 11 test numbers belong only to the clean baseline, not this candidate.

## Engineering evidence and trade-offs

Exact frozen-output checks reproduced **699/699** legacy windows/classes and **64/64** direct-versus-integrated candidate verification sequences with maximum numerical difference 0.0. The Phase 17 suite passed **578 tests**, covering unit, integration, parity, and source-access guards. In a matched, headless Phase 16 local-video CPU benchmark, model-forward P50 was **2.142 ms** for legacy and **3.749 ms** for LSTM32; raw-pipeline P50 was **28.460 / 30.088 ms**. These are machine-specific offline-video measurements, not physical-webcam FPS guarantees. MediaPipe was the dominant raw stage in the earlier benchmark. A separate Phase 9 dashboard cache reduced matched render-application P50 from **34.069 to 31.146 ms (8.58%)**; that is a rendering/application gain, not a neural-model speedup.

## Try the repository

The [README](../README.md) has Windows setup and both profile commands. With the pinned environment active, a local development-video demo is:

```powershell
python main.py --source 'path/to/video.avi' --model-profile lstm32 --no-display --no-audio
```

For an annotated output, add `--save-output --output '.cache/portfolio_demo_01.avi'` with a **new** filename. Supply a local clip; the repository does not bundle recorded videos or full experiment archives. See the [architecture](ARCHITECTURE.md), [evaluation](ML_EVALUATION.md), [performance](PERFORMANCE.md), [claim registry](CLAIMS.md), and [limitations](LIMITATIONS.md) for the exact evidence boundaries. Diverse subject-labeled data, event-level annotations, external validation, and physical-camera testing remain future work.
