# GitHub and application copy

## Short repository description

Local CPU driver-monitoring prototype using MediaPipe facial features, PyTorch temporal models, a state machine, and optional SQLite records.

## Suggested topics

`python` · `pytorch` · `opencv` · `mediapipe` · `lstm` · `computer-vision` · `driver-monitoring` · `sqlite` · `machine-learning` · `pytest`

## Pinned-repository description

Built a local webcam/video driver-monitoring pipeline with five MediaPipe-derived features, dual CNN-LSTM16/LSTM-only32 profiles, bounded temporal inference, smoothed driver states, alerts, dashboard, and optional SQLite. Clean baseline test and candidate development-CV evidence are reported separately.

## Five portfolio-card highlights

1. Local CPU temporal inference from webcam or video, targeting 20 accepted observations/s.
2. Video-independent baseline evaluation with raw-video splitting and train-only normalization.
3. Development ablations across temporal architectures and 8/16/24/32-observation windows.
4. Explicit legacy/default and development-selected LSTM-only32 runtime profiles.
5. Exact frozen-output parity: 699 legacy windows and 64 candidate integration sequences.

## One-line project descriptions

- **A — at most 120 characters:** Local driver monitoring with MediaPipe features, temporal models, state tracking, and SQLite.
- **B — at most 180 characters:** Built a local CPU driver monitor with MediaPipe, PyTorch temporal models, webcam/video input, smoothed alerts, dashboard, and optional SQLite.
- **C — at most 250 characters:** Local driver-monitoring prototype using five MediaPipe facial features, CNN-LSTM16 and LSTM-only32 profiles, timestamp-aware sampling, a seven-state policy, dashboard, and optional SQLite; evaluated with video-level splits.

## LinkedIn Projects paragraph

Built a local CPU-based driver-monitoring prototype in Python with OpenCV, MediaPipe, PyTorch temporal models, a seven-state policy, dashboard, and optional SQLite. I corrected an overlapping-window evaluation leak using video-level splits and train-only normalization; the clean CNN-LSTM16 baseline achieved 94.19% window-level test accuracy and 0.9418 macro-F1 on 27 held-out videos. A separate LSTM-only32 candidate was selected at 0.9498 mean **development cross-validation** macro-F1. This research prototype has not been validated as a safety device.

## Application form: at most 50 words

Built a local CPU driver-monitoring prototype using MediaPipe facial features and PyTorch temporal models. Added smoothed driver states, alerts, and optional SQLite. A clean CNN-LSTM16 baseline achieved 94.19% window-level test accuracy on 27 held-out videos; LSTM-only32 was selected separately through development cross-validation.

## Application form: at most 100 words

I built a local webcam/video driver monitor in Python, OpenCV, MediaPipe, and PyTorch. Five facial features feed rolling CNN-LSTM16 or LSTM-only32 sequences; probability smoothing and a seven-state policy drive dashboard alerts and optional SQLite session records. I replaced leakage-prone random-window evaluation with raw-video splits and train-only normalization. The clean CNN-LSTM16 baseline achieved 94.19% window-level test accuracy and 0.9418 macro-F1 on 27 held-out videos. LSTM-only32 was selected using development cross-validation (0.9498 mean macro-F1), without a pristine external final-test estimate.

## Application form: at most 150 words

I engineered a local CPU driver-monitoring prototype for webcam and video input using OpenCV, MediaPipe, NumPy, and PyTorch. It converts landmarks into five eye, mouth, and pose features, samples accepted observations at a 20 Hz target, and uses temporal models plus smoothing and a seven-state policy to track fatigue and alert episodes. A dashboard, nonblocking local alert requests, and optional SQLite records complete the application. I corrected an earlier evaluation leak by splitting raw videos before overlapping windows and fitting normalization on training videos only. The separately trained clean CNN-LSTM16 baseline achieved 94.19% window-level test accuracy and 0.9418 macro-F1 on 27 held-out videos. Development architecture and window studies selected an opt-in LSTM-only32 candidate at 0.9498 mean development cross-validation macro-F1. It has no pristine external final-test estimate; subject identity and event-onset labels were unavailable.

## Tech stack and evidence boundary

**Used:** Python, PyTorch, OpenCV, MediaPipe, NumPy, scikit-learn, Python `sqlite3`/SQLite, pytest. Supporting scripts use SciPy and Matplotlib for analysis/plots. The source of truth is the [README](../README.md), [evaluation](ML_EVALUATION.md), [performance](PERFORMANCE.md), [testing](TESTING.md), and [limitations](LIMITATIONS.md). The project is designed for local processing; no universal webcam-FPS or clinical-safety claim is made.
