# Driver Drowsiness Detection

A local, CPU-based driver-monitoring research prototype. It extracts five facial measurements from MediaPipe landmarks, classifies rolling observation sequences, and turns smoothed predictions into driver states, timed events, dashboard annotations, and optional SQLite records. It accepts a webcam or a local video. This is an engineering prototype, **not a medically or clinically validated safety device**.

## Highlights

- Two explicit temporal profiles share one runtime: historical **CNN-LSTM16** (`legacy`, CLI default) and development-selected **LSTM-only32** (`lstm32`, recommended for an explicit trial).
- Face detection runs on each decoded frame; timestamp-aware sampling targets 20 accepted observations/s. Missing faces are tracked rather than filled with invented observations.
- Bounded probability smoothing, a seven-state policy, event/session metrics, dashboard rendering, optional nonblocking local audio, and optional SQLite session/event storage.
- Video-independent clean-baseline evaluation, development cross-validation for model selection, and exact frozen-output regression checks. [Evaluation boundaries](docs/ML_EVALUATION.md) and [limitations](docs/LIMITATIONS.md) are essential context for the metrics below.

## Quickstart (Windows)

From the repository root, use Python 3.12 and the pinned [runtime dependencies](requirements.txt). The existing project environment, if present, is `.venv-baseline`. To create a new one:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The examples below use `python` after activating the environment (or replace it with `.\.venv\Scripts\python.exe`). Camera `0` requires a connected, accessible webcam; a local file is easier for a headless demo. Press **q** or **Escape** to close the display.

```powershell
python main.py --source 0 --model-profile legacy
python main.py --source 0 --model-profile lstm32
python main.py --source 'path/to/video.avi' --model-profile lstm32
python main.py --source 'path/to/video.avi' --model-profile lstm32 --no-display --no-audio
python main.py --source 'path/to/video.avi' --model-profile lstm32 --no-audio --save-output --output '.cache/demo_lstm32_01.avi'
python main.py --source 'path/to/video.avi' --model-profile lstm32 --no-display --no-audio --sqlite --db-path '.cache/demo_sessions_01.db'
python tools/query_sessions.py '.cache/demo_sessions_01.db'
```

`--no-display` suppresses OpenCV GUI calls and also disables local audio; `--no-audio` can be used independently. `--output` enables saving and accepts `.avi`, `.mp4`, or `.mov`; it must name a **new** file distinct from the input. `--db-path` requires `--sqlite`; use a new demo database path. These commands do not target canonical `results/` evidence. Full options: `python main.py --help`. For setup, exact artifacts, and safe verification commands, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Model profiles

| | CNN-LSTM16 (`legacy`) | LSTM-only32 (`lstm32`) |
|---|---|---|
| Role | Historical compatibility; **CLI default** | Development-selected recommendation; **explicit opt-in** |
| Architecture / parameters | CNN-LSTM / **246,595** | LSTM-only / **209,667** |
| Accepted observations per window | 16 | 32 |
| Nominal warm-up at 20 Hz | 0.80 s accumulation; 0.75 s first-to-last span | 1.60 s accumulation; 1.55 s first-to-last span |
| Model-only P50, matched Phase 16 CPU | 2.142 ms | 3.749 ms |
| Raw pipeline P50, matched Phase 16 CPU | 28.460 ms | 30.088 ms |
| Headless application P50, matched Phase 16 CPU | 28.745 ms | 30.381 ms |

These are local-video, machine-specific timings; [Performance](docs/PERFORMANCE.md) defines each boundary and includes the full frame loop. The two profiles use separate checkpoints and normalizers. The `lstm32` model's **0.949829 mean window macro-F1 is development cross-validation**, not a final test score. The **94.1909% window test accuracy and 0.941808 macro-F1 belong only to the clean Phase 11 CNN-LSTM16 baseline**, which is distinct from the historical production checkpoint. The LSTM32 candidate has no pristine external final-test estimate. See [ML evaluation](docs/ML_EVALUATION.md) and the [claim registry](docs/CLAIMS.md).

## Demo

Run one of the local-video commands above with a video file you are authorized to use. The repository does not bundle video datasets or recorded demonstrations. The dashboard shows the selected model profile, sequence readiness, raw and smoothed class predictions, driver state, facial features, and session/event counters. It updates per decoded frame; predictions update after each accepted observation once the model window is ready. The historical [Phase 16 integration report](docs/PHASE16_RUNTIME_INTEGRATION.md) documents local characterization separately from accuracy evidence.

## Repository map

```text
main.py                 Local CLI and frame loop
driver_monitor/vision/  Video source, face detector, five features
driver_monitor/inference/  Profiles, normalizer, temporal buffer, policy
driver_monitor/metrics/     Session and event accounting
driver_monitor/alerts/      Local warning audio
driver_monitor/ui/          Annotated dashboard
driver_monitor/storage/     Optional SQLite store
evaluation/             Split, training, and experiment code
models/                 Frozen research checkpoints and metadata
docs/                   Architecture, evaluation, evidence, guides
tests/                  Unit, integration, and parity checks
```

Start with [Architecture](docs/ARCHITECTURE.md), [State machine](docs/STATE_MACHINE.md), [SQLite](docs/SQLITE.md), [Testing](docs/TESTING.md), and [Limitations](docs/LIMITATIONS.md). The [engineering summary](docs/ENGINEERING_SUMMARY.md) traces the project decisions; the [interview guide](docs/INTERVIEW_GUIDE.md) explains what the results support.

For a concise project overview and reusable application material, see the [portfolio brief](docs/PORTFOLIO_BRIEF.md), [resume bullets](docs/RESUME_BULLETS.md), [GitHub/application copy](docs/GITHUB_DESCRIPTION.md), [interview pitch and Q&A](docs/INTERVIEW_PITCH.md), and [STAR stories](docs/STAR_STORIES.md).

The GitHub source package excludes local video corpora, virtual environments, caches, and bulky generated experiment archives. Reproducing full training, video replay, or benchmark studies requires the original data and phase artifacts, which are not bundled.
