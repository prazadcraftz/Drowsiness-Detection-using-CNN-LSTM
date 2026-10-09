# Phase 2 frozen canonical baseline

**Status: PASS. Phase 3 has not been started.**

The baseline is the unchanged `inference.py:main`, executed on real local videos.
No production Python, config, model, preprocessing, checkpoint, or class-mapping
file was modified. The capture tool observes the original loop with line tracing,
feature timers, and model hooks; it does not copy its algorithms into a new
inference engine. The four legacy alternatives were not used.

## Tested local environment

Created `.venv-baseline` from the available bundled Python 3.12.14 using `venv
--copies`. The broken Python 3.14 `.venv` remains untouched. Package downloads
were installed into the new local environment; the initial sandbox network
restriction was resolved through approved package-install tool escalation.
Versions below are installed and import-tested, not proposed pins.

| Component | Tested version |
| --- | --- |
| Python | 3.12.14, Windows x64 |
| PyTorch | 2.14.0+cpu |
| OpenCV distribution | opencv-contrib-python 5.0.0.93 |
| `cv2.__version__` | 5.0.0 |
| MediaPipe | 1.0.1 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| pytest | 9.1.1 |
| matplotlib | 3.11.2 |
| tqdm | 4.70.1 |
| pyserial | 3.5 |
| pip bootstrap | 25.0.1 |

Hardware/OS: Intel Core i5-12450H, 12 logical processors, Windows 11
`10.0.26220`. Inference device is explicitly CPU. CUDA availability is **False**.
Unchanged default thread settings: PyTorch intra-op 8, inter-op 8; OpenCV 12.
MKLDNN is enabled; deterministic-algorithm forcing is disabled. Seed 0 affects
model initialization only; the saved state dict replaces those initialized
weights before inference. No CPU thread optimization was performed.

`torch`, `cv2`, `mediapipe`, `numpy`, `sklearn`, and `pytest` imports succeeded.
The checkpoint loaded with `map_location='cpu'`, `model.eval()` worked, output
shape was `(1,3)`, and the original detector asset loaded successfully.
`pip check` passed. An offline `pip install --dry-run --no-index -r
requirements-dev.txt` confirmed that every pinned requirement is satisfied.

`requirements.txt` pins the runtime dependency graph; `requirements-dev.txt`
includes it and pins the test packages. MediaPipe depends on contrib OpenCV,
so only `opencv-contrib-python` is retained as the `cv2` provider. The redundant
standard distribution selected during initial installation was removed and the
selected contrib wheel was reinstalled before capture. Do not install both
OpenCV providers into this environment. These pins were tested on Windows x64
and Python 3.12; other OS/Python combinations have not been verified.

Compatibility references used while establishing the environment: [PyTorch
local installation](https://pytorch.org/get-started/locally/) and [MediaPipe
Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python).
The tested runtime, rather than the historical unknown training environment,
defines this new reproducible baseline.

## Artifact verification

All four required SHA-256 identities matched `CURRENT_ARCHITECTURE.md` before
model/detector execution and after capture. Fixtures also record all canonical
source/config hashes and each source video's full-file hash. Baseline capture
fails on any required artifact mismatch; it refuses existing fixture outputs
unless replacement is explicitly requested.

| Artifact | Verified SHA-256 |
| --- | --- |
| `saved_model.pt` | `162426d6292a87162188b70d7bac6704e5bf217f8ff1fe5eaafc41085aa0c551` |
| `norm.npy` | `54b2a1fc66cc8eab3b68ba2b5fe081558aabc6e48c79f752d1d49026494e1900` |
| `class_idx.npy` | `4f06c199cfdb8031a6caf92501e19a073dcf08399c4661c3f9336c0fb81818b2` |
| `face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The normalization array is finite float64 `(5,2)`, with positive denominators.
The hash-verified mapping pickle decodes exactly to Normal=0, Yawning=1,
Eye Close=2. No model asset was rewritten. The verified model remains
246,595 parameters and a 991,798-byte checkpoint.

## Fixture coverage and data schema

| Source | Decoded frames | Valid observations | Full windows |
| --- | ---: | ---: | ---: |
| `Normal/Normal_1.avi` | 131 | 131 | 116 |
| `Normal/Normal_60.avi` | 140 | 140 | 125 |
| `Yawning/Yawning_1.avi` | 135 | 135 | 120 |
| `Yawning/Yawning_60.avi` | 137 | 126 | 111 |
| `Eye Close/Eye Close_1.avi` | 116 | 116 | 101 |
| `Eye Close/Eye Close_60.avi` | 141 | 141 | 126 |
| Total | **800** | **789** | **699** |

Every video is 640×480, 20 FPS. Every frame is represented, including the first
15 valid warm-up observations per source and missing-face frames. All 699
ready windows are stored consecutively at stride 1 of valid observations.
Raw predictions cover Normal **321**, Yawning **251**, Eye Close **127**.
These counts describe the model's outputs, not correctness against ground truth.

Four actual face-loss runs, totaling 11 frames, occur in `Yawning_60.avi`:
6 frames at 2.25–2.50 s, 1 at 2.90 s, 1 at 4.85 s, and 3 at 5.40–5.50 s.
Their endpoint spans differ from nominal missing-observation duration; the
longest run has six observations, nominally 0.30 s at 20 FPS. No sustained
long-gap fixture or annotated transition/onset fixture is included. These
selected videos provide whole-video action labels only; no gap, frame class,
or event timing was fabricated. Raw class changes are observable but are not
ground-truth event annotations. The legacy buffer deliberately persists across
these gaps; no new reset policy exists in this phase.

`tests/fixtures/manifest.json` records the environment, source filenames/hashes,
frame-row ranges, video metadata, artifact/source/tool hashes, shape/dtype schema,
feature order, class mapping, and observed face-loss runs. Per-frame indices,
timestamps, and source IDs are stored losslessly in the linked NPZ rather than
duplicated as hundreds of JSON objects. Indices are zero-based; timestamps are
the actual OpenCV `CAP_PROP_POS_MSEC` after read, converted to seconds. No FPS
sampling, timestamp fallback, or rescaling is applied.

`tests/fixtures/legacy_baseline.npz` is 2,871,106 bytes and has SHA-256
`068bc2db30a0a6ca5c1c7261d2b63d2281ed6696c21dc13ef1fe893f0b678fbb`.
It stores numeric/string arrays with no object pickle and no duplicated videos:

| Array | Shape | Meaning |
| --- | --- | --- |
| `source_ids`, `frame_indices`, `timestamps_seconds`, `frame_sha256` | `(800,)` each | Source/frame/media identity and unannotated decoded-frame hash |
| `face_detected`, `sequence_fill`, `warming_up`, `frame_window_indices` | `(800,)` each | Original per-frame availability/warm-up/window lookup |
| `raw_features` | `(800,5)` float64 | EAR-R, EAR-L, MAR, phi, theta |
| `normalized_features` | `(800,5)` float64 | Original per-feature normalization |
| `landmarks_xy` | `(800,478,2)` float64 | Original normalized x/y landmarks |
| `window_frame_rows` | `(699,)` | Frame ending each window |
| `observation_frame_rows` | `(699,16)` | Exact chronology of valid observations |
| `sequences` | `(699,5,16)` float64 | Before conversion to model dtype |
| `input_tensors` | `(699,1,5,16)` float32 | Exact inputs to the original model |
| `logits`, `probabilities` | `(699,1,3)` float32 each | Original outputs and caller softmax |
| `predicted_class_ids`, `predicted_class_names` | `(699,)` each | Exact raw class ID/name |
| `normalization_ranges` | `(5,2)` float64 | Frozen baseline ranges |
| `stage_times_ms` | `(800,6)` float64 | Raw timings, with stage order in performance JSON |

Missing-face raw/normalized features and landmarks are NaN placeholders, not
fabricated observations. `frame_window_indices=-1` denotes no prediction on a
missing or warm-up frame. Those rows do not enter the temporal tensor. Features,
normalization, sequences and model outputs are stored without rounding.

The feature formulas and detector options remain unchanged: BGR tagged SRGB,
legacy normalized-coordinate EAR/MAR/solvePnP inputs, legacy pose order and
extra degree conversion, unmodified min/max normalization, no clipping,
unbounded feature-history stacking, 16 observations, chronological `(1,5,16)`,
stride 1 after warm-up, eval/no_grad and `nn.Softmax(dim=1)`.

## Numerical validation

**9 tests passed, 0 failed, 0 errors, 0 skipped**. Pytest reported 76.27 seconds;
JUnit suite time is 76.245 seconds. Evidence is saved in `docs/phase2_tests.xml`
and `docs/phase2_validation.json`.

| Check | Maximum absolute difference | Class agreement |
| --- | ---: | --- |
| Features recomputed from frozen landmarks | **0.0** | n/a |
| Independent normalization from frozen raw features | **0.0** | n/a |
| Chronological sequence/tensor reconstruction | **0.0** | n/a |
| Model-only logits from frozen tensors | **0.0** | **699/699** |
| Model-only softmax probabilities | **0.0** | **699/699** |
| Second full canonical video replay: landmarks/features | **0.0** | n/a |
| Second full canonical video replay: normalized features/tensors | **0.0** | n/a |
| Second full canonical video replay: logits/probabilities | **0.0** | **699/699** |

Tolerance remains the requested **atol=1e-6, rtol=1e-5**. No adjustment was
needed. Frozen source/frame hashes, frame indices, detected-face flags, window
membership and raw classes agree exactly. Probability sums are checked against
1, all ready outputs are finite, and input/output shapes are asserted.

The model-only test starts a fresh subprocess with an import guard that raises
if OpenCV or MediaPipe is imported. It loads `model.py` and `saved_model.pt`
directly on CPU, evaluates every frozen `(1,5,16)` tensor individually and
checks `(1,3)` logits/probabilities. It does not require a detector or webcam.
Use the model-only selection below when vision packages are unavailable.

Independent feature tests use frozen landmarks and the original formulas;
real-video replay separately reruns the detector through the canonical loop.
This separation allows later feature/model changes to be tested without assuming
landmark equality across MediaPipe versions. The zero observed vision differences
apply to this pinned environment and these videos, not all future environments.
Timing arrays are deliberately excluded from numerical replay equality.

The unchanged direct CLI was also executed, with actual OpenCV presentation:

```powershell
& .\.venv-baseline\Scripts\python.exe -B inference.py --source 'Normal/Normal_1.avi'
```

It exited with code 0 and printed `Inference complete.`. Instrumented executions
of that same main path verified first prediction after 16 valid observations,
probabilities/classes, complete decoding to EOF, and original capture release
for all six sources. The tool restores patched functions and closes detectors
after each run. There is no production runtime compatibility fix in this phase.

## Initial CPU performance reference

Raw per-frame timing arrays live in the NPZ. Full-precision aggregates and
measurement policy live in `tests/fixtures/baseline_performance.json`.
The frozen capture was run before the later validation jobs. No optimization,
sampling, reduced thread count, changed detector options or new model was used.

Steady-state policy excludes the first **20 valid observations per video**:
15 temporal warm-up observations plus five initial model executions. Model and
asset startup costs are excluded. This leaves **669** valid ready frames.
Missing-face frames remain in the separately recorded all-frame distributions,
but not these steady-ready-frame summaries.

| Stage | Samples | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| Frame acquisition/decode | 669 | 2.876 | 2.840 | 3.453 |
| MediaPipe `detect` | 669 | 38.755 | 38.279 | 42.522 |
| EAR/MAR/head-pose calculations | 669 | 3.281 | 3.159 | 3.703 |
| Original normalization arithmetic | 669 | 0.067 | 0.064 | 0.083 |
| CNN-LSTM forward | 669 | 4.361 | 3.999 | 5.719 |
| Processing pipeline, before overlay | 669 | 55.572 | 54.249 | 62.729 |

Processing begins after successful acquisition and observer frame hashing and
ends after original softmax/class handling, immediately before overlay drawing.
It includes image setup, detector, features, normalization, original growing
history/window/tensor operations, model and output handling. Acquisition is
measured separately. GUI rendering/`waitKey`, overlay drawing, disk video writing,
startup, and observer frame hashing are excluded. This is **not** a complete
capture-to-display latency or a sustained UI FPS benchmark.

The tool times original operations but introduces line-trace, hook and snapshot
overhead; that overhead remains in the processing number and to some extent in
the timed stages. These are **instrumented original-path baseline measurements**.
Later optimization comparisons must use the same measurement harness or collect
a new matched original/candidate benchmark with less intrusive timing. No
speedup percentage, sustained FPS, medical performance, or test accuracy is
claimed from this phase. No perf timing was changed to improve a result.

## Existing issues and limitations

- The original script still grows feature histories indefinitely. This is
  intentionally preserved for the baseline and will require a later measured
  change.
- `--save` with the default bare filename still encounters `os.makedirs('')`.
  Saving was not enabled during capture or the direct CLI check; no fix was
  needed for the requested baseline execution path.
- The original script releases capture on normal EOF but does not explicitly
  close its detector or guarantee exception cleanup. The harness closes the
  detector after observing the original path. Production cleanup changes are
  deferred.
- Four short face-loss runs exist in these fixtures, but no annotated sustained
  long gap or objective alert-onset ground truth. Face-gap resets and alert
  policy must be tested in their later phases.
- `Eye Close_1.avi` yields 93 Normal and 8 Yawning predictions; `Eye Close_60.avi`
  yields 126 Eye Close predictions. This is a model/source-label discrepancy,
  not a parity error. Directory labels are not verified frame annotations, and
  these training-corpus clips cannot establish independent model accuracy.
- The historical library versions used to produce the checkpoint/cache are
  unknown. This baseline freezes the current code under tested versions; it
  does not prove equality to undocumented historical executions.
- MediaPipe emitted normal XNNPACK/feedback-signature initialization messages.
  During the initial import smoke check matplotlib could not write its user
  font cache. Capture/tests use a workspace `.cache/matplotlib` directory instead;
  this changes cache storage only, not vision/model calculations.
- Webcam, new UI, audio, smoothing, resampling, resets, state machine, SQLite,
  training/evaluation changes and experiments have not been implemented/tested
  in this phase.

## Files and reproducible commands

Created:

- `.gitignore` — excludes the local environment and generated caches.
- `requirements-dev.txt` — pinned test dependency graph.
- `pytest.ini` — test discovery, root import path, vision marker and JUnit format.
- `tools/capture_baseline.py` — observed canonical capture with hash and overwrite guards.
- `tests/test_inference_parity.py` — nine layered integrity/parity tests.
- `tests/fixtures/manifest.json` — provenance and schema.
- `tests/fixtures/legacy_baseline.npz` — full-precision frozen observations/windows/outputs.
- `tests/fixtures/baseline_performance.json` — machine-readable CPU baseline statistics.
- `docs/phase2_tests.xml` — actual pytest result and numeric measurements.
- `docs/phase2_validation.json` — validation summary and direct CLI execution record.
- `docs/PHASE2_BASELINE.md` — this report.
- `.venv-baseline/` and `.cache/` — local ignored runtime/cache directories.

Modified: `requirements.txt` only. `CURRENT_ARCHITECTURE.md` is unchanged from
Phase 1. The four pre-existing tracked bytecode modifications remain untouched.

Run these commands from the repository root; activation is unnecessary:

```powershell
# Existing tested environment: verify dependencies and run all tests.
& .\.venv-baseline\Scripts\python.exe -B -m pip check
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase2_tests.xml

# Model-only parity: no OpenCV/MediaPipe import, detector, video or webcam.
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q -k model_only

# Run the original application on a video.
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache\matplotlib'
& .\.venv-baseline\Scripts\python.exe -B inference.py --source 'Normal/Normal_1.avi'

# Capture a comparison into a separate directory; preserve frozen fixtures.
& .\.venv-baseline\Scripts\python.exe -B tools/capture_baseline.py --output .cache/baseline-comparison
```

To construct a new environment with the available tested base interpreter:

```powershell
py -3.12 -B -m venv --copies .venv-baseline
& .\.venv-baseline\Scripts\python.exe -B -m pip install -r requirements-dev.txt
```

Alternatively, use a verified installed Windows x64 Python 3.12 interpreter with
`-m venv`; the exact Python patch version used here is recorded above. Creating
the environment/installing wheels may require network approval in a restricted
sandbox. Frozen fixture regeneration is opt-in with `--overwrite`; it is not
part of ordinary test execution. Full replay writes comparison data to pytest's
temporary directory and leaves the frozen fixture untouched.

All required Phase 2 gates pass on this environment. **Stop here; do not begin
Phase 3 or change the baseline preprocessing/model contract automatically.**
