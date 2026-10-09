# Phase 4: unified local application

**PHASE 4 PASS.** All **105 tests passed** (9 Phase 2, 43 Phase 3, 53 Phase 4),
with **0 failed, 0 errors, 0 skipped** in **213.734 seconds**.
The unchanged 699-window parity gates passed. Four real-video application
replays matched the frozen/Phase 3 inference contract exactly at every checked
numeric layer. Three separate command-line processes and actual saved-output
decoding also passed. **Physical webcam smoke test unavailable:** camera 0 could
not open in this environment; mocked webcam tests passed. Stop after Phase 4.

## Files and application boundary

Created application files:

- `main.py`: recommended local CLI, one processing loop, minimal result display,
  optional output writer, cleanup and bounded run-summary counters.
- `driver_monitor/vision/video_source.py`: `VideoSource`, `FramePacket`, source
  parsing and `SourceError`; acquisition and metadata only.
- `tests/test_video_source.py`: 27 source tests.
- `tests/test_main.py`: 26 application tests, including four real-video cases.

Created evidence:

- `docs/PHASE4_APPLICATION.md` (this report).
- `docs/phase4_tests.xml` (complete actual suite results).
- `docs/phase4_integration_initial.xml` (earlier four-video check, superseded by
  the complete suite).
- `docs/phase4_validation.json` (acceptance results, hashes, test cases and timing).
- `docs/phase4_preserved_hashes.json` (39 pre-existing files checked unchanged).
- `docs/phase4_webcam_probe.json` (actual failed camera-0 probe and release).
- `docs/phase4_cli_validation.json` and `docs/phase4_cli/` stdout/stderr logs.

The real annotated video is an ignored validation artifact at
`.cache/phase4_output/normal_annotated.avi`. One-off evidence generation scripts
also live in ignored `.cache/`; they are not production entry points.

**Existing files modified: none.** The Phase 3 package modules/tests/tools and
evidence, all legacy inference scripts, model, checkpoint, detector asset,
normalization, class map, config and dependency pins were preserved. Phase 2
fixtures were read and compared, never overwritten. Package hashes were checked
against the authoritative Phase 3 report, in addition to the 39-file snapshot.
The four previously modified tracked bytecode files were left alone.

`main.py` is the new recommended application. `inference.py` remains unchanged
as the Phase 2/3 historical baseline (Option A). Both webcam and file mode call
the same `_create_engine()` and exactly one `engine.process_frame(packet.frame)`
site. There is no alternative inference pipeline. Source validation and first
decode occur before engine imports/model/detector loading. Assets remain resolved
by the unchanged engine configuration relative to the repository.

## CLI and source contract

```text
python main.py --source SOURCE [--no-display] [--save-output]
               [--output PATH] [--window-name NAME]
```

`--source` is required. Default display uses an OpenCV window named
`Driver Monitor`. `--no-display` avoids `imshow`, `waitKey` and
`destroyAllWindows` entirely and still supports saving. `--save-output` writes
`model_output.avi` by default. Supplying `--output` also enables saving. `q` and
ESC quit displayed mode. Successful completion prints a JSON summary; source,
writer and runtime errors print a useful message to stderr and return 1.
KeyboardInterrupt cleans up and returns 130.

| Supplied source | Interpretation |
| --- | --- |
| `"0"`, `"1"`, `"2"`, `"12"` | Webcam indices 0, 1, 2, 12 |
| Integer `0`, `1`, `2` via Python API | Webcam index |
| `"video.mp4"`, `"123.mp4"`, `"videos/1.mp4"` | Local file |
| `"./123"` or `Path("123")` via API | File named exactly 123 |
| `"1.0"` | Local file, not a camera |
| Negative camera index or empty source | Validation error |

Only a whole ASCII decimal string is a CLI camera index. File paths are
expanded/resolved relative to the invocation directory; no numeric extension
or directory path is mistaken for a camera. Missing files and directories fail
before capture creation. Unavailable cameras, unopenable/corrupt sources,
first-frame decode failures and invalid decoded BGR frames fail cleanly.

`VideoSource` exposes `source_type`, `source_name`, `camera_index`, `file_path`,
`width`, `height`, `nominal_fps`, `total_frames` and `duration`. Invalid/unknown
metadata is `None`; camera frame count/duration is always unknown. Known file
duration is nominal frame count divided by nominal FPS. `FramePacket` contains
the unchanged BGR frame, zero-based index, timestamp, source type and EOF flag.
Every successful decoded frame is passed to the engine once, in order.

File timestamps use `CAP_PROP_POS_MSEC / 1000` **after successful reading**.
Zero is preserved; negative/nonfinite timestamps become `None`. Camera
timestamps use `perf_counter()` relative to source initialization. Timestamps
are exposed per packet without retaining a growing timestamp history and are
not used for pacing, sampling, frame skipping, duplication or buffer reset.

## EOF, display, cleanup and saving

For files, a failed read after the known nominal frame count is normal EOF.
A failure before that count, or before decoding any frame, is a source error.
When frame-count metadata is unavailable, a failed read after at least one
decoded frame is treated as EOF. OpenCV supplies no universal reason code to
separate unknown-length EOF from corruption, so this case is explicitly limited.
All webcam read failures are camera/disconnection errors, never EOF.

`VideoSource.close()` and context exit release the capture idempotently;
read exceptions also release immediately. The application uses `ExitStack` to
release the source, engine and writer on EOF, q/ESC, interruption, initialization
failure or runtime exception. Displayed mode registers window destruction before
its first `imshow`, including partially failed GUI initialization. Tests exercise
engine, acquisition and GUI failures; writer failure before engine initialization;
engine initialization failure after writer opening; and user interruption.

Rendering happens on a copy **after inference**, preserving detector pixels.
The display shows source, frame number, sequence fill, WARMING UP, NO FACE,
or the current raw prediction and confidence. A no-face result displays no stale
prediction. `Normal`, `Yawning` and `Eye Close` remain distinct raw classes.
The first 15 valid observations warm up; observation 16 predicts and subsequent
valid observations use stride 1. No-face frames keep the existing buffer and
produce no prediction, exactly as in Phase 3.

Bare output paths resolve into the current directory; nested parents are created
with `Path.parent.mkdir(parents=True, exist_ok=True)`, avoiding the legacy empty
dirname bug. Output must differ from input and must not already exist. This
prevents accidental overwrite; choose a new name for reruns. AVI uses MJPG;
MP4/MOV use mp4v, subject to installed OpenCV codecs. The writer is checked with
`isOpened()` and released if opening fails. Dimensions come from the first
decoded frame; a later size change is rejected and cleaned up.

The source's finite positive nominal FPS is used for writer metadata. Missing or
invalid FPS falls back explicitly to **20.0 FPS for output metadata only**.
It does not set model cadence. Every decoded frame is annotated and written,
including warm-up and no-face frames. Frame dimensions and nominal FPS were
verified by reopening the actual encoded AVI: **131 frames, 640 x 480, 20 FPS**,
4,959,182 bytes. Bare and nested output paths were also
checked using real OpenCV writers in tests.

## Real-video parity, tests and webcam evidence

| Source | Decoded frames | Valid observations | Predictions | Warm-up observations | No-face frames | Maximum numeric difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal/Normal_1.avi | 131 | 131 | 116 | 15 | 0 | 0.0 |
| Yawning/Yawning_1.avi | 135 | 135 | 120 | 15 | 0 | 0.0 |
| Eye Close/Eye Close_60.avi | 141 | 141 | 126 | 15 | 0 | 0.0 |
| Yawning/Yawning_60.avi | 137 | 126 | 111 | 15 | 11 | 0.0 |

Each application result was compared to the authoritative frozen rows:
face detection, buffer fill, readiness, raw features, normalized features,
chronological sequences, float32 input tensors, logits, probabilities and raw
class IDs. All **473/473** application prediction windows matched exactly.
Every source reached clean EOF, with the engine and detector closed. The gap
source retained all 11 missing-face frames without a reset. The unchanged
Phase 2/3 tests independently passed all **699/699** frozen windows across six
sources, with zero numeric difference and exact raw class agreement.

Test totals: **105 passed, 0 failed, 0 errors, 0 skipped**. Phase 2: **9/9**;
Phase 3: **43/43**; new Phase 4: **53/53**. The Phase 3 real-video comparator,
feature/model parity and 100,000-observation bounded-storage test all still pass.
The full XML records actual cases and measured timings. Tests cover source
parsing, missing files/directories, unavailable captures, metadata, indices,
timestamps, EOF/read failure, cleanup, one engine path, headless behavior,
display semantics, keyboard exit, writer failure, bare/nested paths, FPS fallback,
output protection and frame-size changes.

The actual camera-0 probe returned `opened=false`, `first_frame_read=false`,
and `capture_released=true`. **Physical webcam smoke test unavailable.** Live
display, real-camera face detection, warm-up and manual q exit were not claimed.
Fake cameras exercise indices 0/1/2, relative monotonic timestamps, shared engine
dispatch, read failure, q/ESC and resource release. Other physical camera indices
were not probed.

## Observational local-video throughput

These are separate actual `python main.py ... --no-display` processes, without
saving, on the unchanged pinned Windows CPU environment recorded in Phase 3.

| Source | Frames | Valid | Predictions | Engine processing seconds | Loop wall seconds | Application total seconds | Process wall seconds | Loop throughput FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal/Normal_1.avi | 131 | 131 | 116 | 3.720 | 3.971 | 7.397 | 8.489 | 32.99 |
| Yawning/Yawning_1.avi | 135 | 135 | 120 | 3.867 | 4.127 | 7.062 | 8.000 | 32.72 |
| Eye Close/Eye Close_60.avi | 141 | 141 | 126 | 3.989 | 4.279 | 7.497 | 8.506 | 32.95 |

Engine time is the sum of `process_frame()` calls, including detection/features
and warm-up/no-face calls. Loop wall time includes engine calls and subsequent
decoding; it excludes initialization and the first preflight decode. Application
total time includes source opening, first decode, lazy imports/model/detector
initialization, loop and cleanup. Process wall time also includes interpreter
startup and initial imports. Loop throughput is decoded frames divided by loop
wall seconds. These counts are bounded basic run diagnostics, not a prediction
history or a future session-metrics subsystem. The separate saving run is not
used in the throughput table because annotation/encoding adds work.

These are offline local-file processing measurements, **not real-time webcam
FPS**, controlled performance comparisons, Phase 9 optimization results, accuracy
improvements or resume speedup claims. No model, detector or thread setting was
changed. The measured application source hashes are in the validation JSON.

## Exact commands and limitations

Use the pinned environment for the established parity contract:

```powershell
& .\.venv-baseline\Scripts\python.exe -B main.py --source 0
& .\.venv-baseline\Scripts\python.exe -B main.py --source 1
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Normal/Normal_1.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Yawning/Yawning_1.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Eye Close/Eye Close_60.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Normal/Normal_1.avi" --save-output --output "model_output.avi"
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Normal/Normal_1.avi" --no-display --output "results/model_output.avi"
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase4_tests.xml
```

The saved integration command used the same Normal source with
`--no-display --output .cache/phase4_output/normal_annotated.avi`. Output reruns
require a new/nonexistent filename. Existing Phase 4 evidence should be retained
when collecting a separate measurement. Plain `python main.py` works when that
interpreter has the pinned dependencies; the explicit interpreter above avoids
mixing environments.

Remaining limitations: physical webcam/interactive GUI behavior is unverified
in this host; backend frame counts and media timestamps can be absent or
inaccurate; unknown-length video EOF cannot reliably be separated from later
corruption; OpenCV writer `write()` offers no universal per-frame success status;
failed output runs may leave a partial file for inspection; source resolution
changes are rejected during saving; codec support depends on the local OpenCV
build. Numeric parity is claimed only for the tested pinned environment. The
legacy color/head-pose quirks and source/model label discrepancy remain unchanged.

No sampling, pacing, resampling, gap reset, smoothing, derived alert state, audio,
event tracking, prediction history, SQLite, retraining or alternative model was
added. No Phase 5 or later work has begun. **PHASE 4 PASS; stop here.**
