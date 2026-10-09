# Phase 5: timestamp-aware temporal runtime

**PHASE 5 PASS.** The full suite passed **165 tests**: 9 Phase 2, 43 Phase 3,
53 Phase 4 and 60 Phase 5; **0 failures, 0 errors, 0 skips** in
**296.806 seconds**. All previous tests and assertions remain
unchanged. Frozen raw-engine parity remains **699/699**, with observed maximum
numeric differences **0.0**. Sampling-enabled real application replay matched
all **473/473** prediction windows from the four required 20-FPS sources exactly.
No Phase 6 or later work was implemented.

## Files and compatibility boundary

Created:

- `driver_monitor/inference/sampling.py`: `TemporalSampler`, timestamp error,
  sampling/temporal decisions and bounded `TemporalRuntime` state.
- `tests/test_sampling.py`: deadlines, rates, invalid timestamps, configuration,
  disabled mode, 100,000-timestamp drift/storage and extreme-span tests.
- `tests/test_face_gap_reset.py`: gap boundaries, immediate no-face reset,
  valid-to-valid reset, restarted warm-up, compatibility and identical-sequence parity.
- `tests/test_temporal_application.py`: shared webcam/file synthetic rates,
  minimal display semantics, controlled timestamp error cleanup and four real replays.
- `docs/PHASE5_TEMPORAL_RUNTIME.md` (this report), `docs/phase5_tests.xml`,
  `docs/phase5_validation.json`, `docs/phase5_before_hashes.json`,
  `docs/phase5_cli_validation.json`, and `docs/phase5_cli/` stdout/stderr evidence.
- `docs/phase5_initial_tests.xml` and `docs/phase5_initial_tests2.xml`: superseded
  development gates. The first uncovered an incorrectly constructed new copied-INI
  test fixture; that fixture was corrected using ConfigParser before final validation.
  These early records do not replace the complete passing suite.

Modified exactly five existing runtime files:

- `driver_monitor/config.py`: validated centralized target/gap/sampling settings.
- `driver_monitor/results.py`: per-call temporal metadata; existing positional
  fields preserved and no derived alert-state fields added.
- `driver_monitor/inference/temporal_buffer.py`: bounded buffer `clear()`.
- `driver_monitor/inference/engine.py`: optional timestamp policy around unchanged
  feature extraction, normalization, tensor construction and raw model inference.
- `main.py`: timestamp forwarding, focused CLI settings, temporal display and
  bounded diagnostic counters/rate measurements.

All other **62 snapshotted files** were verified unchanged, including
old tests, frozen fixtures, Phase 2/3/4 evidence, `VideoSource`, detector/features/
normalizer modules, legacy scripts, model architecture, checkpoint, normalization,
mapping, detector asset, config.ini and dependency pins. The pre-existing four
tracked bytecode changes were left untouched. One-off evidence scripts live in
ignored `.cache/` and are not alternative production entry points.

Untimestamped `process_frame(frame)` / `process_features(values)` retain the raw
Phase 3 compatibility behavior: all observations enter, no gap reset. This is
the intentional frozen-fixture API. Explicit `timestamp_seconds=None` is an
invalid timed input, distinct from omitting the keyword. Once a stream is timed,
untimed calls are rejected. Entering timed mode after untimed observations clears
unknown-time history before the first timed observation and marks a reset.
`infer_sequence(sequence)` is unchanged and never mutates temporal runtime state.

The application normally supplies timestamps through one shared
`engine.process_frame(packet.frame, **timestamp_args)` call. It supports old
single-argument adapters by inspecting their signature once; those adapters use
raw compatibility behavior. The production engine always accepts the timestamp
keyword. New timed application tests explicitly verify the production path;
old adapter tests remain unmodified. `--compatibility-mode` explicitly reproduces
Phase 4 untimed behavior and was also verified in a real CLI process.

## Sampling algorithm and timestamp policy

Initial **target_model_fps = 20.0**, interval **0.05 seconds**, is based on the
repository training-video metadata and frozen 16-observation contract. It is an
initial runtime cadence, not a demonstrated accuracy optimum. The centralized
configuration also exposes **max_face_gap_seconds = 0.5** and
**enable_temporal_sampling = True**. Positive finite numeric cadence/gap values
and a real boolean flag are validated; the target interval must be representable.
CLI overrides are validated RuntimeConfig replacements. An optional `[runtime]`
INI section is supported, but the frozen config.ini was not changed.

For first timestamp `t0`, target tick zero is immediately due. The sampler holds
an integer `next_tick`; each timestamp maps to `ticks = (t - t0) * target_fps`.
It accepts when `ticks >= next_tick - epsilon`, with
`epsilon = max(1e-9 ticks, 8 * ulp(ticks))`. On acceptance:

```text
current_tick = floor(ticks + epsilon)
missed_deadlines = max(0, current_tick - next_tick)
next_tick = current_tick + 1
next_deadline = t0 + next_tick / target_fps
```

Deadlines derive from the original anchor and integer tick; no cumulative
`last_accept + interval` drift. An eligible frame is accepted **at most once**,
even if multiple deadlines were missed. Example: after 0.00, a frame at 0.11 is
one eligible frame, one missed deadline, with next deadline 0.15. No duplicated
frame, fabricated feature, interpolation, reordering or sleeping occurs.
The sampler advances on eligible frames even when no face exists; only real,
valid-face features enter the temporal buffer. There is no later catch-up burst.

File selection uses the existing post-read `CAP_PROP_POS_MSEC / 1000` timestamps.
Offline decoding remains as fast as the application permits, without real-time
playback pacing. Webcam selection uses the existing relative monotonic
acquisition timestamps. Both source types use the same policy. The detector runs
on **every decoded frame**, including frames not due for observation; feature
extraction/normalization/model-window work is performed only for sampled faces.
Detector mode, options, color tagging, image dimensions and learned preprocessing
were not optimized or corrected.

In timed mode, `None`, NaN, infinity, negative values, boolean/string timestamps,
repeated timestamps and backward timestamps raise `TemporalTimestampError`, a
controlled ValueError caught by the application. No guessed file timestamps or
synthetic upsampling fallback is used. Invalid timestamp checks do not advance
the sampler or alter the buffer. A span too large for finite target-tick arithmetic
also raises a controlled error. The application stops and releases resources;
users should correct the timestamp source or use raw compatibility mode only for
baseline/testing purposes.

## Face-gap policy and result/display metadata

The runtime tracks the **last accepted valid observation timestamp**. Elapsed
time is checked on every timed frame/feature call, including no-face and
not-sampled frames. Gap is current time minus that timestamp. A reset occurs when
`gap > max_face_gap_seconds + epsilon_seconds`, where
`epsilon_seconds = max(1e-12, 8 * ulp(current_timestamp))` only handles float
roundoff. Equality/shorter gaps retain the buffer; a meaningful just-above gap
resets. Boundary tests use 0.49, 0.50, 0.500001 and 1.0 seconds.

**0.5 seconds is a configurable engineering runtime default, not medically
validated.** It is smaller than the nominal 16/20 = 0.8-second observation
coverage and allows brief tracking loss. The first-to-last timestamp span for
16 exactly 20-Hz observations is 15/20 = 0.75 seconds; this is distinct from
coverage. The threshold was not tuned to prediction labels or to improve
Yawning_60 results.

On an excessive gap, clear the 16-observation feature deque and previous valid
timestamp before appending any eligible new feature. A resumed due valid frame
starts fill 1; prediction requires 16 new accepted observations. This also resets
on a large valid-to-valid jump without intervening no-face calls. A no-face frame
past the threshold clears immediately even if its sampling deadline is not due.
Once cleared, repeated missing frames do not issue repeated resets. The sampler
keeps its original target timeline on a face reset to avoid reanchoring jitter;
there is no feature history or hidden prediction history to retain.

No face always returns `face_detected=False`, `sampled=False`, no features and no
raw class/logits/probabilities, and renders NO FACE. Missing time is never
credited to Normal, Yawning or Eye Close. Short gaps may retain existing sequence
values. Unsampled face frames return `sampled=False`, no new inference, and do
not append or change the sequence unless an excessive gap invalidates it.

Observation results expose `timestamp_seconds`, `sample_due`, `sampled`,
`gap_seconds`, `gap_reset`, the existing `reset_occurred`, sequence fill/readiness,
`effective_observation_interval` and `missed_deadlines`. Accepted intervals across
a reset are unknown/None, rather than reported as ordinary contiguous intervals.
The minimal display adds NOT SAMPLED, TEMPORAL RESET, target observations/sec,
source nominal FPS and current accepted-observation interval FPS when available.
An unsampled frame can show a labeled **Latest prediction** (one bounded
class/confidence tuple); this is never a false new prediction. The tuple clears
on no-face/reset. There is no prediction deque, vote, smoothing or alert state.

## Synthetic timing and long-gap evidence

The following 10-second pure timestamp streams use a 20-Hz target:

| Source FPS | Real input timestamps | Accepted frames | Not selected | Effective observation FPS |
| --- | ---: | ---: | ---: | ---: |
| 10 | 100 | 100 | 0 | 10.000 |
| 20 | 200 | 200 | 0 | 20.000 |
| 30 | 300 | 200 | 100 | 19.967 |
| 60 | 600 | 200 | 400 | 20.000 |

Low-FPS sources accept all real frames and remain at their lower cadence; they
are never synthetically upsampled. Thirty/60-FPS sources select roughly two thirds/
one third, with source-frame quantization around target deadlines. No timestamp
was accepted twice. Separate shared-application tests exercise each rate for
both webcam and file metadata, verify prediction count/warm-up, separate rate
diagnostics, and assert detector calls equal all decoded frames.

The 100,000-timestamp test accepted all exact 20-Hz timestamps at a nonzero
anchor with no accumulated drift. Sampler state has five scalar fields and no
growing list/dictionary/array. TemporalRuntime adds fixed gap state; the feature
buffer remains at most 16. The previous 100,000-feature storage test still passes.

The synthetic long-gap integration supplied 15 accepted observations, then a
1.0-second valid-to-valid jump. All 15 old values were discarded, resumed fill
was 1, and the first prediction occurred only at the 16th new observation with
input shape (1,5,16). Short/long missing-face streams, threshold equality, reset
off a sampling deadline, no-fabricated-class behavior, and disabled sampling
with retained gap protection also passed. Disabling sampling in configuration
does not disable timestamp validation or gap safety; omitting timestamps/raw
compatibility mode disables both policies deliberately.

## Real files, Yawning_60 gaps and rates

Separate actual sampling-enabled CLI processes completed all four required sources:

| Source | Decoded | Valid face frames | Sampled valid | Not sampled valid | Predictions | Gap resets | Effective observation FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal/Normal_1.avi | 131 | 131 | 131 | 0 | 116 | 0 | 20.000 |
| Yawning/Yawning_1.avi | 135 | 135 | 135 | 0 | 120 | 0 | 20.000 |
| Eye Close/Eye Close_60.avi | 141 | 141 | 141 | 0 | 126 | 0 | 20.000 |
| Yawning/Yawning_60.avi | 137 | 126 | 126 | 0 | 111 | 0 | 18.382 |

All these sources have valid increasing 20-FPS media timestamps, so every valid
face was sampled. The new timed application tests independently compared every
frame's face flag, timestamp, sampling decision, fill/readiness, features,
normalization, sequence, input tensor, logits, probabilities and raw class to
the frozen/Phase 3 contract. Maximum numeric difference was 0.0; all 473 windows
matched. Initial warm-up remained 15 accepted faces per source. Raw compatibility
CLI on Yawning_60 also returned 137 frames, 126 valid observations, 111 predictions
and zero resets. Source acquisition/EOF/cleanup/output-saving regression tests pass.

Yawning_60's four missing-face runs (zero-based frame indices) were measured
against the last preceding and next following valid timestamp:

| Missing frames | Count | Valid-to-valid gap seconds | Policy |
| --- | ---: | ---: | --- |
| 45–50 | 6 | 0.350 | Retained |
| 58–58 | 1 | 0.100 | Retained |
| 97–97 | 1 | 0.100 | Retained |
| 108–110 | 3 | 0.200 | Retained |

The longest six-frame loss covers approximately 0.30 seconds of missing frames;
its valid-to-valid interval is approximately 0.35 seconds. All runs deliberately
remain below the 0.5-second threshold and retain the sequence. Lower effective
observation cadence for this file reflects actual missing faces, not duplicates
or an altered target. No ML accuracy inference follows from the matching labels.

| Source | Measured source/decode cadence FPS | Effective observations/sec | Offline loop throughput frames/sec | Engine seconds | Loop wall seconds | Process wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal/Normal_1.avi | 20.000 | 20.000 | 21.54 | 5.703 | 6.082 | 17.116 |
| Yawning/Yawning_1.avi | 20.000 | 20.000 | 21.35 | 5.874 | 6.324 | 18.482 |
| Eye Close/Eye Close_60.avi | 20.000 | 20.000 | 21.15 | 6.277 | 6.667 | 18.815 |
| Yawning/Yawning_60.avi | 20.000 | 18.382 | 21.57 | 5.967 | 6.352 | 18.071 |

Source cadence is `(decoded_frames - 1)/(last_source_time - first_source_time)`;
for files it describes media time, and for webcams acquisition time. Effective
observation cadence is `(sampled_valid - 1)/(last_accepted_time - first_accepted_time)`
over the whole run, including face-loss gaps. This average is different from
the latest inter-observation interval shown in the UI. Offline processing
throughput is decoded frames divided by loop wall time. Engine seconds sum
`process_frame` calls; loop time includes subsequent decoding and excludes
startup/first preflight decode; process time includes interpreter startup,
imports, initialization and cleanup. Counts/rates are fixed-size runtime
diagnostics, without a later session-statistics or event subsystem.

The physical webcam remains unverified: Phase 4 camera-0 probe was unavailable;
no live-camera success is claimed here. Synthetic monotonic webcam streams
verify the same Phase 5 policy for all four rates.

## Temporal-policy overhead and regression evidence

Isolated policy benchmark: 100,000 valid 20-Hz `TemporalRuntime.observe()` calls
after 1,000 warm-up calls, including timestamp checks, integer deadlines, gap
comparison and decision construction. Per-call perf_counter_ns measurements:
**P50 11.600 microseconds, P95 16.100 microseconds**,
mean 12.279 microseconds. It excludes detector/features/model,
buffer append, rendering and diagnostic counters; timer/Python harness overhead
is included. It represents the ordinary valid observation path, not a comprehensive
benchmark of every error/reset case.

The actual engine `_temporal_decision()` helper was also instrumented externally
over every frame in the four real replays (validation harness overhead included):

| Source | Policy P50 microseconds | Policy P95 microseconds |
| --- | ---: | ---: |
| Normal/Normal_1.avi | 53.800 | 75.350 |
| Yawning/Yawning_1.avi | 54.000 | 84.230 |
| Eye Close/Eye Close_60.avi | 53.100 | 89.400 |
| Yawning/Yawning_60.avi | 53.900 | 75.780 |

These policy-only times are small relative to the previously measured MediaPipe
frame costs. They are observational, not a controlled speedup claim. Selecting
fewer observations from 30/60-FPS input reduces temporal/model workload; it does
not make a model execution faster. Detector calls were not skipped for benchmark
numbers, and detector/model/thread settings were not tuned. The measured source
hashes and complete CLI summaries are preserved in the validation JSON.

All existing 105 tests ran unchanged, plus 60 new tests. Phase 2 regression:
9/9 passed, including real canonical replay and all 699 raw classes. Phase 3:
43/43 passed, including model-only parity, chronological windows, all six
legacy-vs-engine videos, bounded memory and cleanup. Phase 4: 53/53 passed,
including source parsing/timestamps/EOF, shared engine, headless mode, writer
failures, bare/nested output, keyboard exit and actual video application checks.
The final XML records zero differences for features, normalization, sequences,
tensors, logits and probabilities in raw compatibility mode. No assertions or
fixtures were weakened/replaced.

## Commands and limitations

The measured sampling-enabled runs used:

```powershell
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Normal/Normal_1.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Yawning/Yawning_1.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Eye Close/Eye Close_60.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Yawning/Yawning_60.avi" --no-display
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Yawning/Yawning_60.avi" --no-display --compatibility-mode
& .\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase5_tests.xml
```

Normal intended webcam/configured usage (physical smoke unavailable here):

```powershell
& .\.venv-baseline\Scripts\python.exe -B main.py --source 0 --target-model-fps 20 --max-face-gap-seconds 0.5
& .\.venv-baseline\Scripts\python.exe -B main.py --source 1
& .\.venv-baseline\Scripts\python.exe -B main.py --source "Normal/Normal_1.avi" --no-display --output "results/phase5_output.avi"
```

Existing output files still require a new filename. Preserve earlier evidence
when collecting new timing runs. Temporal settings do not change writer nominal
FPS or the number of decoded/annotated frames written.

Limitations: the 0.5-second threshold and 20-Hz target are engineering choices,
not medically validated or accuracy-tuned. Low capture rate/face loss can lengthen
the actual 16-observation span; sampling cannot guarantee a fixed 0.8-second
window without forbidden interpolation/duplication. Timestamp quantization
causes selection jitter; absent/repeated/backward file timestamps stop timed
processing explicitly. Extremely large timestamp spans can exceed float
representability. Physical webcam/GUI timing is unverified. OpenCV EOF/metadata/
codec limitations documented in Phase 4 persist. Baseline numerical equality
is established only in the pinned tested environment. Color/head-pose quirks,
model labels and normalization are unchanged.

No smoothing, derived attentive/drowsy/alert state, confidence policy, persistence,
audio, event aggregation, SQLite, retraining, alternative architecture or
preprocessing correction was added. **PHASE 5 PASS. Stop after Phase 5.**
