# Phase 16: local runtime integration and characterization

**Status: PHASE16 PASS**, subject to the final regression record. The local app now supports two explicit model profiles. `legacy` remains the CLI default for historical compatibility; `lstm32` is the **development-selected recommended profile** for an explicit user choice. This is not an externally validated final generalization model. The 27 Phase 11 held-out videos were previously evaluated by the old baseline and received **zero candidate inference passes** here.

## 1–10. Files, profiles, loading, and numerical parity

**Created:** `driver_monitor/inference/model_profile.py`; `tests/test_phase16_runtime_candidate.py`; `tools/phase16_candidate_parity.py`, `tools/characterize_phase16.py`, `tools/benchmark_phase16.py`, `tools/longrun_phase16.py`, `tools/validate_phase16_saved_sqlite.py`, `tools/audit_phase16_preservation.py`; this report, the pre-change hash snapshot, preservation/validation JSON and JUnit record in `docs/`; and the candidate parity, two-profile characterization, probability distributions, matched benchmark jobs/samples, long-run, saved video/preview, SQLite database/verification, profile specification and access audit in `results/phase16/`.

**Modified prior files:** `driver_monitor/config.py`, `driver_monitor/inference/engine.py`, `driver_monitor/inference/temporal_buffer.py`, `driver_monitor/results.py`, `driver_monitor/ui/overlay.py`, `driver_monitor/storage/session_persistence.py`, `main.py`, `tests/test_cpu_optimization.py`, and `tests/test_phase15_final_candidate.py`. The two historical tests were updated only to evaluate their old source-preservation assertions at the correct archived/phase boundary; the prior numerical assertions remain. No feature extraction, detector, sampler, smoother, state machine, audio implementation, SQLite schema, checkpoint, normalizer, source video, or Phase 11–15 evidence was edited. The byte audit records the exact before/after hash for every modified prior file and verifies the other **4,657** pre-existing files unchanged.

[`model_profile.py`](../driver_monitor/inference/model_profile.py) owns profile identity, architecture, asset paths, exact checkpoint and normalizer SHA256, sequence length, parameter count, class mapping, and preprocessing mode. Both asset paths resolve inside the repository, and canonical asset hashes are checked once at engine initialization. The shared engine and bounded buffer use the selected profile; no duplicate runtime engine exists.

| Profile | Architecture | N | Parameters | Checkpoint | Normalizer |
|---|---|---:|---:|---|---|
| `legacy` | historical CNN–LSTM | 16 | 246,595 | `saved_model.pt` | `norm.npy` |
| `lstm32` | frozen Phase 13 LSTM-only through verified Phase 14 variable-window adapter | 32 | 209,667 | `models/phase15/lstm_only_w32_full_development.pt` | `models/phase15/lstm32_dev_norm.npy` |

The candidate checkpoint SHA256 is `1dcdcdbd17378190a6b7c429fa5ccf5b621dcf0adae6541d89d789a1d5b764ed`; normalizer SHA256 is `9b397ae1327b515c5505610fffa37e198eb5ccd24ec7bf1d112eacec0f950edc`. The model factory reuses the frozen architecture constructor and operations; tests verify all saved state keys/shapes, 209,667 parameters, `(B,5,32)` float32 input, `(B,32,5)` internal LSTM input, and `(B,3)` logits. Legacy retains `(B,5,16)`.

The [candidate parity gate](../results/phase16/candidate_parity.json) replayed all **64 frozen Phase 15 development verification tensors** through the direct Phase 15 model and the integrated engine. Normalized features, sequence, float32 tensor, logits, probabilities, and class IDs matched exactly: maximum difference **0.0** for every numeric field, 64/64 classes exact. One offline verification window uses repeat-last padding; it was supplied directly to `infer_sequence`, while live `process_features` never fabricates missing observations. The final [legacy parity](../results/phase16/legacy_parity_final/result.json) reproduced **699/699** frozen windows and classes, with raw features, normalized features, sequences, tensors, logits, probabilities, policy/session traces, landmarks, and dashboard pixels exact; maximum numeric difference **0.0**. An earlier parity attempt recorded a policy-trace mismatch because an added dataclass field entered serialized observations; the final implementation keeps the legacy serialized observation shape, and the final full parity passed.

## 11–15. Sampling, warm-up, gaps, and frozen policy

The target cadence remains **20 accepted observations/s**. The detector still processes every decoded frame; the original one-face MediaPipe mode, BGR-as-SRGB compatibility quirk, EAR/MAR/pose arithmetic, and feature order are unchanged. Both profiles infer on **every newly accepted observation after readiness** (runtime stride 1); the candidate's training stride 16 is not applied at runtime. The bounded buffer uses 16 or 32 slots according to the profile.

LSTM32 predicts only on accepted observation **32**, then on 33 and each subsequent accepted observation. At 20 Hz its nominal accumulation is 1.60 s and first-to-last span 1.55 s. A >0.5 s face gap resets the same existing temporal policy and clears all stale observations; 32 fresh accepted observations are then required. Short no-face gaps retain the buffer according to the original rule. Tests exercise 31/32/33, rolling stride 1, reset re-warm, and short no-face retention. No observations are filled in. The existing five-probability-vector smoother, seven state names, thresholds, persistence durations, DROWSY/ALERT timing, and session/event ownership remain unchanged. Five vectors are predictions, not seconds; their history begins after the longer candidate warm-up.

## 16–20. Six-video behavior and frozen-threshold probabilities

The six [characterization fixtures](../results/phase16/profile_characterization.json) are all Phase 11 **development** videos. They provide compatibility observations, **not candidate accuracy**. Audio was disabled and no threshold was changed. The full per-source artifact includes decoded/face-visible/sampled counts, predictions, first timestamp, resets, raw and smoothed class counts, state frame counts and occupancy, event counts/durations, no-face intervals, and Eye-Evidence Fraction.

| Profile | Development source | Decoded / face / sampled | Predictions; first s | Raw N/Y/E | Smooth N/Y/E | Yawn/Eye/Drowsy/Alert/No-face events | Eye fraction |
|---|---|---|---|---|---|---|---:|
| legacy | Normal_1 | 131/131/131 | 116; 0.75 | 103/12/1 | 111/5/0 | 0/0/0/0/0 | 0 |
| legacy | Normal_60 | 140/140/140 | 125; 0.75 | 125/0/0 | 125/0/0 | 0/0/0/0/0 | 0 |
| legacy | Yawning_1 | 135/135/135 | 120; 0.75 | 0/120/0 | 0/120/0 | 1/0/0/0/0 | 0 |
| legacy | Yawning_60 | 137/126/126 | 111; 0.75 | 0/111/0 | 0/111/0 | 1/0/0/0/4 | 0 |
| legacy | Eye Close_1 | 116/116/116 | 101; 0.75 | 93/8/0 | 101/0/0 | 0/0/0/0/0 | 0 |
| legacy | Eye Close_60 | 141/141/141 | 126; 0.75 | 0/0/126 | 0/0/126 | 0/1/1/1/0 | 1 |
| lstm32 | Normal_1 | 131/131/131 | 100; 1.55 | 100/0/0 | 100/0/0 | 0/0/0/0/0 | 0 |
| lstm32 | Normal_60 | 140/140/140 | 109; 1.55 | 109/0/0 | 109/0/0 | 0/0/0/0/0 | 0 |
| lstm32 | Yawning_1 | 135/135/135 | 104; 1.55 | 0/104/0 | 0/104/0 | 1/0/0/0/0 | 0 |
| lstm32 | Yawning_60 | 137/126/126 | 95; 1.55 | 0/95/0 | 0/95/0 | 1/0/0/0/4 | 0 |
| lstm32 | Eye Close_1 | 116/116/116 | 85; 1.55 | 85/0/0 | 85/0/0 | 0/0/0/0/0 | 0 |
| lstm32 | Eye Close_60 | 141/141/141 | 110; 1.55 | 0/0/110 | 0/0/110 | 0/1/1/1/0 | 1 |

The longer warm-up explains 16 fewer predictions in each clip. Both profiles logged one Yawn event in each Yawning fixture and one Eye, Drowsy, and Alert event in Eye Close_60; this is **not** event-ground-truth validation. Eye Close_60 remained in ALERT for 3.75 s under legacy and 2.95 s under LSTM32 because prediction begins 0.8 s later; the same 1.0 s suspected, 1.5 s drowsy progression rules and 5-prediction smoother applied. Its longest eye evidence was 6.25 versus 5.45 s, longest drowsy progression 5.25 versus 4.45 s, and longest alert 3.75 versus 2.95 s. The only no-face intervals were four in Yawning_60 under each profile. Full seven-state occupancy is stored per source.

The [probability artifact](../results/phase16/probability_characterization.json) supplies **mean, median, P10, P25, P75, P90, minimum, and maximum** for each of Normal, Yawning, Eye Close per source and profile, as well as raw-class-conditioned counts. Candidate mean N/Y/E probabilities by source were: Normal_1 **.866/.112/.022**, Normal_60 **.908/.079/.013**, Yawning_1 **.001/.999/.000**, Yawning_60 **.001/.999/.000**, Eye Close_1 **.852/.124/.024**, Eye Close_60 **.004/.005/.991**. In these six clips, **zero candidate predictions** fell within ±0.05 of the frozen .60/.60/.70 class thresholds; the artifact also records below/above counts. That observation does not justify retuning or guarantee threshold behavior elsewhere. Whole-video folder labels contain no event-onset ground truth.

## 21–24. Dashboard, audio, session metrics, and SQLite

The dashboard still renders at the source frame dimensions and shows `Sequence: x / 32` and a small `Model: LSTM32` label for the candidate. Legacy dashboard pixels remain exact. A saved local candidate video was reopened: **141/141 readable frames, 640×480, nominal 20 FPS**. Its frame 55 preview was visually inspected; profile/readiness/state are legible, with no rendering exception. Those development-video media files are not bundled with the GitHub source package. This characterization is not an accuracy demonstration.

`AlertManager`, `SessionMetrics`, and event trackers are shared. The silent candidate Eye Close_60 run reached ALERT and logged **one disabled audio trigger request**; no device playback was attempted, and audio did not block inference. The same run created one [SQLite database](../results/phase16/candidate.db) under the unchanged version-1 schema. [Validation](../results/phase16/sqlite_validation.json) confirms a finalized `LSTMOnly32` session, 209,667 parameters, sequence length 32, exact candidate checkpoint/normalizer SHA256, aggregate JSON equal to SessionMetrics, three persisted events matching the analytics counts, all events finalized, and no frame/prediction tables. No model-specific event or schema branch was added.

## 25–31. Matched CPU performance and separate warm-up cost

The [matched benchmark](../results/phase16/benchmark_matched.json) used four development sources (Normal_1, Yawning_1, Eye Close_60, Yawning_60), two repetitions per profile/source, **16 fresh processes** in fixed-seed shuffled order, identical 8/8 Torch threads, detector and 20-Hz sampling, headless mode, no audio, no SQLite, no rendering. Only model profile changed. Every per-job raw sample in the local phase archive and the combined [sample archive](../results/phase16/latency_samples.npz) are retained. Warm-up and initialization are outside steady-state latency; results are machine-specific engineering measurements.

| CPU stage, ms (P50 / P95 / P99) | Legacy | LSTM32 |
|---|---:|---:|
| Integrated model forward only | 2.142 / 2.669 / 3.811 | 3.749 / 5.246 / 8.964 |
| Raw inference pipeline | 28.460 / 29.913 / 31.224 | 30.088 / 32.961 / 38.099 |
| Headless application processing, decode excluded | 28.745 / 30.245 / 31.541 | 30.381 / 33.300 / 38.460 |
| Full frame loop including decode | 30.484 / 32.111 / 33.400 | 32.094 / 35.178 / 40.644 |

Offline loop median FPS across the two runs: Normal_1 **36.93 legacy / 32.43 LSTM32**; Yawning_1 **36.20 / 34.00**; Eye Close_60 **34.31 / 33.30**; Yawning_60 **36.72 / 36.94**. Weighted process CPU was **261.4% / 230.9%** respectively (sums cores, so can exceed 100%). Median end-of-run RSS was **319.17 / 319.35 MB**; maximum observed peak RSS **319.89 / 320.67 MB**. The profile's larger model-only cost translated into a smaller, variable end-to-end effect because vision dominated the per-frame pipeline. These observations are not a controlled hardware guarantee. First prediction on each 20-FPS fixture was **0.75 s after 16 observations** for legacy versus **1.55 s after 32** for candidate, an **0.80 s** observed startup difference; nominal accumulation is 0.80 s versus 1.60 s.

## 32–37. Boundedness, test access, regression, decision, and limitations

The [10,000-frame candidate run](../results/phase16/long_run.json) used repeated decoded frames from development video Normal_1 through one engine, detector, sampler, policy, SessionMetrics, and periodic dashboard rendering. It recorded **9,969 predictions**; feature buffer remained ≤32 and ended at 32, smoother ≤5 and ended at 5, exactly five event trackers, dashboard cache 1/4, and no retained per-frame model history. RSS rose from **205.98 MiB at process startup** to **312.84 MiB at frame 1,000**, then to **316.06 MiB at frame 10,000** (peak **316.16 MiB**). Allocation warm-up and a later small rise are visible; one run is **not proof of no memory leak**.

The [test-access audit](../results/phase16/test_access_audit.json) records **0 candidate Phase 11 held-out inference passes, 0 held-out cache reads, and 0 held-out video reads**. Candidate parity used only frozen Phase 15 development tensors/caches; six-video characterization, benchmark, saved video/SQLite, and long-run sources were source-whitelisted Phase 11 development fixtures. Preservation hashed sealed files as bytes without decoding or inference. No external evaluation, new model search, preprocessing correction, or threshold optimization occurred.

The full Phase 2–16 regression suite passed **575 tests**, with no failures/errors/skips. It includes the new profile, warm-up/reset/stride, hash, direct parity, dashboard, SQLite, long-run, and source-exclusion checks, plus the intact 699-window legacy numerical regression. The Phase 9 source-restoration gate now correctly rejects this deliberately changed engine; its historical performance assertions are still exercised using the archived Phase 9 source identity in the test harness. The Phase 15 preservation test still verifies that `main.py` and engine had remained unchanged *through the Phase 16 entry snapshot*. Neither historical claim is silently discarded.

**Default/recommendation:** `legacy` remains the application default for maximum compatibility. `lstm32` is documented as the explicit development-selected recommended profile, justified by Phase 12–14 cross-validation rather than six-video folder labels. Final external generalization remains unestablished.

Reproduction commands from the repository root (new artifact destinations are required for reruns):

```powershell
.\.venv-baseline\Scripts\python.exe -B tools/phase16_candidate_parity.py
$env:MPLCONFIGDIR='.cache/matplotlib'
.\.venv-baseline\Scripts\python.exe -B tools/optimize_phase9.py --parity production --output results/phase16/legacy_parity_final/result.json
.\.venv-baseline\Scripts\python.exe -B tools/characterize_phase16.py
.\.venv-baseline\Scripts\python.exe -B tools/benchmark_phase16.py
.\.venv-baseline\Scripts\python.exe -B tools/longrun_phase16.py
.\.venv-baseline\Scripts\python.exe -B main.py --source 'Eye Close/Eye Close_60.avi' --model-profile lstm32 --no-audio --no-display --save-output --output results/phase16/eye_close_60_lstm32.avi --sqlite --db-path results/phase16/candidate.db
.\.venv-baseline\Scripts\python.exe -B tools/validate_phase16_saved_sqlite.py --output results/phase16
.\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase16_tests_final.xml
.\.venv-baseline\Scripts\python.exe -B tools/audit_phase16_preservation.py
```

For a historical run: `python main.py --source <local source> --model-profile legacy`. Existing output/database creation guards intentionally refuse overwrites; use fresh paths or an isolated pre-Phase-16 workspace to reproduce the listed outputs. Phase 16 ends here; no Phase 17 or documentation polish was begun.
