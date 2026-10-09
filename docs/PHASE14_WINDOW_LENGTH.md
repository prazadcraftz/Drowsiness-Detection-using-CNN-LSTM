# PHASE 14 — Window-length development study

**PHASE14 PASS.** All 108 new runs completed, with 36 immutable window16 controls reused: 144 comparison cells. No final model was selected. Work stops after Phase 14.

## Files and experimental boundary

Created: `evaluation/windows.py`, `tests/test_phase14_windows.py`, `tools/report_phase14.py`, `tools/run_phase14.py`, `tools/audit_phase14_preservation.py`, `docs/phase14_before_hashes.json`, `docs/phase14_unit.xml`, `docs/phase14_recipe_unit.xml`, `docs/phase14_tests.xml`, `docs/phase14_preservation.json`, `docs/phase14_validation.json`, `docs/PHASE14_WINDOW_LENGTH.md`, `results/phase14/**`, `results/phase14_production_parity/**`, `results/phase14_* logs`. Modified prior files: **none**. All 3821 files in the before snapshot retain their SHA256; separately verified by a file-byte preservation audit. That audit hashes sealed artifacts without decoding or inspecting their contents and is separate from candidate processes. The production model, saved weights, normalization, sampler, buffer, inference engine, state machine, events, dashboard, SQLite, main.py and legacy preprocessing are unchanged. No balancing, preprocessing correction, all-development refit, test evaluation, production promotion, full application benchmark or Phase 15 work occurred.

## Data, integrity, and sealed test

Exactly 153 development videos, 51/class; unchanged three video-grouped folds, each 102 train and 51 validation, with 34/17 per class. The persisted Phase 12 fold file is reused directly; no fold file was regenerated. Canonical assignment verification does not create new folds.

CV SHA256: `eb7dd82962e189ceaf5e3acd512773af8df572e1055475e5c9c2f4acd2c15739`. Frozen Phase 13 architecture-spec SHA256: `49e9966c42c822a6d09b2465dbac25b481934e2544b0097790c54205f21f35ab`. Pre-training window-spec SHA256: `95a19973681f08e4236da469be001ae2c9ffdaa9f14cc95459ab4103ead4f895`.

Only verified pre-normalization development caches were loaded: source IDs, raw-video SHA256, cache SHA256, finite features, observation counts and increasing original frame IDs were checked. MediaPipe was not rerun for candidates. All three existing Phase 12 feature-wise normalizers retain their SHA256 and exact 102 training IDs; validation contributions are zero. No normalizer was refitted. No Phase 11 global train_norm.npy was used.

Window16 arrays were reconstructed exactly for normalized float64 values, labels, source order, frame and observation identities, repeat-last flags and starts. The original arrays omit start indices; starts were independently derived from contiguous source window ordinals ×8 and verified against raw chronology. All 36 checkpoint/config/history/metrics/probability artifacts are referenced by path and SHA256 in `reused_window16/references.json`; saved metrics were recomputed from those probabilities. The initial shell training attempt ended before its first checkpoint/metrics were written. Its config, start record and access entry remain in `interrupted_attempt_1`, with an explicit interruption record. The identical frozen matrix was restarted via `tools/run_phase14.py` in a hidden background process; no completed run was discarded. The variable-length adapter inherits frozen construction/initialization and keeps state keys and parameters; only the fixed16 input guard is relaxed. N16 training/evaluation outputs, including seeded dropout, are bit exact. A two-epoch recipe unit test reproduced frozen history and weights; it produced no control experiment checkpoint.

The canonical 27 Phase 11 test sources remain sealed. Candidate source whitelists and a permanent process file-open audit reject their raw videos, raw feature caches, normalized test arrays, predictions and metrics. **Phase 11 test model inference passes during Phase 14 = 0; test-cache reads = 0; attempted blocked opens in candidate experiments = 0.** A separate subprocess unit test deliberately checks six forbidden opens and confirms rejection. No candidate test predictions or test confusion matrices exist.

## Frozen experiment recipe and temporal contract

Four models: cnn_lstm, lstm_only, cnn_gru, cnn_bilstm. CNN-only is excluded after its Phase 13 class-recall gate failure. Seeds 1201/1202/1203, folds 0/1/2. New runs: 3 lengths ×4 architectures ×3 folds ×3 seeds =108. Controls: 4×3×3=36. Unweighted CE, ordinary seeded shuffle, Adam 0.001, batch 64, no scheduler, max 60 epochs, patience 10, minimum improvement 1e-12, earliest ties, validation WINDOW macro F1 selection. CPU float32; deterministic algorithms; num_workers = 0; Torch 8/8 threads. Same-seed initial state digests match Phase 13 for every run. Every source uses range(0,valid_observations,N/2); incomplete tails repeat its final valid observation. Windows never cross sources. No source is removed for face coverage or padding.

| window | training_stride | accepted_observations_before_first_prediction | nominal_coverage_seconds | nominal_warmup_seconds | first_to_last_span_seconds | float64_feature_payload_bytes | potential_runtime_prediction_stride |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 8 | 0.400000 | 0.400000 | 0.350000 | 320 | 1 |
| 16 | 8 | 16 | 0.800000 | 0.800000 | 0.750000 | 640 | 1 |
| 24 | 12 | 24 | 1.200000 | 1.200000 | 1.150000 | 960 | 1 |
| 32 | 16 | 32 | 1.600000 | 1.600000 | 1.550000 | 1280 | 1 |

Nominal coverage N/20 and first-to-last span (N−1)/20 are distinct. Face loss can make actual wall-time span longer. Theoretical float64 payload excludes Python/deque/tensor/model overhead. Potential runtime prediction stride remains1 after warm-up, independent of training stride N/2. The production16-entry buffer remains unchanged. Whole-video labels provide no trustworthy event-onset annotation; these are temporal responsiveness characteristics, not measured drowsiness/yawn/alert response times.

CNN lengths: 8→6→4→pool2; 16→14→12→pool6; 24→22→20→pool10; 32→30→28→pool14. Hard forward tests verify all 16 architecture/length combinations output finite (B,3).

## Window counts and padding

Counts below are shared by every architecture and seed; `padding_analysis.csv` also expands architecture×window×fold×split×class. Full per-source counts, decoded/valid frame counts, and padding shares are in `source_window_counts.json` and `.csv`.

| window | fold | split | class_id | windows | padded_windows | padded_percentage |
|---|---|---|---|---|---|---|
| 8 | 0 | train | 0 | 1184 | 56 | 4.729730 |
| 8 | 0 | train | 1 | 1164 | 58 | 4.982818 |
| 8 | 0 | train | 2 | 1186 | 49 | 4.131535 |
| 8 | 0 | validation | 0 | 600 | 24 | 4.000000 |
| 8 | 0 | validation | 1 | 573 | 30 | 5.235602 |
| 8 | 0 | validation | 2 | 563 | 30 | 5.328597 |
| 8 | 1 | train | 0 | 1185 | 54 | 4.556962 |
| 8 | 1 | train | 1 | 1150 | 59 | 5.130435 |
| 8 | 1 | train | 2 | 1158 | 54 | 4.663212 |
| 8 | 1 | validation | 0 | 599 | 26 | 4.340568 |
| 8 | 1 | validation | 1 | 587 | 29 | 4.940375 |
| 8 | 1 | validation | 2 | 591 | 25 | 4.230118 |
| 8 | 2 | train | 0 | 1199 | 50 | 4.170142 |
| 8 | 2 | train | 1 | 1160 | 59 | 5.086207 |
| 8 | 2 | train | 2 | 1154 | 55 | 4.766031 |
| 8 | 2 | validation | 0 | 585 | 30 | 5.128205 |
| 8 | 2 | validation | 1 | 577 | 29 | 5.025997 |
| 8 | 2 | validation | 2 | 595 | 24 | 4.033613 |
| 16 | 0 | train | 0 | 600 | 67 | 11.166667 |
| 16 | 0 | train | 1 | 592 | 66 | 11.148649 |
| 16 | 0 | train | 2 | 604 | 65 | 10.761589 |
| 16 | 0 | validation | 0 | 306 | 34 | 11.111111 |
| 16 | 0 | validation | 1 | 292 | 34 | 11.643836 |
| 16 | 0 | validation | 2 | 287 | 33 | 11.498258 |
| 16 | 1 | train | 0 | 602 | 68 | 11.295681 |
| 16 | 1 | train | 1 | 586 | 66 | 11.262799 |
| 16 | 1 | train | 2 | 592 | 67 | 11.317568 |
| 16 | 1 | validation | 0 | 304 | 33 | 10.855263 |
| 16 | 1 | validation | 1 | 298 | 34 | 11.409396 |
| 16 | 1 | validation | 2 | 299 | 31 | 10.367893 |
| 16 | 2 | train | 0 | 610 | 67 | 10.983607 |
| 16 | 2 | train | 1 | 590 | 68 | 11.525424 |
| 16 | 2 | train | 2 | 586 | 64 | 10.921502 |
| 16 | 2 | validation | 0 | 296 | 34 | 11.486486 |
| 16 | 2 | validation | 1 | 294 | 32 | 10.884354 |
| 16 | 2 | validation | 2 | 305 | 34 | 11.147541 |
| 24 | 0 | train | 0 | 401 | 68 | 16.957606 |
| 24 | 0 | train | 1 | 399 | 68 | 17.042607 |
| 24 | 0 | train | 2 | 404 | 68 | 16.831683 |
| 24 | 0 | validation | 0 | 204 | 34 | 16.666667 |
| 24 | 0 | validation | 1 | 196 | 34 | 17.346939 |
| 24 | 0 | validation | 2 | 192 | 33 | 17.187500 |
| 24 | 1 | train | 0 | 402 | 68 | 16.915423 |
| 24 | 1 | train | 1 | 394 | 68 | 17.258883 |
| 24 | 1 | train | 2 | 395 | 67 | 16.962025 |
| 24 | 1 | validation | 0 | 203 | 34 | 16.748768 |
| 24 | 1 | validation | 1 | 201 | 34 | 16.915423 |
| 24 | 1 | validation | 2 | 201 | 34 | 16.915423 |
| 24 | 2 | train | 0 | 407 | 68 | 16.707617 |
| 24 | 2 | train | 1 | 397 | 68 | 17.128463 |
| 24 | 2 | train | 2 | 393 | 67 | 17.048346 |
| 24 | 2 | validation | 0 | 198 | 34 | 17.171717 |
| 24 | 2 | validation | 1 | 198 | 34 | 17.171717 |
| 24 | 2 | validation | 2 | 203 | 34 | 16.748768 |
| 32 | 0 | train | 0 | 302 | 68 | 22.516556 |
| 32 | 0 | train | 1 | 300 | 68 | 22.666667 |
| 32 | 0 | train | 2 | 304 | 67 | 22.039474 |
| 32 | 0 | validation | 0 | 153 | 34 | 22.222222 |
| 32 | 0 | validation | 1 | 148 | 34 | 22.972973 |
| 32 | 0 | validation | 2 | 145 | 33 | 22.758621 |
| 32 | 1 | train | 0 | 302 | 68 | 22.516556 |
| 32 | 1 | train | 1 | 298 | 68 | 22.818792 |
| 32 | 1 | train | 2 | 298 | 67 | 22.483221 |
| 32 | 1 | validation | 0 | 153 | 34 | 22.222222 |
| 32 | 1 | validation | 1 | 150 | 34 | 22.666667 |
| 32 | 1 | validation | 2 | 151 | 33 | 21.854305 |
| 32 | 2 | train | 0 | 306 | 68 | 22.222222 |
| 32 | 2 | train | 1 | 298 | 68 | 22.818792 |
| 32 | 2 | train | 2 | 296 | 66 | 22.297297 |
| 32 | 2 | validation | 0 | 149 | 34 | 22.818792 |
| 32 | 2 | validation | 1 | 150 | 34 | 22.666667 |
| 32 | 2 | validation | 2 | 153 | 34 | 22.222222 |

The retained short source is `32f470d0caf9847104ed`, Eye Close/Eye Close_51.avi: **10 valid observations /135 decoded frames**. Its validation counts by length follow; train inclusion in other folds uses the same policy.

| class_id | decoded_frames | fold | padded_percentage | padded_windows | source_id | split | valid_observations | window | windows |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 135 | 0 | 66.666667 | 2 | 32f470d0caf9847104ed | validation | 10 | 8 | 3 |
| 2 | 135 | 0 | 100.000000 | 2 | 32f470d0caf9847104ed | validation | 10 | 16 | 2 |
| 2 | 135 | 0 | 100.000000 | 1 | 32f470d0caf9847104ed | validation | 10 | 24 | 1 |
| 2 | 135 | 0 | 100.000000 | 1 | 32f470d0caf9847104ed | validation | 10 | 32 | 1 |


## Primary development quality

All deterministic held-out windows remain primary. Mean, sample SD (ddof1), min and max summarize nine fold/seed cells, not nine independent datasets.

| architecture | N | mean | std | minimum | maximum |
|---|---|---|---|---|---|
| cnn_lstm | 8 | 0.916609 | 0.007399 | 0.909809 | 0.930879 |
| cnn_lstm | 16 | 0.922965 | 0.008070 | 0.914583 | 0.934732 |
| cnn_lstm | 24 | 0.938884 | 0.005215 | 0.930781 | 0.949407 |
| cnn_lstm | 32 | 0.945269 | 0.006134 | 0.937362 | 0.955280 |
| lstm_only | 8 | 0.920359 | 0.007833 | 0.912678 | 0.930580 |
| lstm_only | 16 | 0.932276 | 0.008192 | 0.919782 | 0.942491 |
| lstm_only | 24 | 0.942690 | 0.007146 | 0.932945 | 0.952775 |
| lstm_only | 32 | 0.949829 | 0.007323 | 0.941053 | 0.959849 |
| cnn_gru | 8 | 0.916307 | 0.006446 | 0.908202 | 0.928474 |
| cnn_gru | 16 | 0.929740 | 0.006279 | 0.919182 | 0.940653 |
| cnn_gru | 24 | 0.938772 | 0.005793 | 0.930629 | 0.946244 |
| cnn_gru | 32 | 0.944414 | 0.006529 | 0.933198 | 0.953118 |
| cnn_bilstm | 8 | 0.916099 | 0.006898 | 0.908680 | 0.927700 |
| cnn_bilstm | 16 | 0.925335 | 0.007185 | 0.913565 | 0.939350 |
| cnn_bilstm | 24 | 0.939968 | 0.006622 | 0.928950 | 0.950857 |
| cnn_bilstm | 32 | 0.947787 | 0.005246 | 0.941186 | 0.955131 |


## Secondary window and class behavior

| configuration | accuracy | macro_precision | macro_recall | weighted_f1 | worst_class_recall | Yawning_recall |
|---|---|---|---|---|---|---|
| cnn_lstm_8 | 0.916562 | 0.921446 | 0.916135 | 0.916592 | 0.849808 | 0.849808 |
| cnn_lstm_16 | 0.923083 | 0.927611 | 0.922721 | 0.922971 | 0.857955 | 0.857955 |
| cnn_lstm_24 | 0.938959 | 0.941580 | 0.938872 | 0.938868 | 0.886708 | 0.886964 |
| cnn_lstm_32 | 0.945302 | 0.947908 | 0.945137 | 0.945248 | 0.895155 | 0.895155 |
| lstm_only_8 | 0.920292 | 0.924450 | 0.919891 | 0.920326 | 0.857994 | 0.857994 |
| lstm_only_16 | 0.932277 | 0.935097 | 0.931944 | 0.932264 | 0.881262 | 0.881262 |
| lstm_only_24 | 0.942682 | 0.944869 | 0.942485 | 0.942677 | 0.898015 | 0.898576 |
| lstm_only_32 | 0.949755 | 0.951726 | 0.949599 | 0.949806 | 0.910781 | 0.910781 |
| cnn_gru_8 | 0.916100 | 0.921806 | 0.915625 | 0.916263 | 0.850384 | 0.850384 |
| cnn_gru_16 | 0.929782 | 0.933177 | 0.929413 | 0.929758 | 0.873714 | 0.873714 |
| cnn_gru_24 | 0.938773 | 0.940791 | 0.938636 | 0.938750 | 0.895819 | 0.895819 |
| cnn_gru_32 | 0.944312 | 0.946248 | 0.944165 | 0.944410 | 0.910751 | 0.910751 |
| cnn_bilstm_8 | 0.915973 | 0.919914 | 0.915640 | 0.916067 | 0.856723 | 0.856723 |
| cnn_bilstm_16 | 0.925288 | 0.929647 | 0.924880 | 0.925336 | 0.862061 | 0.862061 |
| cnn_bilstm_24 | 0.940043 | 0.942458 | 0.939905 | 0.939943 | 0.890204 | 0.891326 |
| cnn_bilstm_32 | 0.947744 | 0.949814 | 0.947623 | 0.947761 | 0.906286 | 0.906286 |

| configuration | class | precision | recall | f1 |
|---|---|---|---|---|
| cnn_lstm_8 | Normal | 0.853120 | 0.949093 | 0.898183 |
| cnn_lstm_8 | Yawning | 0.954205 | 0.849808 | 0.898762 |
| cnn_lstm_8 | Eye Close | 0.957013 | 0.949502 | 0.952882 |
| cnn_lstm_16 | Normal | 0.866708 | 0.949900 | 0.905713 |
| cnn_lstm_16 | Yawning | 0.963748 | 0.857955 | 0.907217 |
| cnn_lstm_16 | Eye Close | 0.952378 | 0.960307 | 0.955966 |
| cnn_lstm_24 | Normal | 0.896701 | 0.955861 | 0.924646 |
| cnn_lstm_24 | Yawning | 0.967466 | 0.886964 | 0.925139 |
| cnn_lstm_24 | Eye Close | 0.960574 | 0.973792 | 0.966866 |
| cnn_lstm_32 | Normal | 0.903786 | 0.965458 | 0.933212 |
| cnn_lstm_32 | Yawning | 0.979790 | 0.895155 | 0.935429 |
| cnn_lstm_32 | Eye Close | 0.960149 | 0.974797 | 0.967166 |
| lstm_only_8 | Normal | 0.862635 | 0.946347 | 0.902166 |
| lstm_only_8 | Yawning | 0.954311 | 0.857994 | 0.903436 |
| lstm_only_8 | Eye Close | 0.956405 | 0.955331 | 0.955475 |
| lstm_only_16 | Normal | 0.887931 | 0.944863 | 0.914841 |
| lstm_only_16 | Yawning | 0.957069 | 0.881262 | 0.917420 |
| lstm_only_16 | Eye Close | 0.960290 | 0.969708 | 0.964567 |
| lstm_only_24 | Normal | 0.904579 | 0.957250 | 0.929568 |
| lstm_only_24 | Yawning | 0.961383 | 0.898576 | 0.928651 |
| lstm_only_24 | Eye Close | 0.968646 | 0.971629 | 0.969850 |
| lstm_only_32 | Normal | 0.911350 | 0.966165 | 0.937602 |
| lstm_only_32 | Yawning | 0.974017 | 0.910781 | 0.941210 |
| lstm_only_32 | Eye Close | 0.969811 | 0.971852 | 0.970674 |
| cnn_gru_8 | Normal | 0.848987 | 0.952245 | 0.897257 |
| cnn_gru_8 | Yawning | 0.956935 | 0.850384 | 0.900296 |
| cnn_gru_8 | Eye Close | 0.959496 | 0.944246 | 0.951369 |
| cnn_gru_16 | Normal | 0.876152 | 0.953264 | 0.912711 |
| cnn_gru_16 | Yawning | 0.959262 | 0.873714 | 0.914138 |
| cnn_gru_16 | Eye Close | 0.964116 | 0.961261 | 0.962371 |
| cnn_gru_24 | Normal | 0.897297 | 0.953586 | 0.924172 |
| cnn_gru_24 | Yawning | 0.959232 | 0.895819 | 0.926397 |
| cnn_gru_24 | Eye Close | 0.965844 | 0.966502 | 0.965747 |
| cnn_gru_32 | Normal | 0.901372 | 0.960316 | 0.929730 |
| cnn_gru_32 | Yawning | 0.970062 | 0.910751 | 0.939377 |
| cnn_gru_32 | Eye Close | 0.967309 | 0.961427 | 0.964136 |
| cnn_bilstm_8 | Normal | 0.857728 | 0.939696 | 0.896342 |
| cnn_bilstm_8 | Yawning | 0.944037 | 0.856723 | 0.898066 |
| cnn_bilstm_8 | Eye Close | 0.957977 | 0.950502 | 0.953889 |
| cnn_bilstm_16 | Normal | 0.866291 | 0.953148 | 0.907104 |
| cnn_bilstm_16 | Yawning | 0.959962 | 0.862061 | 0.908051 |
| cnn_bilstm_16 | Eye Close | 0.962689 | 0.959431 | 0.960852 |
| cnn_bilstm_24 | Normal | 0.901617 | 0.954081 | 0.926367 |
| cnn_bilstm_24 | Yawning | 0.962891 | 0.891326 | 0.925284 |
| cnn_bilstm_24 | Eye Close | 0.962865 | 0.974308 | 0.968253 |
| cnn_bilstm_32 | Normal | 0.909865 | 0.964084 | 0.935880 |
| cnn_bilstm_32 | Yawning | 0.976125 | 0.906286 | 0.939764 |
| cnn_bilstm_32 | Eye Close | 0.963451 | 0.972498 | 0.967718 |

Yawning recall is tracked descriptively; no recipe was tuned specifically to it. Raw per-class fold/seed metrics for window and video levels are in `per_class_results.csv`; summary dispersion is in aggregate_results.json.

## Video-level development results

Mean window probabilities per held-out source → argmax; exactly one vote per source. Each cell has 51 decisions. Across folds, each 153-video cohort is evaluated once per seed (459 repeated-seed video decisions/configuration).

| configuration | accuracy | macro_f1 | Normal_recall | Yawning_recall | Eye Close_recall |
|---|---|---|---|---|---|
| cnn_lstm_8 | 0.986928 | 0.986917 | 1.000000 | 1.000000 | 0.960784 |
| cnn_lstm_16 | 0.984749 | 0.984738 | 0.993464 | 1.000000 | 0.960784 |
| cnn_lstm_24 | 0.986928 | 0.986917 | 1.000000 | 1.000000 | 0.960784 |
| cnn_lstm_32 | 0.986928 | 0.986917 | 1.000000 | 1.000000 | 0.960784 |
| lstm_only_8 | 0.980392 | 0.980377 | 0.980392 | 1.000000 | 0.960784 |
| lstm_only_16 | 0.982571 | 0.982556 | 0.986928 | 1.000000 | 0.960784 |
| lstm_only_24 | 0.982571 | 0.982556 | 0.986928 | 1.000000 | 0.960784 |
| lstm_only_32 | 0.984749 | 0.984736 | 0.993464 | 1.000000 | 0.960784 |
| cnn_gru_8 | 0.982571 | 0.982486 | 0.993464 | 1.000000 | 0.954248 |
| cnn_gru_16 | 0.984749 | 0.984736 | 0.993464 | 1.000000 | 0.960784 |
| cnn_gru_24 | 0.984749 | 0.984736 | 0.993464 | 1.000000 | 0.960784 |
| cnn_gru_32 | 0.986928 | 0.986917 | 1.000000 | 1.000000 | 0.960784 |
| cnn_bilstm_8 | 0.978214 | 0.978197 | 0.973856 | 1.000000 | 0.960784 |
| cnn_bilstm_16 | 0.984749 | 0.984736 | 0.993464 | 1.000000 | 0.960784 |
| cnn_bilstm_24 | 0.982571 | 0.982556 | 0.986928 | 1.000000 | 0.960784 |
| cnn_bilstm_32 | 0.986928 | 0.986917 | 1.000000 | 1.000000 | 0.960784 |


## Padding diagnostics

Non-padded and padded metrics are secondary only; all windows remain primary. Fixed three-class order and zero-division = 0 are used even for subsets. Support totals below include three seeds of the same physical windows; raw per-cell supports and metrics are stored. Padded subsets are smaller and should not be interpreted as independent validation cohorts.

| configuration | non_padded_F1 | non_padded_accuracy | non_padded_support_3seeds | padded_F1 | padded_accuracy | padded_support_3seeds |
|---|---|---|---|---|---|---|
| cnn_lstm_8 | 0.921585 | 0.921484 | 15069 | 0.810952 | 0.815995 | 741 |
| cnn_lstm_16 | 0.933096 | 0.933001 | 7146 | 0.836520 | 0.843513 | 897 |
| cnn_lstm_24 | 0.952846 | 0.952808 | 4473 | 0.866768 | 0.871104 | 915 |
| cnn_lstm_32 | 0.964171 | 0.964075 | 3144 | 0.877819 | 0.880476 | 912 |
| lstm_only_8 | 0.924831 | 0.924739 | 15069 | 0.826633 | 0.829717 | 741 |
| lstm_only_16 | 0.941044 | 0.940981 | 7146 | 0.859281 | 0.862572 | 897 |
| lstm_only_24 | 0.956726 | 0.956649 | 4473 | 0.870939 | 0.874318 | 915 |
| lstm_only_32 | 0.966743 | 0.966635 | 3144 | 0.889866 | 0.891413 | 912 |
| cnn_gru_8 | 0.921179 | 0.920931 | 15069 | 0.813445 | 0.817418 | 741 |
| cnn_gru_16 | 0.940181 | 0.940120 | 7146 | 0.842249 | 0.846892 | 897 |
| cnn_gru_24 | 0.953327 | 0.953268 | 4473 | 0.863952 | 0.867803 | 915 |
| cnn_gru_32 | 0.960360 | 0.960253 | 3144 | 0.887289 | 0.889266 | 912 |
| cnn_bilstm_8 | 0.920907 | 0.920732 | 15069 | 0.813783 | 0.818773 | 741 |
| cnn_bilstm_16 | 0.936820 | 0.936598 | 7146 | 0.828179 | 0.834633 | 897 |
| cnn_bilstm_24 | 0.955512 | 0.955469 | 4473 | 0.859477 | 0.864503 | 915 |
| cnn_bilstm_32 | 0.966402 | 0.966286 | 3144 | 0.881590 | 0.883766 | 912 |


## Paired within-architecture comparisons

Candidate minus SAME architecture at16, matched fold/seed. No cross-architecture reference is substituted.

| configuration | mean | std | minimum | maximum | positive_pairs | negative_pairs | zero_pairs |
|---|---|---|---|---|---|---|---|
| cnn_lstm_8 | -0.006356 | 0.003940 | -0.011682 | -0.001595 | 0 | 9 | 0 |
| cnn_lstm_24 | 0.015919 | 0.005300 | 0.008096 | 0.024590 | 9 | 0 | 0 |
| cnn_lstm_32 | 0.022304 | 0.005282 | 0.015942 | 0.031158 | 9 | 0 | 0 |
| lstm_only_8 | -0.011917 | 0.003107 | -0.016224 | -0.007104 | 0 | 9 | 0 |
| lstm_only_24 | 0.010414 | 0.003790 | 0.003533 | 0.016060 | 9 | 0 | 0 |
| lstm_only_32 | 0.017553 | 0.005918 | 0.009067 | 0.028962 | 9 | 0 | 0 |
| cnn_gru_8 | -0.013433 | 0.002881 | -0.019588 | -0.009510 | 0 | 9 | 0 |
| cnn_gru_24 | 0.009032 | 0.003786 | 0.004181 | 0.015178 | 9 | 0 | 0 |
| cnn_gru_32 | 0.014674 | 0.002331 | 0.011751 | 0.019411 | 9 | 0 | 0 |
| cnn_bilstm_8 | -0.009236 | 0.006098 | -0.016709 | 0.001696 | 1 | 8 | 0 |
| cnn_bilstm_24 | 0.014632 | 0.006382 | 0.004817 | 0.022415 | 9 | 0 | 0 |
| cnn_bilstm_32 | 0.022452 | 0.006014 | 0.012675 | 0.032196 | 9 | 0 | 0 |

12 candidate configurations ×9 pairs =108 paired differences are retained in `paired_window_comparisons.csv`, with F1, worst recall and video F1. These overlaps and repeated seeds preclude interpreting pair counts as independent replications or formal significance tests.

## Model-only latency and parameter/storage cost

Prepared normalized float32 batch1 tensor →logits only. Fixed Torch 8/8. 64 development inputs at common source/start anchors (51 first windows and13 starts at48), same identities across lengths; every architecture within a length uses exactly the same tensors. For each configuration the predeclared fold 0/seed 1201 checkpoint is measured in three repetitions, 128 warm-up calls and 1024 measured calls per repetition: 3072 raw samples/configuration, 49152 total. Job order is randomized with seed 1402. This checkpoint choice was fixed before quality results; latency is not a nine-checkpoint distribution. Wall and processCPU arrays are preserved both per job and pooled, with source/frame/start/tensor hashes. No input preparation or model loading is timed.

| configuration | mean_ms | p50_ms | p95_ms | p99_ms | sample_count | process_cpu_mean_ms |
|---|---|---|---|---|---|---|
| cnn_lstm_8 | 2.456932 | 2.293250 | 3.403895 | 4.078848 | 3072 | 6.388346 |
| cnn_lstm_16 | 3.144974 | 2.901000 | 4.438550 | 5.419891 | 3072 | 7.420858 |
| cnn_lstm_24 | 3.681927 | 3.352300 | 5.185135 | 6.688987 | 3072 | 9.180705 |
| cnn_lstm_32 | 4.052535 | 3.864300 | 5.380550 | 6.464288 | 3072 | 10.660807 |
| lstm_only_8 | 2.719650 | 2.534700 | 3.809380 | 4.746368 | 3072 | 6.708781 |
| lstm_only_16 | 3.802278 | 3.480950 | 5.308230 | 6.439005 | 3072 | 9.394328 |
| lstm_only_24 | 4.947977 | 4.565900 | 6.785975 | 8.900612 | 3072 | 11.967977 |
| lstm_only_32 | 5.665406 | 5.339750 | 7.542410 | 8.926206 | 3072 | 14.526367 |
| cnn_gru_8 | 2.056423 | 1.903350 | 2.919880 | 3.656802 | 3072 | 4.928589 |
| cnn_gru_16 | 3.777276 | 3.537400 | 5.137315 | 6.267395 | 3072 | 9.526571 |
| cnn_gru_24 | 5.821619 | 5.337700 | 8.335555 | 10.922414 | 3072 | 13.875326 |
| cnn_gru_32 | 7.251909 | 6.833500 | 9.585200 | 10.963058 | 3072 | 18.152873 |
| cnn_bilstm_8 | 3.412343 | 3.146350 | 4.626915 | 6.072109 | 3072 | 8.356730 |
| cnn_bilstm_16 | 3.518976 | 3.279950 | 4.743425 | 5.774419 | 3072 | 8.850098 |
| cnn_bilstm_24 | 3.886785 | 3.608700 | 5.307430 | 6.608077 | 3072 | 9.318034 |
| cnn_bilstm_32 | 3.823470 | 3.600100 | 5.086350 | 6.097640 | 3072 | 9.745280 |

All raw samples remain in the means and quantiles; no trimming. Extreme wall stalls are predeclared as≥100ms. Observed extreme stalls: **1**. They are listed below with paired CPU time; OS/scheduler cause is not inferred solely from elapsed time. Windows process CPU accounting sums multiple threads and is coarse; zeros or values exceeding wall time are possible. It is not another wall latency estimate.

| input_index | job | process_cpu_ms | sample_index | wall_ms |
|---|---|---|---|---|
| 45 | lstm_only_window_24_repeat_2 | 15.625000 | 365 | 101.599500 |

| configuration | parameters | checkpoint_bytes_min | checkpoint_bytes_max |
|---|---|---|---|
| cnn_lstm_8 | 246595 | 991089.000000 | 991089.000000 |
| cnn_lstm_16 | 246595 | 991089.000000 | 991089.000000 |
| cnn_lstm_24 | 246595 | 991089.000000 | 991089.000000 |
| cnn_lstm_32 | 246595 | 991089.000000 | 991089.000000 |
| lstm_only_8 | 209667 | 842505.000000 | 842505.000000 |
| lstm_only_16 | 209667 | 842505.000000 | 842505.000000 |
| lstm_only_24 | 209667 | 842505.000000 | 842505.000000 |
| lstm_only_32 | 209667 | 842505.000000 | 842505.000000 |
| cnn_gru_8 | 188739 | 759665.000000 | 759665.000000 |
| cnn_gru_16 | 188739 | 759665.000000 | 759665.000000 |
| cnn_gru_24 | 188739 | 759665.000000 | 759665.000000 |
| cnn_gru_32 | 188739 | 759665.000000 | 759665.000000 |
| cnn_bilstm_8 | 181059 | 730625.000000 | 730625.000000 |
| cnn_bilstm_16 | 181059 | 730625.000000 | 730625.000000 |
| cnn_bilstm_24 | 181059 | 730625.000000 | 730625.000000 |
| cnn_bilstm_32 | 181059 | 730625.000000 | 730625.000000 |

Parameter count is invariant across lengths within each frozen model. Container/metadata differences in checkpoint bytes do not indicate a change in learned parameter count.

## Later-consideration filters and Pareto sets

Rules were persisted before training: quality gain≥0.003, ≥5/9 positive pairs, mean worst-class recall regression≤0.005, and no class mean recall regression>0.01. Shorter windows are responsiveness-interesting when no more than0.002 below their own16-control and safety passes; quality gains also qualify under the persisted noninferiority-margin interpretation. Longer windows are context-interesting only when the quality rule passes. These filters describe evidence for later consideration and do not select a final architecture/window. Window16 configurations are retained reference evidence, not tested as new window candidates.

| configuration | quality | responsiveness | context | safety | later_consideration | failed_conditions |
|---|---|---|---|---|---|---|
| cnn_lstm_8 | False | False | False | False | False | mean worst-class recall regresses >0.005; Eye Close mean recall regresses >0.01; quality gain <0.003; fewer than 5 positive pairs; F1 loss exceeds responsiveness margin 0.002 |
| cnn_lstm_24 | True | False | True | True | True | none |
| cnn_lstm_32 | True | False | True | True | True | none |
| lstm_only_8 | False | False | False | False | False | mean worst-class recall regresses >0.005; Yawning mean recall regresses >0.01; Eye Close mean recall regresses >0.01; quality gain <0.003; fewer than 5 positive pairs; F1 loss exceeds responsiveness margin 0.002 |
| lstm_only_24 | True | False | True | True | True | none |
| lstm_only_32 | True | False | True | True | True | none |
| cnn_gru_8 | False | False | False | False | False | mean worst-class recall regresses >0.005; Yawning mean recall regresses >0.01; Eye Close mean recall regresses >0.01; quality gain <0.003; fewer than 5 positive pairs; F1 loss exceeds responsiveness margin 0.002 |
| cnn_gru_24 | True | False | True | True | True | none |
| cnn_gru_32 | True | False | True | True | True | none |
| cnn_bilstm_8 | False | False | False | False | False | mean worst-class recall regresses >0.005; Normal mean recall regresses >0.01; quality gain <0.003; fewer than 5 positive pairs; F1 loss exceeds responsiveness margin 0.002 |
| cnn_bilstm_24 | True | False | True | True | True | none |
| cnn_bilstm_32 | True | False | True | True | True | none |

- quality_p50: cnn_bilstm_16, cnn_bilstm_32, cnn_gru_8, cnn_lstm_16, cnn_lstm_24, cnn_lstm_8, lstm_only_32, lstm_only_8
- quality_nominal_warmup: lstm_only_16, lstm_only_24, lstm_only_32, lstm_only_8
- quality_parameters: cnn_bilstm_32, lstm_only_32

Pareto dominance uses higher mean F1 and lower stated cost; no hidden composite score. The sets can differ with descriptive timing variation and are not final selection.

## Architecture × window observations

cnn_lstm: largest observed mean F1 at N=32 (0.945269); N8-to-N32 change +0.028660. N32-vs16 mean recall deltas: Normal +0.015559, Yawning +0.037200, Eye Close +0.014490.

lstm_only: largest observed mean F1 at N=32 (0.949829); N8-to-N32 change +0.029470. N32-vs16 mean recall deltas: Normal +0.021302, Yawning +0.029519, Eye Close +0.002145.

cnn_gru: largest observed mean F1 at N=32 (0.944414); N8-to-N32 change +0.028107. N32-vs16 mean recall deltas: Normal +0.007052, Yawning +0.037037, Eye Close +0.000166.

cnn_bilstm: largest observed mean F1 at N=32 (0.947787); N8-to-N32 change +0.031688. N32-vs16 mean recall deltas: Normal +0.010936, Yawning +0.044226, Eye Close +0.013067.

These are descriptive maxima within the evaluated matrix, not model selection. CNN recurrent lengths are 2/6/10/14, while LSTM-only reads 8/16/24/32 raw observations; the frozen CNN representation compresses the temporal sequence.
Changes in temporal context also change the number of overlapping training windows and padding share. This experiment measures their combined deterministic protocol effect; it does not isolate context from exposure/optimization changes. No formal interaction significance model was predeclared or run.

## Confusions and plots

All 16 configurations have raw window and video confusion CSVs, per seed and pooled three-seed totals, rows actual and columns predicted in Normal/Yawning/Eye Close order. Pooled confusion counts repeat the development cohort across seeds; quality summaries are means of cell metrics, not F1 from the pooled matrix. The eight plots visualize macro F1, worst recall, Yawning recall, model P50, quality/P50, quality/coverage, padding shares and paired F1 deltas. RawCSV/JSON is authoritative.

## Regression and acceptance

**537 tests passed** (42 Phase 14 tests; all 495 Phase2–13 tests retained). **Production 699/699 windows and 699/699 classes exact; every maximum numerical difference 0.0.** Frozen raw-production parity is a separate authorized regression path, using the existing production model and fixtures, never an experimental checkpoint or candidate test evaluation. All 108 new runs completed with no failure/discard; 36 controls accounted for; 153 development sources preserved; normalizer refits 0; candidate sealed-test inference 0. No final model selection, all-development refit, final test evaluation or promotion. **PHASE14 PASS.**

## Limitations and exact reproduction

Development folds and seed runs overlap; reported SD is descriptive, not an independent-video confidence interval. Different lengths yield different correlated window counts; video aggregation gives each source one vote but depends on whole-video labels. Very short/padded windows are retained and disclosed; subset performance is not a new selection target. Legacy BGR/SRGB tagging, pose geometry/landmark order/angle conversion, EAR/MAR and zero-anchored normalization remain as frozen. CPU timing is specific to this host, fixed threads and representative checkpoint; wall tails may include scheduling stalls. Nominal 20 observations/s properties do not measure actual event response. Development evidence cannot establish sealed-test generalization.

Commands run from repository root using a fresh output path (artifact creation is exclusive; do not overwrite existing evidence):

```powershell
.venv-baseline/Scripts/python.exe -B -m evaluation.windows prepare
.venv-baseline/Scripts/python.exe -B -m pytest -q tests/test_phase14_windows.py
.venv-baseline/Scripts/python.exe -B -m evaluation.windows run
.venv-baseline/Scripts/python.exe -B -m evaluation.windows latency
.venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase14_tests.xml
.venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase14_production_parity/result.json
.venv-baseline/Scripts/python.exe -B tools/audit_phase14_preservation.py
$env:MPLCONFIGDIR='.cache/matplotlib'
.venv-baseline/Scripts/python.exe -B tools/report_phase14.py
```
The recorded execution redirected Python stdout/stderr to exclusive line-buffered phase14 log files. The before/after preservation audit is separate from experiment execution; its result is required by the reporter. No Phase 15 work is authorized by this report.
