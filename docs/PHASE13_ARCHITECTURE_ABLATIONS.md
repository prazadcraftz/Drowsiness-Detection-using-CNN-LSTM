# Phase 13 — architecture ablations on fixed development folds

**PHASE 13 PASS.** All 45 planned runs completed. The nine CNN-LSTM controls reproduce Phase 12 unweighted weights, epoch histories, checkpoint selection and validation probabilities exactly. Development-only Pareto set (mean window macro F1 / model-only CPU P50): **cnn_bilstm, cnn_lstm, cnn_only, lstm_only**. Additional models meeting the predeclared later-consideration rules: **lstm_only, cnn_gru, cnn_bilstm**. No architecture was promoted. Phase 11 test model inference passes during Phase 13: **0**. Stop after Phase 13.

## 1–4. Files, preservation and boundary

Created evaluation/models.py (central fixed registry), evaluation/architectures.py (prepare/run/latency driver), tests/test_phase13_architectures.py, tools/report_phase13.py, tools/confirm_phase13_latency.py (separate confirmation after a timing anomaly), this report, docs/phase13_before_hashes.json, unit/full-suite XML, docs/phase13_validation.json, experiment/training/latency/report logs and all new results/phase13 artifacts. New production regression evidence is in results/phase13_production_parity. Exact created-file inventory: results/phase13/created_files.json.

**Files modified from before Phase 13: none.** All 3505 pre-existing non-environment files are byte-identical, including all Phase 11/12 evidence, raw corpus, production assets and runtime code. Existing unrelated working-tree changes are preserved. No balancing, window study, corrected preprocessing, final test evaluation, all-development refit, policy/runtime/storage change, deployment, model promotion or Phase 14+ work. No production benchmark repeated or profiling dependency added. Only the five primary requested architectures were trained.

Canonical test IDs are checked against the original split before every experiment. Guarded fold data cannot contain a sealed source; process file-open seal rejects the actual sealed arrays, raw caches/videos and test metric files. The reused guard's error message names its originating Phase 12 implementation; it enforces the same seal during Phase 13. Preservation checks only hash prior bytes and do not load candidate test tensors or infer on test sources.

## 5–8. Exact reused protocol, normalization and recipe

153 development sources, 51/class. Byte-identical reuse of Phase 12 cv_folds.json (SHA `eb7dd82962e189ceaf5e3acd512773af8df572e1055475e5c9c2f4acd2c15739`), three fixed content-group-disjoint, class-stratified folds: 102 train / 51 validation videos per fold, 34 train / 17 validation per class. Fold assignment SHA `c233eaebad55d74365743e69804df9122531e0a2849a5de27e526c7cb3fa87b2`. Each source is validation exactly once per seed. The 27 Phase 11 test IDs remain excluded. Video-independent only; unknown subject identities are not invented.

Verified raw video and PRE-normalization cache hashes for all 153 development sources against Phase 12 provenance, finite observations and increasing original frame indices. Reconstructed every fold's zero-initialized extrema from fold TRAIN only and compared train/validation windows, frame identities, padding and labels exactly with the frozen prepared arrays. Normalizers and prepared arrays are read directly from Phase 12; no normalized arrays or range files are overwritten. Input/code/spec SHA-256 hashes are locked in experiment_config.json before training. No MediaPipe extraction repeated.

Fixed feature order EAR-R, EAR-L, MAR, phi, theta; input (B,5,16), stride 8, repeat-last valid tail padding and historical every-frame valid-observation compression. No BGR/SRGB, landmark, pose, angle, zero-anchor or face-gap changes. Same unweighted CE, seeded shuffled DataLoader (no weighted sampler), Adam .001, batch 64, CPU float32, max 60 epochs, patience 10, no scheduler, minimum improvement 1e-12, earliest exact ties, eval/inference_mode for validation, deterministic algorithms, Torch intra/interop 8/8 and num_workers 0. Only architecture changes.

## 9–13. Predeclared models, parameters, initialization and runs

| Name | Recurrent hidden | Layers | Readout | Parameters |
|---|---|---|---|---|
| cnn_only | None | 0 | global temporal mean (B,64) | 11075 |
| lstm_only | 128 | 2 | last recurrent output (B,128) | 209667 |
| cnn_lstm | 128 | 2 | last recurrent output (B,128) | 246,595 |
| cnn_gru | 128 | 2 | last recurrent output (B,128) | 188739 |
| cnn_bilstm | 64 | 2 | top-layer forward/reverse final hidden concatenation (B,128) | 181059 |

Reference cnn_lstm directly instantiates model.py CNN_LSTM, preserving state keys, construction RNG draws, module shapes and the exact forward implementation (246,595 parameters). Its front-end uses Conv1d 5→32→64, kernel3/no padding, ReLU after each, MaxPool1d2: temporal lengths 16→14→12→6, Conv dropout .25; two-layer LSTM hidden 128, internal recurrent dropout 0, external output dropout .25, last output, dense 128→64(ReLU)→3.

CNN-only chooses ONE fixed global temporal mean readout, producing 64 features from the same convolution stack. It retains external readout dropout .25 and dense 64→64→3; no flatten/readout sweep. LSTM-only feeds exactly permuted (B,16,5) values to two-layer hidden 128 LSTM, external dropout then last temporal output; dense 128→64→3. CNN-GRU replaces only the recurrent block with two-layer hidden 128 GRU; same convolution, external dropout, last-output readout and dense head. CNN-BiLSTM uses two layers, hidden 64/direction; concatenates top-layer final forward and final reverse hidden summaries, yielding 128features, external dropout .25 and the same dense head. This avoids the incomplete reverse summary in a last-position bidirectional output. It can use later positions within the already completed window and does not see future frames beyond that window.

Initialization policy is fixed across models: PyTorch module construction followed by Conv He/ReLU, dense Xavier, recurrent input He/linear, recurrent hidden orthogonal, all biases zero. Structural differences imply different weight tensors/RNG consumption; same seed/policy and data order rules are used, not an impossible identical-weight requirement across architectures. Train seeds 1201/1202/1203, three folds, five models: 45 runs. All 9 reference runs completed and passed exact Phase 12 reproduction BEFORE the 36 alternative runs. Technical failures would be persisted as failure.json and stop the study; **none occurred**. Best checkpoints only are retained per run; raw histories, best/stopping epochs, selected validation metrics/probabilities and configurations remain auditable. No final model fit on all 153 sources.

## 14–22. Development quality, variability and class trade-offs

Equal averaging of nine fold/seed runs per architecture. Window macro F1 is primary; checkpoint epochs selected only on fold validation macro F1. Video mean window probabilities→argmax, one decision per source; 51 validation sources/fold. Per-run and per-class raw CSV/JSON is authoritative, not a single best run. SD below is sample SD (ddof=1), describing correlated folds/seeds; no independence/significance claim.

| Architecture | F1 mean | F1 SD | Accuracy | Macro precision | Macro recall | Worst recall | Video F1 | Video accuracy |
|---|---|---|---|---|---|---|---|---|
| cnn_only | 0.921083 | 0.005321 | 0.920953 | 0.924212 | 0.920596 | 0.875756 | 0.982556 | 0.982571 |
| lstm_only | 0.932276 | 0.008192 | 0.932277 | 0.935097 | 0.931944 | 0.881262 | 0.982556 | 0.982571 |
| cnn_lstm | 0.922965 | 0.008070 | 0.923083 | 0.927611 | 0.922721 | 0.857955 | 0.984738 | 0.984749 |
| cnn_gru | 0.929740 | 0.006279 | 0.929782 | 0.933177 | 0.929413 | 0.873714 | 0.984736 | 0.984749 |
| cnn_bilstm | 0.925335 | 0.007185 | 0.925288 | 0.929647 | 0.924880 | 0.862061 | 0.984736 | 0.984749 |

| Architecture | Level | Class | precision | recall | f1 |
|---|---|---|---|---|---|
| cnn_only | window | Normal | 0.876950 | 0.933816 | 0.903295 |
| cnn_only | window | Yawning | 0.932825 | 0.880194 | 0.905056 |
| cnn_only | window | Eye Close | 0.962860 | 0.947778 | 0.954897 |
| cnn_only | video | Normal | 0.981481 | 0.986928 | 0.983742 |
| cnn_only | video | Yawning | 0.969136 | 1.000000 | 0.984127 |
| cnn_only | video | Eye Close | 1.000000 | 0.960784 | 0.979798 |
| lstm_only | window | Normal | 0.887931 | 0.944863 | 0.914841 |
| lstm_only | window | Yawning | 0.957069 | 0.881262 | 0.917420 |
| lstm_only | window | Eye Close | 0.960290 | 0.969708 | 0.964567 |
| lstm_only | video | Normal | 0.981481 | 0.986928 | 0.983742 |
| lstm_only | video | Yawning | 0.969136 | 1.000000 | 0.984127 |
| lstm_only | video | Eye Close | 1.000000 | 0.960784 | 0.979798 |
| cnn_lstm | window | Normal | 0.866708 | 0.949900 | 0.905713 |
| cnn_lstm | window | Yawning | 0.963748 | 0.857955 | 0.907217 |
| cnn_lstm | window | Eye Close | 0.952378 | 0.960307 | 0.955966 |
| cnn_lstm | video | Normal | 0.981481 | 0.993464 | 0.987109 |
| cnn_lstm | video | Yawning | 0.981481 | 1.000000 | 0.990476 |
| cnn_lstm | video | Eye Close | 0.993464 | 0.960784 | 0.976629 |
| cnn_gru | window | Normal | 0.876152 | 0.953264 | 0.912711 |
| cnn_gru | window | Yawning | 0.959262 | 0.873714 | 0.914138 |
| cnn_gru | window | Eye Close | 0.964116 | 0.961261 | 0.962371 |
| cnn_gru | video | Normal | 0.981481 | 0.993464 | 0.987109 |
| cnn_gru | video | Yawning | 0.975309 | 1.000000 | 0.987302 |
| cnn_gru | video | Eye Close | 1.000000 | 0.960784 | 0.979798 |
| cnn_bilstm | window | Normal | 0.866291 | 0.953148 | 0.907104 |
| cnn_bilstm | window | Yawning | 0.959962 | 0.862061 | 0.908051 |
| cnn_bilstm | window | Eye Close | 0.962689 | 0.959431 | 0.960852 |
| cnn_bilstm | video | Normal | 0.981481 | 0.993464 | 0.987109 |
| cnn_bilstm | video | Yawning | 0.975309 | 1.000000 | 0.987302 |
| cnn_bilstm | video | Eye Close | 1.000000 | 0.960784 | 0.979798 |

| Architecture | Metric | mean | std | minimum | maximum |
|---|---|---|---|---|---|
| cnn_only | window_macro_f1 | 0.921083 | 0.005321 | 0.915742 | 0.931435 |
| cnn_only | window_accuracy | 0.920953 | 0.005267 | 0.914539 | 0.931073 |
| cnn_only | window_macro_precision | 0.924212 | 0.005029 | 0.918299 | 0.932733 |
| cnn_only | window_macro_recall | 0.920596 | 0.005443 | 0.914048 | 0.931087 |
| cnn_only | window_worst_class_recall | 0.875756 | 0.022548 | 0.835570 | 0.897260 |
| cnn_only | video_macro_f1 | 0.982556 | 0.006542 | 0.980375 | 1.000000 |
| cnn_only | video_accuracy | 0.982571 | 0.006536 | 0.980392 | 1.000000 |
| lstm_only | window_macro_f1 | 0.932276 | 0.008192 | 0.919782 | 0.942491 |
| lstm_only | window_accuracy | 0.932277 | 0.007803 | 0.920670 | 0.942373 |
| lstm_only | window_macro_precision | 0.935097 | 0.008603 | 0.920001 | 0.944566 |
| lstm_only | window_macro_recall | 0.931944 | 0.008111 | 0.919992 | 0.942288 |
| lstm_only | window_worst_class_recall | 0.881262 | 0.012982 | 0.860544 | 0.907534 |
| lstm_only | video_macro_f1 | 0.982556 | 0.006542 | 0.980375 | 1.000000 |
| lstm_only | video_accuracy | 0.982571 | 0.006536 | 0.980392 | 1.000000 |
| cnn_lstm | window_macro_f1 | 0.922965 | 0.008070 | 0.914583 | 0.934732 |
| cnn_lstm | window_accuracy | 0.923083 | 0.007580 | 0.914539 | 0.934463 |
| cnn_lstm | window_macro_precision | 0.927611 | 0.006641 | 0.916409 | 0.935057 |
| cnn_lstm | window_macro_recall | 0.922721 | 0.007976 | 0.914140 | 0.934884 |
| cnn_lstm | window_worst_class_recall | 0.857955 | 0.034490 | 0.795918 | 0.910959 |
| cnn_lstm | video_macro_f1 | 0.984738 | 0.013079 | 0.960767 | 1.000000 |
| cnn_lstm | video_accuracy | 0.984749 | 0.013072 | 0.960784 | 1.000000 |
| cnn_gru | window_macro_f1 | 0.929740 | 0.006279 | 0.919182 | 0.940653 |
| cnn_gru | window_accuracy | 0.929782 | 0.005880 | 0.920670 | 0.940113 |
| cnn_gru | window_macro_precision | 0.933177 | 0.005859 | 0.924024 | 0.942952 |
| cnn_gru | window_macro_recall | 0.929413 | 0.006064 | 0.919772 | 0.939846 |
| cnn_gru | window_worst_class_recall | 0.873714 | 0.031127 | 0.819728 | 0.907534 |
| cnn_gru | video_macro_f1 | 0.984736 | 0.008654 | 0.980375 | 1.000000 |
| cnn_gru | video_accuracy | 0.984749 | 0.008646 | 0.980392 | 1.000000 |
| cnn_bilstm | window_macro_f1 | 0.925335 | 0.007185 | 0.913565 | 0.939350 |
| cnn_bilstm | window_accuracy | 0.925288 | 0.007195 | 0.912994 | 0.938983 |
| cnn_bilstm | window_macro_precision | 0.929647 | 0.007434 | 0.916501 | 0.941243 |
| cnn_bilstm | window_macro_recall | 0.924880 | 0.007224 | 0.912919 | 0.938888 |
| cnn_bilstm | window_worst_class_recall | 0.862061 | 0.020660 | 0.829932 | 0.897260 |
| cnn_bilstm | video_macro_f1 | 0.984736 | 0.008654 | 0.980375 | 1.000000 |
| cnn_bilstm | video_accuracy | 0.984749 | 0.008646 | 0.980392 | 1.000000 |

## 23–24. Development confusion and paired differences

Rows actual, columns predicted Normal/Yawning/Eye Close. Separate raw CSV matrices sum three validation folds for each fixed training seed. Pooled tables additionally sum three seeds:8,043 window predictions  / 459 video predictions, representing 2,681 unique development windows  / 153 videos each predicted three times. These repeated predictions are not independent subjects and are not test matrices.

**cnn_only, window, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2539 | 128 | 51 |
| Yawning | 270 | 2334 | 48 |
| Eye Close | 95 | 44 | 2534 |

**cnn_only, video, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 151 | 2 | 0 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**lstm_only, window, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2569 | 88 | 61 |
| Yawning | 266 | 2337 | 49 |
| Eye Close | 63 | 18 | 2592 |

**lstm_only, video, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 151 | 2 | 0 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**cnn_lstm, window, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2582 | 71 | 65 |
| Yawning | 312 | 2275 | 65 |
| Eye Close | 90 | 16 | 2567 |

**cnn_lstm, video, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 152 | 0 | 1 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**cnn_gru, window, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2591 | 78 | 49 |
| Yawning | 287 | 2317 | 48 |
| Eye Close | 82 | 21 | 2570 |

**cnn_gru, video, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 152 | 1 | 0 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**cnn_bilstm, window, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2591 | 77 | 50 |
| Yawning | 316 | 2286 | 50 |
| Eye Close | 88 | 20 | 2565 |

**cnn_bilstm, video, pooled 3 seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 152 | 1 | 0 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

Each paired delta is candidate minus CNN-LSTM at identical fold/seed. Raw 9 deltas/model are in paired_comparisons.csv; summaries below include mean, SD, min/max, positive/negative/zero pairs.

| Architecture | Paired metric | mean | std | minimum | maximum | positive_pairs | negative_pairs | zero_pairs |
|---|---|---|---|---|---|---|---|---|
| cnn_only | window_macro_f1 | -0.001882 | 0.005744 | -0.011218 | 0.005290 | 4 | 5 | 0 |
| cnn_only | window_worst_class_recall | 0.017801 | 0.032117 | -0.033557 | 0.075703 | 6 | 2 | 1 |
| cnn_only | video_macro_f1 | -0.002182 | 0.011789 | -0.019625 | 0.019608 | 1 | 2 | 6 |
| lstm_only | window_macro_f1 | 0.009311 | 0.003489 | 0.004584 | 0.017281 | 9 | 0 | 0 |
| lstm_only | window_worst_class_recall | 0.023307 | 0.027070 | -0.027397 | 0.064626 | 7 | 2 | 0 |
| lstm_only | video_macro_f1 | -0.002182 | 0.011789 | -0.019625 | 0.019608 | 1 | 2 | 6 |
| cnn_gru | window_macro_f1 | 0.006775 | 0.006052 | -0.000208 | 0.016220 | 8 | 1 | 0 |
| cnn_gru | window_worst_class_recall | 0.015759 | 0.021961 | -0.013605 | 0.050336 | 6 | 3 | 0 |
| cnn_gru | video_macro_f1 | -0.000002 | 0.009808 | -0.019625 | 0.019608 | 1 | 1 | 7 |
| cnn_bilstm | window_macro_f1 | 0.002370 | 0.009377 | -0.017752 | 0.013807 | 7 | 2 | 0 |
| cnn_bilstm | window_worst_class_recall | 0.004106 | 0.031729 | -0.030822 | 0.054422 | 4 | 5 | 0 |
| cnn_bilstm | video_macro_f1 | -0.000002 | 0.009808 | -0.019625 | 0.019608 | 1 | 1 | 7 |

## 25–27. Model-only CPU latency, sizes and Pareto trade-off

| Architecture | Parameters | Checkpoint bytes min | Checkpoint bytes max | mean_ms | p50_ms | p95_ms | p99_ms | sample_count |
|---|---|---|---|---|---|---|---|---|
| cnn_only | 11075 | 47395 | 47395 | 0.405792 | 0.428500 | 0.561925 | 0.753230 | 9216 |
| lstm_only | 209667 | 842505 | 842505 | 1.922764 | 2.130000 | 2.694650 | 3.412895 | 9216 |
| cnn_lstm | 246,595 | 991089 | 991089 | 47.614986 | 1.736300 | 2.453700 | 4.085615 | 9216 |
| cnn_gru | 188739 | 759665 | 759665 | 2.382594 | 2.259800 | 3.107225 | 4.552300 | 9216 |
| cnn_bilstm | 181059 | 730625 | 730625 | 1.955278 | 2.059200 | 2.564625 | 3.126280 | 9216 |

Latency is **prepared batch-1 tensor→model logits on CPU**, excluding decode, MediaPipe, features, rendering, SQLite, audio, loader, checkpoint loading and normalization. Thread counts fixed8/8 for every architecture. Shared 64 development tensors from fold 0 validation, deterministic seed 1301, source/window/frame IDs and exact tensor hashes persisted in latency_inputs.json and latency_inputs.npz. All candidates see the same tensor order. Each of 45 best checkpoints has 128 warm-up calls and 1,024 timed calls (16 cycles of 64 inputs). Jobs have predeclared seed 1302 shuffled order; no concurrent agent training/tests/production regression during measurement. All logits evaluated in eval/inference_mode. Individual raw timing samples are in latency_samples.npz; per-checkpoint quantiles and pooled 9,216 samples/architecture are in latency_results.json. Quantiles use NumPy percentile (linear interpolation). Checkpoint bytes reflect serialized state overhead and are separate from parameter counts. No batch 64headline or FLOP estimate was used.

Descriptive non-dominated quality/P50 set: **cnn_bilstm, cnn_lstm, cnn_only, lstm_only**. A dominates B only if mean window F1≥ and pooled P50≤, with at least one strict inequality. No arbitrary composite score. Plots show variability and quality versus latency/parameters; proximity is not statistical equivalence. Tail quantiles are reported rather than hidden by a mean. Host CPU/OS scheduling, thread overhead and background system activity can affect timing; one hardware/software environment, no realtime deadline guarantee. Per-checkpoint latency distributions are also available.

### Preserved timing anomaly and separate full-design confirmation

The primary `cnn_lstm_fold_1_seed_1203` sample 29 took **423,121.8114 ms** (about 423 seconds), with neighboring calls around1–3ms. Every sample is retained: this single wall-clock stall dominates the primary pooled CNN-LSTM mean 47.614986 ms, while its pooled P50/P95/P99 are 1.736300/2.453700/4.085615 ms. A first run's extreme stall is not evidence that ordinary CNN-LSTM computation costs 47.6 ms; its underlying cause is unverified because that pass did not record process CPU time. There were two samples above 100 ms in that checkpoint. No sample was trimmed, no architecture was retrained and no recommendation threshold was changed.

After the complete regression suite/parity finished, all 45 timing jobs were repeated once with the SAME weights, inputs, seed/order, threads, warm-up and repetitions. This separate confirmation records process CPU time as an additional coarse scheduling diagnostic and preserves the primary data. Confirmation results are under latency_confirmation, including protocol, raw wall/process-CPU samples and per-checkpoint results. They are not used to substitute a favorable primary P50 or to alter the predeclared recommendation rules. The extra timer calls add a small diagnostic overhead outside the wall interval; raw wall latency remains authoritative.

| Architecture | mean_ms | p50_ms | p95_ms | p99_ms | sample_count |
|---|---|---|---|---|---|
| cnn_only | 0.441209 | 0.438300 | 0.553975 | 0.796550 | 9216 |
| lstm_only | 2.189633 | 2.176300 | 2.792550 | 3.406585 | 9216 |
| cnn_lstm | 1.781675 | 1.775150 | 2.249875 | 2.860495 | 9216 |
| cnn_gru | 2.289496 | 2.263500 | 2.914800 | 3.569315 | 9216 |
| cnn_bilstm | 2.084427 | 2.091100 | 2.561100 | 3.133090 | 9216 |

Confirmation quality/P50 Pareto set: **cnn_bilstm, cnn_lstm, cnn_only, lstm_only**. Primary and confirmation differences describe host variability rather than a new tuning experiment. All measured means and tails are disclosed; no whole-application speedup is claimed.

## 28–33. Engineering interpretation and later recommendations

- CNN-only vs CNN-LSTM: removing recurrence changes mean window F1 by **-0.001882**. Its mean is close to the control in the predeclared practical margin and its model is much smaller/faster, but class recall and paired consistency still determine acceptance. Near means alone do not establish equivalence or justify removing recurrence; this is one fixed global-mean CNN readout.
- LSTM-only vs CNN-LSTM: removing convolution changes mean F1 by **+0.009311**. Under this fixed recipe, LSTM-only has higher mean quality, so this study does not justify the CNN front-end as necessary for classification quality. It processes 16 raw temporal positions rather than 6 convolution outputs; primary P50 is **1.227×** the CNN-LSTM value. Convolution affects representation and compute, and this result does not prove it useless across other recipes.
- CNN-GRU vs CNN-LSTM: mean F1 change **+0.006775**, parameters 188,739 vs 246,595; primary P50 is **1.302×** reference. The smaller GRU therefore does not give lower median CPU latency in this measurement, despite its quality/size benefit. Fewer parameters alone are not a speed claim.
- CNN-BiLSTM vs CNN-LSTM: mean F1 change **+0.002370**, parameters 181,059 vs 246,595. Two-direction completed-window context and smaller per-direction states change the architecture; no claim of future-frame access beyond the current window or biological causality.

Recommendation guidance was locked before training: an efficient candidate requires mean F1 within 0.002 of control, at least 5/9 paired deltas within that margin, and at least 20% lower pooled P50 or parameter count. Alternatively, a strong-quality candidate requires mean F1 gain≥0.005 and at least 5/9 positive pairs; higher cost is disclosed for later judgment. Both routes require mean worst-class recall regression≤0.005, no class mean recall regression>0.01, and F1 sample SD increase≤0.01. This is practical engineering guidance, not statistical significance or authorization to promote.

- cnn_only: {"efficient_route": true, "interesting_for_later_consideration": false, "noninferior_pairs": 5, "safety_gates": {"class_recalls": false, "variability": true, "worst_class_recall": true}, "strong_quality_route": false}.
- lstm_only: {"efficient_route": false, "interesting_for_later_consideration": true, "noninferior_pairs": 9, "safety_gates": {"class_recalls": true, "variability": true, "worst_class_recall": true}, "strong_quality_route": true}.
- cnn_gru: {"efficient_route": true, "interesting_for_later_consideration": true, "noninferior_pairs": 9, "safety_gates": {"class_recalls": true, "variability": true, "worst_class_recall": true}, "strong_quality_route": true}.
- cnn_bilstm: {"efficient_route": true, "interesting_for_later_consideration": true, "noninferior_pairs": 7, "safety_gates": {"class_recalls": true, "variability": true, "worst_class_recall": true}, "strong_quality_route": false}.

Reference CNN-LSTM remains the verified control. Additional architectures recommended for later Phase 14/15 consideration: **lstm_only, cnn_gru, cnn_bilstm**. These recommendations and the descriptive Pareto set answer different questions; an efficient Pareto point can still fail recall/stability safeguards. **No architecture was promoted**, no production state or normalizer replaced, and no new test or resume metric claimed. Phase 11 clean final test metrics remain historical and are not reused for candidate selection. No Phase 14 study begun.

Phase 9 established MediaPipe at approximately 83% of raw runtime. These new model-only numbers cannot establish whole-application speedup; even a faster model may produce only modest end-to-end improvement. The complete Phase 9/SQLite benchmarks were not repeated.

## 34–37. Access audit, regression and limitations

**Phase 11 test model inference passes during Phase 13 = 0.** test_access_audit.json and 45-entry candidate_access_audit.jsonl persist exact candidate source IDs and the zero-test invariant. Latency inputs also contain development sources only. No sealed cache/video/tensor/metric is opened by candidate training, validation or latency code. No candidate test_prediction files, test confusion matrices or test-driven decisions exist. Source guards, immutable artifact hashes, file-open refusal inherited from Phase 12 and explicit validation coverage checks enforce this technically. Report analysis reconstructs raw development data and recomputes saved validation metrics independently with sklearn; it does not run candidate inference on any test or validation source.

Full suite: **495 passed**, comprising all 475 Phase 2–12 tests and 20 new Phase 13 tests; no failures/errors/skips. Fresh production regression **699/699 windows and classes exact**, max difference 0 across compared fields, using existing frozen production fixtures. That authorized production-model check is separate from candidate inference on the sealed Phase 11 cohort. All old artifacts and runtime preserved byte-for-byte.

Limitations: video independence only, unknown driver overlap, single dataset and single hardware/software environment; validation optimizes epochs and assesses architectures, so it is not untouched final evaluation. Folds share training videos and seeds reuse validation observations; SD/paired comparisons are descriptive correlated runs. Fixed equal recipe isolates this study but does not find each architecture's optimal hyperparameters; global-mean CNN and 64/direction BiLSTM are the single predeclared designs. Whole-video labels do not validate event timing, clinical safety or alert accuracy. Model-only timing excludes the dominant vision pipeline and cannot predict deployment latency. No architecture/data/window/preprocessing tuning, new subjects, all-development refit or final test occurred.

## 38–39. Reproduction and status

Keep the frozen Phase 11/12 artifacts and recorded software environment. Commands below target NEW output paths; canonical artifacts refuse overwrite. From repository root:

```powershell
$env:MPLCONFIGDIR = '.cache/matplotlib'
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures prepare --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures run --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures latency --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase13_tests_reproduction.xml
.venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase13_production_parity_reproduction/result.json
```

Canonical execution used the same prepare/run/latency commands with results/phase13, full-suite docs/phase13_tests.xml, production parity results/phase13_production_parity/result.json, then tools/report_phase13.py. The reporter reads canonical paths and writes new-only outputs, so do not run it over completed report artifacts. Reproduction histories, weights and saved validation probabilities can be compared directly; byte-level checkpoint serialization hashes and latency samples need not repeat. Control reproducibility was verified across two phases on this CPU, not all environments. No test inference command exists in this workflow.

The separate anomaly investigation used `.venv-baseline/Scripts/python.exe -B tools/confirm_phase13_latency.py` after regression. It reads canonical trained checkpoints and writes a NEW latency_confirmation directory, refusing overwrite. It does not train models or modify the primary benchmark. Keep this confirmation optional when reproducing the original predeclared design; retain both passes whenever investigating the reported stall.

**PHASE 13 PASS. Stop after Phase 13; Phase 14 not begun.**

## Fixed layer definitions (predeclared)

### cnn_only

Hypothesis: Does recurrence add value beyond the fixed convolutional stack?

```text
Alternative(
  (conv1): Conv1d(5, 32, kernel_size=(3,), stride=(1,))
  (conv2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_cnn): Dropout(p=0.25, inplace=False)
  (dropout_lstm): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=64, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
)
```

### lstm_only

Hypothesis: Does the convolutional front-end add useful representation learning?

```text
Alternative(
  (lstm): LSTM(5, 128, num_layers=2, batch_first=True)
  (dropout_lstm): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
)
```

### cnn_lstm

Hypothesis: Exact established control

```text
CNN_LSTM(
  (conv1): Conv1d(5, 32, kernel_size=(3,), stride=(1,))
  (conv2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_cnn): Dropout(p=0.25, inplace=False)
  (lstm): LSTM(64, 128, num_layers=2, batch_first=True)
  (dropout_lstm): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
)
```

### cnn_gru

Hypothesis: Can GRU retain quality with smaller recurrent state/parameter cost?

```text
Alternative(
  (conv1): Conv1d(5, 32, kernel_size=(3,), stride=(1,))
  (conv2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_cnn): Dropout(p=0.25, inplace=False)
  (gru): GRU(64, 128, num_layers=2, batch_first=True)
  (dropout_lstm): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
)
```

### cnn_bilstm

Hypothesis: Does both-direction context inside a completed window add value?

```text
Alternative(
  (conv1): Conv1d(5, 32, kernel_size=(3,), stride=(1,))
  (conv2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_cnn): Dropout(p=0.25, inplace=False)
  (lstm): LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (dropout_lstm): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
)
```

