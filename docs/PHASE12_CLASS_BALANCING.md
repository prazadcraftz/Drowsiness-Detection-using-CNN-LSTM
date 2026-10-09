# Phase 12 — class balancing on sealed-test development folds

**PHASE 12 PASS.** Recommendation: **unweighted** for later experimental model selection. Production remains unchanged. Three fixed video-grouped folds × three fixed seeds × three strategies completed, with zero Phase 11 test model inference passes. This measures development validation performance only and creates no new final test claim.

## 1–4. Files, boundary, and sealed test

Created `evaluation/balancing.py`, `tests/test_phase12_balancing.py`, `tools/report_phase12.py`, this report, preservation/test/validation records in docs, new experiment artifacts under results/phase12, training logs, and new production parity evidence under results/phase12_production_parity. The exact created-file inventory is `results/phase12/created_files.json`. **No pre-existing files modified**: all 3298 prior files are byte-identical, including every Phase 11 artifact, production model/normalizer, raw corpus and runtime source. Existing unrelated working tree changes were preserved.

No architecture, window, feature/geometry, normalization-semantics, focal-loss, video-balanced sampling, resume, production promotion, deployment, or Phase 13+ work. No production performance benchmark repeated. Only the three requested primary strategies were run.

## 5–10. Development protocol and quantified balance

153 development videos, 51/class, from Phase 11 TRAIN + VALIDATION. The 27 test source IDs are explicitly persisted as excluded; metadata is read only to establish the seal. 3-fold stratified content-group CV: 102 train videos (34/class) and 51 validation videos (17/class) per fold. Each development source is validated exactly once per seed. Raw content duplicates co-group; none are present in this corpus. Seed 20260927; deterministic sorted identities, seeded group shuffle and balanced class allocation.

Assignment SHA-256: `c233eaebad55d74365743e69804df9122531e0a2849a5de27e526c7cb3fa87b2`. CV file SHA-256: `eb7dd82962e189ceaf5e3acd512773af8df572e1055475e5c9c2f4acd2c15739`. Phase 11 raw manifest SHA-256: `f58679d38752cd25640980779ed86a4494c8010f137d540ddd1fd4c537e45444`. Phase 11 split SHA-256: `5e3256ea3e3becd61e05c54474875091a71466c28b5e8ee46089cff37542f602`. Persisted assignments were independently regenerated and compared exactly. Unknown subject identities are not inferred.

Corpus video counts: 60/60/60 (ratio 1.0); development and every fold video ratios 1.0. Counts below use Normal / Yawning / Eye Close order. Window and observation ratios are largest/smallest. This corpus is nearly balanced; train window ratios are only about 1.02–1.04.

| Fold | Split | Videos N/Y/E | Windows N/Y/E | Observations N/Y/E | Window ratio | Observation ratio |
|---|---|---|---|---|---|---|
| 0 | train | [34, 34, 34] | [600, 592, 604] | [4679, 4603, 4704] | 1.020270 | 1.021942 |
| 0 | validation | [17, 17, 17] | [306, 292, 287] | [2382, 2265, 2220] | 1.066202 | 1.072973 |
| 1 | train | [34, 34, 34] | [602, 586, 592] | [4688, 4547, 4584] | 1.027304 | 1.031009 |
| 1 | validation | [17, 17, 17] | [304, 298, 299] | [2373, 2321, 2340] | 1.020134 | 1.022404 |
| 2 | train | [34, 34, 34] | [610, 590, 586] | [4755, 4586, 4560] | 1.040956 | 1.042763 |
| 2 | validation | [17, 17, 17] | [296, 294, 305] | [2306, 2282, 2364] | 1.037415 | 1.035933 |

## 11. Fold-train normalization and raw cache provenance

Reused only the 153 development PRE-normalization raw-feature caches. Before each cache load, its SHA-256 is compared with the pre-experiment preservation snapshot, and its raw video hash matches the immutable raw manifest. Cache keys, observation counts, finite feature values and frame ordering are validated. `cache_provenance.json` records all 153 IDs/hashes. No Phase 11 normalized windows or train_norm.npy is loaded. No sealed raw cache or video is opened by candidate code.

Each fold independently fits five-feature extrema on its 102 train sources, starting min/max at zero, then applies that fixed range to train and validation. Exact train IDs and norm hash are persisted under fold_N/train_norm.json and train_norm.npy. Held-out validation does not contribute or get clipped. Reuses unchanged feature order, length 16, stride 8, last-observation tail padding and historical compression of valid face observations. No runtime sampling/gap resets are introduced. Missing-face observations and the short Eye Close_51 source are retained as in Phase 11; no coverage-based tuning or exclusion.

## 12–18. Controlled treatment, weights, recipe and selection

UNWEIGHTED: ordinary seeded shuffled batches and `CrossEntropyLoss()`. WEIGHTED_CE: train-window counts only, weight_c = N/(3*n_c), PyTorch weighted mean CE. WEIGHTED_SAMPLER: per-window weights proportional to 1/n_c, normalized probability 1/(3*n_c), total class probability 1/3 in expectation, N samples/epoch with replacement; unweighted CE. Actual sampled class counts per epoch are retained in each history. Sampler and weighted loss are never combined. No optional video-balanced strategy was run.

| Fold | Normal | Yawning | Eye Close | Sampler epoch N | Sampler p/window N/Y/E |
|---|---|---|---|---|---|
| 0 | 0.997778 | 1.011261 | 0.991170 | 1796 | [0.0005555555555555556, 0.0005630630630630631, 0.0005518763796909492] |
| 1 | 0.985604 | 1.012514 | 1.002252 | 1780 | [0.0005537098560354374, 0.0005688282138794084, 0.0005630630630630631] |
| 2 | 0.975956 | 1.009040 | 1.015927 | 1786 | [0.000546448087431694, 0.0005649717514124294, 0.0005688282138794084] |

Training seeds locked before execution: 1201, 1202, 1203; 27 completed runs. Within each fold/seed, all strategies use bit-identical initial weights (hashes audited) and identical normalized data. Fixed original CNN_LSTM: 246,595 parameters, 5 features ×16 observations, dropout 0.25/0.25, CPU float32, Adam .001, batch 64, no scheduler, max 60 epochs, patience 10, minimum improvement 1e-12, earliest exact ties, num_workers 0, 8/8 Torch threads and deterministic algorithms. Only balancing changes. Run configurations, environment and code hashes are preserved; control synthetic training reproduces the Phase 11 fit history and weights exactly.

Every checkpoint uses validation WINDOW macro F1 for epoch selection. Validation is eval/inference_mode with dropout off and ordinary CE diagnostic loss. Best checkpoint is restored and saved once per run; no per-epoch checkpoints. Each run stores config, best.pt plus hash, raw epoch history, selected metrics and validation predictions. No test-loader path exists in candidate code. Hard source guards precede training/each validation inference; process file-open audit also rejects sealed test arrays, predictions, metrics, caches and raw videos.

## 19–26. Development results and trade-offs

All following are validation results. Equal weight per fold/seed run, sample SD (ddof=1) across 9 runs/strategy. Window macro F1 is primary. Video predictions average window probabilities within a source then argmax; each source gets one vote. Per-fold video support 51; 153 distinct development videos overall, repeated across 3 training seeds.

| Strategy | Window F1 mean | SD | Accuracy mean | Macro recall | Worst recall | Video F1 | Video accuracy |
|---|---|---|---|---|---|---|---|
| unweighted | 0.922965 | 0.008070 | 0.923083 | 0.922721 | 0.857955 | 0.984738 | 0.984749 |
| weighted_ce | 0.922372 | 0.007059 | 0.922456 | 0.922146 | 0.861333 | 0.982558 | 0.982571 |
| weighted_sampler | 0.923142 | 0.005660 | 0.922938 | 0.922569 | 0.868453 | 0.982556 | 0.982571 |

Per-class mean precision/recall/F1 below; full run-specific support and metrics and mean/SD/min/max per class are in per_class_results.csv and aggregate_results.json.

| Strategy | Level | Class | precision | recall | f1 |
|---|---|---|---|---|---|
| unweighted | window | Normal | 0.866708 | 0.949900 | 0.905713 |
| unweighted | window | Yawning | 0.963748 | 0.857955 | 0.907217 |
| unweighted | window | Eye Close | 0.952378 | 0.960307 | 0.955966 |
| unweighted | video | Normal | 0.981481 | 0.993464 | 0.987109 |
| unweighted | video | Yawning | 0.981481 | 1.000000 | 0.990476 |
| unweighted | video | Eye Close | 0.993464 | 0.960784 | 0.976629 |
| weighted_ce | window | Normal | 0.869855 | 0.945050 | 0.905057 |
| weighted_ce | window | Yawning | 0.959870 | 0.861333 | 0.907340 |
| weighted_ce | window | Eye Close | 0.949969 | 0.960056 | 0.954719 |
| weighted_ce | video | Normal | 0.981481 | 0.986928 | 0.983742 |
| weighted_ce | video | Yawning | 0.975309 | 1.000000 | 0.987302 |
| weighted_ce | video | Eye Close | 0.993464 | 0.960784 | 0.976629 |
| weighted_sampler | window | Normal | 0.864048 | 0.949832 | 0.904381 |
| weighted_sampler | window | Yawning | 0.953947 | 0.868453 | 0.908542 |
| weighted_sampler | window | Eye Close | 0.964010 | 0.949421 | 0.956503 |
| weighted_sampler | video | Normal | 0.981481 | 0.986928 | 0.983742 |
| weighted_sampler | video | Yawning | 0.969136 | 1.000000 | 0.984127 |
| weighted_sampler | video | Eye Close | 1.000000 | 0.960784 | 0.979798 |

## 27–28. Confusion matrices and variability

Rows actual, columns predicted; fixed Normal/Yawning/Eye Close order. CSV matrices separately sum all 3 folds for each fixed seed. Pooled tables below additionally sum 3 seeds, so support is 8,043 window predictions and 459 video predictions, representing 2,681 distinct windows  / 153 videos each evaluated 3 times. Repeated seed predictions are not independent data or subjects. These are development matrices, not test matrices.

**unweighted, window, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2582 | 71 | 65 |
| Yawning | 312 | 2275 | 65 |
| Eye Close | 90 | 16 | 2567 |

**unweighted, video, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 152 | 0 | 1 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**weighted_ce, window, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2569 | 80 | 69 |
| Yawning | 301 | 2284 | 67 |
| Eye Close | 89 | 18 | 2566 |

**weighted_ce, video, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 151 | 1 | 1 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

**weighted_sampler, window, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 2582 | 88 | 48 |
| Yawning | 302 | 2303 | 47 |
| Eye Close | 108 | 27 | 2538 |

**weighted_sampler, video, pooled three seeds**

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---|---|---|
| Normal | 151 | 2 | 0 |
| Yawning | 0 | 153 | 0 |
| Eye Close | 3 | 3 | 147 |

| Strategy | Metric | mean | std | minimum | maximum |
|---|---|---|---|---|---|
| unweighted | window_macro_f1 | 0.922965 | 0.008070 | 0.914583 | 0.934732 |
| unweighted | window_accuracy | 0.923083 | 0.007580 | 0.914539 | 0.934463 |
| unweighted | macro_recall | 0.922721 | 0.007976 | 0.914140 | 0.934884 |
| unweighted | worst_class_recall | 0.857955 | 0.034490 | 0.795918 | 0.910959 |
| unweighted | video_macro_f1 | 0.984738 | 0.013079 | 0.960767 | 1.000000 |
| unweighted | video_accuracy | 0.984749 | 0.013072 | 0.960784 | 1.000000 |
| weighted_ce | window_macro_f1 | 0.922372 | 0.007059 | 0.914775 | 0.933705 |
| weighted_ce | window_accuracy | 0.922456 | 0.006682 | 0.914539 | 0.933333 |
| weighted_ce | macro_recall | 0.922146 | 0.007017 | 0.914140 | 0.933213 |
| weighted_ce | worst_class_recall | 0.861333 | 0.029633 | 0.795918 | 0.886986 |
| weighted_ce | video_macro_f1 | 0.982558 | 0.011789 | 0.960767 | 1.000000 |
| weighted_ce | video_accuracy | 0.982571 | 0.011783 | 0.960784 | 1.000000 |
| weighted_sampler | window_macro_f1 | 0.923142 | 0.005660 | 0.916470 | 0.929464 |
| weighted_sampler | window_accuracy | 0.922938 | 0.005261 | 0.917318 | 0.928814 |
| weighted_sampler | macro_recall | 0.922569 | 0.005430 | 0.916616 | 0.928744 |
| weighted_sampler | worst_class_recall | 0.868453 | 0.025729 | 0.809524 | 0.893836 |
| weighted_sampler | video_macro_f1 | 0.982556 | 0.006542 | 0.980375 | 1.000000 |
| weighted_sampler | video_accuracy | 0.982571 | 0.006536 | 0.980392 | 1.000000 |

Raw JSON/CSV is authoritative. Plots: strategy_variability.png (all 9 run points), per_class_recall.png (mean±sampleSD), paired_delta.png and development_confusion.png under results/phase12/plots.

## 22, 29–30. Paired differences and decision

Each delta is strategy minus unweighted at identical fold and seed. Both the 9 raw deltas and their descriptive distribution are reported; pooled means alone are insufficient.

| Strategy | Metric delta | mean | std | minimum | maximum | positive_pairs | negative_pairs |
|---|---|---|---|---|---|---|---|
| weighted_ce | window_macro_f1 | -0.000593 | 0.002191 | -0.004890 | 0.002299 | 4 | 5 |
| weighted_ce | window_accuracy | -0.000628 | 0.002107 | -0.004520 | 0.002220 | 3 | 4 |
| weighted_ce | macro_recall | -0.000574 | 0.002024 | -0.004081 | 0.002259 | 4 | 4 |
| weighted_ce | worst_class_recall | 0.003378 | 0.018335 | -0.023973 | 0.034014 | 4 | 3 |
| weighted_ce | video_macro_f1 | -0.002181 | 0.006542 | -0.019625 | 0.000000 | 0 | 1 |
| weighted_ce | video_accuracy | -0.002179 | 0.006536 | -0.019608 | 0.000000 | 0 | 1 |
| weighted_sampler | window_macro_f1 | 0.000177 | 0.005380 | -0.005490 | 0.012697 | 3 | 6 |
| weighted_sampler | window_accuracy | -0.000145 | 0.005330 | -0.005650 | 0.012209 | 3 | 6 |
| weighted_sampler | macro_recall | -0.000152 | 0.005469 | -0.006257 | 0.012429 | 3 | 6 |
| weighted_sampler | worst_class_recall | 0.010498 | 0.040664 | -0.047619 | 0.078231 | 6 | 3 |
| weighted_sampler | video_macro_f1 | -0.002182 | 0.011789 | -0.019625 | 0.019608 | 1 | 2 |
| weighted_sampler | video_accuracy | -0.002179 | 0.011783 | -0.019608 | 0.019608 | 1 | 2 |

Predeclared before any training: mean F1 gain at least 0.002 (0.2 percentage points), positive on at least 5/9 pairs, mean worst-class recall regression no more than 0.005, no class mean recall regression above 0.01, F1 sample SD increase no more than 0.01. These are engineering benefit gates, not a statistical significance claim. They were neither derived from test results nor revised after seeing CV results.

- weighted_ce: rejected under the predeclared benefit rule; gates {"class_recalls": true, "majority_positive": false, "mean_gain": false, "variability": true, "worst_recall": true}.
- weighted_sampler: rejected under the predeclared benefit rule; gates {"class_recalls": false, "majority_positive": false, "mean_gain": false, "variability": true, "worst_recall": true}.

weighted_ce: mean paired window macro-F1 change -0.000593, positive in 4/9 pairs, mean worst-class recall change +0.003378. weighted_sampler: mean paired window macro-F1 change +0.000177, positive in 3/9 pairs, mean worst-class recall change +0.010498.

Recommendation: **unweighted**. If neither balancing treatment passes every gate, ordinary unweighted CE is retained for simplicity. An accepted treatment is only an experimental recommendation for later Phase 15 selection; no application checkpoint is replaced. All class-recall gains/losses are visible above. No new resume metric is added.

## 31–33. Access audit and regression

**Number of Phase 11 test model inference passes in Phase 12: 0.** test_access_audit.json and the 27-entry candidate_access_audit.jsonl list exact training/validation IDs and zero test access. No candidate test_predictions files or candidate test metrics/confusions exist. Raw cache provenance lists 153 development reads and zero test cache reads. Source guards, sealed duplicate protection and actual file-open refusal have unit tests. The report recomputes metrics from saved validation probabilities and independently verifies window F1/matrices with sklearn; it performs no candidate model inference.

Full suite: **475 passed**, including all 459 Phase 2–11tests and 16 Phase 12 tests; no failures/errors/skips. Fresh authorized production regression, using the existing immutable raw fixture inputs: **699/699 windows/classes exact**, maximum difference 0 in every compared field. This is production parity, not candidate evaluation on the sealed Phase 11 test cohort. No Phase 8/9/SQLite benchmark artifacts overwritten.

## 34. Limitations

Video independence only: trustworthy subject IDs are absent, and driver overlap may remain. Single dataset, 153 development videos and three deterministic folds/seeds; folds share training data and seeds reuse validation sources. Reported SD/deltas are descriptive correlated observations, not independent confidence intervals or statistical superiority proof. Validation selects epochs and strategies; these are optimistic development estimates, not untouched final test performance. Whole-video labels do not measure event onset, alert detection or clinical safety. Near-balanced classes limit the rationale for balancing, and window sampling with replacement may duplicate/omit windows and still gives longer videos greater exposure. No other hardware/library reproducibility claim. Production/model promotion remains out of scope.

## 35–36. Reproduction and status

From repository root, use baseline environment, `-B`, and `MPLCONFIGDIR=.cache/matplotlib`. To reproduce the complete experiment in a NEW output directory (existing canonical outputs are guarded against overwriting):

```powershell
$env:MPLCONFIGDIR = '.cache/matplotlib'
.venv-baseline/Scripts/python.exe -B -m evaluation.balancing prepare --output results/phase12_reproduction
.venv-baseline/Scripts/python.exe -B -m evaluation.balancing run --output results/phase12_reproduction
.venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase12_tests_reproduction.xml
.venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase12_production_parity_reproduction/result.json
```

The frozen pre-experiment docs/phase12_before_hashes.json is required for cache provenance. Same software/CPU seed protocol is recorded; runtime seconds and serialized checkpoint file hashes need not reproduce across environments. Authoritative canonical commands were `prepare --output results/phase12`, `run --output results/phase12`, pytest with docs/phase12_tests.xml, production parity with results/phase12_production_parity/result.json, and `tools/report_phase12.py`. Reporter consumes canonical paths and writes new-only files; do not rerun it over completed outputs. Reproduction runs can be compared directly using each config/history/validation_predictions/metrics JSON/NPZ. No test inference command is part of this protocol.

**PHASE 12 PASS. Stop after Phase 12. Phase 13 not begun.**
