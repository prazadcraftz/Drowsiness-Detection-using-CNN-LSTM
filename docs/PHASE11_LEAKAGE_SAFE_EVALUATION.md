# Phase 11 — leakage-safe, video-independent evaluation

**PHASE 11 PASS.** One predeclared CNN-LSTM baseline recipe from scratch, selected on validation window macro F1; final test evaluated once. A separate train/validation-only repeat verifies exact weights, history and selection without further test inference. Clean window accuracy **94.1909%**, macro F1 **0.941808**. Video accuracy **100.0000%**, macro F1 **1.000000**, 27 held-out videos. These are video-independent results, not subject-independent or event-level accuracy. Production runtime/artifacts remain unchanged. Stop after Phase 11.

## 1–2. Files and preservation

Created code/tests: `evaluation/__init__.py`, `evaluation/evaluate.py`, `evaluation/manifest.py`, `evaluation/metrics.py`, `evaluation/preprocess.py`, `evaluation/reproducibility.py`, `evaluation/splits.py`, `evaluation/train.py`, `tests/test_evaluation_manifest.py`, `tests/test_evaluation_metrics.py`, `tests/test_evaluation_preprocess.py`, `tests/test_evaluation_splits.py`, `tests/test_evaluation_training.py`, `tools/report_phase11.py`, `tools/verify_phase11_features.py`, `tools/verify_phase11_training.py`. Created this report, docs/phase11_before_hashes.json, docs/phase11_validation.json, docs/phase11_unit.xml, docs/phase11_tests.xml, new preprocessing/training/evaluation logs, all offline artifacts in results/phase11/, new parity evidence in results/phase11_production_parity/, and the two experimental checkpoint files in models/phase11/.

**Modified prior files: none.** All 2703 inventoried prior files are byte-identical, including all production code, old tests and Phase 2–10 evidence. saved_model.pt, saved_model_0.95.pt, norm.npy, preprocess_window.npy, preprocess_label.npy and class_idx.npy retain their exact pre-phase hashes. No runtime imports evaluation; no model promotion, dependencies or application settings changed. Existing unrelated working-tree changes were preserved.

## 3–8. Dataset manifest, mapping and grouping limitation

180 raw videos discovered, 180 usable, 0 excluded. Only the three explicit directories Normal, Yawning, Eye Close enter the dataset; sorted raw inventory and stable relative-path source IDs avoid filesystem-enumeration label order. Source content SHA-256 defines grouping; exact byte duplicates would be co-grouped, conflicting labels fail. Observed duplicate-content count: 0.

Mapping: **0=Normal, 1=Yawning, 2=Eye Close**. No trustworthy subject/recording IDs were supplied or found in repository metadata; filename numbers, appearance and guessed recording order were not used as identity. Each unique raw-video content group is independent at the split boundary. Shared people/scenes/recording conditions across groups may still exist. This is **video-independent**, never subject-independent.

dataset_manifest_raw.csv is the immutable pre-extraction inventory (source_id, class_name/id, relative_path, file_sha256, FPS, width/height, header count, group_id). splits.json references that file's SHA-256. dataset_manifest.csv adds assigned split, decoded/valid observations, window/padded counts, status and notes after processing. Both CSVs have the same deterministic sorted source identities. Final manifest SHA-256: `b70d087c660077ae5734ff9f59a7698937fbfe95906820f0c1074a1d23564827`. Headers are audited separately from actual successful decode counts; missing-face frames remain counted. Material decode/feature/empty-source failures record the source/split/error and abort instead of silently dropping data.

## 9–13. Split protocol, seed and leakage checks

Seed **20260927**; class-stratified 70/15/15 raw-video group split. Each class: **42 train / 9 validation / 9 test**; total **126 / 27 / 27** videos. Assignments are persisted before any extraction/window creation. Exact assignments, group strategy, proportions, class counts, source-ID lists, assignment hash, raw manifest hash, code/config hashes are in splits.json.

Split manifest SHA-256: `5e3256ea3e3becd61e05c54474875091a71466c28b5e8ee46089cff37542f602`.
Assignment SHA-256: `3257e54bc938300ddade930de581465fa636fc926571658a0e7ebe1a4f8b65e5`.

Automated hard gates: complete disjoint source/group coverage; every window's source and class matches its split; observation identities are (source_id, decoded_frame_index), not bare frame numbers; no observation identity crosses splits; the normalizer's source list equals exactly TRAIN. Overlapping windows remain within one source/split. integrity.json: `{"counts": {"test": {"sources": 27, "unique_source_observation_identities": 3733, "windows": 482}, "train": {"sources": 126, "unique_source_observation_identities": 17144, "windows": 2204}, "validation": {"sources": 27, "unique_source_observation_identities": 3709, "windows": 477}}, "cross_split_observation_overlap": 0, "normalization_train_only": true, "passed": true, "source_group_overlap": 0}`. Tests rerun splitting with the same seed and reversed input enumeration, requiring identical assignments; a different seed changes the synthetic assignments. No split leakage was detected.

## 14–17. Feature extraction, windows and train-only normalization

Unchanged legacy feature order EAR-R, EAR-L, MAR, phi, theta. The same FaceDetector/get_detector and extract_features/draw_landmarks implementations retain IMAGE mode, one face, blendshape/matrix outputs, BGR tagged SRGB, the legacy landmark order, normalized-coordinate solvePnP and extra degree conversion. No geometry/preprocessing correction occurred. Independent comparison of the newly extracted Phase 11 raw caches against all six frozen clips matches 789 valid observation identities and feature vectors exactly (max difference 0.0); evidence: legacy_feature_parity.json.

For historical training comparability, every decoded frame is passed through the detector; valid face observations are retained in chronology, compressing missing-face gaps as historical training did. Runtime 20-Hz sampling/reset logic is not substituted into training. No window crosses a video boundary. Input **(5,16)**, stride **8 observations in all splits**, starts range(0,n,8), short tails repeat the last observation exactly. This can create one or two padded windows/source, including windows from very short videos; padding shares below are disclosed, not removed post hoc. Whole source-video class is used as the window label. Held-out normalized values are not clipped to [0,1].

Primary normalizer preserves **legacy zero-initialized extrema**, fitted only from 126 training sources. TRAIN extraction completes, train_norm.npy is fitted/frozen, then validation/test are extracted and use that frozen artifact. Neither validation/test extrema nor norm.npy enter fitting. A dedicated regression changes held-out extremes to 1e9 and confirms identical train ranges/source IDs. No correct-extrema experiment was run. Artifact min/max in feature order:

```json
[
  [
    0.0,
    0.9574378410040649
  ],
  [
    0.0,
    0.8781379045532656
  ],
  [
    0.0,
    2.07656651642746
  ],
  [
    -10264.639129681691,
    10280.159706808952
  ],
  [
    -1946.106035891749,
    4764.083494977571
  ]
]
```

All ranges finite; all max-minus-min denominators positive. Normalizer SHA-256: `bd9523c2b3fc57d2ca588765400b84471242cc377e7ef8986983da1c7c4a50f6`. train_norm.json records feature names, exact training source IDs, denominators, method and split hash. Raw feature caches contain finite offline numerical observations/frame identities, not images or video blobs; caches/window arrays are experimental training artifacts, separate from runtime SQLite.

## Dataset/split statistics and failure disclosure

| Split | Class | Videos | Decoded | Valid observations | Windows | Padded | Padded share | Window share | No-face frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | Normal | 42 | 5880 | 5804 | 745 | 83 | 11.14% | 33.80% | 76 |
| train | Yawning | 42 | 5844 | 5671 | 729 | 82 | 11.25% | 33.08% | 173 |
| train | Eye Close | 42 | 5815 | 5669 | 730 | 80 | 10.96% | 33.12% | 146 |
| validation | Normal | 9 | 1257 | 1257 | 161 | 18 | 11.18% | 33.75% | 0 |
| validation | Yawning | 9 | 1246 | 1197 | 155 | 18 | 11.61% | 32.49% | 49 |
| validation | Eye Close | 9 | 1255 | 1255 | 161 | 18 | 11.18% | 33.75% | 0 |
| test | Normal | 9 | 1260 | 1248 | 161 | 18 | 11.18% | 33.40% | 12 |
| test | Yawning | 9 | 1257 | 1239 | 160 | 17 | 10.62% | 33.20% | 18 |
| test | Eye Close | 9 | 1246 | 1246 | 161 | 18 | 11.18% | 33.40% | 0 |


Excluded sources: `[]`. Sources with fewer than 16 valid observations: `[{"source": "Eye Close/Eye Close_51.avi", "split": "train", "decoded": 135, "valid": 10, "windows": 2, "padded": 2}]`. They remain included with disclosed historical padding; no face-coverage exclusion threshold was introduced. Preprocessing completed in 810.768 s on this CPU; this descriptive wall time is not a new performance benchmark. Full source-level progress/counts are in preprocessing_log.jsonl and dataset_manifest.csv. Window shares are natural; there is no class weighting, weighted sampler, oversampling or undersampling.

## 18–22. Training recipe and validation selection

Fresh existing **CNN_LSTM, 246595 parameters**, unchanged convolution/LSTM dimensions, 5 inputs / 16 observations / 3 outputs, CNN and LSTM dropout 0.25. CPU, float32 inputs, batch size 64, shuffle train only; standard unweighted CrossEntropyLoss, Adam fixed learning rate 0.001, no scheduler. This deliberately replaces the historical batch-level focal-like transform; it is not called focal loss. One recipe, no loss/hyperparameter/architecture/window search.

Predeclared maximum 60 epochs, patience 10 non-improving validation macro-F1 epochs, minimum improvement 1e-12, earliest epoch wins ties. Test arrays are never loaded in train.py. Validation/test use model.eval() and torch.inference_mode(); dropout-active evaluation is blocked by synthetic tests. Each best validation checkpoint is saved separately; the selected state is restored rather than using the last epoch.

Completed **15 epochs**, stop reason **patience**, training-loop wall time 15.135 s. Selected epoch **5**, validation macro F1 **0.923752**. Raw curves: training_history.json and training_history.csv (train/validation loss/accuracy, validation macro F1, LR). Plot: results/phase11/plots/training_curves.png. Selected train/validation eval-mode metrics are in selected_train_validation_metrics.json.

| Selected checkpoint, eval mode | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Windows |
|---|---:|---:|---:|---:|---:|---:|
| train | 92.4682% | 0.927621 | 0.924487 | 0.924697 | 0.924593 | 2204 |
| validation | 92.4528% | 0.927564 | 0.923623 | 0.923752 | 0.923960 | 477 |


Training per-class metrics at the selected checkpoint (eval mode):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.870690 | 0.948993 | 0.908157 | 745 |
| Yawning | 0.954407 | 0.861454 | 0.905552 | 729 |
| Eye Close | 0.957766 | 0.963014 | 0.960383 | 730 |


Validation per-class metrics at the selected checkpoint (eval mode):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.898810 | 0.937888 | 0.917933 | 161 |
| Yawning | 0.970588 | 0.851613 | 0.907216 | 155 |
| Eye Close | 0.913295 | 0.981366 | 0.946108 | 161 |


History train accuracy/loss are accumulated during training with dropout and changing minibatch weights; this selected-checkpoint table uses fixed weights and dropout disabled. Both meanings are explicit.

## 23–28. One-time final window-level test

Accuracy **94.1909%**, macro precision **0.943419**, macro recall **0.941874**, macro F1 **0.941808**, weighted F1 **0.941781**, support **482 windows**. Classes are fixed in the order below; zero-support precision/recall/F1 are zero, macro averages always include all three classes.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.924051 | 0.906832 | 0.915361 | 161 |
| Yawning | 0.986667 | 0.925000 | 0.954839 | 160 |
| Eye Close | 0.919540 | 0.993789 | 0.955224 | 161 |


Window confusion matrix: rows actual, columns predicted. CSV: confusion_matrix.csv; PNG: plots/confusion_matrix.png.

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---:|---:|---:|
| Normal | 146 | 2 | 13 |
| Yawning | 11 | 148 | 1 |
| Eye Close | 1 | 0 | 160 |


## 29–33. Video-level test and supports

Mean window probabilities per held-out source, then argmax; every video contributes one decision, so longer videos do not get extra votes. Accuracy **100.0000%**, macro F1 **1.000000**, support **27 videos**, nine per class.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 1.000000 | 1.000000 | 1.000000 | 9 |
| Yawning | 1.000000 | 1.000000 | 1.000000 | 9 |
| Eye Close | 1.000000 | 1.000000 | 1.000000 | 9 |


Video confusion matrix: rows actual, columns predicted. CSV: video_confusion_matrix.csv; PNG: plots/video_confusion_matrix.png.

| Actual / predicted | Normal | Yawning | Eye Close |
|---|---:|---:|---:|
| Normal | 9 | 0 | 0 |
| Yawning | 0 | 9 | 0 |
| Eye Close | 0 | 0 | 9 |


One final forward-evaluation pass generated test_predictions.npz; both levels derive from those same persisted probabilities. test_evaluation_started.json is an exclusive guard preventing a second canonical test invocation, created only after checkpoint selection. test_metrics.json, video_test_metrics.json and classification_report.csv retain exact values; per-video labels, mean probabilities and predictions are included in video_test_metrics.json. The report recomputes both levels from saved probabilities, and independently cross-checks window metrics/confusion against sklearn; it performs no second model inference on test. No hyperparameter/checkpoint decisions were made after seeing test results.

## 34–35. Historical vs clean methodology

| Aspect | Historical repository artifact | Phase 11 clean baseline |
|---|---|---|
| Split unit | Random overlapping windows, correlated same-video observations | Raw videos/content groups first |
| Normalization | Whole corpus extrema before split, zero initialization | TRAIN only, preserving zero initialization |
| Assignment evidence | No fixed source split manifest | Hashed raw inventory and exact seeded split manifest |
| Identity limitation | No subject/video isolation | Video-independent holdout; subject IDs unavailable |
| Evaluation | Dropout/mismatched-checkpoint issues in old scripts | eval + inference_mode, selected checkpoint |
| Selection | No true validation; final epoch saved | Predeclared validation macro F1, early stopping |
| Loss/recipe | Batch-level focal-like transform and varying training scripts | Unweighted CE, fixed Adam recipe |
| Test use | Historical methodology cannot establish independence | One final evaluation after selection |

Historical displayed confusion matrix: approximately **97.6842% accuracy / 0.976785 macro F1** under the old methodology (CURRENT_ARCHITECTURE.md audit), **not the clean benchmark**. Do not interpret a difference as proof that the architecture improved/worsened: source isolation, normalization fit domain, seeded recipe, loss, selection and evaluation correctness differ. The stricter evaluation supersedes historical numbers for generalization/resume claims. Historical production checkpoint characterization was **not run**; its unknown training exposure prevents clean independence claims.

## 36–37. Candidate checkpoint and reproducibility

Experimental checkpoint: `models/phase11/cnn_lstm_clean_split.pt`.
SHA-256: `145fcf0ad66a85102cfb1f3c0179bc1fb159005a84d8270131e57e9c465e0b24`.
Metadata: models/phase11/cnn_lstm_clean_split.json records architecture/parameter count, feature order, sequence length, train normalizer/split hashes, seed, best epoch/validation metric, final test metrics, recipe, training-history hash and environment. **Production saved_model.pt was not replaced.**

Python/NumPy/PyTorch seeds fixed; DataLoader generator fixed, worker seeding defined, num_workers=0. Deterministic algorithms enabled; Torch intra/inter threads 8/8, CPU. Environment: `{"bit_identical_training_verified": true, "cpu": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel", "deterministic_algorithms": true, "device": "cpu", "logical_cpu_count": 12, "mkldnn_enabled": true, "platform": "Windows-11-10.0.26220-SP0", "python": "3.12.14", "software": {"mediapipe": "1.0.1", "numpy": "2.5.3", "opencv-contrib-python": "5.0.0.93", "torch": "2.14.0+cpu"}, "torch_interop_threads": 8, "torch_threads": 8}`. Source/config/asset hashes are saved before training and checked again before training begins. Same-seed split reproducibility and synthetic loop behavior are tested. A fresh-process repetition of the identical full train/validation recipe reproduced **all selected weight tensors bit-for-bit, all 15 epoch metric rows exactly, best epoch 5 and stopping epoch 15**. It loaded only train and validation, made zero test inference passes, did not overwrite the canonical checkpoint and made no recipe/selection changes. Evidence: reproducibility_check.json and phase11_reproducibility_stdout.log. This establishes repeatability for these two runs on this CPU/software; it does not guarantee bit identity across other hardware or library versions. Artifacts provide recipe/data/split/normalization/prediction/checkpoint reproduction evidence.

Run commands from the repo root with the baseline venv and MPLCONFIGDIR=.cache/matplotlib:

```powershell
& .venv-baseline/Scripts/python.exe -B -m evaluation.preprocess --output results/phase11
& .venv-baseline/Scripts/python.exe -B -m evaluation.train --artifacts results/phase11 --models models/phase11
& .venv-baseline/Scripts/python.exe -B -m evaluation.evaluate --artifacts results/phase11 --models models/phase11
& .venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase11_tests.xml
& .venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase11_production_parity/result.json
```

Existing evidence is exclusive/immutable. For independent reproduction use fresh results/models directories and the same unchanged code/data/protocol, never overwrite this canonical run or re-evaluate its final test. Reproduction is a verification exercise, not permission to tune on the exposed holdout. Importing evaluation modules never launches training; main()/CLI are explicit entry points.

## 38–42. Regression, limitations, resume wording and outcome

**459 tests passed**, including all 435 prior Phase 2–10 cases and 24 Phase 11 tests; no failures/errors/skips. Counts by phase: `{'Phase 2': 9, 'Phase 3': 43, 'Phase 4': 53, 'Phase 5': 60, 'Phase 6': 64, 'Phase 7': 60, 'Phase 8': 27, 'Phase 9': 64, 'Phase 10': 55, 'Phase 11': 24}`. Prior tests/assertions are byte-identical. Fresh production replay: **699/699 frozen raw windows and classes exact**, maximum numeric difference **0.0**; landmarks/policy/event/session/dashboard traces also exact. New training normalization/checkpoint never enter the production parity run. Evidence: docs/phase11_tests.xml and results/phase11_production_parity/result.json.

Limitations: one fixed split/seed/recipe, with a same-seed verification repeat; 27 test videos; correlated overlapping/padded windows within each video; no repeated-seed uncertainty estimate; no trustworthy driver/recording IDs, so shared people/scenes across videos may remain. Whole-video action labels are coarse and provide no reliable onset/frame/event ground truth. No alert precision, eye/yawn event recall or onset latency accuracy can be calculated here. Features retain known legacy geometry/color/pose quirks; normalization retains zero-anchor quirks. No balancing, architectures, window sizes, corrected-extrema/preprocessing experiments, production promotion, deployment/cloud work or Phase 12+ work was done. Results do not establish clinical validity or new-driver generalization. Test performance was not used to tune this recipe.

Resume-safe wording: “Built a leakage-safe CNN-LSTM evaluation pipeline with raw-video splits and train-only normalization, achieving **94.19% window-level video-independent test accuracy / 0.9418 macro F1** and **100.00% video-level accuracy / 1.0000 macro F1** on 27 held-out videos, while preserving 699/699 exact production outputs.” Use these clean results only; never substitute the historical number or claim subject/event independence.

**PHASE 11 PASS. STOP after Phase 11.**
