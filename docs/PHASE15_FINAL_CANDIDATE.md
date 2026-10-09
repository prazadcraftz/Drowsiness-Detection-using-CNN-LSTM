# Phase 15: frozen LSTM-only full-development candidate

**Status: PHASE15 PASS.** This phase froze one choice, fitted it on the 153 Phase 11 development videos, and stopped before runtime integration or further evaluation. The 27 Phase 11 held-out videos were excluded from every Phase 15 candidate fit and inference. They **were already evaluated by the historical Phase 11 baseline** and are therefore not a pristine untouched final test set. No final generalization estimate is claimed for the new candidate.

## 1–10. Selection boundary, evidence, and trade-offs

The choice in [`selection_spec.json`](../results/phase15/selection_spec.json) was written **before any Phase 15 model training**. The quality-first final-choice rule was frozen **after observing the completed Phase 14 development study**, not predeclared before Phase 14. Its sole choice is LSTM-only, 32 observations, training stride 16, and unweighted cross-entropy. No architecture, window, loss, optimizer, or hyperparameter search was reopened.

| Phase 14 development architecture, N=32 | Mean window macro F1 across nine runs |
|---|---:|
| CNN–LSTM | 0.945269 |
| **LSTM-only** | **0.949829** |
| CNN–GRU | 0.944414 |
| CNN–BiLSTM | 0.947787 |

The selected configuration had the highest observed mean in the complete Phase 14 matrix, 9/9 positive paired F1 deltas against LSTM-only N=16 (mean +0.017553), mean window accuracy 0.949755, macro recall 0.949599, and worst-class/Yawning recall 0.910781. These are **development cross-validation metrics**, not final or held-out candidate performance. The historical Phase 11 baseline's 94.1909% window accuracy, 0.941808 macro F1, and 100% video accuracy do not belong to this candidate.

The selected model has 209,667 trainable parameters versus the original CNN–LSTM's 246,595. Phase 14 model-only P50 was 5.33975 ms for LSTM-only N=32 versus 3.8643 ms for CNN–LSTM N=32, so the selection accepts a slower model-only forward pass for the observed quality. N=32 means 1.60 s nominal coverage and 1.55 s first-to-last span at 20 Hz, twice the N=16 nominal warm-up, and more padded windows. All N=32 architectures have the same observation warm-up. Raw application timing has been dominated by vision, but Phase 15 does not measure full application latency or establish that integration has no cost.

## 11–17. Cohort, preprocessing, windows, and model

The exact 153 development source IDs are stored in the selection specification, normalizer metadata, and window manifest. There are 51 Normal, 51 Yawning, and 51 Eye Close videos. Their set has empty intersection with the exact 27 excluded Phase 11 source IDs. The pre-normalization feature caches were checked against source and cache SHA256, declared source IDs, valid observation counts, finite features, and increasing original frame indices; the source guard was installed before access. No MediaPipe extraction was rerun. Feature order remains EAR-R, EAR-L, MAR, phi, theta.

The **new** [`lstm32_dev_norm.npy`](../models/phase15/lstm32_dev_norm.npy) fits legacy zero-initialized extrema over **all and only** those 153 development videos. Neither the Phase 11 train normalizer nor a Phase 12 fold normalizer was used. Its SHA256 is `9b397ae1327b515c5505610fffa37e198eb5ccd24ec7bf1d112eacec0f950edc`. Its [metadata](../models/phase15/lstm32_dev_norm.json) records source IDs, source-set hash, per-feature extrema, denominators, method, and artifact hash.

| Feature | Minimum | Maximum |
|---|---:|---:|
| EAR-R | 0 | 0.9574378410 |
| EAR-L | 0 | 0.8781379046 |
| MAR | 0 | 2.0765665164 |
| phi | -10264.6391296817 | 10280.1597068090 |
| theta | -3165.2259874313 | 4764.0834949776 |

The full-development [window manifest](../results/phase15/window_dataset.json) records 1,352 source-contained N=32 windows at stride 16, with repeat-last padding in 304 windows (22.485%). Normal has 455 windows/102 padded/7,061 valid observations; Yawning 448/102/6,868; Eye Close 449/100/6,924. Every source appears exactly once in the source manifest. Eye Close_51 remains present with 10 valid observations from 135 decoded frames, yielding one padded N=32 window. The [windows](../results/phase15/development_windows.npz) are generated from the new normalizer and include source, start, observation, frame, label, and padding identities. The external input contract is float32 `(B,5,32)` in chronological order; the LSTM receives `(B,32,5)` and outputs `(B,3)` logits for 0 Normal, 1 Yawning, 2 Eye Close.

The model uses the frozen Phase 13 LSTM-only module registry through the Phase 14 variable-length wrapper: two-layer, hidden-128, batch-first LSTM with input size 5, external dropout 0.25, and dense 128→64→3 with ReLU. The parameter count is exactly **209,667**. No Phase 13/14 architecture implementation was modified.

## 18–27. Epoch rule, refit, checkpoint, repeat, and latency

The nine frozen Phase 14 LSTM-only N=32 best validation epochs, ordered by fold 0/1/2 then seeds 1201/1202/1203, were **13, 14, 7, 5, 21, 22, 20, 12, 5**. Their exact integer median is **13**; the nine history paths and SHA256 hashes were captured in the selection specification before training. The new seed **1501** was fixed at the same time and was not selected from performance.

The one canonical fit used all 1,352 development windows, CPU float32, deterministic algorithms, Torch intra/inter threads 8/8, batch size 64, shuffle enabled, no workers, unweighted `CrossEntropyLoss`, Adam at 0.001, and exactly 13 epochs. There was no validation/test loader, scheduler, early stopping, or checkpoint selection. The final epoch weights are the canonical [checkpoint](../models/phase15/lstm_only_w32_full_development.pt), with [metadata](../models/phase15/lstm_only_w32_full_development.json). SHA256 is `1dcdcdbd17378190a6b7c429fa5ccf5b621dcf0adae6541d89d789a1d5b764ed`; size is **843,805 bytes**. The normalizer is **208 bytes**. Serialized byte sizes are distinct from the parameter count. Neither production `saved_model.pt` nor the Phase 11 checkpoint was replaced.

The [training history](../results/phase15/training_history.json) and [CSV](../results/phase15/training_history.csv) record all 13 epochs. Last-epoch training loss was 0.189521 and training accuracy 0.947485. These are **in-sample training diagnostics only**, not validation, test, final, or generalization accuracy.

The one fresh-process reproducibility fit used the same source set, saved full-development normalization, windows, seed, epochs, and environment. Every state tensor, all history entries, and logits on the fixed 64-development-window [verification set](../results/phase15/verification_inputs.json) were bit-identical; maximum tensor/history/logit difference was **0.0**. The separately saved repeat weights were not used to replace the canonical checkpoint. Reloading the canonical checkpoint reproduced finite `(64,3)` logits and stable class IDs. See [repeat result](../results/phase15/reproducibility/result.json).

The final-candidate [latency result](../results/phase15/latency_results.json) uses 64 fixed development inputs under the **new** normalizer, batch 1, 128 warm-ups and 1,024 measured calls in each of three repetitions, 8/8 Torch threads, and no trimming. Across 3,072 retained wall-clock samples, mean was **3.3543 ms**, P50 **3.3396 ms**, P95 **4.1861 ms**, P99 **5.2618 ms**; no ≥100 ms stall occurred. The [raw wall and process CPU samples](../results/phase15/latency_samples.npz) are retained. Windows process CPU timings are coarse and sum threads. This is **model-only CPU latency**, not full application latency; it is a new checkpoint and input set, so the Phase 14 timing is contextual rather than an exact paired benchmark.

## 28–35. Runtime boundary, regression, files, and limitations

The [runtime contract](../results/phase15/runtime_contract.json) documents possible Phase 16 integration: current production CNN–LSTM N=16 `(B,5,16)` would need LSTM-only N=32 `(B,5,32)`, candidate checkpoint and normalizer, a 32-accepted-observation temporal buffer, and 32-observation first-prediction warm-up. Raw feature order, target 20 Hz sampling, class IDs, and ready-state inference stride 1 remain the intended contracts; face-gap semantics may remain unless future integration tests reveal an issue. **No production runtime integration occurred here.**

Phase 6 thresholds Normal ≥0.60, Yawning ≥0.60, Eye Close ≥0.70 were characterized with historical CNN–LSTM16 probabilities. LSTM-only32 and a new normalizer may change probability distributions. Phase 16 should first characterize compatibility with **frozen** thresholds; Phase 15 did not tune them. Whole-video labels lack event-onset ground truth and cannot justify optimizing alert thresholds against directory labels.

The [candidate access audit](../results/phase15/test_access_audit.json) records **0 candidate Phase 11 test inference passes, 0 test cache reads, 0 test video reads, and 0 blocked-open attempts**. The 27 Phase 11 videos were previously used for the historical baseline; they cannot now serve as pristine external final evidence. Phase 15 performed no external dataset evaluation and no held-out candidate inference.

The complete Phase 2–14 plus Phase 15 regression suite passed: **560 tests**. The fresh production parity [result](../results/phase15/production_parity/result.json) has **699/699 frozen windows exact**, 699/699 classes exact, and maximum raw-feature, normalized-feature, sequence, input-tensor, logit, and probability differences **0.0** on every source. All six parity video fixtures are Phase 11 development sources, not held-out sources. The preservation audit found all **4,629 pre-Phase 15 files byte-identical**, including production, model artifacts, and Phase 11–14 evidence. The preservation audit only hashed held-out bytes; candidate code did not decode or infer on them.

**Created files:** `evaluation/final_candidate.py`, `tests/test_phase15_final_candidate.py`, `tools/audit_phase15_preservation.py`; this report, `docs/phase15_before_hashes.json`, `docs/phase15_preservation.json`, `docs/phase15_tests_final.xml`, and `docs/phase15_validation.json`; the candidate checkpoint/metadata and normalizer/metadata in `models/phase15/`; the selection, window data/manifest, fixed verification inputs/identities, training history/CSV, canonical logits, reproducibility repeat and comparison, latency samples/results, runtime contract, test-access audit, and production parity artifacts in `results/phase15/`. **Modified pre-Phase 15 files: none.**

Reproduction commands, from the repository root with the existing baseline Python environment:

```powershell
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate freeze
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate prepare
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate train
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate repeat
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate latency
.\.venv-baseline\Scripts\python.exe -B -m evaluation.final_candidate audit
$env:MPLCONFIGDIR='.cache/matplotlib'
.\.venv-baseline\Scripts\python.exe -B -m pytest -q --junitxml=docs/phase15_tests_final.xml
.\.venv-baseline\Scripts\python.exe -B tools/optimize_phase9.py --parity production --output results/phase15/production_parity/result.json
.\.venv-baseline\Scripts\python.exe -B tools/audit_phase15_preservation.py
```

These commands are for a **fresh isolated copy** of the pre-Phase 15 workspace: artifact creation intentionally fails if the canonical outputs already exist, protecting the frozen selection and checkpoint. The `freeze` command reconstructs the exact selection-spec bytes from Phase 12–14 evidence and must run before `prepare` or `train`; its output hash was verified against this run. No Phase 16 work was begun.
