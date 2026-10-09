"""Generate the Phase 11 report only from completed experiment/regression evidence."""

from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.manifest import sha256, write_json
from evaluation.manifest import read_manifest
from evaluation.metrics import classification_metrics, aggregate_videos


def table_metrics(metrics):
    text = "| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n"
    for name in ("Normal", "Yawning", "Eye Close"):
        row = metrics["per_class"][name]
        text += f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |\n"
    return text


def table_matrix(metrics):
    text = "| Actual / predicted | Normal | Yawning | Eye Close |\n|---|---:|---:|---:|\n"
    for name, row in zip(("Normal", "Yawning", "Eye Close"), metrics["confusion_matrix"]):
        text += f"| {name} | " + " | ".join(map(str, row)) + " |\n"
    return text


def main():
    directory = ROOT / "results/phase11"
    load = lambda filename: json.loads((directory / filename).read_text())
    stats, splits, norm, protocol, history = [load(filename) for filename in ("dataset_statistics.json", "splits.json", "train_norm.json", "protocol.json", "training_history.json")]
    window, video, integrity = [load(filename) for filename in ("test_metrics.json", "video_test_metrics.json", "integrity.json")]
    selected_metrics = load("selected_train_validation_metrics.json")
    repeat = load("reproducibility_check.json")
    assert repeat["selected_weights_bit_exact"] and repeat["epoch_metrics_exact"] and repeat["selection_exact"]
    assert repeat["test_inference_passes"] == 0 and repeat["loaded_splits"] == ["train", "validation"]
    metadata = json.loads((ROOT / "models/phase11/cnn_lstm_clean_split.json").read_text())
    parity = json.loads((ROOT / "results/phase11_production_parity/result.json").read_text())
    feature_parity = load("legacy_feature_parity.json")
    assert feature_parity["exact"] and feature_parity["maximum_difference"] == 0.0
    xml = ET.parse(ROOT / "docs/phase11_tests.xml")
    cases = xml.findall(".//testcase")
    assert not any(xml.findall('.//'+tag) for tag in ("failure", "error", "skipped"))
    counts = Counter(case.attrib["classname"] for case in cases)
    new_tests = sum(value for name, value in counts.items() if name.startswith("tests.test_evaluation_"))
    assert len(cases)-new_tests == 435
    assert integrity["passed"] and integrity["cross_split_observation_overlap"] == 0
    assert norm["train_source_ids"] == splits["source_ids"]["train"]
    assert metadata["test_evaluated"] and window["final_test_evaluation"] and video["final_test_evaluation"]
    import numpy as np
    # Recompute metrics from already saved probabilities, without another model
    # test inference. This also gives an independent sklearn cross-check.
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as predictions:
        labels, probabilities, source_ids = [predictions[name] for name in ("labels", "probabilities", "source_ids")]
    recomputed = classification_metrics(labels, probabilities.argmax(1))
    assert all(window[key] == value for key, value in recomputed.items())
    recomputed_videos = aggregate_videos(labels, probabilities, source_ids)
    assert all(video[key] == value for key, value in recomputed_videos.items())
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    predicted = probabilities.argmax(1)
    assert accuracy_score(labels, predicted) == window["accuracy"]
    assert confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist() == window["confusion_matrix"]
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[0, 1, 2], average=average, zero_division=0)
        assert abs(f1-window["macro_f1" if average == "macro" else "weighted_f1"]) < 1e-12
        if average == "macro":
            assert abs(precision-window["macro_precision"]) < 1e-12 and abs(recall-window["macro_recall"]) < 1e-12
    assert parity["exact_parity"] and parity["windows_exact"] == parity["classes_exact"] == 699
    assert all(all(value == 0 for value in source["maximum_differences"].values()) for source in parity["sources"])
    before = json.loads((ROOT / "docs/phase11_before_hashes.json").read_text(encoding="utf-8-sig"))
    changed = {name: {"before": expected, "after": sha256(ROOT / name)} for name, expected in before.items() if sha256(ROOT / name) != expected}
    assert not changed, changed
    assert splits["manifest_sha256"] == sha256(directory / "dataset_manifest_raw.csv")
    assert metadata["checkpoint_sha256"] == sha256(ROOT / "models/phase11/cnn_lstm_clean_split.pt")
    stats_table = "| Split | Class | Videos | Decoded | Valid observations | Windows | Padded | Padded share | Window share | No-face frames |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for split in ("train", "validation", "test"):
        for name in ("Normal", "Yawning", "Eye Close"):
            row = stats["splits"][split][name]
            stats_table += f"| {split} | {name} | {row['videos']} | {row['decoded_frames']} | {row['valid_observations']} | {row['windows']} | {row['padded_windows']} | {100*row['padded_share']:.2f}% | {100*row['window_share']:.2f}% | {row['face_detection_failures']} |\n"
    counts_by_phase = json.loads((ROOT / "docs/phase10_validation.json").read_text())["phase_test_counts"]
    counts_by_phase["Phase 11"] = new_tests
    created = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "evaluation").glob("*.py")) + sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_evaluation_*.py")) + ["tools/report_phase11.py", "tools/verify_phase11_features.py", "tools/verify_phase11_training.py"]
    env = metadata["environment"]
    selected_table = "| Selected checkpoint, eval mode | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Windows |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for split in ("train", "validation"):
        metrics = selected_metrics[split]
        selected_table += f"| {split} | {100*metrics['accuracy']:.4f}% | {metrics['macro_precision']:.6f} | {metrics['macro_recall']:.6f} | {metrics['macro_f1']:.6f} | {metrics['weighted_f1']:.6f} | {metrics['support']} |\n"
    low_coverage = [{"source": row["relative_path"], "split": row["split"], "decoded": int(row["successfully_decoded_frame_count"]),
                     "valid": int(row["valid_face_observation_count"]), "windows": int(row["generated_window_count"]), "padded": int(row["padded_window_count"])}
                    for row in read_manifest(directory / "dataset_manifest.csv") if int(row["valid_face_observation_count"]) < 16]
    text = f"""# Phase 11 — leakage-safe, video-independent evaluation

**PHASE 11 PASS.** One predeclared CNN-LSTM baseline recipe from scratch, selected on validation window macro F1; final test evaluated once. A separate train/validation-only repeat verifies exact weights, history and selection without further test inference. Clean window accuracy **{100*window['accuracy']:.4f}%**, macro F1 **{window['macro_f1']:.6f}**. Video accuracy **{100*video['accuracy']:.4f}%**, macro F1 **{video['macro_f1']:.6f}**, {video['support']} held-out videos. These are video-independent results, not subject-independent or event-level accuracy. Production runtime/artifacts remain unchanged. Stop after Phase 11.

## 1–2. Files and preservation

Created code/tests: {', '.join('`'+name+'`' for name in created)}. Created this report, docs/phase11_before_hashes.json, docs/phase11_validation.json, docs/phase11_unit.xml, docs/phase11_tests.xml, new preprocessing/training/evaluation logs, all offline artifacts in results/phase11/, new parity evidence in results/phase11_production_parity/, and the two experimental checkpoint files in models/phase11/.

**Modified prior files: none.** All {len(before)} inventoried prior files are byte-identical, including all production code, old tests and Phase 2–10 evidence. saved_model.pt, saved_model_0.95.pt, norm.npy, preprocess_window.npy, preprocess_label.npy and class_idx.npy retain their exact pre-phase hashes. No runtime imports evaluation; no model promotion, dependencies or application settings changed. Existing unrelated working-tree changes were preserved.

## 3–8. Dataset manifest, mapping and grouping limitation

{stats['sources_discovered']} raw videos discovered, {stats['usable_sources']} usable, {len(stats['excluded_sources'])} excluded. Only the three explicit directories Normal, Yawning, Eye Close enter the dataset; sorted raw inventory and stable relative-path source IDs avoid filesystem-enumeration label order. Source content SHA-256 defines grouping; exact byte duplicates would be co-grouped, conflicting labels fail. Observed duplicate-content count: {stats['duplicate_content_groups']}.

Mapping: **0=Normal, 1=Yawning, 2=Eye Close**. No trustworthy subject/recording IDs were supplied or found in repository metadata; filename numbers, appearance and guessed recording order were not used as identity. Each unique raw-video content group is independent at the split boundary. Shared people/scenes/recording conditions across groups may still exist. This is **video-independent**, never subject-independent.

dataset_manifest_raw.csv is the immutable pre-extraction inventory (source_id, class_name/id, relative_path, file_sha256, FPS, width/height, header count, group_id). splits.json references that file's SHA-256. dataset_manifest.csv adds assigned split, decoded/valid observations, window/padded counts, status and notes after processing. Both CSVs have the same deterministic sorted source identities. Final manifest SHA-256: `{stats['final_manifest_sha256']}`. Headers are audited separately from actual successful decode counts; missing-face frames remain counted. Material decode/feature/empty-source failures record the source/split/error and abort instead of silently dropping data.

## 9–13. Split protocol, seed and leakage checks

Seed **{splits['seed']}**; class-stratified 70/15/15 raw-video group split. Each class: **42 train / 9 validation / 9 test**; total **126 / 27 / 27** videos. Assignments are persisted before any extraction/window creation. Exact assignments, group strategy, proportions, class counts, source-ID lists, assignment hash, raw manifest hash, code/config hashes are in splits.json.

Split manifest SHA-256: `{sha256(directory/'splits.json')}`.
Assignment SHA-256: `{splits['assignment_sha256']}`.

Automated hard gates: complete disjoint source/group coverage; every window's source and class matches its split; observation identities are (source_id, decoded_frame_index), not bare frame numbers; no observation identity crosses splits; the normalizer's source list equals exactly TRAIN. Overlapping windows remain within one source/split. integrity.json: `{json.dumps(integrity, sort_keys=True)}`. Tests rerun splitting with the same seed and reversed input enumeration, requiring identical assignments; a different seed changes the synthetic assignments. No split leakage was detected.

## 14–17. Feature extraction, windows and train-only normalization

Unchanged legacy feature order EAR-R, EAR-L, MAR, phi, theta. The same FaceDetector/get_detector and extract_features/draw_landmarks implementations retain IMAGE mode, one face, blendshape/matrix outputs, BGR tagged SRGB, the legacy landmark order, normalized-coordinate solvePnP and extra degree conversion. No geometry/preprocessing correction occurred. Independent comparison of the newly extracted Phase 11 raw caches against all six frozen clips matches {feature_parity['total_valid_observations']} valid observation identities and feature vectors exactly (max difference 0.0); evidence: legacy_feature_parity.json.

For historical training comparability, every decoded frame is passed through the detector; valid face observations are retained in chronology, compressing missing-face gaps as historical training did. Runtime 20-Hz sampling/reset logic is not substituted into training. No window crosses a video boundary. Input **(5,16)**, stride **8 observations in all splits**, starts range(0,n,8), short tails repeat the last observation exactly. This can create one or two padded windows/source, including windows from very short videos; padding shares below are disclosed, not removed post hoc. Whole source-video class is used as the window label. Held-out normalized values are not clipped to [0,1].

Primary normalizer preserves **legacy zero-initialized extrema**, fitted only from 126 training sources. TRAIN extraction completes, train_norm.npy is fitted/frozen, then validation/test are extracted and use that frozen artifact. Neither validation/test extrema nor norm.npy enter fitting. A dedicated regression changes held-out extremes to 1e9 and confirms identical train ranges/source IDs. No correct-extrema experiment was run. Artifact min/max in feature order:

```json
{json.dumps(norm['min_max'], indent=2)}
```

All ranges finite; all max-minus-min denominators positive. Normalizer SHA-256: `{norm['sha256']}`. train_norm.json records feature names, exact training source IDs, denominators, method and split hash. Raw feature caches contain finite offline numerical observations/frame identities, not images or video blobs; caches/window arrays are experimental training artifacts, separate from runtime SQLite.

## Dataset/split statistics and failure disclosure

{stats_table}

Excluded sources: `{stats['excluded_sources']}`. Sources with fewer than 16 valid observations: `{json.dumps(low_coverage)}`. They remain included with disclosed historical padding; no face-coverage exclusion threshold was introduced. Preprocessing completed in {stats['preprocessing_wall_seconds']:.3f} s on this CPU; this descriptive wall time is not a new performance benchmark. Full source-level progress/counts are in preprocessing_log.jsonl and dataset_manifest.csv. Window shares are natural; there is no class weighting, weighted sampler, oversampling or undersampling.

## 18–22. Training recipe and validation selection

Fresh existing **CNN_LSTM, {metadata['parameter_count']} parameters**, unchanged convolution/LSTM dimensions, 5 inputs / 16 observations / 3 outputs, CNN and LSTM dropout 0.25. CPU, float32 inputs, batch size 64, shuffle train only; standard unweighted CrossEntropyLoss, Adam fixed learning rate 0.001, no scheduler. This deliberately replaces the historical batch-level focal-like transform; it is not called focal loss. One recipe, no loss/hyperparameter/architecture/window search.

Predeclared maximum 60 epochs, patience 10 non-improving validation macro-F1 epochs, minimum improvement 1e-12, earliest epoch wins ties. Test arrays are never loaded in train.py. Validation/test use model.eval() and torch.inference_mode(); dropout-active evaluation is blocked by synthetic tests. Each best validation checkpoint is saved separately; the selected state is restored rather than using the last epoch.

Completed **{history['stopping_epoch']} epochs**, stop reason **{history['stop_reason']}**, training-loop wall time {history['training_wall_seconds']:.3f} s. Selected epoch **{history['best_epoch']}**, validation macro F1 **{history['best_validation_macro_f1']:.6f}**. Raw curves: training_history.json and training_history.csv (train/validation loss/accuracy, validation macro F1, LR). Plot: results/phase11/plots/training_curves.png. Selected train/validation eval-mode metrics are in selected_train_validation_metrics.json.

{selected_table}

Training per-class metrics at the selected checkpoint (eval mode):

{table_metrics(selected_metrics['train'])}

Validation per-class metrics at the selected checkpoint (eval mode):

{table_metrics(selected_metrics['validation'])}

History train accuracy/loss are accumulated during training with dropout and changing minibatch weights; this selected-checkpoint table uses fixed weights and dropout disabled. Both meanings are explicit.

## 23–28. One-time final window-level test

Accuracy **{100*window['accuracy']:.4f}%**, macro precision **{window['macro_precision']:.6f}**, macro recall **{window['macro_recall']:.6f}**, macro F1 **{window['macro_f1']:.6f}**, weighted F1 **{window['weighted_f1']:.6f}**, support **{window['support']} windows**. Classes are fixed in the order below; zero-support precision/recall/F1 are zero, macro averages always include all three classes.

{table_metrics(window)}

Window confusion matrix: rows actual, columns predicted. CSV: confusion_matrix.csv; PNG: plots/confusion_matrix.png.

{table_matrix(window)}

## 29–33. Video-level test and supports

Mean window probabilities per held-out source, then argmax; every video contributes one decision, so longer videos do not get extra votes. Accuracy **{100*video['accuracy']:.4f}%**, macro F1 **{video['macro_f1']:.6f}**, support **{video['support']} videos**, nine per class.

{table_metrics(video)}

Video confusion matrix: rows actual, columns predicted. CSV: video_confusion_matrix.csv; PNG: plots/video_confusion_matrix.png.

{table_matrix(video)}

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
SHA-256: `{metadata['checkpoint_sha256']}`.
Metadata: models/phase11/cnn_lstm_clean_split.json records architecture/parameter count, feature order, sequence length, train normalizer/split hashes, seed, best epoch/validation metric, final test metrics, recipe, training-history hash and environment. **Production saved_model.pt was not replaced.**

Python/NumPy/PyTorch seeds fixed; DataLoader generator fixed, worker seeding defined, num_workers=0. Deterministic algorithms enabled; Torch intra/inter threads 8/8, CPU. Environment: `{json.dumps(env, sort_keys=True)}`. Source/config/asset hashes are saved before training and checked again before training begins. Same-seed split reproducibility and synthetic loop behavior are tested. A fresh-process repetition of the identical full train/validation recipe reproduced **all selected weight tensors bit-for-bit, all 15 epoch metric rows exactly, best epoch 5 and stopping epoch 15**. It loaded only train and validation, made zero test inference passes, did not overwrite the canonical checkpoint and made no recipe/selection changes. Evidence: reproducibility_check.json and phase11_reproducibility_stdout.log. This establishes repeatability for these two runs on this CPU/software; it does not guarantee bit identity across other hardware or library versions. Artifacts provide recipe/data/split/normalization/prediction/checkpoint reproduction evidence.

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

**{len(cases)} tests passed**, including all 435 prior Phase 2–10 cases and {new_tests} Phase 11 tests; no failures/errors/skips. Counts by phase: `{counts_by_phase}`. Prior tests/assertions are byte-identical. Fresh production replay: **699/699 frozen raw windows and classes exact**, maximum numeric difference **0.0**; landmarks/policy/event/session/dashboard traces also exact. New training normalization/checkpoint never enter the production parity run. Evidence: docs/phase11_tests.xml and results/phase11_production_parity/result.json.

Limitations: one fixed split/seed/recipe, with a same-seed verification repeat; 27 test videos; correlated overlapping/padded windows within each video; no repeated-seed uncertainty estimate; no trustworthy driver/recording IDs, so shared people/scenes across videos may remain. Whole-video action labels are coarse and provide no reliable onset/frame/event ground truth. No alert precision, eye/yawn event recall or onset latency accuracy can be calculated here. Features retain known legacy geometry/color/pose quirks; normalization retains zero-anchor quirks. No balancing, architectures, window sizes, corrected-extrema/preprocessing experiments, production promotion, deployment/cloud work or Phase 12+ work was done. Results do not establish clinical validity or new-driver generalization. Test performance was not used to tune this recipe.

Resume-safe wording: “Built a leakage-safe CNN-LSTM evaluation pipeline with raw-video splits and train-only normalization, achieving **{100*window['accuracy']:.2f}% window-level video-independent test accuracy / {window['macro_f1']:.4f} macro F1** and **{100*video['accuracy']:.2f}% video-level accuracy / {video['macro_f1']:.4f} macro F1** on 27 held-out videos, while preserving 699/699 exact production outputs.” Use these clean results only; never substitute the historical number or claim subject/event independence.

**PHASE 11 PASS. STOP after Phase 11.**
"""
    path = ROOT / "docs/PHASE11_LEAKAGE_SAFE_EVALUATION.md"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    validation = {"status": "PHASE11_PASS", "tests_passed": len(cases), "new_tests": new_tests, "tests_by_phase": counts_by_phase,
                  "source_count": stats["sources_discovered"], "usable_sources": stats["usable_sources"], "integrity": integrity,
                  "window_test": window, "video_test": {key: value for key, value in video.items() if key != "videos"},
                  "best_epoch": history["best_epoch"], "best_validation_macro_f1": history["best_validation_macro_f1"],
                  "production_windows_exact": 699, "maximum_production_difference": 0.0, "prior_files_preserved": len(before), "changed_prior_files": changed,
                  "normalizer_sha256": norm["sha256"], "split_sha256": sha256(directory / "splits.json"), "checkpoint_sha256": metadata["checkpoint_sha256"],
                  "preprocessed_frozen_raw_observations_exact": feature_parity["total_valid_observations"],
                  "report_sha256": sha256(path), "full_training_repeat_performed": True, "repeat_selected_weights_bit_exact": True,
                  "repeat_epoch_metrics_exact": True, "repeat_test_inference_passes": 0, "phase12_started": False,
                  "persisted_prediction_metrics_reproduced": True, "sklearn_metric_cross_check": True,
                  "new_code_sha256": {name: sha256(ROOT / name) for name in created},
                  "evidence_sha256": {filename: sha256(directory / filename) for filename in ("training_history.json", "test_metrics.json", "video_test_metrics.json", "test_predictions.npz", "protocol.json", "integrity.json", "reproducibility_check.json")}}
    write_json(ROOT / "docs/phase11_validation.json", validation)
    print(json.dumps({key: value for key, value in validation.items() if key not in ("window_test", "video_test", "new_code_sha256", "evidence_sha256")}, indent=2))


if __name__ == "__main__":
    main()
