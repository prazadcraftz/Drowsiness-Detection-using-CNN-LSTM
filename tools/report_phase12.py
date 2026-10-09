"""Audit and summarize development-only balancing runs; never open test arrays."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import CLASS_NAMES
from evaluation.balancing import STRATEGIES, SEEDS, create_folds, guard_sources
from evaluation.manifest import read_manifest, sha256, write_json
from evaluation.metrics import classification_metrics, aggregate_videos
from evaluation.preprocess import create_windows, fit_train_normalization

D = ROOT/'results/phase12'


def load(path):
    return json.loads(Path(path).read_text())


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def csv_out(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def table(rows):
    columns = list(rows[0])
    def cell(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---']*len(columns)) + '|\n' + ''.join('| '+' | '.join(cell(r[c]) for c in columns)+' |\n' for r in rows)


def main():
    folds, config, balance = [load(D/name) for name in ('cv_folds.json', 'experiment_config.json', 'balance_report.json')]
    canonical = create_folds(read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv'), load(ROOT/'results/phase11/splits.json'))
    assert all(folds[k] == canonical[k] for k in canonical)
    assert config['cv_sha256'] == sha256(D/'cv_folds.json')
    for name, expected in config['source_code_sha256'].items():
        assert sha256(ROOT/name) == expected
    assert load(D/'integrity.json')['passed']
    # Independently reconstruct each fold normalizer and its windows from raw
    # development observations, without touching sealed caches or arrays.
    rows = read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv')
    by_id = {r['source_id']: r for r in rows}
    provenance = load(D/'cache_provenance.json')
    assert provenance['verified_source_ids'] == folds['development_source_ids']
    features, frames = {}, {}
    for source in folds['development_source_ids']:
        path = ROOT/'results/phase11/raw_features'/(source+'.npz')
        assert sha256(path) == provenance['cache_sha256'][source]
        assert sha256(ROOT/by_id[source]['relative_path']) == by_id[source]['file_sha256']
        with np.load(path, allow_pickle=False) as archive:
            features[source], frames[source] = archive['features'], archive['frame_indices']
    for fold in folds['folds']:
        assignments = {s: 'train' if s in fold['train_source_ids'] else 'validation' for s in features}
        norm, fitted = fit_train_normalization(features, [by_id[s] for s in sorted(features)], assignments)
        directory = D/('fold_'+str(fold['fold']))
        assert fitted == fold['train_source_ids'] == load(directory/'train_norm.json')['train_source_ids']
        np.testing.assert_array_equal(norm, np.load(directory/'train_norm.npy', allow_pickle=False))
        for split in ('train', 'validation'):
            with np.load(directory/(split+'.npz'), allow_pickle=False) as archive:
                sources = archive['source_ids']
                assert set(sources) == set(fold[split+'_source_ids'])
                for source in fold[split+'_source_ids']:
                    select = sources == source
                    windows, indices, padded = create_windows(features[source], frames[source], norm)
                    np.testing.assert_array_equal(archive['windows'][select], windows)
                    np.testing.assert_array_equal(archive['frame_indices'][select], indices)
                    np.testing.assert_array_equal(archive['padded'][select], padded)
                    assert (archive['labels'][select] == by_id[source]['class_id']).all()
    audit = load(D/'test_access_audit.json')
    assert audit['phase11_test_model_inference_passes'] == audit['test_cache_reads'] == 0
    access = [json.loads(line) for line in (D/'candidate_access_audit.jsonl').read_text().splitlines()]
    assert len(access) == audit['completed_runs'] == 27
    assert audit['candidate_access_log_sha256'] == sha256(D/'candidate_access_audit.jsonl')
    for entry in access:
        fold = folds['folds'][entry['fold']]
        assert entry['training_sources'] == fold['train_source_ids']
        assert entry['validation_inference_sources'] == fold['validation_source_ids']
        guard_sources(entry['training_sources'] + entry['validation_inference_sources'], folds['development_source_ids'], folds['excluded_test_source_ids'])
        assert entry['test_inference_passes'] == 0
    runs, class_rows = [], []
    all_metrics = {}
    initial = {}
    from sklearn.metrics import f1_score, confusion_matrix
    for strategy in STRATEGIES:
        for fold in folds['folds']:
            for seed in SEEDS:
                directory = D/strategy/('fold_'+str(fold['fold']))/('seed_'+str(seed))
                run_config, metrics, history = [load(directory/name) for name in ('config.json', 'metrics.json', 'history.json')]
                assert metrics['test_inference_passes'] == 0 and not history['selection_used_test']
                assert run_config['normalization_sha256'] == sha256(D/('fold_'+str(fold['fold']))/'train_norm.npy')
                assert metrics['checkpoint_sha256'] == sha256(directory/'best.pt')
                key = (fold['fold'], seed)
                assert initial.setdefault(key, run_config['initial_weights_sha256']) == run_config['initial_weights_sha256']
                best = max(history['epochs'], key=lambda e: e['validation_macro_f1'])
                assert best['epoch'] == history['best_epoch'] == metrics['best_epoch']
                assert best['validation_macro_f1'] == metrics['window']['macro_f1']
                with np.load(directory/'validation_predictions.npz', allow_pickle=False) as archive:
                    labels, probabilities, sources = [archive[k] for k in ('labels', 'probabilities', 'source_ids')]
                guard_sources(sources, fold['validation_source_ids'], folds['excluded_test_source_ids'])
                assert set(sources) == set(fold['validation_source_ids'])
                recomputed = classification_metrics(labels, probabilities.argmax(1))
                assert all(metrics['window'][k] == v for k, v in recomputed.items())
                assert metrics['video'] == aggregate_videos(labels, probabilities, sources)
                assert abs(f1_score(labels, probabilities.argmax(1), labels=[0, 1, 2], average='macro')-recomputed['macro_f1']) < 1e-12
                assert confusion_matrix(labels, probabilities.argmax(1), labels=[0, 1, 2]).tolist() == recomputed['confusion_matrix']
                all_metrics[(strategy, fold['fold'], seed)] = metrics
                runs.append({'strategy': strategy, 'fold': fold['fold'], 'seed': seed, 'best_epoch': metrics['best_epoch'], 'stopping_epoch': history['stopping_epoch'],
                             'window_macro_f1': metrics['window']['macro_f1'], 'window_accuracy': metrics['window']['accuracy'],
                             'macro_recall': metrics['window']['macro_recall'], 'worst_class_recall': metrics['window']['worst_class_recall'],
                             'video_macro_f1': metrics['video']['macro_f1'], 'video_accuracy': metrics['video']['accuracy'],
                             'window_support': metrics['window']['support'], 'video_support': metrics['video']['support'], 'training_seconds': history['training_wall_seconds']})
                for level in ('window', 'video'):
                    for name in CLASS_NAMES:
                        class_rows.append({'strategy': strategy, 'fold': fold['fold'], 'seed': seed, 'level': level, 'class': name, **metrics[level]['per_class'][name]})
    csv_out(D/'run_results.csv', runs)
    csv_out(D/'per_class_results.csv', class_rows)
    names = ('window_macro_f1', 'window_accuracy', 'macro_recall', 'worst_class_recall', 'video_macro_f1', 'video_accuracy')
    aggregates = {s: {name: stats([r[name] for r in runs if r['strategy'] == s]) for name in names} for s in STRATEGIES}
    per_class = {s: {level: {c: {m: stats([r[m] for r in class_rows if r['strategy'] == s and r['level'] == level and r['class'] == c]) for m in ('precision', 'recall', 'f1')} for c in CLASS_NAMES} for level in ('window', 'video')} for s in STRATEGIES}
    paired = []
    by_key = {(r['strategy'], r['fold'], r['seed']): r for r in runs}
    for strategy in STRATEGIES[1:]:
        for fold in range(3):
            for seed in SEEDS:
                treatment, control = [by_key[(s, fold, seed)] for s in (strategy, 'unweighted')]
                paired.append({'strategy': strategy, 'fold': fold, 'seed': seed, **{name+'_delta': treatment[name]-control[name] for name in names}})
    csv_out(D/'paired_comparisons.csv', paired)
    paired_summary = {s: {name: dict(stats([r[name+'_delta'] for r in paired if r['strategy'] == s]),
                                     positive_pairs=sum(r[name+'_delta'] > 0 for r in paired if r['strategy'] == s),
                                     negative_pairs=sum(r[name+'_delta'] < 0 for r in paired if r['strategy'] == s)) for name in names} for s in STRATEGIES[1:]}
    rule, decisions = config['benefit_rule'], {}
    for s in STRATEGIES[1:]:
        gates = {'mean_gain': paired_summary[s]['window_macro_f1']['mean'] >= rule['minimum_mean_macro_f1_gain'],
                 'majority_positive': paired_summary[s]['window_macro_f1']['positive_pairs'] >= rule['minimum_positive_pairs'],
                 'worst_recall': paired_summary[s]['worst_class_recall']['mean'] >= -rule['maximum_mean_worst_recall_regression'],
                 'class_recalls': all(per_class[s]['window'][c]['recall']['mean']-per_class['unweighted']['window'][c]['recall']['mean'] >= -rule['maximum_mean_class_recall_regression'] for c in CLASS_NAMES),
                 'variability': aggregates[s]['window_macro_f1']['std']-aggregates['unweighted']['window_macro_f1']['std'] <= rule['maximum_macro_f1_std_increase']}
        decisions[s] = {'gates': gates, 'accepted_for_later_selection': all(gates.values())}
    accepted = [s for s in STRATEGIES[1:] if decisions[s]['accepted_for_later_selection']]
    recommendation = max(accepted, key=lambda s: aggregates[s]['window_macro_f1']['mean']) if accepted else 'unweighted'
    (D/'confusion_matrices').mkdir()
    matrices = {}
    for strategy in STRATEGIES:
        matrices[strategy] = {}
        for level in ('window', 'video'):
            seed_matrices = []
            for seed in SEEDS:
                matrix = sum(np.asarray(all_metrics[(strategy, f, seed)][level]['confusion_matrix']) for f in range(3))
                seed_matrices.append(matrix)
                rows = [{'actual': c, **dict(zip(CLASS_NAMES, map(int, row)))} for c, row in zip(CLASS_NAMES, matrix)]
                csv_out(D/'confusion_matrices'/f'{strategy}_{level}_seed_{seed}.csv', rows)
            matrices[strategy][level] = {'per_seed': {str(s): m.tolist() for s, m in zip(SEEDS, seed_matrices)},
                                         'pooled_three_seeds': sum(seed_matrices).tolist()}
    aggregate = {'strategies': aggregates, 'per_class': per_class, 'paired': paired_summary, 'decisions': decisions, 'recommendation': recommendation,
                 'confusion_matrices': matrices, 'std_definition': 'sample SD, ddof=1 across 9 correlated fold/seed runs; descriptive, not independent replicates',
                 'test_model_inference_passes': 0, 'number_of_runs': len(runs)}
    write_json(D/'aggregate_results.json', aggregate)
    plots(aggregate, runs, paired)
    xml = ET.parse(ROOT/'docs/phase12_tests.xml')
    cases = xml.findall('.//testcase')
    assert not any(xml.findall('.//'+tag) for tag in ('error', 'failure', 'skipped'))
    new_count = sum(c.attrib['classname'] == 'tests.test_phase12_balancing' for c in cases)
    assert len(cases)-new_count == 459
    parity = load(ROOT/'results/phase12_production_parity/result.json')
    assert parity['exact_parity'] and parity['windows_exact'] == parity['classes_exact'] == 699
    assert all(all(v == 0 for v in s['maximum_differences'].values()) for s in parity['sources'])
    before = load(ROOT/'docs/phase12_before_hashes.json')
    changed = {name: sha256(ROOT/name) for name, expected in before.items() if sha256(ROOT/name) != expected}
    assert not changed, changed
    write_json(ROOT/'docs/phase12_validation.json', {'status': 'PHASE12_PASS', 'tests': len(cases), 'new_tests': new_count, 'prior_files_preserved': len(before),
               'changed_prior_files': changed, 'production_exact_windows': 699, 'candidate_test_inference_passes': 0,
               'run_metrics_recomputed': True, 'fold_normalizers_and_windows_recomputed': True,
               'sklearn_metrics_crosschecked': True, 'same_initial_weights_paired': True,
               'recommendation': recommendation, 'phase13_started': False})
    report(aggregate, config, folds, balance, runs, len(cases), new_count, len(before))
    print(json.dumps({'status': 'PHASE12_PASS', 'recommendation': recommendation, 'tests': len(cases), 'test_inference_passes': 0, 'strategies': aggregates}, indent=2))


def plots(aggregate, runs, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = D/'plots'
    directory.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for axis, metric, title in zip(axes, ('window_macro_f1', 'worst_class_recall'), ('Validation window macro F1', 'Validation worst-class recall')):
        axis.boxplot([[r[metric] for r in runs if r['strategy'] == s] for s in STRATEGIES], tick_labels=['Unweighted', 'Weighted CE', 'Sampler'])
        for i, s in enumerate(STRATEGIES):
            axis.scatter(np.arange(9)*.015+i+1-.06, [r[metric] for r in runs if r['strategy'] == s], s=15, alpha=.7)
        axis.set(title=title, ylabel='Score', ylim=(.7, 1.005)); axis.grid(axis='y', alpha=.2)
    fig.savefig(directory/'strategy_variability.png', dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), layout='constrained')
    x = np.arange(3)
    for i, s in enumerate(STRATEGIES):
        mean = [aggregate['per_class'][s]['window'][c]['recall']['mean'] for c in CLASS_NAMES]
        sd = [aggregate['per_class'][s]['window'][c]['recall']['std'] for c in CLASS_NAMES]
        axis.errorbar(x+(i-1)*.08, mean, yerr=sd, fmt='o', capsize=4, label=s.replace('_', ' '))
    axis.set(xticks=x, xticklabels=CLASS_NAMES, ylim=(.7, 1.02), ylabel='Validation recall (mean ± sample SD)', title='Per-class recall across 9 fold/seed runs')
    axis.legend(); axis.grid(axis='y', alpha=.2)
    fig.savefig(directory/'per_class_recall.png', dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5), layout='constrained')
    for s in STRATEGIES[1:]:
        rows = [r for r in paired if r['strategy'] == s]
        axis.plot(range(9), [r['window_macro_f1_delta'] for r in rows], 'o-', label=s.replace('_', ' '))
    axis.axhline(0, color='black', linewidth=1)
    axis.axhline(.002, color='gray', linestyle='--', label='Predeclared mean-gain threshold (0.002)')
    axis.set(xticks=range(9), xticklabels=[f'f{f}/s{s}' for f in range(3) for s in SEEDS], ylabel='Macro F1 change vs paired unweighted', title='Paired development differences')
    axis.tick_params(axis='x', rotation=30); axis.legend(); axis.grid(alpha=.2)
    fig.savefig(directory/'paired_delta.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for column, s in enumerate(STRATEGIES):
        for row, level in enumerate(('window', 'video')):
            axis = axes[row, column]
            matrix = np.array(aggregate['confusion_matrices'][s][level]['pooled_three_seeds'])
            axis.imshow(matrix, cmap='Blues')
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i,j]>matrix.max()/2 else 'black')
            axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                     xlabel='Predicted', ylabel='Actual', title=f'{s.replace("_", " ")} / {level}')
    fig.suptitle('Development confusion counts summed across folds and 3 seeds\nEach source counted once per seed; repeated predictions are not independent subjects', fontsize=11)
    fig.savefig(directory/'development_confusion.png', dpi=150); plt.close(fig)


def report(a, config, folds, balance, runs, tests, new_count, preserved):
    summary = []
    for s in STRATEGIES:
        r = a['strategies'][s]
        summary.append({'Strategy': s, 'Window F1 mean': r['window_macro_f1']['mean'], 'SD': r['window_macro_f1']['std'],
                        'Accuracy mean': r['window_accuracy']['mean'], 'Macro recall': r['macro_recall']['mean'],
                        'Worst recall': r['worst_class_recall']['mean'], 'Video F1': r['video_macro_f1']['mean'], 'Video accuracy': r['video_accuracy']['mean']})
    variance = [{'Strategy': s, 'Metric': m, **v} for s, values in a['strategies'].items() for m, v in values.items()]
    pairs = [{'Strategy': s, 'Metric delta': m, **v} for s, values in a['paired'].items() for m, v in values.items()]
    classes = [{'Strategy': s, 'Level': level, 'Class': c, **{m: a['per_class'][s][level][c][m]['mean'] for m in ('precision', 'recall', 'f1')}} for s in STRATEGIES for level in ('window', 'video') for c in CLASS_NAMES]
    counts = [{'Fold': f['fold'], 'Split': split, 'Videos N/Y/E': str(f[split]['videos_per_class']),
               'Windows N/Y/E': str(f[split]['windows_per_class']), 'Observations N/Y/E': str(f[split]['observations_per_class']),
               'Window ratio': f[split]['window_imbalance_ratio'], 'Observation ratio': f[split]['observation_imbalance_ratio']}
              for f in balance['folds'] for split in ('train', 'validation')]
    weights = [{'Fold': f['fold'], 'Normal': f['class_weights'][0], 'Yawning': f['class_weights'][1], 'Eye Close': f['class_weights'][2],
                'Sampler epoch N': f['sampler']['epoch_sample_count'], 'Sampler p/window N/Y/E': str(f['sampler']['per_window_probability_by_class'])} for f in balance['folds']]
    before = load(ROOT/'docs/phase12_before_hashes.json')
    created = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                     and not any(x.startswith('.venv') or x in ('.git', '.cache', '__pycache__', '.pytest_cache') for x in p.relative_to(ROOT).parts)
                     and p.relative_to(ROOT).as_posix() not in before)
    write_json(D/'created_files.json', created + ['results/phase12/created_files.json', 'docs/PHASE12_CLASS_BALANCING.md'])
    matrix_text = ''
    for s in STRATEGIES:
        for level in ('window', 'video'):
            matrix_text += f'\n**{s}, {level}, pooled three seeds**\n\n'
            matrix_text += table([{'Actual / predicted': c, **dict(zip(CLASS_NAMES, row))} for c, row in zip(CLASS_NAMES, a['confusion_matrices'][s][level]['pooled_three_seeds'])])
    recommendation = a['recommendation']
    decision_text = '\n'.join(f'- {s}: {"accepted for later selection" if d["accepted_for_later_selection"] else "rejected under the predeclared benefit rule"}; gates {json.dumps(d["gates"], sort_keys=True)}.' for s, d in a['decisions'].items())
    interpretation = ' '.join(
        f"{s}: mean paired window macro-F1 change {a['paired'][s]['window_macro_f1']['mean']:+.6f}, "
        f"positive in {a['paired'][s]['window_macro_f1']['positive_pairs']}/9 pairs, "
        f"mean worst-class recall change {a['paired'][s]['worst_class_recall']['mean']:+.6f}."
        for s in STRATEGIES[1:])
    text = f'''# Phase 12 — class balancing on sealed-test development folds

**PHASE 12 PASS.** Recommendation: **{recommendation}** for later experimental model selection. Production remains unchanged. Three fixed video-grouped folds × three fixed seeds × three strategies completed, with zero Phase 11 test model inference passes. This measures development validation performance only and creates no new final test claim.

## 1–4. Files, boundary, and sealed test

Created `evaluation/balancing.py`, `tests/test_phase12_balancing.py`, `tools/report_phase12.py`, this report, preservation/test/validation records in docs, new experiment artifacts under results/phase12, training logs, and new production parity evidence under results/phase12_production_parity. The exact created-file inventory is `results/phase12/created_files.json`. **No pre-existing files modified**: all {preserved} prior files are byte-identical, including every Phase 11 artifact, production model/normalizer, raw corpus and runtime source. Existing unrelated working tree changes were preserved.

No architecture, window, feature/geometry, normalization-semantics, focal-loss, video-balanced sampling, resume, production promotion, deployment, or Phase 13+ work. No production performance benchmark repeated. Only the three requested primary strategies were run.

## 5–10. Development protocol and quantified balance

153 development videos, 51/class, from Phase 11 TRAIN + VALIDATION. The 27 test source IDs are explicitly persisted as excluded; metadata is read only to establish the seal. 3-fold stratified content-group CV: 102 train videos (34/class) and 51 validation videos (17/class) per fold. Each development source is validated exactly once per seed. Raw content duplicates co-group; none are present in this corpus. Seed {folds['seed']}; deterministic sorted identities, seeded group shuffle and balanced class allocation.

Assignment SHA-256: `{folds['assignment_sha256']}`. CV file SHA-256: `{sha256(D/'cv_folds.json')}`. Phase 11 raw manifest SHA-256: `{folds['raw_manifest_sha256']}`. Phase 11 split SHA-256: `{folds['phase11_split_sha256']}`. Persisted assignments were independently regenerated and compared exactly. Unknown subject identities are not inferred.

Corpus video counts: 60/60/60 (ratio 1.0); development and every fold video ratios 1.0. Counts below use Normal / Yawning / Eye Close order. Window and observation ratios are largest/smallest. This corpus is nearly balanced; train window ratios are only about 1.02–1.04.

{table(counts)}
## 11. Fold-train normalization and raw cache provenance

Reused only the 153 development PRE-normalization raw-feature caches. Before each cache load, its SHA-256 is compared with the pre-experiment preservation snapshot, and its raw video hash matches the immutable raw manifest. Cache keys, observation counts, finite feature values and frame ordering are validated. `cache_provenance.json` records all 153 IDs/hashes. No Phase 11 normalized windows or train_norm.npy is loaded. No sealed raw cache or video is opened by candidate code.

Each fold independently fits five-feature extrema on its 102 train sources, starting min/max at zero, then applies that fixed range to train and validation. Exact train IDs and norm hash are persisted under fold_N/train_norm.json and train_norm.npy. Held-out validation does not contribute or get clipped. Reuses unchanged feature order, length 16, stride 8, last-observation tail padding and historical compression of valid face observations. No runtime sampling/gap resets are introduced. Missing-face observations and the short Eye Close_51 source are retained as in Phase 11; no coverage-based tuning or exclusion.

## 12–18. Controlled treatment, weights, recipe and selection

UNWEIGHTED: ordinary seeded shuffled batches and `CrossEntropyLoss()`. WEIGHTED_CE: train-window counts only, weight_c = N/(3*n_c), PyTorch weighted mean CE. WEIGHTED_SAMPLER: per-window weights proportional to 1/n_c, normalized probability 1/(3*n_c), total class probability 1/3 in expectation, N samples/epoch with replacement; unweighted CE. Actual sampled class counts per epoch are retained in each history. Sampler and weighted loss are never combined. No optional video-balanced strategy was run.

{table(weights)}
Training seeds locked before execution: 1201, 1202, 1203; 27 completed runs. Within each fold/seed, all strategies use bit-identical initial weights (hashes audited) and identical normalized data. Fixed original CNN_LSTM: 246,595 parameters, 5 features ×16 observations, dropout 0.25/0.25, CPU float32, Adam .001, batch 64, no scheduler, max 60 epochs, patience 10, minimum improvement 1e-12, earliest exact ties, num_workers 0, 8/8 Torch threads and deterministic algorithms. Only balancing changes. Run configurations, environment and code hashes are preserved; control synthetic training reproduces the Phase 11 fit history and weights exactly.

Every checkpoint uses validation WINDOW macro F1 for epoch selection. Validation is eval/inference_mode with dropout off and ordinary CE diagnostic loss. Best checkpoint is restored and saved once per run; no per-epoch checkpoints. Each run stores config, best.pt plus hash, raw epoch history, selected metrics and validation predictions. No test-loader path exists in candidate code. Hard source guards precede training/each validation inference; process file-open audit also rejects sealed test arrays, predictions, metrics, caches and raw videos.

## 19–26. Development results and trade-offs

All following are validation results. Equal weight per fold/seed run, sample SD (ddof=1) across 9 runs/strategy. Window macro F1 is primary. Video predictions average window probabilities within a source then argmax; each source gets one vote. Per-fold video support 51; 153 distinct development videos overall, repeated across 3 training seeds.

{table(summary)}
Per-class mean precision/recall/F1 below; full run-specific support and metrics and mean/SD/min/max per class are in per_class_results.csv and aggregate_results.json.

{table(classes)}
## 27–28. Confusion matrices and variability

Rows actual, columns predicted; fixed Normal/Yawning/Eye Close order. CSV matrices separately sum all 3 folds for each fixed seed. Pooled tables below additionally sum 3 seeds, so support is 8,043 window predictions and 459 video predictions, representing 2,681 distinct windows  / 153 videos each evaluated 3 times. Repeated seed predictions are not independent data or subjects. These are development matrices, not test matrices.
{matrix_text}
{table(variance)}
Raw JSON/CSV is authoritative. Plots: strategy_variability.png (all 9 run points), per_class_recall.png (mean±sampleSD), paired_delta.png and development_confusion.png under results/phase12/plots.

## 22, 29–30. Paired differences and decision

Each delta is strategy minus unweighted at identical fold and seed. Both the 9 raw deltas and their descriptive distribution are reported; pooled means alone are insufficient.

{table(pairs)}
Predeclared before any training: mean F1 gain at least 0.002 (0.2 percentage points), positive on at least 5/9 pairs, mean worst-class recall regression no more than 0.005, no class mean recall regression above 0.01, F1 sample SD increase no more than 0.01. These are engineering benefit gates, not a statistical significance claim. They were neither derived from test results nor revised after seeing CV results.

{decision_text}

{interpretation}

Recommendation: **{recommendation}**. If neither balancing treatment passes every gate, ordinary unweighted CE is retained for simplicity. An accepted treatment is only an experimental recommendation for later Phase 15 selection; no application checkpoint is replaced. All class-recall gains/losses are visible above. No new resume metric is added.

## 31–33. Access audit and regression

**Number of Phase 11 test model inference passes in Phase 12: 0.** test_access_audit.json and the 27-entry candidate_access_audit.jsonl list exact training/validation IDs and zero test access. No candidate test_predictions files or candidate test metrics/confusions exist. Raw cache provenance lists 153 development reads and zero test cache reads. Source guards, sealed duplicate protection and actual file-open refusal have unit tests. The report recomputes metrics from saved validation probabilities and independently verifies window F1/matrices with sklearn; it performs no candidate model inference.

Full suite: **{tests} passed**, including all 459 Phase 2–11tests and {new_count} Phase 12 tests; no failures/errors/skips. Fresh authorized production regression, using the existing immutable raw fixture inputs: **699/699 windows/classes exact**, maximum difference 0 in every compared field. This is production parity, not candidate evaluation on the sealed Phase 11 test cohort. No Phase 8/9/SQLite benchmark artifacts overwritten.

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
'''
    with (ROOT/'docs/PHASE12_CLASS_BALANCING.md').open('x', encoding='utf-8') as handle:
        handle.write(text)


if __name__ == '__main__':
    main()
