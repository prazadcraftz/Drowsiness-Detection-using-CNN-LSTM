"""Phase 12: sealed-test, fixed grouped CV; balancing is the only treatment."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model import CNN_LSTM
from . import CLASS_NAMES
from .manifest import read_manifest, sha256, write_json
from .metrics import aggregate_videos, classification_metrics
from .preprocess import create_windows, fit_train_normalization
from .reproducibility import RECIPE, environment, seed_everything, seed_worker
from .train import MODEL_CONFIG, predict

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('unweighted', 'weighted_ce', 'weighted_sampler')
SEEDS = (1201, 1202, 1203)
FOLD_SEED = 20260927
RULE = {'minimum_mean_macro_f1_gain': .002, 'minimum_positive_pairs': 5,
        'maximum_mean_worst_recall_regression': .005,
        'maximum_mean_class_recall_regression': .01,
        'maximum_macro_f1_std_increase': .01}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def guard_sources(source_ids, allowed, sealed):
    sources = set(map(str, source_ids))
    if not sources or sources & set(sealed) or not sources <= set(allowed):
        raise ValueError('Sealed test or unauthorized/empty candidate sources')


def create_folds(rows, splits, *, seed=FOLD_SEED, count=3):
    development = sorted(splits['source_ids']['train'] + splits['source_ids']['validation'])
    sealed = sorted(splits['source_ids']['test'])
    guard_sources(development, development, sealed)
    by_id = {r['source_id']: r for r in rows}
    if len(by_id) != len(rows) or set(development) | set(sealed) != set(by_id):
        raise ValueError('Incomplete or duplicate source inventory')
    sealed_groups = {by_id[s]['group_id'] for s in sealed}
    groups = {}
    for source in development:
        row = by_id[source]
        if row['group_id'] in sealed_groups:
            raise ValueError('Development content duplicates sealed test')
        groups.setdefault(row['group_id'], []).append(source)
    buckets = [[] for _ in range(count)]
    rng = random.Random(seed)
    for label in range(3):
        selected = []
        for group, sources in sorted(groups.items()):
            labels = {by_id[s]['class_id'] for s in sources}
            if len(labels) != 1:
                raise ValueError('Conflicting group labels')
            if labels == {label}:
                selected.append(group)
        rng.shuffle(selected)
        for group in selected:
            target = min(range(count), key=lambda i: (sum(by_id[s]['class_id'] == label for s in buckets[i]), i))
            buckets[target].extend(groups[group])
    assignments = {s: i for i, bucket in enumerate(buckets) for s in sorted(bucket)}
    folds = []
    for index, bucket in enumerate(buckets):
        validation = sorted(bucket)
        train = sorted(set(development) - set(validation))
        if {by_id[s]['group_id'] for s in train} & {by_id[s]['group_id'] for s in validation}:
            raise ValueError('Group overlap')
        for sources in (train, validation):
            guard_sources(sources, development, sealed)
        folds.append({'fold': index, 'train_source_ids': train, 'validation_source_ids': validation,
                      'class_counts': {name: dict(Counter(str(by_id[s]['class_id']) for s in ids))
                                       for name, ids in (('train', train), ('validation', validation))}})
    if sorted(s for b in buckets for s in b) != development:
        raise ValueError('Validation coverage is not exactly once')
    return {'seed': seed, 'number_of_folds': count, 'development_source_ids': development,
            'excluded_test_source_ids': sealed, 'assignments': assignments,
            'assignment_sha256': digest(assignments), 'folds': folds}


def install_seal(phase11, rows, sealed):
    """Reject opens of real held-out arrays/caches/videos for this experiment process."""
    blocked = {(phase11 / name).resolve() for name in ('test.npz', 'test_predictions.npz', 'test_metrics.json', 'video_test_metrics.json')}
    for row in rows:
        if row['source_id'] in sealed:
            blocked.add((phase11 / 'raw_features' / (row['source_id'] + '.npz')).resolve())
            blocked.add((ROOT / row['relative_path']).resolve())
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)) and Path(args[0]).resolve() in blocked:
            raise PermissionError('Phase 11 test is sealed for Phase 12')
    sys.addaudithook(audit)


def class_weights(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, range(3)).all():
        raise ValueError('Invalid training labels')
    counts = np.bincount(labels.astype(int), minlength=3)
    if (counts == 0).any():
        raise ValueError('Training class missing')
    return len(labels) / (3 * counts)


def training_loader(data, strategy, seed, batch_size=64):
    weights = class_weights(data['labels'])
    dataset = TensorDataset(torch.from_numpy(data['windows'].astype(np.float32)), torch.from_numpy(data['labels'].astype(np.int64)))
    sampler = None
    if strategy == 'weighted_sampler':
        sampler = WeightedRandomSampler(torch.as_tensor(weights[data['labels']], dtype=torch.double),
                                        len(dataset), replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                      num_workers=0, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def validate_data(data, expected, sealed):
    guard_sources(data['source_ids'], expected, sealed)
    if set(data['source_ids']) != set(expected) or data['windows'].shape != (len(data['labels']), 5, 16):
        raise ValueError('Fold coverage or shape mismatch')
    if not np.isfinite(data['windows']).all():
        raise ValueError('Nonfinite windows')


def fit(model, train, validation, strategy, recipe, fold, sealed):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown balancing strategy')
    validate_data(train, fold['train_source_ids'], sealed)
    validate_data(validation, fold['validation_source_ids'], sealed)
    weights = class_weights(train['labels'])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32) if strategy == 'weighted_ce' else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'])
    batches = training_loader(train, strategy, recipe['seed'], recipe['batch_size'])
    history, best, score, best_epoch, stale = [], None, -1., None, 0
    exposure = []
    for epoch in range(1, recipe['max_epochs'] + 1):
        model.train()
        loss_sum = correct = count = 0
        seen = np.zeros(3, dtype=int)
        for inputs, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
            seen += np.bincount(labels.numpy(), minlength=3)
        validate_data(validation, fold['validation_source_ids'], sealed)
        probabilities, val_loss = predict(model, validation, batch_size=recipe['batch_size'])
        metrics = classification_metrics(validation['labels'], probabilities.argmax(1))
        entry = {'epoch': epoch, 'train_loss': loss_sum/count, 'train_accuracy': correct/count,
                 'validation_loss': val_loss, 'validation_accuracy': metrics['accuracy'],
                 'validation_macro_f1': metrics['macro_f1'], 'learning_rate': recipe['learning_rate']}
        history.append(entry)
        exposure.append(seen.tolist())
        if metrics['macro_f1'] > score + recipe['minimum_improvement']:
            score, best_epoch, stale = metrics['macro_f1'], epoch, 0
            best = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= recipe['patience']:
            break
    model.load_state_dict(best)
    model.eval()
    return {'epochs': history, 'best_epoch': best_epoch, 'stopping_epoch': len(history),
            'best_validation_macro_f1': score, 'selection_used_test': False, 'epoch_class_exposure': exposure,
            'stop_reason': 'patience' if stale >= recipe['patience'] else 'max_epochs'}


def prepare(phase11, output):
    phase11, output = Path(phase11).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = read_manifest(phase11 / 'dataset_manifest_raw.csv')
    splits = json.loads((phase11 / 'splits.json').read_text())
    if splits['manifest_sha256'] != sha256(phase11 / 'dataset_manifest_raw.csv'):
        raise ValueError('Phase 11 raw manifest changed')
    folds = create_folds(rows, splits)
    install_seal(phase11, rows, folds['excluded_test_source_ids'])
    folds.update(raw_manifest_sha256=sha256(phase11 / 'dataset_manifest_raw.csv'), phase11_split_sha256=sha256(phase11 / 'splits.json'))
    write_json(output / 'cv_folds.json', folds)
    config = {'strategies': list(STRATEGIES), 'training_seeds': list(SEEDS), 'number_of_runs': 27,
              'recipe': dict(RECIPE, test_policy='sealed: zero candidate test inference'), 'model_config': MODEL_CONFIG,
              'benefit_rule': RULE, 'declared_before_training': True, 'cv_sha256': sha256(output / 'cv_folds.json'),
              'source_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [Path(__file__), ROOT/'model.py', ROOT/'evaluation/train.py', ROOT/'evaluation/preprocess.py', ROOT/'evaluation/metrics.py', ROOT/'evaluation/reproducibility.py']}}
    write_json(output / 'experiment_config.json', config)
    # Snapshot predates cache reuse: exact bytes + raw source provenance verified.
    before = json.loads((ROOT / 'docs/phase12_before_hashes.json').read_text())
    features, frames, cache_hashes = {}, {}, {}
    by_id = {row['source_id']: row for row in rows}
    final_rows = {r['source_id']: r for r in read_manifest(phase11 / 'dataset_manifest.csv')}
    for source in folds['development_source_ids']:
        guard_sources([source], folds['development_source_ids'], folds['excluded_test_source_ids'])
        row = by_id[source]
        path = phase11 / 'raw_features' / (source + '.npz')
        expected = before[path.relative_to(ROOT).as_posix()]
        if sha256(path) != expected or sha256(ROOT / row['relative_path']) != row['file_sha256']:
            raise ValueError('Cache/source bytes changed')
        with np.load(path, allow_pickle=False) as archive:
            features[source], frames[source] = archive['features'], archive['frame_indices']
        if len(features[source]) != int(final_rows[source]['valid_face_observation_count']):
            raise ValueError('Raw observation count differs')
        cache_hashes[source] = expected
    write_json(output / 'cache_provenance.json', {'pre_normalization': True, 'verified_source_ids': sorted(features), 'cache_sha256': cache_hashes, 'test_cache_reads': 0})
    balance = {'corpus_videos_per_class': [sum(r['class_id'] == c for r in rows) for c in range(3)], 'folds': []}
    for fold in folds['folds']:
        directory = output / ('fold_' + str(fold['fold']))
        directory.mkdir()
        assignments = {s: 'train' if s in fold['train_source_ids'] else 'validation' for s in features}
        dev_rows = [by_id[s] for s in sorted(features)]
        norm, norm_sources = fit_train_normalization(features, dev_rows, assignments)
        assert norm_sources == fold['train_source_ids']
        np.save(directory / 'train_norm.npy', norm)
        write_json(directory / 'train_norm.json', {'train_source_ids': norm_sources, 'sha256': sha256(directory / 'train_norm.npy'), 'method': 'legacy zero-initialized extrema; fold train only'})
        stats = {'fold': fold['fold']}
        for split in ('train', 'validation'):
            data = {name: [] for name in ('windows', 'labels', 'source_ids', 'frame_indices', 'padded')}
            for source in fold[split + '_source_ids']:
                windows, indices, padding = create_windows(features[source], frames[source], norm)
                data['windows'].extend(windows); data['frame_indices'].extend(indices); data['padded'].extend(padding)
                data['labels'].extend([by_id[source]['class_id']]*len(windows)); data['source_ids'].extend([source]*len(windows))
            data = {k: np.asarray(v) for k, v in data.items()}
            validate_data(data, fold[split + '_source_ids'], folds['excluded_test_source_ids'])
            np.savez_compressed(directory / (split + '.npz'), **data)
            counts = np.bincount(data['labels'], minlength=3).tolist()
            observations = [sum(len(features[s]) for s in fold[split + '_source_ids'] if by_id[s]['class_id'] == c) for c in range(3)]
            stats[split] = {'videos_per_class': [sum(by_id[s]['class_id'] == c for s in fold[split + '_source_ids']) for c in range(3)],
                            'windows_per_class': counts, 'observations_per_class': observations,
                            'window_imbalance_ratio': max(counts)/min(counts), 'observation_imbalance_ratio': max(observations)/min(observations),
                            'artifact_sha256': sha256(directory / (split + '.npz'))}
            if split == 'train':
                stats['class_weights'] = class_weights(data['labels']).tolist()
                stats['sampler'] = {'per_window_probability_by_class': [1/(3*n) for n in counts], 'class_probability': [1/3]*3,
                                    'epoch_sample_count': sum(counts), 'replacement': True, 'loss': 'unweighted CE'}
        balance['folds'].append(stats)
    write_json(output / 'balance_report.json', balance)
    write_json(output / 'integrity.json', {'passed': True, 'normalizers_fold_train_only': True, 'group_disjoint': True,
               'test_excluded': True, 'validation_coverage_once': True, 'fold_reproducible': folds['assignments'] == create_folds(rows, splits)['assignments']})


def run(output):
    output = Path(output).resolve()
    config = json.loads((output/'experiment_config.json').read_text())
    folds = json.loads((output/'cv_folds.json').read_text())
    if config['cv_sha256'] != sha256(output/'cv_folds.json'):
        raise ValueError('Fold file changed')
    for name, expected in config['source_code_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Predeclared source changed')
    phase11 = ROOT/'results/phase11'
    if sha256(phase11/'splits.json') != folds['phase11_split_sha256']:
        raise ValueError('Canonical sealed split changed')
    rows = read_manifest(phase11/'dataset_manifest_raw.csv')
    canonical = create_folds(rows, json.loads((phase11/'splits.json').read_text()))
    if any(folds[k] != canonical[k] for k in canonical):
        raise ValueError('Fold or seal assignments changed')
    install_seal(phase11, rows, folds['excluded_test_source_ids'])
    balance = json.loads((output/'balance_report.json').read_text())
    seed_everything(SEEDS[0])
    write_json(output/'experiment_started.json', {'config_sha256': sha256(output/'experiment_config.json'), 'environment': environment(), 'test_inference_passes': 0})
    audit_path = output/'candidate_access_audit.jsonl'
    with audit_path.open('x', encoding='utf-8') as audit:
        for fold in folds['folds']:
            data = {}
            directory = output/('fold_'+str(fold['fold']))
            norm = json.loads((directory/'train_norm.json').read_text())
            if norm['train_source_ids'] != fold['train_source_ids'] or norm['sha256'] != sha256(directory/'train_norm.npy'):
                raise ValueError('Fold normalization changed')
            for split in ('train', 'validation'):
                path = directory/(split+'.npz')
                if sha256(path) != balance['folds'][fold['fold']][split]['artifact_sha256']:
                    raise ValueError('Fold arrays changed')
                with np.load(path, allow_pickle=False) as archive:
                    data[split] = {k: archive[k] for k in archive.files}
                validate_data(data[split], fold[split+'_source_ids'], folds['excluded_test_source_ids'])
            for seed in SEEDS:
                for strategy in STRATEGIES:
                    target = output/strategy/('fold_'+str(fold['fold']))/('seed_'+str(seed))
                    target.mkdir(parents=True, exist_ok=False)
                    recipe = dict(config['recipe'], seed=seed, class_weighting=strategy=='weighted_ce', weighted_sampling=strategy=='weighted_sampler')
                    seed_everything(seed)
                    model = CNN_LSTM(MODEL_CONFIG).to('cpu')
                    assert sum(p.numel() for p in model.parameters()) == 246595
                    initial_hash = hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
                    run_config = {'strategy': strategy, 'fold': fold['fold'], 'seed': seed, 'recipe': recipe, 'model_config': MODEL_CONFIG,
                                  'class_weights': class_weights(data['train']['labels']).tolist(), 'normalization_sha256': norm['sha256'],
                                  'train_data_sha256': balance['folds'][fold['fold']]['train']['artifact_sha256'],
                                  'validation_data_sha256': balance['folds'][fold['fold']]['validation']['artifact_sha256'], 'initial_weights_sha256': initial_hash}
                    write_json(target/'config.json', run_config)
                    audit.write(json.dumps({'strategy': strategy, 'fold': fold['fold'], 'seed': seed,
                                            'training_sources': fold['train_source_ids'], 'validation_inference_sources': fold['validation_source_ids'], 'test_inference_passes': 0})+'\n')
                    audit.flush()
                    started = time.perf_counter()
                    history = fit(model, data['train'], data['validation'], strategy, recipe, fold, folds['excluded_test_source_ids'])
                    history['training_wall_seconds'] = time.perf_counter()-started
                    write_json(target/'history.json', history)
                    checkpoint = target/'best.pt'
                    torch.save(model.state_dict(), checkpoint)
                    validate_data(data['validation'], fold['validation_source_ids'], folds['excluded_test_source_ids'])
                    probabilities, loss = predict(model, data['validation'])
                    np.savez_compressed(target/'validation_predictions.npz', probabilities=probabilities, labels=data['validation']['labels'], source_ids=data['validation']['source_ids'])
                    window = classification_metrics(data['validation']['labels'], probabilities.argmax(1))
                    window['worst_class_recall'] = min(c['recall'] for c in window['per_class'].values())
                    video = aggregate_videos(data['validation']['labels'], probabilities, data['validation']['source_ids'])
                    write_json(target/'metrics.json', {'window': window, 'video': video, 'validation_loss': loss,
                               'checkpoint_sha256': sha256(checkpoint), 'best_epoch': history['best_epoch'], 'test_inference_passes': 0})
                    print(f"{strategy} fold={fold['fold']} seed={seed} best={history['best_epoch']} F1={window['macro_f1']:.6f}", flush=True)
    write_json(output/'test_access_audit.json', {'phase11_test_model_inference_passes': 0, 'test_cache_reads': 0,
               'guard': 'canonical sealed source IDs checked before candidate fit and each validation inference; process open audit blocks test arrays/caches/videos/metrics',
               'candidate_access_log_sha256': sha256(audit_path), 'completed_runs': 27, 'excluded_test_source_ids': folds['excluded_test_source_ids']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--output', default=str(ROOT/'results/phase12'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(ROOT/'results/phase11', args.output)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
