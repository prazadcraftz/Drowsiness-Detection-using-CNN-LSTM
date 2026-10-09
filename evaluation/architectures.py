"""Phase 13 prepare/run/latency CLI; no sealed test or production model path."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .balancing import create_folds, fit, guard_sources, install_seal, validate_data
from .manifest import read_manifest, sha256, write_json
from .metrics import classification_metrics, aggregate_videos
from .models import NAMES, architecture_spec, build_model
from .preprocess import fit_train_normalization, create_windows
from .reproducibility import environment, seed_everything
from .train import predict

ROOT = Path(__file__).resolve().parents[1]
P11 = ROOT/'results/phase11'
P12 = ROOT/'results/phase12'
GUIDANCE = {'efficient_quality_margin': .002, 'minimum_noninferior_pairs': 5,
            'minimum_latency_or_parameter_reduction_fraction': .20,
            'strong_quality_gain': .005, 'minimum_positive_pairs_for_quality_gain': 5,
            'maximum_worst_recall_regression': .005, 'maximum_class_recall_regression': .01,
            'maximum_macro_f1_std_increase': .01}
LATENCY = {'batch_size': 1, 'warmup_calls': 128, 'measured_calls_per_checkpoint': 1024,
           'input_count': 64, 'input_seed': 1301, 'order_seed': 1302,
           'torch_threads': 8, 'torch_interop_threads': 8, 'device': 'cpu'}


def load(path):
    return json.loads(Path(path).read_text())


def verify_protocol(output=None, *, raw=False, enforce_seal=True):
    folds = load(P12/'cv_folds.json')
    rows = read_manifest(P11/'dataset_manifest_raw.csv')
    splits = load(P11/'splits.json')
    canonical = create_folds(rows, splits)
    if any(canonical[k] != folds[k] for k in canonical):
        raise ValueError('Phase 12 canonical folds or seal changed')
    if folds['raw_manifest_sha256'] != sha256(P11/'dataset_manifest_raw.csv') or folds['phase11_split_sha256'] != sha256(P11/'splits.json'):
        raise ValueError('Canonical manifest/split hash changed')
    if enforce_seal:
        install_seal(P11, rows, folds['excluded_test_source_ids'])
    original_config = load(P12/'experiment_config.json')
    if original_config['cv_sha256'] != sha256(P12/'cv_folds.json') or original_config['training_seeds'] != [1201,1202,1203]:
        raise ValueError('Phase 12 configuration changed')
    for name, expected in original_config['source_code_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Established recipe implementation changed')
    expected_files = {'results/phase12/cv_folds.json': sha256(P12/'cv_folds.json')}
    for name in ('experiment_config.json', 'balance_report.json', 'cache_provenance.json'):
        expected_files['results/phase12/'+name] = sha256(P12/name)
    balance = load(P12/'balance_report.json')
    data = {}
    for fold in folds['folds']:
        directory = P12/f"fold_{fold['fold']}"
        norm = load(directory/'train_norm.json')
        if norm['train_source_ids'] != fold['train_source_ids'] or norm['sha256'] != sha256(directory/'train_norm.npy'):
            raise ValueError('Fold normalizer provenance changed')
        for name in ('train_norm.npy', 'train_norm.json', 'train.npz', 'validation.npz'):
            expected_files[(directory/name).relative_to(ROOT).as_posix()] = sha256(directory/name)
        data[fold['fold']] = {}
        for split in ('train', 'validation'):
            path = directory/(split+'.npz')
            if sha256(path) != balance['folds'][fold['fold']][split]['artifact_sha256']:
                raise ValueError('Phase 12 prepared data changed')
            with np.load(path, allow_pickle=False) as archive:
                values = {k: archive[k] for k in archive.files}
            validate_data(values, fold[split+'_source_ids'], folds['excluded_test_source_ids'])
            data[fold['fold']][split] = values
    if output is not None:
        config = load(Path(output)/'experiment_config.json')
        for name, expected in config['input_artifact_sha256'].items():
            if sha256(ROOT/name) != expected:
                raise ValueError('Predeclared input artifact changed')
        if sha256(Path(output)/'architecture_spec.json') != config['architecture_spec_sha256']:
            raise ValueError('Architecture specification changed')
        for name, expected in config['source_code_sha256'].items():
            if sha256(ROOT/name) != expected:
                raise ValueError('Predeclared experimental code changed')
    if raw:
        provenance = load(P12/'cache_provenance.json')
        by_id = {r['source_id']: r for r in rows}
        features, frames = {}, {}
        for source in folds['development_source_ids']:
            guard_sources([source], folds['development_source_ids'], folds['excluded_test_source_ids'])
            path = P11/'raw_features'/(source+'.npz')
            if sha256(path) != provenance['cache_sha256'][source] or sha256(ROOT/by_id[source]['relative_path']) != by_id[source]['file_sha256']:
                raise ValueError('Raw feature cache/source bytes changed')
            with np.load(path, allow_pickle=False) as archive:
                features[source], frames[source] = archive['features'], archive['frame_indices']
        for fold in folds['folds']:
            assignment = {s: 'train' if s in fold['train_source_ids'] else 'validation' for s in features}
            norm, sources = fit_train_normalization(features, [by_id[s] for s in sorted(features)], assignment)
            if sources != fold['train_source_ids']:
                raise ValueError('Normalizer is not fold-train only')
            np.testing.assert_array_equal(norm, np.load(P12/f"fold_{fold['fold']}"/'train_norm.npy', allow_pickle=False))
            for split in ('train','validation'):
                values = data[fold['fold']][split]
                for source in fold[split+'_source_ids']:
                    windows, indices, padded = create_windows(features[source], frames[source], norm)
                    select = values['source_ids'] == source
                    np.testing.assert_array_equal(windows, values['windows'][select])
                    np.testing.assert_array_equal(indices, values['frame_indices'][select])
                    np.testing.assert_array_equal(padded, values['padded'][select])
                    if not (values['labels'][select] == by_id[source]['class_id']).all():
                        raise ValueError('Source label changed')
    return folds, original_config, data, expected_files


def state_digest(state):
    return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in state.values())).hexdigest()


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    folds, original, data, hashes = verify_protocol(raw=True)
    seed_everything(1201)
    spec = architecture_spec()
    assert spec['cnn_lstm']['trainable_parameters'] == 246595
    write_json(output/'architecture_spec.json', spec)
    shutil.copyfile(P12/'cv_folds.json', output/'cv_folds.json')
    config = {'architecture_names': list(NAMES), 'training_seeds': original['training_seeds'], 'number_of_runs': 45,
              'recipe': original['recipe'], 'architecture_spec_sha256': sha256(output/'architecture_spec.json'),
              'input_artifact_sha256': hashes, 'guidance': GUIDANCE, 'latency_protocol': LATENCY,
              'execution_order': 'all nine cnn_lstm controls first; verify against Phase12; then other 36 runs',
              'declared_before_training': True,
              'source_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [Path(__file__),ROOT/'evaluation/models.py',ROOT/'evaluation/balancing.py',ROOT/'evaluation/train.py',ROOT/'evaluation/preprocess.py',ROOT/'evaluation/metrics.py',ROOT/'evaluation/reproducibility.py',ROOT/'model.py']}}
    write_json(output/'experiment_config.json', config)
    values = data[0]['validation']
    indices = np.sort(np.random.default_rng(LATENCY['input_seed']).choice(len(values['labels']), LATENCY['input_count'], replace=False))
    tensors = values['windows'][indices].astype(np.float32)
    np.savez_compressed(output/'latency_inputs.npz', windows=tensors, source_ids=values['source_ids'][indices], frame_indices=values['frame_indices'][indices], window_indices=indices)
    write_json(output/'latency_inputs.json', {'sha256': sha256(output/'latency_inputs.npz'), 'fold': 0, 'split': 'validation',
               'sources': values['source_ids'][indices].tolist(), 'window_indices': indices.tolist(),
               'tensor_sha256': [hashlib.sha256(t.tobytes()).hexdigest() for t in tensors], 'batch_shape': [1,5,16]})
    write_json(output/'integrity.json', {'passed': True, 'fold_file_byte_identical': sha256(output/'cv_folds.json') == sha256(P12/'cv_folds.json'),
               'raw_development_caches_verified': 153, 'fold_normalizers_and_windows_reconstructed': True, 'sealed_test_excluded': True})


def compare_control(model, history, metrics, predictions, reference):
    previous = load(reference/'history.json')
    assert history['epochs'] == previous['epochs'], 'CNN-LSTM epoch metrics do not reproduce Phase12'
    for key in ('best_epoch','stopping_epoch','stop_reason','best_validation_macro_f1'):
        assert history[key] == previous[key], 'CNN-LSTM selection does not reproduce Phase12'
    state = torch.load(reference/'best.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items()), 'CNN-LSTM weights differ from Phase12'
    old = load(reference/'metrics.json')
    assert metrics['window'] == old['window'] and metrics['video'] == old['video'], 'CNN-LSTM validation metrics differ'
    with np.load(reference/'validation_predictions.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(predictions, archive['probabilities'])
    return {'weights_bit_exact': True, 'epoch_metrics_exact': True, 'selection_exact': True, 'validation_probabilities_exact': True,
            'phase12_checkpoint_sha256': sha256(reference/'best.pt')}


def run(output):
    output = Path(output).resolve()
    folds, _, data, _ = verify_protocol(output, raw=True)
    config = load(output/'experiment_config.json')
    seed_everything(1201)
    write_json(output/'experiment_started.json', {'environment': environment(), 'config_sha256': sha256(output/'experiment_config.json'), 'candidate_test_inference_passes': 0})
    control = []
    with (output/'candidate_access_audit.jsonl').open('x', encoding='utf-8') as audit:
        for architecture in ('cnn_lstm',) + tuple(n for n in NAMES if n != 'cnn_lstm'):
            for fold in folds['folds']:
                for seed in config['training_seeds']:
                    target = output/architecture/f"fold_{fold['fold']}"/f'seed_{seed}'
                    target.mkdir(parents=True, exist_ok=False)
                    try:
                        seed_everything(seed)
                        model = build_model(architecture).to('cpu')
                        recipe = dict(config['recipe'], seed=seed)
                        write_json(target/'config.json', {'architecture': architecture, 'fold': fold['fold'], 'seed': seed, 'recipe': recipe,
                                   'architecture_spec_sha256': config['architecture_spec_sha256'], 'initial_weights_sha256': state_digest(model.state_dict()),
                                   'normalization_sha256': sha256(P12/f"fold_{fold['fold']}"/'train_norm.npy')})
                        audit.write(json.dumps({'architecture': architecture, 'fold': fold['fold'], 'seed': seed,
                                               'train_source_ids': fold['train_source_ids'], 'validation_source_ids': fold['validation_source_ids'], 'test_inference_passes': 0})+'\n')
                        audit.flush()
                        started = time.perf_counter()
                        history = fit(model, data[fold['fold']]['train'], data[fold['fold']]['validation'], 'unweighted', recipe, fold, folds['excluded_test_source_ids'])
                        history['training_wall_seconds'] = time.perf_counter()-started
                        write_json(target/'history.json', history)
                        checkpoint = target/'best.pt'
                        torch.save(model.state_dict(), checkpoint)
                        values = data[fold['fold']]['validation']
                        validate_data(values, fold['validation_source_ids'], folds['excluded_test_source_ids'])
                        probabilities, loss = predict(model, values)
                        np.savez_compressed(target/'validation_predictions.npz', probabilities=probabilities, labels=values['labels'], source_ids=values['source_ids'])
                        window = classification_metrics(values['labels'], probabilities.argmax(1))
                        window['worst_class_recall'] = min(c['recall'] for c in window['per_class'].values())
                        video = aggregate_videos(values['labels'], probabilities, values['source_ids'])
                        metrics = {'window': window, 'video': video, 'validation_loss': loss, 'best_epoch': history['best_epoch'],
                                   'checkpoint_sha256': sha256(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
                                   'trainable_parameters': sum(p.numel() for p in model.parameters()), 'test_inference_passes': 0}
                        write_json(target/'metrics.json', metrics)
                        if architecture == 'cnn_lstm':
                            control.append(dict(compare_control(model, history, metrics, probabilities, P12/'unweighted'/f"fold_{fold['fold']}"/f'seed_{seed}'),fold=fold['fold'],seed=seed))
                        print(f"{architecture} fold={fold['fold']} seed={seed} best={history['best_epoch']} F1={window['macro_f1']:.6f}", flush=True)
                    except Exception as error:
                        write_json(target/'failure.json', {'type': type(error).__name__, 'message': str(error), 'action': 'stop; no silent discard'})
                        raise
            if architecture == 'cnn_lstm':
                write_json(output/'control_reproduction.json', {'passed': True, 'runs': control, 'test_inference_passes': 0})
    write_json(output/'test_access_audit.json', {'phase11_test_model_inference_passes': 0, 'test_cache_reads': 0,
               'completed_runs': 45, 'excluded_test_source_ids': folds['excluded_test_source_ids'],
               'candidate_access_log_sha256': sha256(output/'candidate_access_audit.jsonl'),
               'guard': 'canonical source guard before fit and each validation prediction; process file-open seal; no candidate test loader'})


def latency_statistics(samples):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError('Invalid latency samples')
    return {'p50_ms': float(np.percentile(samples,50)), 'p95_ms': float(np.percentile(samples,95)),
            'p99_ms': float(np.percentile(samples,99)), 'mean_ms': float(samples.mean()), 'sample_count': len(samples)}


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def paired_summary(candidate, reference):
    if set(candidate) != set(reference):
        raise ValueError('Unmatched fold/seed pairs')
    deltas = [candidate[key]-reference[key] for key in sorted(reference)]
    return dict(summary(deltas), positive_pairs=sum(v>0 for v in deltas), negative_pairs=sum(v<0 for v in deltas),
                zero_pairs=sum(v==0 for v in deltas))


def pareto_front(quality_latency):
    return sorted(name for name,(quality,latency) in quality_latency.items() if not any(
        q>=quality and t<=latency and (q>quality or t<latency)
        for other,(q,t) in quality_latency.items() if other!=name))


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def paired_summary(candidate, reference):
    if set(candidate) != set(reference):
        raise ValueError('Unmatched fold/seed pairs')
    deltas = [candidate[key]-reference[key] for key in sorted(reference)]
    return dict(summary(deltas), positive_pairs=sum(v>0 for v in deltas), negative_pairs=sum(v<0 for v in deltas),
                zero_pairs=sum(v==0 for v in deltas))


def pareto_front(quality_latency):
    return sorted(name for name,(quality,latency) in quality_latency.items() if not any(
        q>=quality and t<=latency and (q>quality or t<latency)
        for other,(q,t) in quality_latency.items() if other!=name))


def latency(output):
    output = Path(output).resolve()
    folds, _, _, _ = verify_protocol(output)
    config = load(output/'experiment_config.json')
    assert load(output/'control_reproduction.json')['passed']
    assert load(output/'test_access_audit.json')['completed_runs'] == 45
    seed_everything(1201)
    protocol = config['latency_protocol']
    inputs_meta = load(output/'latency_inputs.json')
    if sha256(output/'latency_inputs.npz') != inputs_meta['sha256']:
        raise ValueError('Latency inputs changed')
    with np.load(output/'latency_inputs.npz', allow_pickle=False) as archive:
        guard_sources(archive['source_ids'], folds['folds'][0]['validation_source_ids'], folds['excluded_test_source_ids'])
        tensors = archive['windows']
    assert [hashlib.sha256(t.tobytes()).hexdigest() for t in tensors] == inputs_meta['tensor_sha256']
    prepared = [torch.from_numpy(t[None].copy()) for t in tensors]
    jobs = [(name, fold, seed) for fold in range(3) for seed in config['training_seeds'] for name in NAMES]
    random.Random(protocol['order_seed']).shuffle(jobs)
    write_json(output/'latency_started.json', {'environment': environment(), 'protocol': protocol, 'job_order': jobs,
               'input_sha256': inputs_meta['sha256'], 'model_only': True, 'test_inference_passes': 0})
    raw, records = {}, []
    for name, fold, seed in jobs:
        checkpoint = output/name/f'fold_{fold}'/f'seed_{seed}'/'best.pt'
        assert sha256(checkpoint) == load(checkpoint.with_name('metrics.json'))['checkpoint_sha256']
        model = build_model(name).to('cpu')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        samples = []
        with torch.inference_mode():
            for i in range(protocol['warmup_calls']):
                model(prepared[i % len(prepared)])
            for i in range(protocol['measured_calls_per_checkpoint']):
                value = prepared[i % len(prepared)]
                started = time.perf_counter_ns()
                logits = model(value)
                samples.append((time.perf_counter_ns()-started)/1e6)
            assert logits.shape == (1,3) and torch.isfinite(logits).all()
        key = f'{name}_fold_{fold}_seed_{seed}'
        raw[key] = np.asarray(samples)
        records.append({'architecture': name, 'fold': fold, 'seed': seed, 'checkpoint_sha256': sha256(checkpoint), **latency_statistics(samples)})
        print(f"latency {key} p50={records[-1]['p50_ms']:.4f} ms", flush=True)
    np.savez_compressed(output/'latency_samples.npz', **raw)
    write_json(output/'latency_results.json', {'protocol': protocol, 'runs': records, 'aggregate': {
               name: latency_statistics(np.concatenate([v for k,v in raw.items() if k.startswith(name+'_fold_')])) for name in NAMES},
               'input_sha256': inputs_meta['sha256'], 'raw_samples_sha256': sha256(output/'latency_samples.npz'), 'test_inference_passes': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','latency'))
    parser.add_argument('--output', default=str(ROOT/'results/phase13'))
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'latency':latency}[args.action](args.output)


if __name__ == '__main__':
    main()
