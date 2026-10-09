"""Phase 15 offline full-development refit. Importing this module does no work."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from . import CLASS_NAMES, FEATURE_NAMES
from .architectures import latency_statistics
from .balancing import guard_sources, training_loader
from .manifest import sha256, write_json
from .preprocess import fit_train_normalization
from .reproducibility import environment, seed_everything
from .windows import build_model, development, install_seal, assemble, validate_data, temporal, LATENCY

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/phase15'
MODELS = ROOT / 'models/phase15'
P12 = ROOT / 'results/phase12'
P14 = ROOT / 'results/phase14'
CHECKPOINT = MODELS / 'lstm_only_w32_full_development.pt'
NORM = MODELS / 'lstm32_dev_norm.npy'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hash(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_selection():
    """Freeze one quality-first choice using completed development artifacts only."""
    folds = load(P12 / 'cv_folds.json')
    ids, sealed = folds['development_source_ids'], folds['excluded_test_source_ids']
    if len(ids) != 153 or len(set(ids)) != 153 or len(sealed) != 27 or set(ids) & set(sealed):
        raise ValueError('Development/held-out boundary changed')
    aggregate = load(P14 / 'aggregate_results.json')
    scores = {k: v['window_macro_f1']['mean'] for k, v in aggregate['aggregates'].items()}
    if max(scores, key=scores.get) != 'lstm_only_32':
        raise ValueError('Completed Phase14 ranking changed')
    epochs = []
    for fold in range(3):
        for seed in (1201, 1202, 1203):
            path = P14 / f'windows_32/lstm_only/fold_{fold}/seed_{seed}/history.json'
            epochs.append({'fold': fold, 'seed': seed, 'best_epoch': load(path)['best_epoch'],
                           'artifact': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)})
    median = int(statistics.median(e['best_epoch'] for e in epochs))
    with (P14 / 'paired_window_comparisons.csv').open(newline='', encoding='utf-8') as stream:
        pairs = [r for r in csv.DictReader(stream) if r['architecture'] == 'lstm_only' and r['window'] == '32']
    if len(pairs) != 9 or not all(float(r['window_macro_f1_delta']) > 0 for r in pairs):
        raise ValueError('Completed Phase14 pairing changed')
    per = aggregate['per_class']['lstm_only_32']['window']
    phase14_latency = load(P14 / 'latency_results.json')['aggregate']
    inputs = [P12 / 'cv_folds.json', ROOT / 'results/phase13/architecture_spec.json', P14 / 'window_spec.json',
              P14 / 'aggregate_results.json', P14 / 'latency_results.json', P14 / 'paired_window_comparisons.csv']
    spec = {'phase': 15, 'frozen_before_new_training': True,
        'selection_timing': 'Quality-first final choice frozen AFTER observing completed Phase 14 development evidence; not prospectively predeclared before Phase 14.',
        'architecture': 'lstm_only', 'window_length': 32, 'training_stride': 16,
        'loss': 'unweighted CrossEntropyLoss', 'seed': 1501, 'full_development_source_ids': ids,
        'excluded_phase11_test_source_ids': sealed, 'source_set_sha256': source_hash(ids),
        'class_video_counts': [51,51,51], 'phase14_best_epochs': epochs, 'fixed_epoch_count': median,
        'epoch_rule': 'integer median of nine Phase14 LSTM-only N32 validation-selected best epochs, fixed before full-development training',
        'evidence': {'phase14_mean_window_macro_f1': scores,
            'selected_mean_window_accuracy': aggregate['aggregates']['lstm_only_32']['window_accuracy']['mean'],
            'selected_mean_macro_recall': aggregate['aggregates']['lstm_only_32']['window_macro_recall']['mean'],
            'selected_mean_worst_recall': aggregate['aggregates']['lstm_only_32']['window_worst_class_recall']['mean'],
            'selected_yawning_recall': per['Yawning']['recall']['mean'], 'paired_n32_vs_n16_positive': 9,
            'paired_n32_vs_n16_mean_macro_f1_delta': sum(float(r['window_macro_f1_delta']) for r in pairs)/9,
            'selected_parameter_count': 209667, 'original_cnn_lstm_parameter_count': 246595,
            'selected_phase14_model_only_p50_ms': phase14_latency['lstm_only_32']['wall']['p50_ms'],
            'cnn_lstm32_model_only_p50_ms': phase14_latency['cnn_lstm_32']['wall']['p50_ms'],
            'phase14_metrics_scope': 'development cross-validation, not final generalization'},
        'tradeoffs': ['LSTM-only N32 is slower model-only than CNN-LSTM N32',
            'N32 nominal 1.60 s warm-up doubles N16 and has higher padding share',
            'Vision dominates raw application timing but candidate full-application latency is unmeasured'],
        'input_artifact_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        'recipe': {'optimizer': 'Adam', 'lr': .001, 'batch_size': 64, 'shuffle': True,
            'scheduler': None, 'num_workers': 0, 'device': 'cpu', 'dtype': 'float32',
            'torch_threads': [8,8], 'deterministic_algorithms': True,
            'validation_loader': False, 'test_loader': False, 'checkpoint_selection': False,
            'checkpoint': 'final epoch'},
        'heldout_boundary': '27 Phase11 held-out sources excluded from candidate training and inference. This cohort already had historical Phase11 baseline evaluation and is not pristine untouched final test.'}
    RESULTS.mkdir(exist_ok=False)
    with (RESULTS / 'selection_spec.json').open('x', encoding='utf-8') as stream:
        json.dump(spec, stream, indent=2)
        stream.write('\n')
    print('selection frozen', sha256(RESULTS / 'selection_spec.json'), 'fixed epochs', median)


def context():
    spec = load(RESULTS / 'selection_spec.json')
    if not spec['frozen_before_new_training'] or (spec['architecture'], spec['window_length'], spec['training_stride'], spec['seed']) != ('lstm_only', 32, 16, 1501):
        raise ValueError('Phase 15 frozen choice changed')
    for name, expected in spec['input_artifact_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen development evidence changed: ' + name)
    for entry in spec['phase14_best_epochs']:
        history = ROOT / entry['artifact']
        if sha256(history) != entry['sha256'] or load(history)['best_epoch'] != entry['best_epoch']:
            raise ValueError('Phase 14 best epoch changed')
    folds = load(P12 / 'cv_folds.json')
    if folds['development_source_ids'] != spec['full_development_source_ids'] or folds['excluded_test_source_ids'] != spec['excluded_phase11_test_source_ids']:
        raise ValueError('Development/test source boundary changed')
    ids, sealed = spec['full_development_source_ids'], spec['excluded_phase11_test_source_ids']
    if len(ids) != 153 or len(set(ids)) != 153 or len(sealed) != 27 or set(ids) & set(sealed) or source_hash(ids) != spec['source_set_sha256']:
        raise ValueError('Source cohort invalid')
    guard_sources(ids, ids, sealed)
    audit = install_seal(folds)
    return spec, folds, audit


def source_data(spec, folds):
    raw, rows, final = development(folds)
    ids = spec['full_development_source_ids']
    assert sorted(raw) == ids
    counts = np.bincount([rows[s]['class_id'] for s in ids], minlength=3).tolist()
    if counts != [51, 51, 51]:
        raise ValueError('Expected 51 development videos per class')
    return raw, rows, final


def read_dataset(spec):
    meta = load(MODELS / 'lstm32_dev_norm.json')
    if meta['source_ids'] != spec['full_development_source_ids'] or meta['source_set_sha256'] != spec['source_set_sha256'] or meta['feature_order'] != list(FEATURE_NAMES) or meta['artifact_sha256'] != sha256(NORM):
        raise ValueError('Full-development normalizer provenance differs')
    norm = np.load(NORM, allow_pickle=False)
    if norm.shape != (5, 2) or not np.isfinite(norm).all() or not (norm[:, 1] > norm[:, 0]).all():
        raise ValueError('Invalid full-development extrema')
    manifest = load(RESULTS / 'window_dataset.json')
    path = RESULTS / 'development_windows.npz'
    if manifest['artifact_sha256'] != sha256(path):
        raise ValueError('Development window artifact changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_data(data, spec['full_development_source_ids'], spec['excluded_phase11_test_source_ids'], 32)
    if len(data['source_ids']) != manifest['total_windows']:
        raise ValueError('Development window count differs')
    return data, norm


def prepare():
    spec, folds, audit = context()  # Install file-open seal before raw cache access.
    raw, rows, final = source_data(spec, folds)
    ids = spec['full_development_source_ids']
    assignments = {s: 'train' for s in ids}
    norm, sources = fit_train_normalization({s: raw[s][0] for s in ids}, [rows[s] for s in ids], assignments)
    if sources != ids:
        raise ValueError('Normalizer source coverage differs')
    MODELS.mkdir(exist_ok=False)
    np.save(NORM, norm)
    write_json(MODELS / 'lstm32_dev_norm.json', {'source_ids': ids, 'source_set_sha256': spec['source_set_sha256'],
        'feature_order': list(FEATURE_NAMES), 'min_max': norm.tolist(), 'denominators': (norm[:, 1] - norm[:, 0]).tolist(),
        'fit_method': 'legacy zero-initialized extrema over all and only 153 Phase11 development sources',
        'artifact_sha256': sha256(NORM), 'class_video_counts': [51, 51, 51], 'excluded_test_count': 27})
    data = assemble(ids, raw, rows, norm, 32)
    validate_data(data, ids, spec['excluded_phase11_test_source_ids'], 32)
    np.savez_compressed(RESULTS / 'development_windows.npz', **data)
    by_source = []
    for s in ids:
        mask = data['source_ids'] == s
        by_source.append({'source_id': s, 'class_id': rows[s]['class_id'], 'valid_observations': len(raw[s][0]),
            'decoded_frames': int(final[s]['successfully_decoded_frame_count']), 'windows': int(mask.sum()),
            'padded_windows': int(data['padded'][mask].sum())})
    per_class = []
    for c in range(3):
        mask = data['labels'] == c
        per_class.append({'class_id': c, 'class_name': CLASS_NAMES[c], 'videos': 51,
            'valid_observations': sum(r['valid_observations'] for r in by_source if r['class_id'] == c),
            'windows': int(mask.sum()), 'padded_windows': int(data['padded'][mask].sum()),
            'padded_share': float(data['padded'][mask].mean())})
    short = next((x for x in by_source if rows[x['source_id']]['relative_path'] == 'Eye Close/Eye Close_51.avi'), None)
    if short is None or short['valid_observations'] != 10 or short['decoded_frames'] != 135 or short['windows'] != 1 or short['padded_windows'] != 1:
        raise ValueError('Known short source changed or excluded')
    write_json(RESULTS / 'window_dataset.json', {'source_ids': ids, 'source_set_sha256': spec['source_set_sha256'],
        'feature_order': list(FEATURE_NAMES), 'window_length': 32, 'training_stride': 16,
        'tail_rule': 'repeat last valid observation, within source only', 'total_windows': len(data['labels']),
        'padded_windows': int(data['padded'].sum()), 'padded_share': float(data['padded'].mean()),
        'by_class': per_class, 'by_source': by_source, 'short_source': short,
        'artifact_sha256': sha256(RESULTS / 'development_windows.npz'),
        'normalizer_sha256': sha256(NORM), 'test_cache_reads': audit['test_cache_reads']})
    # Fixed source/start identities: first valid window for the first 64 sorted sources.
    indexes = [int(np.flatnonzero(data['source_ids'] == s)[0]) for s in ids[:64]]
    fixed = data['windows'][indexes].astype(np.float32)
    identities = [{'source_id': str(data['source_ids'][i]), 'start_index': int(data['start_indices'][i]),
        'frame_indices': data['frame_indices'][i].tolist(),
        'tensor_sha256': hashlib.sha256(fixed[j].tobytes()).hexdigest()} for j, i in enumerate(indexes)]
    np.savez_compressed(RESULTS / 'verification_inputs.npz', windows=fixed)
    write_json(RESULTS / 'verification_inputs.json', {'identities': identities, 'shape': list(fixed.shape),
        'dtype': 'float32', 'artifact_sha256': sha256(RESULTS / 'verification_inputs.npz'),
        'source_set_sha256': spec['source_set_sha256'], 'normalizer_sha256': sha256(NORM)})
    print('prepared', len(ids), 'sources', len(data['labels']), 'windows; test opens', audit['blocked_open_attempts'], flush=True)


def fixed_inputs(spec):
    meta = load(RESULTS / 'verification_inputs.json')
    if meta['artifact_sha256'] != sha256(RESULTS / 'verification_inputs.npz') or meta['source_set_sha256'] != spec['source_set_sha256'] or meta['normalizer_sha256'] != sha256(NORM):
        raise ValueError('Verification inputs provenance changed')
    with np.load(RESULTS / 'verification_inputs.npz', allow_pickle=False) as archive:
        x = archive['windows']
    if x.shape != (64, 5, 32) or x.dtype != np.float32 or any(s['source_id'] not in spec['full_development_source_ids'] for s in meta['identities']):
        raise ValueError('Verification inputs invalid')
    if [hashlib.sha256(t.tobytes()).hexdigest() for t in x] != [i['tensor_sha256'] for i in meta['identities']]:
        raise ValueError('Verification tensor hashes changed')
    return x


def fit_fixed(data, epochs, seed):
    """Train every epoch; no validation, test, checkpoint selection, or scheduler."""
    seed_everything(seed)
    model = build_model('lstm_only', 32).to('cpu')
    assert sum(p.numel() for p in model.parameters()) == 209667
    batches = training_loader(data, 'unweighted', seed, 64)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = count = 0
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
        history.append({'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
            'training_examples': count})
        print(f'epoch {epoch}/{epochs} training_loss={history[-1]["train_loss"]:.6f} training_accuracy={history[-1]["train_accuracy"]:.6f}', flush=True)
    model.eval()
    return model, history


def train(repeat=False):
    spec, folds, audit = context()
    data, norm = read_dataset(spec)
    fixed = fixed_inputs(spec)
    if repeat:
        if not CHECKPOINT.exists():
            raise ValueError('Canonical checkpoint must exist before repeat')
        target = RESULTS / 'reproducibility' / 'repeat.pt'
        target.parent.mkdir(exist_ok=False)
    else:
        if CHECKPOINT.exists():
            raise FileExistsError('Canonical checkpoint already exists')
        target = CHECKPOINT
    model, history = fit_fixed(data, spec['fixed_epoch_count'], spec['seed'])
    torch.save(model.state_dict(), target)
    with torch.inference_mode():
        logits = model(torch.from_numpy(fixed)).numpy()
    if logits.shape != (64, 3) or not np.isfinite(logits).all():
        raise ValueError('Final development logits invalid')
    prefix = RESULTS / 'reproducibility' if repeat else RESULTS
    write_json(prefix / ('repeat_history.json' if repeat else 'training_history.json'),
        {'epochs': history, 'scope': 'in-sample training diagnostics only; no validation or generalization estimate',
         'seed': spec['seed'], 'fixed_epoch_count': spec['fixed_epoch_count'], 'training_sources': spec['full_development_source_ids'],
         'test_inference_passes': 0, 'test_cache_reads': audit['test_cache_reads'], 'test_video_reads': 0})
    if not repeat:
        with (RESULTS / 'training_history.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    np.save(prefix / ('repeat_logits.npy' if repeat else 'canonical_logits.npy'), logits)
    if not repeat:
        reloaded = build_model('lstm_only', 32)
        reloaded.load_state_dict(torch.load(target, map_location='cpu', weights_only=True)); reloaded.eval()
        with torch.inference_mode():
            replay = reloaded(torch.from_numpy(fixed)).numpy()
        if not np.array_equal(logits, replay) or not np.isin(logits.argmax(1), [0, 1, 2]).all():
            raise ValueError('Checkpoint reload or class IDs changed')
        write_json(MODELS / 'lstm_only_w32_full_development.json', {'architecture': 'lstm_only', 'parameters': 209667,
            'feature_order': list(FEATURE_NAMES), 'class_mapping': {name: i for i, name in enumerate(CLASS_NAMES)},
            'window_length': 32, 'training_stride': 16, 'expected_runtime_stride': 1,
            'development_source_count': 153, 'class_video_counts': [51,51,51],
            'normalizer_sha256': sha256(NORM), 'source_set_sha256': spec['source_set_sha256'],
            'phase12_fold_sha256': sha256(P12 / 'cv_folds.json'),
            'phase13_architecture_spec_sha256': sha256(ROOT / 'results/phase13/architecture_spec.json'),
            'phase14_window_spec_sha256': sha256(P14 / 'window_spec.json'),
            'phase15_selection_spec_sha256': sha256(RESULTS / 'selection_spec.json'),
            'seed': spec['seed'], 'epoch_count': spec['fixed_epoch_count'],
            'epoch_derivation': 'median of nine Phase14 LSTM-only N32 CV best epochs',
            'optimizer': 'Adam lr=0.001', 'loss': 'unweighted CrossEntropyLoss',
            'software_environment': environment(), 'checkpoint_sha256': sha256(target),
            'checkpoint_bytes': target.stat().st_size, 'normalization_bytes': NORM.stat().st_size,
            'final_epoch_weights': True, 'test_inference_passes': 0})
    else:
        compare_repeat(spec, target, audit)
    print('saved', target, sha256(target), flush=True)


def compare_repeat(spec, target, audit):
    canonical = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    other = torch.load(target, map_location='cpu', weights_only=True)
    if canonical.keys() != other.keys():
        raise ValueError('State keys differ')
    tensor_diff = {key: float((canonical[key] - other[key]).abs().max()) for key in canonical}
    history_a = load(RESULTS / 'training_history.json')['epochs']
    history_b = load(RESULTS / 'reproducibility/repeat_history.json')['epochs']
    logits_a = np.load(RESULTS / 'canonical_logits.npy', allow_pickle=False)
    logits_b = np.load(RESULTS / 'reproducibility/repeat_logits.npy', allow_pickle=False)
    identical = all(torch.equal(canonical[k], other[k]) for k in canonical) and history_a == history_b and np.array_equal(logits_a, logits_b)
    report = {'bit_identical': identical, 'state_tensor_count': len(canonical),
        'max_state_tensor_absolute_difference': max(tensor_diff.values()), 'per_tensor_max_absolute_difference': tensor_diff,
        'history_identical': history_a == history_b, 'max_history_loss_difference': max(abs(a['train_loss']-b['train_loss']) for a,b in zip(history_a,history_b)),
        'max_history_accuracy_difference': max(abs(a['train_accuracy']-b['train_accuracy']) for a,b in zip(history_a,history_b)),
        'final_logits_identical': np.array_equal(logits_a, logits_b),
        'max_logit_absolute_difference': float(np.abs(logits_a-logits_b).max()),
        'class_ids_identical': np.array_equal(logits_a.argmax(1),logits_b.argmax(1)),
        'canonical_checkpoint_sha256': sha256(CHECKPOINT), 'repeat_checkpoint_sha256': sha256(target),
        'fixed_development_inputs_sha256': sha256(RESULTS / 'verification_inputs.npz'),
        'seed': spec['seed'], 'epochs': spec['fixed_epoch_count'], 'fresh_process': True,
        'test_cache_reads': audit['test_cache_reads'], 'test_video_reads': 0, 'test_inference_passes': 0}
    write_json(RESULTS / 'reproducibility/result.json', report)
    if not identical:
        raise ValueError('Fresh-process full fit was not bit identical; inspect saved per-tensor differences')


def latency():
    spec, folds, audit = context()
    seed_everything(1501)
    fixed = fixed_inputs(spec)
    meta = load(MODELS / 'lstm_only_w32_full_development.json')
    if sha256(CHECKPOINT) != meta['checkpoint_sha256']:
        raise ValueError('Canonical checkpoint changed')
    model = build_model('lstm_only', 32)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True)); model.eval()
    tensors = [torch.from_numpy(x[None].copy()) for x in fixed]
    protocol = dict(LATENCY, normalization='Phase15 full-development normalizer', input_artifact_sha256=sha256(RESULTS/'verification_inputs.npz'))
    records=[]; walls=[]; cpus=[]; stalls=[]
    for repeat in range(3):
        wall=[];cpu=[]
        with torch.inference_mode():
            for i in range(128): model(tensors[i%64])
            for i in range(1024):
                c0=time.process_time_ns();w0=time.perf_counter_ns()
                logits=model(tensors[i%64])
                w1=time.perf_counter_ns();c1=time.process_time_ns()
                wall.append((w1-w0)/1e6);cpu.append((c1-c0)/1e6)
        if logits.shape != (1,3) or not torch.isfinite(logits).all():raise ValueError('Invalid latency logits')
        walls.extend(wall);cpus.extend(cpu)
        flagged=[{'repeat':repeat,'sample_index':i,'wall_ms':v,'process_cpu_ms':cpu[i]} for i,v in enumerate(wall) if v>=100]
        stalls.extend(flagged)
        records.append({'repeat':repeat,'wall':latency_statistics(wall),'process_cpu':latency_statistics(cpu),'extreme_stalls':flagged})
        print('latency repetition',repeat,'P50',records[-1]['wall']['p50_ms'],flush=True)
    np.savez_compressed(RESULTS/'latency_samples.npz',wall_ms=np.asarray(walls),process_cpu_ms=np.asarray(cpus))
    write_json(RESULTS/'latency_results.json',{'scope':'model-only CPU inference, batch 1; not full application latency',
        'protocol':protocol,'runs':records,'aggregate':{'wall':latency_statistics(walls),'process_cpu':latency_statistics(cpus)},
        'extreme_stalls':stalls,'raw_samples_sha256':sha256(RESULTS/'latency_samples.npz'),
        'checkpoint_sha256':sha256(CHECKPOINT),'test_cache_reads':audit['test_cache_reads'],'test_video_reads':0,'test_inference_passes':0,
        'clock_note':'Windows process CPU clock can be coarse and sums threads; all samples retained'})


def finalize_audit():
    spec, folds, audit = context()
    data, norm = read_dataset(spec)
    if not CHECKPOINT.exists() or not (RESULTS/'reproducibility/result.json').exists():
        raise ValueError('Candidate/repeat missing')
    if not load(RESULTS/'reproducibility/result.json')['bit_identical']:
        raise ValueError('Repeat differs')
    write_json(RESULTS/'test_access_audit.json',{'candidate_test_inference_passes':0,'test_cache_reads':audit['test_cache_reads'],
        'test_video_reads':0,'blocked_open_attempts':audit['blocked_open_attempts'],
        'excluded_test_source_ids':spec['excluded_phase11_test_source_ids'],
        'guard':'Permanent process file-open audit hook rejects held-out arrays, caches, videos and saved metrics; source whitelist applied to full fit and verification',
        'historical_disclosure':'The 27 Phase11 held-out sources were previously evaluated by the historical Phase11 baseline; they are not pristine untouched final test sources.',
        'candidate_processes':['prepare','canonical train','fresh-process repeat','latency']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','train','repeat','latency','audit'))
    action=parser.parse_args().action
    {'freeze':freeze_selection,'prepare':prepare,'train':lambda:train(False),'repeat':lambda:train(True),'latency':latency,'audit':finalize_audit}[action]()


if __name__=='__main__':main()
