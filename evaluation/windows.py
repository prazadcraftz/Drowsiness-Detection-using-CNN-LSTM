"""Phase 14 offline windows: prepare/run/latency. Importing never runs experiments."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import CNN_LSTM
from .architectures import (verify_protocol, load, state_digest, summary, paired_summary,
                            pareto_front, latency_statistics, ROOT, P11, P12)
from .balancing import guard_sources, training_loader
from .manifest import read_manifest, sha256, write_json
from .metrics import classification_metrics, aggregate_videos
from .models import Alternative, build_model as frozen_model
from .reproducibility import seed_everything, environment
from .train import MODEL_CONFIG, predict

P13 = ROOT/'results/phase13'
LENGTHS = (8, 16, 24, 32)
STRIDES = {8: 4, 16: 8, 24: 12, 32: 16}
ARCHITECTURES = ('cnn_lstm', 'lstm_only', 'cnn_gru', 'cnn_bilstm')
SEEDS = (1201, 1202, 1203)
COUNTS = {'cnn_lstm': 246595, 'lstm_only': 209667, 'cnn_gru': 188739, 'cnn_bilstm': 181059}
RULES = {'quality_gain': .003, 'minimum_positive_pairs': 5,
         'responsiveness_f1_margin': .002, 'maximum_worst_recall_regression': .005,
         'maximum_class_recall_regression': .01,
         'responsiveness_margin_interpretation': 'candidate mean >= control mean - .002; gains also qualify'}
LATENCY = {'batch_size': 1, 'input_count': 64, 'warmup_calls': 128,
           'measured_calls_per_job': 1024, 'repetitions': 3, 'fold': 0, 'seed': 1201,
           'order_seed': 1402, 'torch_threads': 8, 'torch_interop_threads': 8,
           'extreme_wall_stall_ms': 100., 'outlier_policy': 'retain all samples, no trimming',
           'clocks': ['perf_counter_ns', 'process_time_ns'], 'device': 'cpu'}


def check_length(n):
    if n not in LENGTHS:
        raise ValueError('Only predeclared lengths 8/16/24/32 are allowed')


def check_input(x):
    if x.ndim != 3 or x.shape[1] != 5:
        raise ValueError('Expected (B,5,N)')
    check_length(x.shape[2])


class WindowReference(CNN_LSTM):
    def forward(self, x):
        check_input(x)
        return super().forward(x)


class WindowAlternative(Alternative):
    """Same inherited construction/initialization/state; relax only the input guard.

    The operations below are the frozen Phase 13 forward operations, without its
    fixed-16 guard or the excluded CNN-only branch. No modules/parameters added.
    """
    def forward(self, x):
        check_input(x)
        if self.name != 'lstm_only':
            x = self.dropout_cnn(self.maxpool(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = x.permute(0, 2, 1)
        if self.name == 'cnn_gru':
            x, _ = self.gru(x)
            x = self.dropout_lstm(x)[:, -1, :]
        elif self.name == 'cnn_bilstm':
            _, (hidden, _) = self.lstm(x)
            x = self.dropout_lstm(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            x, _ = self.lstm(x)
            x = self.dropout_lstm(x)[:, -1, :]
        return self.fc2(F.relu(self.fc1(x)))


def build_model(name, n):
    check_length(n)
    if name not in ARCHITECTURES:
        raise ValueError('Architecture excluded from Phase 14')
    model = WindowReference(dict(MODEL_CONFIG)) if name == 'cnn_lstm' else WindowAlternative(name)
    if sum(p.numel() for p in model.parameters()) != COUNTS[name]:
        raise ValueError('Frozen parameter count changed')
    return model


def forward_gate():
    """Actual tensor shapes, state/init identity and N16 train/eval equivalence."""
    result = {}
    for name in ARCHITECTURES:
        seed_everything(1201)
        old = frozen_model(name)
        seed_everything(1201)
        model = build_model(name, 16)
        if old.state_dict().keys() != model.state_dict().keys() or any(
                not torch.equal(v, model.state_dict()[k]) for k, v in old.state_dict().items()):
            raise ValueError('Initialization/state keys differ from Phase 13')
        x = torch.arange(160, dtype=torch.float32).reshape(2,5,16)/160
        for training in (False, True):
            old.train(training); model.train(training)
            torch.manual_seed(1401); a = old(x)
            torch.manual_seed(1401); b = model(x)
            if not torch.equal(a,b):
                raise ValueError('Window16 forward/dropout differs from frozen architecture')
        model.eval()
        shapes = {}
        with torch.inference_mode():
            for n in LENGTHS:
                captured = []
                hooks = [] if name == 'lstm_only' else [getattr(model,k).register_forward_hook(
                    lambda module,args,output: captured.append(list(output.shape))) for k in ('conv1','conv2','maxpool')]
                logits = model(torch.zeros(2,5,n))
                for h in hooks:
                    h.remove()
                if logits.shape != (2,3) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid variable-length output')
                if name != 'lstm_only' and captured != [[2,32,n-2],[2,64,n-4],[2,64,(n-4)//2]]:
                    raise ValueError('CNN dimensions changed')
                shapes[str(n)] = captured
        result[name] = {'parameters': COUNTS[name], 'cnn_shapes': shapes,
                        'initial_state_exact': True, 'window16_train_eval_outputs_exact': True}
    return result


def create_windows(features, frames, ranges, n):
    check_length(n)
    features, frames, ranges = np.asarray(features), np.asarray(frames), np.asarray(ranges)
    if features.ndim != 2 or features.shape[1] != 5 or not len(features) or frames.shape != (len(features),):
        raise ValueError('Invalid per-source observations')
    if not np.isfinite(features).all() or not np.issubdtype(frames.dtype,np.integer) or not (np.diff(frames)>0).all():
        raise ValueError('Nonfinite features or nonchronological frames')
    if ranges.shape != (5,2) or not np.isfinite(ranges).all() or not (ranges[:,1]>ranges[:,0]).all():
        raise ValueError('Invalid frozen fold normalization')
    normalized = (features-ranges[:,0])/(ranges[:,1]-ranges[:,0])
    starts = np.arange(0,len(features),STRIDES[n],dtype=np.int64)
    indices = np.minimum(starts[:,None]+np.arange(n),len(features)-1)
    return {'windows': normalized[indices].transpose(0,2,1), 'frame_indices': frames[indices],
            'observation_indices': indices, 'start_indices': starts, 'padded': starts+n>len(features)}


def validate_data(data, expected, sealed, n):
    check_length(n)
    guard_sources(data['source_ids'],expected,sealed)
    size = len(data['labels'])
    if set(data['source_ids']) != set(expected) or data['windows'].shape != (size,5,n):
        raise ValueError('Fold coverage or variable-length shape mismatch')
    if not np.isfinite(data['windows']).all() or not np.isin(data['labels'],range(3)).all():
        raise ValueError('Invalid windows/labels')
    if data['frame_indices'].shape != (size,n) or (np.diff(data['frame_indices'],axis=1)<0).any():
        raise ValueError('Invalid window chronology')
    if data['padded'].shape != (size,) or data['padded'].dtype != np.bool_:
        raise ValueError('Invalid padding metadata')


def verify_normalizer(fold, expected_hash):
    directory = P12/f"fold_{fold['fold']}"
    meta = load(directory/'train_norm.json')
    if meta['train_source_ids'] != fold['train_source_ids'] or set(meta['train_source_ids']) & set(fold['validation_source_ids']):
        raise ValueError('Normalizer includes validation sources or omits train sources')
    if len(meta['train_source_ids']) != 102 or meta['sha256'] != expected_hash or sha256(directory/'train_norm.npy') != expected_hash:
        raise ValueError('Normalizer provenance/hash changed')
    return np.load(directory/'train_norm.npy',allow_pickle=False)


def development(folds):
    provenance = load(P12/'cache_provenance.json')
    if provenance['verified_source_ids'] != folds['development_source_ids'] or not provenance['pre_normalization']:
        raise ValueError('Cache provenance differs')
    rows = {r['source_id']:r for r in read_manifest(P11/'dataset_manifest_raw.csv')}
    final = {r['source_id']:r for r in read_manifest(P11/'dataset_manifest.csv')}
    values = {}
    for source in folds['development_source_ids']:
        guard_sources([source],folds['development_source_ids'],folds['excluded_test_source_ids'])
        path = P11/'raw_features'/(source+'.npz')
        if sha256(path) != provenance['cache_sha256'][source] or sha256(ROOT/rows[source]['relative_path']) != rows[source]['file_sha256']:
            raise ValueError('Raw development source/cache hash differs')
        with np.load(path,allow_pickle=False) as a:
            features, frames = a['features'],a['frame_indices']
        if len(features) != int(final[source]['valid_face_observation_count']):
            raise ValueError('Observation count differs')
        # Exercise strict per-source validation before constructing any windows.
        create_windows(features,frames,np.tile([0.,1.],(5,1)),16)
        values[source] = (features,frames)
    return values, rows, final


def assemble(ids, raw, rows, norm, n):
    pieces = []
    for source in ids:
        part = create_windows(*raw[source],norm,n)
        count = len(part['windows'])
        part.update(labels=np.full(count,rows[source]['class_id'],dtype=np.int64),source_ids=np.asarray([source]*count))
        pieces.append(part)
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def reconstruct16(folds, original, raw, rows, hashes):
    checks = []
    for fold in folds['folds']:
        f = fold['fold']
        norm = verify_normalizer(fold,hashes[f'results/phase12/fold_{f}/train_norm.npy'])
        for split in ('train','validation'):
            rebuilt = assemble(fold[split+'_source_ids'],raw,rows,norm,16)
            previous = original[f][split]
            for k in previous:
                np.testing.assert_array_equal(rebuilt[k],previous[k],err_msg='Window16 reconstruction differs: '+k)
            # Original archive omits starts: derive from each contiguous source's
            # window ordinal and historical stride, then verify frame identities.
            starts = np.concatenate([np.arange((previous['source_ids']==s).sum())*8 for s in fold[split+'_source_ids']])
            np.testing.assert_array_equal(starts,rebuilt['start_indices'])
            for s in fold[split+'_source_ids']:
                mask = previous['source_ids']==s
                indices = np.minimum(starts[mask,None]+np.arange(16),len(raw[s][0])-1)
                np.testing.assert_array_equal(rebuilt['observation_indices'][mask],indices)
                np.testing.assert_array_equal(previous['frame_indices'][mask],raw[s][1][indices])
            checks.append({'fold':f,'split':split,'windows':len(starts),'exact_keys':list(rebuilt),
                           'starts_verified_from_frozen_source_ordinals':True})
    return checks


def padding_stats(labels, padded):
    return [{'class_id':c,'windows':int((labels==c).sum()),'padded_windows':int(padded[labels==c].sum()),
             'padded_percentage':float(padded[labels==c].mean()*100) if (labels==c).any() else 0.} for c in range(3)]


def temporal(n):
    check_length(n)
    return {'window':n,'training_stride':STRIDES[n],'accepted_observations_before_first_prediction':n,
            'nominal_coverage_seconds':n/20,'nominal_warmup_seconds':n/20,
            'first_to_last_span_seconds':(n-1)/20,'float64_feature_payload_bytes':n*5*8,
            'potential_runtime_prediction_stride':1}


def control_references(folds):
    references = []
    for name in ARCHITECTURES:
        for fold in folds['folds']:
            f = fold['fold']
            for seed in SEEDS:
                directory = P13/name/f'fold_{f}'/f'seed_{seed}'
                cfg, metrics = load(directory/'config.json'),load(directory/'metrics.json')
                if cfg['architecture'] != name or cfg['fold'] != f or cfg['seed'] != seed or cfg['architecture_spec_sha256'] != sha256(P13/'architecture_spec.json'):
                    raise ValueError('Control configuration differs')
                if cfg['normalization_sha256'] != sha256(P12/f'fold_{f}'/'train_norm.npy') or cfg['recipe'] != dict(load(P13/'experiment_config.json')['recipe'],seed=seed):
                    raise ValueError('Control normalization/recipe differs')
                if metrics['checkpoint_sha256'] != sha256(directory/'best.pt') or metrics['test_inference_passes'] != 0:
                    raise ValueError('Control checkpoint or seal differs')
                with np.load(P12/f'fold_{f}'/'validation.npz',allow_pickle=False) as a:
                    labels,sources = a['labels'],a['source_ids']
                with np.load(directory/'validation_predictions.npz',allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['labels'],labels)
                    np.testing.assert_array_equal(a['source_ids'],sources)
                    window = classification_metrics(labels,a['probabilities'].argmax(1))
                    window['worst_class_recall'] = min(c['recall'] for c in window['per_class'].values())
                    video = aggregate_videos(labels,a['probabilities'],sources)
                if metrics['window'] != window or metrics['video'] != video or metrics['trainable_parameters'] != COUNTS[name]:
                    raise ValueError('Control saved probabilities/metrics/parameters differ')
                history = load(directory/'history.json')
                if metrics['best_epoch'] != history['best_epoch'] or window['macro_f1'] != history['best_validation_macro_f1']:
                    raise ValueError('Control selection differs')
                references.append({'architecture':name,'window':16,'fold':f,'seed':seed,'directory':directory.relative_to(ROOT).as_posix(),
                    'files':{p.name:sha256(p) for p in [directory/k for k in ('config.json','history.json','metrics.json','best.pt','validation_predictions.npz')]}})
    return references


def install_seal(folds, audit_path=None):
    """Technical file-open rejection, installed before candidate cache/model access."""
    rows = read_manifest(P11/'dataset_manifest_raw.csv')
    blocked = {(P11/name).resolve() for name in ('test.npz','test_predictions.npz','test_metrics.json','video_test_metrics.json')}
    for r in rows:
        if r['source_id'] in folds['excluded_test_source_ids']:
            blocked.add((P11/'raw_features'/(r['source_id']+'.npz')).resolve())
            blocked.add((ROOT/r['relative_path']).resolve())
    state = {'blocked_open_attempts':0,'test_cache_reads':0,'phase11_test_model_inference_passes':0}
    def audit(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes)) and Path(args[0]).resolve() in blocked:
            state['blocked_open_attempts'] += 1
            if audit_path:
                with Path(audit_path).open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'rejected':str(args[0]),'event':event})+'\n')
            raise PermissionError('Phase 11 test is sealed for Phase 14')
    sys.addaudithook(audit)
    return state


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Phase 14 artifacts already exist')
    folds,original,data,hashes = verify_protocol(P13,enforce_seal=False)
    seal = install_seal(folds)
    raw,rows,final = development(folds)
    gate = reconstruct16(folds,data,raw,rows,hashes)
    shapes = forward_gate()
    references = control_references(folds)
    output.mkdir(parents=True,exist_ok=False)
    (output/'reused_window16').mkdir()
    write_json(output/'reused_window16/references.json',references)
    write_json(output/'window16_integrity.json',{'passed':True,'checks':gate,'forward_gate':shapes})
    recipe = dict(load(P13/'experiment_config.json')['recipe'])
    recipe['padding'] = 'repeat last valid observation per source; range(0,n,N/2)'
    spec = {'window_sizes':list(LENGTHS),'stride_mapping':{str(k):v for k,v in STRIDES.items()},
            'architectures':list(ARCHITECTURES),'folds':[0,1,2],'seeds':list(SEEDS),'recipe':recipe,
            'selection_metric':'validation_window_macro_f1','padding_rule':recipe['padding'],
            'later_consideration_rules':RULES,'test_seal':'27 canonical Phase11 test sources; file opens blocked; zero inference',
            'test_seal_sha256':hashlib.sha256(json.dumps(folds['excluded_test_source_ids'],sort_keys=True).encode()).hexdigest(),
            'new_training_runs':108,'reused_controls':36,'total_comparison_cells':144,
            'latency_protocol':LATENCY,'declared_before_training':True}
    write_json(output/'window_spec.json',spec)
    input_hashes = dict(hashes,**{'results/phase13/experiment_config.json':sha256(P13/'experiment_config.json'),
                                 'results/phase13/architecture_spec.json':sha256(P13/'architecture_spec.json')})
    for reference in references:
        input_hashes.update({reference['directory']+'/'+k:v for k,v in reference['files'].items()})
    padding,source_stats,artifacts = [],[],{}
    for n in LENGTHS:
        for fold in folds['folds']:
            f = fold['fold']
            norm = verify_normalizer(fold,hashes[f'results/phase12/fold_{f}/train_norm.npy'])
            for split in ('train','validation'):
                values = assemble(fold[split+'_source_ids'],raw,rows,norm,n)
                validate_data(values,fold[split+'_source_ids'],folds['excluded_test_source_ids'],n)
                if n != 16:
                    path = output/f'windows_{n}'/f'fold_{f}'/(split+'.npz')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**values)
                    artifacts[path.relative_to(output).as_posix()] = sha256(path)
                for name in ARCHITECTURES:
                    padding.extend(dict(row,architecture=name,window=n,fold=f,split=split) for row in padding_stats(values['labels'],values['padded']))
                for s in fold[split+'_source_ids']:
                    mask = values['source_ids']==s
                    source_stats.append({'source_id':s,'class_id':rows[s]['class_id'],'window':n,'fold':f,'split':split,
                        'valid_observations':len(raw[s][0]),'decoded_frames':int(final[s]['successfully_decoded_frame_count']),
                        'windows':int(mask.sum()),'padded_windows':int(values['padded'][mask].sum()),
                        'padded_percentage':float(values['padded'][mask].mean()*100)})
    write_json(output/'padding_counts.json',padding)
    write_json(output/'source_window_counts.json',source_stats)
    # 51 fold0-validation starts at zero + 13 starts at observation48, selected
    # lexicographically among sources long enough. 48 is common to all strides.
    ids = folds['folds'][0]['validation_source_ids']
    anchors = [(s,0) for s in ids]+[(s,48) for s in ids if len(raw[s][0])>48][:13]
    if len(anchors) != LATENCY['input_count']:
        raise ValueError('Cannot construct predeclared shared anchors')
    norm = np.load(P12/'fold_0/train_norm.npy',allow_pickle=False)
    for n in LENGTHS:
        parts = [create_windows(*raw[s],norm,n) for s,start in anchors]
        selected = [int(np.flatnonzero(p['start_indices']==start)[0]) for p,(s,start) in zip(parts,anchors)]
        tensors = np.asarray([p['windows'][i] for p,i in zip(parts,selected)],dtype=np.float32)
        frames = np.asarray([p['frame_indices'][i] for p,i in zip(parts,selected)])
        path = output/f'latency_inputs_{n}.npz'
        np.savez_compressed(path,windows=tensors,source_ids=np.asarray([s for s,start in anchors]),
                            start_indices=np.asarray([start for s,start in anchors]),frame_indices=frames)
        artifacts[path.name] = sha256(path)
        write_json(output/f'latency_inputs_{n}.json',{'sha256':sha256(path),'window':n,'fold':0,'split':'validation',
            'sources':[s for s,start in anchors],'start_indices':[start for s,start in anchors],
            'frame_indices':frames.tolist(),'tensor_sha256':[hashlib.sha256(t.tobytes()).hexdigest() for t in tensors]})
    artifacts.update({p.name:sha256(p) for p in output.glob('latency_inputs_*.json')})
    artifacts.update({name:sha256(output/name) for name in ('padding_counts.json','source_window_counts.json','window16_integrity.json','reused_window16/references.json')})
    write_json(output/'experiment_config.json',{'window_spec_sha256':sha256(output/'window_spec.json'),
        'architecture_spec_sha256':sha256(P13/'architecture_spec.json'),'cv_sha256':sha256(P12/'cv_folds.json'),
        'input_artifact_sha256':input_hashes,'prepared_artifact_sha256':artifacts,
        'source_code_sha256':{'evaluation/windows.py':sha256(Path(__file__))},
        'test_seal_sha256':hashlib.sha256(json.dumps(folds['excluded_test_source_ids'],sort_keys=True).encode()).hexdigest()})
    write_json(output/'integrity.json',{'passed':True,'development_sources':153,'normalization_refits':0,
        'fold_normalizers_train_ids_exact':True,'validation_contributions_to_normalizers':0,
        'reconstructed_window16_exact':True,'controls_referenced':36,'seal':seal})


def verify(output, enforce_seal=True):
    output = Path(output)
    folds,original,data,hashes = verify_protocol(P13,enforce_seal=False)
    seal = install_seal(folds,output/'rejected_test_opens.jsonl') if enforce_seal else None
    config = load(output/'experiment_config.json')
    if sha256(output/'window_spec.json') != config['window_spec_sha256']:
        raise ValueError('Predeclared window spec changed')
    for name,expected in config['input_artifact_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen input artifact changed: '+name)
    for name,expected in config['prepared_artifact_sha256'].items():
        if sha256(output/name) != expected:
            raise ValueError('Prepared window artifact changed: '+name)
    for name,expected in config['source_code_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Predeclared Phase14 source changed')
    return folds,data,config,seal


def fit(model,train,validation,recipe,fold,sealed,n):
    """Phase13 unweighted fit operations, with variable-length validation only."""
    validate_data(train,fold['train_source_ids'],sealed,n)
    validate_data(validation,fold['validation_source_ids'],sealed,n)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'])
    batches = training_loader(train,'unweighted',recipe['seed'],recipe['batch_size'])
    history,best,score,best_epoch,stale,exposure = [],None,-1.,None,0,[]
    for epoch in range(1,recipe['max_epochs']+1):
        model.train()
        loss_sum = correct = count = 0
        seen = np.zeros(3,dtype=int)
        for inputs,labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits,labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(labels)
            correct += int((logits.argmax(1)==labels).sum()); count += len(labels)
            seen += np.bincount(labels.numpy(),minlength=3)
        validate_data(validation,fold['validation_source_ids'],sealed,n)
        probabilities,val_loss = predict(model,validation,batch_size=recipe['batch_size'])
        metrics = classification_metrics(validation['labels'],probabilities.argmax(1))
        history.append({'epoch':epoch,'train_loss':loss_sum/count,'train_accuracy':correct/count,
            'validation_loss':val_loss,'validation_accuracy':metrics['accuracy'],
            'validation_macro_f1':metrics['macro_f1'],'learning_rate':recipe['learning_rate']})
        exposure.append(seen.tolist())
        if metrics['macro_f1']>score+recipe['minimum_improvement']:
            score,best_epoch,stale = metrics['macro_f1'],epoch,0
            best = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale>=recipe['patience']:
            break
    model.load_state_dict(best); model.eval()
    return {'epochs':history,'best_epoch':best_epoch,'stopping_epoch':len(history),
        'best_validation_macro_f1':score,'selection_used_test':False,'epoch_class_exposure':exposure,
        'stop_reason':'patience' if stale>=recipe['patience'] else 'max_epochs'}


def diagnostics(values,probabilities):
    result = {}
    for name,mask in (('non_padded',~values['padded']),('padded',values['padded'])):
        if mask.any():
            result[name] = classification_metrics(values['labels'][mask],probabilities[mask].argmax(1))
        else:
            result[name] = {'support':0,'macro_f1':None,'accuracy':None}
    return result


def run_directory(output,name,n,fold,seed):
    return (P13/name if n==16 else Path(output)/f'windows_{n}'/name)/f'fold_{fold}'/f'seed_{seed}'


def run(output):
    output = Path(output).resolve()
    folds,old,config,seal = verify(output)
    seed_everything(1201)
    write_json(output/'experiment_started.json',{'environment':environment(),'window_spec_sha256':config['window_spec_sha256']})
    completed = 0
    with (output/'candidate_access_audit.jsonl').open('x',encoding='utf-8') as audit:
        for n in (8,24,32):
            for name in ARCHITECTURES:
                for fold in folds['folds']:
                    f = fold['fold']
                    values = {}
                    for split in ('train','validation'):
                        with np.load(output/f'windows_{n}'/f'fold_{f}'/(split+'.npz'),allow_pickle=False) as a:
                            values[split] = {k:a[k] for k in a.files}
                        validate_data(values[split],fold[split+'_source_ids'],folds['excluded_test_source_ids'],n)
                    for seed in SEEDS:
                        target = run_directory(output,name,n,f,seed)
                        target.mkdir(parents=True,exist_ok=False)
                        try:
                            seed_everything(seed)
                            model = build_model(name,n).to('cpu')
                            if state_digest(model.state_dict()) != load(P13/name/f'fold_{f}'/f'seed_{seed}'/'config.json')['initial_weights_sha256']:
                                raise ValueError('Initial weights differ from same-seed frozen architecture')
                            recipe = dict(load(output/'window_spec.json')['recipe'],seed=seed,window_length=n,window_stride=STRIDES[n])
                            write_json(target/'config.json',{'architecture':name,'window':n,'fold':f,'seed':seed,'recipe':recipe,
                                'architecture_spec_sha256':config['architecture_spec_sha256'],'window_spec_sha256':config['window_spec_sha256'],
                                'normalization_sha256':sha256(P12/f'fold_{f}'/'train_norm.npy'),'initial_weights_sha256':state_digest(model.state_dict())})
                            audit.write(json.dumps({'architecture':name,'window':n,'fold':f,'seed':seed,
                                'training_sources':fold['train_source_ids'],'validation_inference_sources':fold['validation_source_ids'],
                                'test_inference_passes':0})+'\n'); audit.flush()
                            started = time.perf_counter()
                            history = fit(model,values['train'],values['validation'],recipe,fold,folds['excluded_test_source_ids'],n)
                            history['training_wall_seconds'] = time.perf_counter()-started
                            write_json(target/'history.json',history)
                            torch.save(model.state_dict(),target/'best.pt')
                            probabilities,loss = predict(model,values['validation'])
                            val = values['validation']
                            np.savez_compressed(target/'validation_predictions.npz',probabilities=probabilities,labels=val['labels'],source_ids=val['source_ids'])
                            window = classification_metrics(val['labels'],probabilities.argmax(1))
                            window['worst_class_recall'] = min(c['recall'] for c in window['per_class'].values())
                            video = aggregate_videos(val['labels'],probabilities,val['source_ids'])
                            write_json(target/'metrics.json',{'window':window,'video':video,'diagnostics':diagnostics(val,probabilities),
                                'validation_loss':loss,'best_epoch':history['best_epoch'],'checkpoint_sha256':sha256(target/'best.pt'),
                                'checkpoint_bytes':(target/'best.pt').stat().st_size,'trainable_parameters':COUNTS[name],'test_inference_passes':0})
                            completed += 1
                            print(f"{completed}/108 window={n} {name} fold={f} seed={seed} epochs={len(history['epochs'])} F1={window['macro_f1']:.6f}",flush=True)
                        except Exception as error:
                            write_json(target/'failure.json',{'type':type(error).__name__,'message':str(error),'action':'stop, no discard'})
                            raise
    write_json(output/'test_access_audit.json',dict(seal,completed_new_runs=completed,reused_controls=36,
        excluded_test_source_ids=folds['excluded_test_source_ids'],candidate_access_log_sha256=sha256(output/'candidate_access_audit.jsonl'),
        guard='source whitelist before fit/every epoch validation; permanent process file-open rejection for test caches/videos/arrays/predictions/metrics'))


def latency(output):
    output = Path(output).resolve()
    folds,old,config,seal = verify(output)
    if load(output/'test_access_audit.json')['completed_new_runs'] != 108:
        raise ValueError('Training matrix incomplete')
    seed_everything(1201)
    protocol = load(output/'window_spec.json')['latency_protocol']
    inputs = {}
    for n in LENGTHS:
        meta = load(output/f'latency_inputs_{n}.json')
        with np.load(output/f'latency_inputs_{n}.npz',allow_pickle=False) as a:
            guard_sources(a['source_ids'],folds['folds'][0]['validation_source_ids'],folds['excluded_test_source_ids'])
            tensors = a['windows']
        if [hashlib.sha256(t.tobytes()).hexdigest() for t in tensors] != meta['tensor_sha256']:
            raise ValueError('Latency tensors changed')
        inputs[n] = [torch.from_numpy(t[None].copy()) for t in tensors]
    jobs = [(name,n,repeat) for repeat in range(protocol['repetitions']) for n in LENGTHS for name in ARCHITECTURES]
    random.Random(protocol['order_seed']).shuffle(jobs)
    write_json(output/'latency_started.json',{'environment':environment(),'protocol':protocol,'job_order':jobs,'model_only':True})
    raw,records,stalls = {},[],[]
    directory = output/'latency_raw'
    directory.mkdir(exist_ok=False)
    for name,n,repeat in jobs:
        checkpoint = run_directory(output,name,n,0,1201)/'best.pt'
        if sha256(checkpoint) != load(checkpoint.with_name('metrics.json'))['checkpoint_sha256']:
            raise ValueError('Latency checkpoint changed')
        model = build_model(name,n).to('cpu')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)); model.eval()
        wall,cpu = [],[]
        with torch.inference_mode():
            for i in range(protocol['warmup_calls']):
                model(inputs[n][i%len(inputs[n])])
            for i in range(protocol['measured_calls_per_job']):
                x = inputs[n][i%len(inputs[n])]
                c0 = time.process_time_ns(); w0 = time.perf_counter_ns()
                logits = model(x)
                w1 = time.perf_counter_ns(); c1 = time.process_time_ns()
                wall.append((w1-w0)/1e6); cpu.append((c1-c0)/1e6)
            if logits.shape != (1,3) or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite timing output')
        key = f'{name}_window_{n}_repeat_{repeat}'
        raw[key+'_wall_ms'],raw[key+'_cpu_ms'] = np.asarray(wall),np.asarray(cpu)
        np.savez_compressed(directory/(key+'.npz'),wall_ms=wall,process_cpu_ms=cpu)
        extreme = [{'sample_index':i,'wall_ms':w,'process_cpu_ms':cpu[i],'input_index':i%64}
                   for i,w in enumerate(wall) if w>=protocol['extreme_wall_stall_ms']]
        stalls.extend(dict(s,job=key) for s in extreme)
        records.append({'architecture':name,'window':n,'repeat':repeat,'checkpoint_sha256':sha256(checkpoint),
            'wall':latency_statistics(wall),'process_cpu':latency_statistics(cpu),'extreme_stalls':extreme})
        print(f"latency {key} P50={records[-1]['wall']['p50_ms']:.4f} ms stalls={len(extreme)}",flush=True)
    np.savez_compressed(output/'latency_samples.npz',**raw)
    aggregate = {}
    for name in ARCHITECTURES:
        for n in LENGTHS:
            aggregate[f'{name}_{n}'] = {clock:latency_statistics(np.concatenate([
                v for k,v in raw.items() if k.startswith(f'{name}_window_{n}_') and k.endswith(suffix)]))
                for clock,suffix in (('wall','_wall_ms'),('process_cpu','_cpu_ms'))}
    write_json(output/'latency_results.json',{'protocol':protocol,'runs':records,'aggregate':aggregate,'extreme_stalls':stalls,
        'raw_samples_sha256':sha256(output/'latency_samples.npz'),'seal':seal,
        'clock_note':'process CPU sums threads and on Windows can be coarse/zero per call; differs from elapsed wall time; clocks include no input preparation'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','latency'))
    parser.add_argument('--output',default=str(ROOT/'results/phase14'))
    args = parser.parse_args()
    globals()[args.action](args.output)


if __name__ == '__main__':
    main()
