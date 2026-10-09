"""Phase 15 frozen full-development candidate and held-out boundary checks."""

import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import pytest
import torch

from evaluation import FEATURE_NAMES
import evaluation.final_candidate as final_candidate
from evaluation.final_candidate import CHECKPOINT, MODELS, NORM, RESULTS, ROOT, fit_fixed, source_hash
from evaluation.manifest import sha256
from evaluation.windows import build_model, create_windows


@pytest.fixture(scope='module')
def artifacts():
    spec = json.loads((RESULTS/'selection_spec.json').read_text())
    folds = json.loads((ROOT/'results/phase12/cv_folds.json').read_text())
    norm = json.loads((MODELS/'lstm32_dev_norm.json').read_text())
    dataset = json.loads((RESULTS/'window_dataset.json').read_text())
    metadata = json.loads((MODELS/'lstm_only_w32_full_development.json').read_text())
    return spec,folds,norm,dataset,metadata


def test_selection_frozen_before_training(artifacts):
    spec,*_ = artifacts
    assert spec['frozen_before_new_training'] and 'AFTER observing' in spec['selection_timing']
    assert spec['architecture'] == 'lstm_only' and spec['window_length'] == 32 and spec['training_stride'] == 16
    assert spec['loss'] == 'unweighted CrossEntropyLoss' and spec['seed'] == 1501
    assert spec['recipe']['validation_loader'] is False and spec['recipe']['test_loader'] is False
    assert spec['recipe']['checkpoint'] == 'final epoch'


def test_freeze_reproduces_exact_spec(tmp_path, monkeypatch):
    target = tmp_path/'phase15'
    monkeypatch.setattr(final_candidate, 'RESULTS', target)
    final_candidate.freeze_selection()
    assert sha256(target/'selection_spec.json') == sha256(RESULTS/'selection_spec.json')


def test_cohort_exact_and_sealed(artifacts):
    spec,folds,norm,dataset,_ = artifacts
    ids=spec['full_development_source_ids'];sealed=spec['excluded_phase11_test_source_ids']
    assert ids == folds['development_source_ids'] and sealed == folds['excluded_test_source_ids']
    assert len(ids)==len(set(ids))==153 and len(sealed)==27 and not set(ids)&set(sealed)
    assert dataset['source_ids']==norm['source_ids']==ids and len(dataset['by_source'])==153
    assert [r['source_id'] for r in dataset['by_source']] == ids
    assert spec['source_set_sha256']==norm['source_set_sha256']==source_hash(ids)
    assert spec['class_video_counts']==norm['class_video_counts']==[51,51,51]
    assert [x['videos'] for x in dataset['by_class']] == [51,51,51]


def test_normalizer_semantics_and_serialization(artifacts):
    spec,_,meta,_,modelmeta=artifacts
    values=np.load(NORM,allow_pickle=False)
    assert values.shape==(5,2) and values.dtype==np.float64
    assert np.isfinite(values).all() and (values[:,1]>values[:,0]).all()
    assert meta['feature_order']==list(FEATURE_NAMES)==['EAR-R','EAR-L','MAR','phi','theta']
    np.testing.assert_array_equal(values,np.asarray(meta['min_max']))
    np.testing.assert_array_equal(values[:,1]-values[:,0],meta['denominators'])
    assert 'zero-initialized extrema' in meta['fit_method']
    assert meta['source_ids']==spec['full_development_source_ids']
    assert meta['artifact_sha256']==modelmeta['normalizer_sha256']==sha256(NORM)


def test_windows_and_short_source(artifacts):
    spec,_,_,meta,_=artifacts
    with np.load(RESULTS/'development_windows.npz',allow_pickle=False) as a:
        assert a['windows'].shape==(1352,5,32)
        assert a['labels'].shape==(1352,)
        assert np.isfinite(a['windows']).all()
        assert (np.diff(a['frame_indices'],axis=1)>=0).all()
        assert set(a['source_ids'])==set(spec['full_development_source_ids'])
        assert list(np.bincount(a['labels'],minlength=3)) == [x['windows'] for x in meta['by_class']]
        assert int(a['padded'].sum()) == meta['padded_windows']
    assert meta['window_length']==32 and meta['training_stride']==16
    assert meta['short_source']['valid_observations']==10
    assert meta['short_source']['decoded_frames']==135
    assert meta['short_source']['windows']==meta['short_source']['padded_windows']==1
    assert meta['artifact_sha256']==sha256(RESULTS/'development_windows.npz')


def test_repeat_last_is_within_source():
    values=create_windows(np.arange(50,dtype=float).reshape(10,5),np.arange(10),np.tile([0.,50.],(5,1)),32)
    assert values['windows'].shape==(1,5,32)
    assert values['padded'].tolist()==[True]
    np.testing.assert_array_equal(values['observation_indices'][0,10:],np.full(22,9))


def test_frozen_model_contract_and_orientation():
    model=build_model('lstm_only',32).eval()
    assert sum(p.numel() for p in model.parameters())==209667
    assert model.lstm.input_size==5 and model.lstm.hidden_size==128 and model.lstm.num_layers==2
    assert model.lstm.batch_first and model.fc1.in_features==128 and model.fc1.out_features==64
    assert model.fc2.out_features==3 and model.dropout_lstm.p==.25
    captured=[]
    handle=model.lstm.register_forward_pre_hook(lambda module,args: captured.append(tuple(args[0].shape)))
    with torch.inference_mode(): logits=model(torch.zeros(2,5,32,dtype=torch.float32))
    handle.remove()
    assert captured==[(2,32,5)] and logits.shape==(2,3) and torch.isfinite(logits).all()


def test_epoch_derivation_and_frozen_inputs(artifacts):
    spec,*_=artifacts
    entries=spec['phase14_best_epochs']
    assert len(entries)==9 and {(e['fold'],e['seed']) for e in entries}=={(f,s) for f in range(3) for s in (1201,1202,1203)}
    for e in entries:
        path=ROOT/e['artifact']
        assert e['sha256']==sha256(path)
        assert e['best_epoch']==json.loads(path.read_text())['best_epoch']
    assert spec['fixed_epoch_count']==statistics.median(e['best_epoch'] for e in entries)==13
    assert spec['seed']==1501


def test_no_validation_or_test_loader_in_fixed_fit():
    source=inspect.getsource(fit_fixed)
    assert 'training_loader(data' in source
    assert 'validation_loader' not in source and 'test_loader' not in source
    assert 'predict(' not in source and 'scheduler=' not in source


def test_checkpoint_metadata_and_replay(artifacts):
    spec,_,_,_,meta=artifacts
    assert meta['architecture']=='lstm_only' and meta['parameters']==209667
    assert meta['window_length']==32 and meta['training_stride']==16 and meta['expected_runtime_stride']==1
    assert meta['checkpoint_sha256']==sha256(CHECKPOINT)
    assert meta['checkpoint_bytes']==CHECKPOINT.stat().st_size and meta['normalization_bytes']==NORM.stat().st_size
    assert meta['phase15_selection_spec_sha256']==sha256(RESULTS/'selection_spec.json')
    assert meta['phase12_fold_sha256']==sha256(ROOT/'results/phase12/cv_folds.json')
    assert meta['phase13_architecture_spec_sha256']==sha256(ROOT/'results/phase13/architecture_spec.json')
    assert meta['phase14_window_spec_sha256']==sha256(ROOT/'results/phase14/window_spec.json')
    assert meta['source_set_sha256']==spec['source_set_sha256']
    model=build_model('lstm_only',32).eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    with np.load(RESULTS/'verification_inputs.npz',allow_pickle=False) as a:x=a['windows']
    with torch.inference_mode():logits=model(torch.from_numpy(x)).numpy()
    assert logits.shape==(64,3) and np.isfinite(logits).all()
    np.testing.assert_array_equal(logits,np.load(RESULTS/'canonical_logits.npy',allow_pickle=False))
    assert np.isin(logits.argmax(1),[0,1,2]).all()


def test_fixed_dev_tensor_identities(artifacts):
    spec,*_=artifacts
    meta=json.loads((RESULTS/'verification_inputs.json').read_text())
    assert meta['artifact_sha256']==sha256(RESULTS/'verification_inputs.npz')
    with np.load(RESULTS/'verification_inputs.npz',allow_pickle=False) as a:x=a['windows']
    assert x.shape==(64,5,32) and x.dtype==np.float32
    assert all(i['source_id'] in spec['full_development_source_ids'] and i['start_index']==0 for i in meta['identities'])
    assert [hashlib.sha256(t.tobytes()).hexdigest() for t in x]==[i['tensor_sha256'] for i in meta['identities']]


def test_fresh_process_repeat_and_history():
    repeat=json.loads((RESULTS/'reproducibility/result.json').read_text())
    assert repeat['fresh_process'] and repeat['bit_identical'] and repeat['history_identical'] and repeat['final_logits_identical']
    assert repeat['max_state_tensor_absolute_difference']==repeat['max_logit_absolute_difference']==0
    assert all(x==0 for x in repeat['per_tensor_max_absolute_difference'].values())
    history=json.loads((RESULTS/'training_history.json').read_text())
    assert len(history['epochs'])==13 and [e['epoch'] for e in history['epochs']]==list(range(1,14))
    assert 'training diagnostics only' in history['scope']
    with (RESULTS/'training_history.csv').open(newline='') as stream:rows=list(__import__('csv').DictReader(stream))
    assert len(rows)==13 and all(int(x['training_examples'])==1352 for x in rows)


def test_latency_samples_and_scope():
    meta=json.loads((RESULTS/'latency_results.json').read_text())
    with np.load(RESULTS/'latency_samples.npz',allow_pickle=False) as a:
        assert a['wall_ms'].shape==(3072,) and a['process_cpu_ms'].shape==(3072,)
        assert np.isfinite(a['wall_ms']).all() and (a['wall_ms']>=0).all()
    assert meta['aggregate']['wall']['sample_count']==3072
    assert 'model-only' in meta['scope'] and 'not full application' in meta['scope']
    assert meta['raw_samples_sha256']==sha256(RESULTS/'latency_samples.npz')


def test_runtime_contract_and_test_audit(artifacts):
    spec,*_=artifacts
    contract=json.loads((RESULTS/'runtime_contract.json').read_text())
    assert contract['current_production']['architecture']=='cnn_lstm' and contract['current_production']['window_length']==16
    assert contract['candidate']['architecture']=='lstm_only' and contract['candidate']['window_length']==32
    assert contract['candidate']['temporal_buffer_length']==32 and contract['candidate']['inference_stride_after_ready']==1
    assert contract['candidate']['feature_order']==list(FEATURE_NAMES)
    assert contract['candidate']['class_mapping']=={'Normal':0,'Yawning':1,'Eye Close':2}
    assert 'Phase16' in contract['policy_warning']
    audit=json.loads((RESULTS/'test_access_audit.json').read_text())
    assert audit['excluded_test_source_ids']==spec['excluded_phase11_test_source_ids']
    assert audit['candidate_test_inference_passes']==audit['test_cache_reads']==audit['test_video_reads']==audit['blocked_open_attempts']==0


def test_import_does_not_train_or_write():
    before=CHECKPOINT.stat().st_mtime_ns
    subprocess.run([sys.executable,'-B','-c','import evaluation.final_candidate'],cwd=ROOT,check=True,capture_output=True)
    assert CHECKPOINT.stat().st_mtime_ns==before


@pytest.mark.parametrize('name',[
    'saved_model.pt','norm.npy','model.py','main.py','driver_monitor/inference/engine.py',
    'results/phase12/cv_folds.json','results/phase13/architecture_spec.json','results/phase14/window_spec.json'])
def test_prior_artifact_unchanged(name):
    baseline=json.loads((ROOT/'docs/phase15_before_hashes.json').read_text())
    phase16_before=json.loads((ROOT/'docs/phase16_before_hashes.json').read_text())
    # Phase15's historical preservation claim is checked at the Phase16 entry
    # boundary for runtime files deliberately changed during Phase16.
    if name in {'main.py','driver_monitor/inference/engine.py'}:
        assert phase16_before[name]==baseline[name]
    else:
        assert sha256(ROOT/name)==baseline[name]
