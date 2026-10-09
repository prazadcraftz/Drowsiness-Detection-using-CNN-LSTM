"""Phase14 frozen architecture adaptation, window provenance and experiment boundaries."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from evaluation import CLASS_NAMES
from evaluation.windows import (LENGTHS, STRIDES, ARCHITECTURES, COUNTS, RULES, LATENCY,
    ROOT, P12, P13, build_model, create_windows, validate_data, verify_normalizer,
    verify, forward_gate, reconstruct16, development, install_seal, temporal,
    padding_stats, diagnostics, paired_summary, latency_statistics, run_directory)
from evaluation.architectures import verify_protocol, load
from evaluation.manifest import sha256, write_json
from evaluation.balancing import guard_sources
from evaluation.reproducibility import seed_everything

D = ROOT/'results/phase14'


def test_exact_matrix_lengths_strides_seeds_and_rules():
    assert LENGTHS == (8,16,24,32) and STRIDES == {8:4,16:8,24:12,32:16}
    assert ARCHITECTURES == ('cnn_lstm','lstm_only','cnn_gru','cnn_bilstm')
    spec = load(D/'window_spec.json')
    assert spec['seeds'] == [1201,1202,1203]
    assert spec['new_training_runs'] == 108 and spec['reused_controls'] == 36
    assert spec['later_consideration_rules'] == RULES and spec['latency_protocol'] == LATENCY
    assert spec['recipe']['max_epochs'] == 60 and spec['recipe']['patience'] == 10
    assert spec['recipe']['class_weighting'] is False and spec['recipe']['weighted_sampling'] is False


@pytest.mark.parametrize('n',LENGTHS)
@pytest.mark.parametrize('name',ARCHITECTURES)
def test_variable_length_shapes_finite_cnn_dimensions_and_count(name,n):
    seed_everything(1201)
    model = build_model(name,n).eval()
    assert sum(p.numel() for p in model.parameters()) == COUNTS[name]
    assert model.state_dict().keys() == torch.load(P13/name/'fold_0/seed_1201/best.pt',weights_only=True).keys()
    shapes = []
    hooks = [] if name == 'lstm_only' else [getattr(model,k).register_forward_hook(
        lambda m,a,o:shapes.append(tuple(o.shape))) for k in ('conv1','conv2','maxpool')]
    with torch.inference_mode():
        out = model(torch.arange(10*n,dtype=torch.float32).reshape(2,5,n)/(10*n))
    for h in hooks:
        h.remove()
    assert out.shape == (2,3) and torch.isfinite(out).all()
    if name != 'lstm_only':
        assert shapes == [(2,32,n-2),(2,64,n-4),(2,64,(n-4)//2)]


@pytest.mark.parametrize('n',(4,12,20,40,64))
def test_undeclared_lengths_rejected(n):
    with pytest.raises(ValueError):
        build_model('cnn_lstm',n)
    with pytest.raises(ValueError):
        create_windows(np.ones((100,5)),np.arange(100),np.tile([0.,1.],(5,1)),n)


def test_frozen_initialization_state_and_training_dropout_bit_exact():
    gate = forward_gate()
    assert all(v['initial_state_exact'] and v['window16_train_eval_outputs_exact'] for v in gate.values())


@pytest.mark.parametrize('n',LENGTHS)
def test_exact_tail_padding_and_chronology_per_source(n):
    features = np.arange(50,dtype=np.float64).reshape(10,5)
    frames = np.arange(10)*7+2
    norm = np.tile([0.,50.],(5,1))
    data = create_windows(features,frames,norm,n)
    assert data['start_indices'].tolist() == list(range(0,10,n//2))
    for i,start in enumerate(data['start_indices']):
        idx = np.minimum(np.arange(start,start+n),9)
        np.testing.assert_array_equal(data['windows'][i],(features[idx]/50).T)
        np.testing.assert_array_equal(data['frame_indices'][i],frames[idx])
        assert data['padded'][i] == (start+n>10)
    separate = create_windows(features+1000,frames+1000,norm,n)
    assert (data['windows']<1).all() and (separate['windows']>=20).all()


def test_nonfinite_and_nonchronological_caches_rejected():
    norm = np.tile([0.,1.],(5,1))
    with pytest.raises(ValueError):
        create_windows(np.full((10,5),np.nan),np.arange(10),norm,8)
    with pytest.raises(ValueError):
        create_windows(np.ones((10,5)),np.zeros(10,dtype=int),norm,8)


def test_normalizer_exact_train_sources_and_frozen_hash():
    folds,_,_,hashes = verify_protocol(P13,enforce_seal=False)
    for fold in folds['folds']:
        assert verify_normalizer(fold,hashes[f"results/phase12/fold_{fold['fold']}/train_norm.npy"]).shape == (5,2)
        with pytest.raises(ValueError):
            verify_normalizer(fold,'bad hash')
        bad = dict(fold,train_source_ids=fold['train_source_ids'][:-1]+fold['validation_source_ids'][:1])
        with pytest.raises(ValueError):
            verify_normalizer(bad,hashes[f"results/phase12/fold_{fold['fold']}/train_norm.npy"])


def test_window16_raw_reconstruction_all_keys_starts_and_frames_exact():
    folds,_,old,hashes = verify_protocol(P13,enforce_seal=False)
    raw,rows,final = development(folds)
    gate = reconstruct16(folds,old,raw,rows,hashes)
    assert len(gate) == 6 and all(v['starts_verified_from_frozen_source_ordinals'] for v in gate)
    short = [s for s,r in rows.items() if r['relative_path'] == 'Eye Close/Eye Close_51.avi'][0]
    assert len(raw[short][0]) == 10 and int(final[short]['successfully_decoded_frame_count']) == 135


def test_frozen_protocol_spec_controls_and_prepared_artifacts_verified():
    folds,old,cfg,seal = verify(D,enforce_seal=False)
    assert len(folds['development_source_ids']) == 153 and seal is None
    assert len(load(D/'reused_window16/references.json')) == 36
    assert sha256(D/'window_spec.json') == cfg['window_spec_sha256']
    assert load(D/'integrity.json')['normalization_refits'] == 0


def test_variable_length_fit_reproduces_frozen_training_recipe_at_16():
    from evaluation.windows import fit
    from evaluation.balancing import fit as frozen_fit
    from evaluation.models import build_model as frozen_model
    folds,original,old,_ = verify_protocol(P13,enforce_seal=False)
    fold = folds['folds'][0]
    recipe = dict(original['recipe'],seed=1201,max_epochs=2)
    seed_everything(1201)
    previous = frozen_model('cnn_lstm')
    history = frozen_fit(previous,old[0]['train'],old[0]['validation'],'unweighted',recipe,fold,folds['excluded_test_source_ids'])
    seed_everything(1201)
    current = build_model('cnn_lstm',16)
    reproduced = fit(current,old[0]['train'],old[0]['validation'],recipe,fold,folds['excluded_test_source_ids'],16)
    assert reproduced == history
    assert all(torch.equal(v,current.state_dict()[k]) for k,v in previous.state_dict().items())


def test_candidate_source_rejection_and_no_test_open_in_subprocess():
    folds = load(P12/'cv_folds.json')
    with pytest.raises(ValueError):
        guard_sources(folds['excluded_test_source_ids'][:1],folds['development_source_ids'],folds['excluded_test_source_ids'])
    code = '''
from evaluation.windows import *
folds=load(P12/'cv_folds.json')
state=install_seal(folds)
rows=read_manifest(P11/'dataset_manifest_raw.csv')
source=folds['excluded_test_source_ids'][0]
row=next(r for r in rows if r['source_id']==source)
paths=[P11/'raw_features'/(source+'.npz'),ROOT/row['relative_path']]
paths += [P11/name for name in ('test.npz','test_predictions.npz','test_metrics.json','video_test_metrics.json')]
for path in paths:
    try: path.open('rb')
    except PermissionError: pass
    else: raise AssertionError('test open allowed')
assert state['blocked_open_attempts']==6 and state['test_cache_reads']==0
'''
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_paired_comparison_exact_pairs_and_sample_sd():
    p = paired_summary({(0,1):.9,(0,2):.7,(1,1):.8},{(0,1):.8,(0,2):.8,(1,1):.8})
    assert p['positive_pairs']==1 and p['negative_pairs']==1 and p['zero_pairs']==1
    assert p['mean'] == pytest.approx(0,abs=1e-15) and p['std']==pytest.approx(.1)
    with pytest.raises(ValueError):
        paired_summary({(0,1):1},{(1,1):1})


@pytest.mark.parametrize('n',LENGTHS)
def test_nominal_coverage_span_startup_and_buffer_distinct(n):
    t = temporal(n)
    assert t['nominal_coverage_seconds'] == n/20
    assert t['first_to_last_span_seconds'] == (n-1)/20
    assert t['accepted_observations_before_first_prediction'] == n
    assert t['float64_feature_payload_bytes'] == n*40
    assert t['potential_runtime_prediction_stride'] == 1


def test_padding_rates_and_secondary_subset_metrics():
    labels = np.array([0,0,1,1,2,2]); padded = np.array([True,False,True,True,False,False])
    stats = padding_stats(labels,padded)
    assert [v['padded_percentage'] for v in stats] == [50,100,0]
    probabilities = np.eye(3)[labels]
    d = diagnostics({'labels':labels,'padded':padded},probabilities)
    assert d['padded']['support']==3 and d['non_padded']['support']==3
    d = diagnostics({'labels':labels,'padded':np.zeros(6,dtype=bool)},probabilities)
    assert d['padded']['support']==0 and d['padded']['macro_f1'] is None


def test_latency_serialization_retains_extreme_samples_and_shared_anchors(tmp_path):
    values = [1.,2.,3.,400000.]
    stats = latency_statistics(values)
    write_json(tmp_path/'latency.json',stats)
    assert load(tmp_path/'latency.json') == stats and stats['mean_ms'] == 100001.5
    previous = None
    for n in LENGTHS:
        with np.load(D/f'latency_inputs_{n}.npz',allow_pickle=False) as a:
            ids = (a['source_ids'].tolist(),a['start_indices'].tolist())
            assert a['windows'].shape == (64,5,n)
            if previous is not None:
                assert ids == previous
            previous = ids
    assert LATENCY['fold']==0 and LATENCY['seed']==1201 and LATENCY['repetitions']==3


def test_import_side_effect_free_no_training(monkeypatch,tmp_path):
    import evaluation.windows as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('training on import'))
    importlib.reload(module)
    assert not list(tmp_path.iterdir())


def test_result_seal_no_final_model_or_candidate_test_files():
    assert not list(D.rglob('*test_predictions*'))
    assert not list(D.rglob('final_model*'))
    if (D/'test_access_audit.json').exists():
        a = load(D/'test_access_audit.json')
        assert a['phase11_test_model_inference_passes']==0 and a['test_cache_reads']==0
        assert a['completed_new_runs']==108 and a['reused_controls']==36
