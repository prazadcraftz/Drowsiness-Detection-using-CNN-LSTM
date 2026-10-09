"""Registry structure, controlled inputs, protocol seal and result semantics."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from model import CNN_LSTM
from evaluation.models import NAMES, build_model, architecture_spec
from evaluation.architectures import latency_statistics, paired_summary, pareto_front, verify_protocol
from evaluation.balancing import guard_sources
from evaluation.manifest import write_json
from evaluation.metrics import classification_metrics, aggregate_videos
from evaluation.reproducibility import seed_everything
from evaluation.train import MODEL_CONFIG

COUNTS = {'cnn_only': 11075, 'lstm_only': 209667, 'cnn_lstm': 246595, 'cnn_gru': 188739, 'cnn_bilstm': 181059}


def test_exact_five_registry_names():
    assert set(NAMES) == set(COUNTS)
    with pytest.raises(ValueError):
        build_model('extra_variant')


@pytest.mark.parametrize('name', NAMES)
def test_finite_output_deterministic_eval_and_stable_parameter_count(name):
    seed_everything(1201)
    model = build_model(name).eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == COUNTS[name]
    inputs = torch.arange(160, dtype=torch.float32).reshape(2,5,16)/160
    with torch.inference_mode():
        first, second = model(inputs), model(inputs)
    assert first.shape == (2,3) and torch.isfinite(first).all()
    assert torch.equal(first,second)
    seed_everything(1201)
    repeated = build_model(name)
    assert all(torch.equal(v,repeated.state_dict()[k]) for k,v in model.state_dict().items())


def test_reference_identical_structure_keys_shapes_weights_outputs():
    seed_everything(1201)
    original = CNN_LSTM(MODEL_CONFIG).eval()
    seed_everything(1201)
    reference = build_model('cnn_lstm').eval()
    assert [(k,type(m),str(m)) for k,m in original.named_modules()] == [(k,type(m),str(m)) for k,m in reference.named_modules()]
    assert original.state_dict().keys() == reference.state_dict().keys()
    assert sum(p.numel() for p in reference.parameters()) == 246595
    for k,v in original.state_dict().items():
        assert v.shape == reference.state_dict()[k].shape and torch.equal(v,reference.state_dict()[k])
    inputs = torch.arange(80,dtype=torch.float32).reshape(1,5,16)/80
    with torch.inference_mode():
        assert torch.equal(original(inputs),reference(inputs))


def test_lstm_only_temporal_orientation_unchanged_values():
    seed_everything(1201)
    model = build_model('lstm_only').eval()
    seen = []
    hook = model.lstm.register_forward_pre_hook(lambda module,args: seen.append(args[0].clone()))
    values = torch.arange(80,dtype=torch.float32).reshape(1,5,16)
    model(values); hook.remove()
    assert seen[0].shape == (1,16,5)
    assert torch.equal(seen[0],values.permute(0,2,1))


def test_bilstm_full_direction_summaries_and_128_readout():
    seed_everything(1201)
    model = build_model('cnn_bilstm').eval()
    observed, dense = [], []
    hook = model.lstm.register_forward_hook(lambda module,args,output: observed.append(output[1][0].clone()))
    head = model.fc1.register_forward_pre_hook(lambda module,args: dense.append(args[0].clone()))
    model(torch.ones(2,5,16)); hook.remove(); head.remove()
    assert model.lstm.bidirectional and model.lstm.hidden_size == 64 and model.lstm.num_layers == 2
    assert dense[0].shape == (2,128)
    assert torch.equal(dense[0],torch.cat((observed[0][-2],observed[0][-1]),dim=1))


def test_cnn_frontends_preserve_layer_shapes():
    seed_everything(1201)
    reference = build_model('cnn_lstm')
    for name in ('cnn_only','cnn_gru','cnn_bilstm'):
        alternative = build_model(name)
        for key in ('conv1','conv2','maxpool','dropout_cnn'):
            assert str(getattr(alternative,key)) == str(getattr(reference,key))


def test_no_internal_recurrent_dropout_and_fixed_external_dropout():
    for name in NAMES:
        model = build_model(name)
        assert model.dropout_lstm.p == .25
        for module in model.modules():
            if isinstance(module,(torch.nn.LSTM,torch.nn.GRU)):
                assert module.dropout == 0


def test_spec_shapes_counts_and_rng_preservation():
    seed_everything(1201)
    before = torch.get_rng_state().clone()
    spec = architecture_spec()
    assert torch.equal(before,torch.get_rng_state())
    assert {k:v['trainable_parameters'] for k,v in spec.items()} == COUNTS
    assert all(v['output_shape']==['B',3] for v in spec.values())


def test_phase12_folds_and_normalizer_provenance_reused_exactly():
    folds, config, data, hashes = verify_protocol(enforce_seal=False)
    canonical = json.loads(Path('results/phase12/cv_folds.json').read_text())
    assert folds == canonical and config['training_seeds'] == [1201,1202,1203]
    assert len(folds['development_source_ids']) == 153
    assert len(data) == 3
    assert 'results/phase12/fold_0/train_norm.npy' in hashes


def test_changed_normalizer_provenance_rejected(monkeypatch):
    import evaluation.architectures as experiment
    original = experiment.load
    def altered(path):
        value = original(path)
        if Path(path).name == 'train_norm.json':
            value['train_source_ids'] = value['train_source_ids']+['sealed']
        return value
    monkeypatch.setattr(experiment,'load',altered)
    with pytest.raises(ValueError,match='normalizer provenance'):
        verify_protocol(enforce_seal=False)


def test_sealed_test_sources_rejected():
    with pytest.raises(ValueError,match='Sealed'):
        guard_sources(['sealed'],['sealed'],['sealed'])


def test_metric_video_aggregation_and_order():
    video = aggregate_videos(np.array([0,0,1,2]),np.eye(3)[[0,1,1,2]],np.array(['a','a','b','c']))
    assert video['support'] == 3 and video['accuracy'] == 1
    assert video['class_order'] == ['Normal','Yawning','Eye Close']


def test_paired_comparison_exact_pair_alignment():
    reference = {(0,1201):.8,(1,1201):.9,(2,1201):.95}
    candidate = {(2,1201):.95,(0,1201):.9,(1,1201):.8}
    result = paired_summary(candidate,reference)
    assert abs(result['mean'])<1e-12 and result['positive_pairs']==1 and result['negative_pairs']==1 and result['zero_pairs']==1
    with pytest.raises(ValueError):
        paired_summary({(0,1201):1},reference)


def test_latency_serialization_and_quantiles(tmp_path):
    result = latency_statistics([1,2,3,4])
    assert result['p50_ms']==2.5 and result['sample_count']==4
    write_json(tmp_path/'latency.json',result)
    assert json.loads((tmp_path/'latency.json').read_text())==result
    with pytest.raises(ValueError):
        latency_statistics([float('nan')])


def test_pareto_quality_latency_without_composite_score():
    assert pareto_front({'a':(.9,2),'b':(.9,3),'c':(.8,1),'d':(.7,4)}) == ['a','c']


def test_imports_have_no_training_side_effects():
    script = "import evaluation.architectures as a; import evaluation.models; assert not hasattr(a,'model'); print('definitions only')"
    result = subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0 and result.stdout.strip()=='definitions only'
