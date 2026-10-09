"""Enforce sealed sources, train-only treatment, and the unchanged control."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from evaluation.balancing import (class_weights, create_folds, fit, guard_sources,
                                  training_loader, validate_data)
from evaluation.metrics import classification_metrics
from evaluation.preprocess import fit_train_normalization
from evaluation.reproducibility import RECIPE, seed_everything
from evaluation.train import fit as baseline_fit


def inventory():
    rows = [{'source_id': f'{c}_{i}', 'class_id': c, 'group_id': f'g{c}_{i}'} for c in range(3) for i in range(7)]
    splits = {'source_ids': {'train': [r['source_id'] for r in rows if int(r['source_id'][-1]) < 4],
                            'validation': [r['source_id'] for r in rows if 4 <= int(r['source_id'][-1]) < 6],
                            'test': [r['source_id'] for r in rows if r['source_id'].endswith('6')]}}
    return rows, splits


def test_sealed_source_guard():
    with pytest.raises(ValueError, match='Sealed'):
        guard_sources(['safe', 'test'], ['safe', 'test'], ['test'])
    with pytest.raises(ValueError):
        guard_sources(['unknown'], ['safe'], ['test'])


def test_folds_reproduce_disjoint_stratified_exact_coverage():
    rows, splits = inventory()
    folds = create_folds(rows, splits)
    assert folds == create_folds(list(reversed(rows)), splits)
    seen = []
    for fold in folds['folds']:
        train, val = map(set, (fold['train_source_ids'], fold['validation_source_ids']))
        assert not train & val and not (train | val) & set(splits['source_ids']['test'])
        assert fold['class_counts']['validation'] == {'0': 2, '1': 2, '2': 2}
        seen.extend(val)
    assert sorted(seen) == folds['development_source_ids']


def test_duplicate_content_groups_stay_together():
    rows, splits = inventory()
    rows[1]['group_id'] = rows[0]['group_id']
    folds = create_folds(rows, splits)
    assert folds['assignments']['0_0'] == folds['assignments']['0_1']


def test_sealed_duplicate_group_rejected():
    rows, splits = inventory()
    rows[0]['group_id'] = rows[6]['group_id']
    with pytest.raises(ValueError, match='sealed'):
        create_folds(rows, splits)


def test_fold_normalizer_does_not_use_heldout_extremes():
    rows = [{'source_id': 'train'}, {'source_id': 'val'}]
    features = {'train': np.ones((2, 5))*2, 'val': np.ones((2, 5))*999}
    norm, sources = fit_train_normalization(features, rows, {'train': 'train', 'val': 'validation'})
    np.testing.assert_array_equal(norm, np.tile([0., 2.], (5, 1)))
    assert sources == ['train']


def test_weights_from_training_only_formula():
    labels = np.array([0]*2 + [1]*3 + [2]*5)
    np.testing.assert_allclose(class_weights(labels), 10/(3*np.array([2, 3, 5])))
    # Function has no validation-label argument or lookup.
    with pytest.raises(TypeError):
        class_weights(labels, np.array([0]*100))


@pytest.mark.parametrize('labels', [[], [0, 0], [0, 1, 3]])
def test_weights_require_all_known_classes(labels):
    with pytest.raises(ValueError):
        class_weights(labels)


def test_sampler_equalizes_training_distribution_and_is_seeded():
    labels = np.array([0]*100 + [1]*300 + [2]*600)
    data = {'windows': np.ones((1000, 5, 16)), 'labels': labels}
    first = training_loader(data, 'weighted_sampler', 1201)
    second = training_loader(data, 'weighted_sampler', 1201)
    a, b = list(first.sampler), list(second.sampler)
    assert a == b and len(a) == 1000 and first.sampler.replacement
    counts = np.bincount(labels[a], minlength=3)
    assert (abs(counts/1000 - 1/3) < .055).all()
    assert first.sampler.weights[0]/first.sampler.weights[-1] == 6


def data(source):
    return {'windows': np.arange(240).reshape(3, 5, 16)/240,
            'labels': np.arange(3), 'source_ids': np.array([source]*3)}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(80, 3)
    def forward(self, x):
        return self.layer(x.reshape(len(x), -1))


def test_unweighted_control_exactly_matches_phase11_recipe():
    recipe = dict(RECIPE, seed=1201, max_epochs=3, patience=2, batch_size=3)
    seed_everything(1201)
    baseline = Tiny()
    candidate = deepcopy(baseline)
    seed_everything(1201)
    original = baseline_fit(baseline, data('train'), data('val'), recipe=recipe)
    seed_everything(1201)
    result = fit(candidate, data('train'), data('val'), 'unweighted', recipe,
                 {'train_source_ids': ['train'], 'validation_source_ids': ['val']}, ['sealed'])
    assert result['epochs'] == original['epochs'] and result['best_epoch'] == original['best_epoch']
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])


def test_validation_only_selection_and_earliest_tie(monkeypatch):
    import evaluation.balancing as experiment
    predictions = iter((np.eye(3), np.eye(3), np.eye(3)[[1, 2, 0]]))
    calls = []
    def predict(model, values, **kwargs):
        calls.append(set(values['source_ids']))
        return next(predictions), 1.
    monkeypatch.setattr(experiment, 'predict', predict)
    history = fit(Tiny(), data('train'), data('val'), 'weighted_ce',
                  dict(RECIPE, seed=1201, max_epochs=6, patience=2, batch_size=3),
                  {'train_source_ids': ['train'], 'validation_source_ids': ['val']}, ['sealed'])
    assert calls == [{'val'}]*3
    assert history['best_epoch'] == 1 and history['stopping_epoch'] == 3
    assert not history['selection_used_test']


def test_candidate_rejects_sealed_before_model_call():
    class Never(Tiny):
        def forward(self, x):
            raise AssertionError('Model inference must never happen')
    with pytest.raises(ValueError, match='Sealed'):
        fit(Never(), data('train'), data('sealed'), 'unweighted', RECIPE,
            {'train_source_ids': ['train'], 'validation_source_ids': ['sealed']}, ['sealed'])


def test_process_open_guard_blocks_real_test_files(tmp_path):
    phase11 = tmp_path/'phase11'
    phase11.mkdir()
    (phase11/'test.npz').write_bytes(b'sealed')
    script = "from pathlib import Path; from evaluation.balancing import install_seal; p=Path(" + repr(str(phase11)) + "); install_seal(p, [], []); p.joinpath('test.npz').read_bytes()"
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode != 0 and 'Phase 11 test is sealed' in result.stderr


def test_metrics_class_order_and_worst_recall():
    metrics = classification_metrics([0, 1, 2], [0, 0, 2])
    assert metrics['class_order'] == ['Normal', 'Yawning', 'Eye Close']
    assert min(c['recall'] for c in metrics['per_class'].values()) == 0


def test_fold_data_coverage_guard():
    with pytest.raises(ValueError):
        validate_data(data('train'), ['train', 'missing'], ['sealed'])
