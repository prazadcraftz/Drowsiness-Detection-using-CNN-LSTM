"""Small fixtures enforce eval mode and validation-based selection; no full training."""

from copy import deepcopy

import numpy as np
import torch

from evaluation.reproducibility import RECIPE
from evaluation.train import fit, predict


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.9)
        self.linear = torch.nn.Linear(80, 3)
        self.calls = []

    def forward(self, inputs):
        self.calls.append((self.training, torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return self.linear(self.dropout(inputs.reshape(len(inputs), -1)))


def data():
    return {"windows": np.ones((3, 5, 16)), "labels": np.arange(3), "source_ids": np.array(["a", "b", "c"])}


def test_validation_eval_inference_mode_dropout_off_and_repeatable():
    model = Tiny()
    model.train()
    first, _ = predict(model, data(), batch_size=2)
    second, _ = predict(model, data(), batch_size=2)
    np.testing.assert_array_equal(first, second)
    assert all(call == (False, True, False) for call in model.calls)


def test_best_validation_epoch_restored_and_early_stop_without_test(monkeypatch):
    import evaluation.train as training
    scores = iter((np.eye(3), np.eye(3)[[1, 2, 0]], np.eye(3)[[1, 2, 0]]))
    monkeypatch.setattr(training, "predict", lambda model, validation, **kwargs: (next(scores), 1.0))
    saved = []
    model = Tiny()
    recipe = dict(RECIPE, max_epochs=6, patience=2, batch_size=3)
    history = fit(model, data(), data(), recipe=recipe, on_best=lambda state, entry: saved.append(deepcopy(state)))
    assert history["best_epoch"] == 1 and history["stopping_epoch"] == 3 and history["stop_reason"] == "patience"
    assert history["selection_used_test"] is False and len(saved) == 1
    assert not model.training
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, saved[0][name], rtol=0, atol=0)
    assert any(training_mode and not inference and grad for training_mode, inference, grad in model.calls)


def test_final_test_guard_prevents_second_model_inference(tmp_path, monkeypatch):
    import json
    import pytest
    from model import CNN_LSTM
    import evaluation.evaluate as final
    from evaluation.manifest import sha256, write_json
    from evaluation.train import MODEL_CONFIG
    artifacts, models = tmp_path / "artifacts", tmp_path / "models"
    artifacts.mkdir(); models.mkdir()
    (artifacts / "splits.json").write_text("{}")
    np.save(artifacts / "train_norm.npy", np.tile([0., 2.], (5, 1)))
    fixture = data()
    np.savez_compressed(artifacts / "test.npz", **fixture)
    write_json(artifacts / "dataset_statistics.json", {"artifact_sha256": {"test.npz": sha256(artifacts / "test.npz")}})
    checkpoint = models / "cnn_lstm_clean_split.pt"
    torch.save(CNN_LSTM(MODEL_CONFIG).state_dict(), checkpoint)
    write_json(models / "cnn_lstm_clean_split.json", {"checkpoint_sha256": sha256(checkpoint), "test_evaluated": False,
        "split_manifest_sha256": sha256(artifacts / "splits.json"), "train_normalization_sha256": sha256(artifacts / "train_norm.npy"),
        "best_epoch": 1, "training_seed": 123, "model_config": MODEL_CONFIG})
    calls = []
    monkeypatch.setattr(final, "seed_everything", lambda seed: None)
    monkeypatch.setattr(final, "predict", lambda model, values: (calls.append(1) or np.eye(3), 0.0))
    final.evaluate(artifacts, models)
    with pytest.raises(ValueError, match="already evaluated"):
        final.evaluate(artifacts, models)
    assert calls == [1]
    assert json.loads((models / "cnn_lstm_clean_split.json").read_text())["test_evaluated"]
