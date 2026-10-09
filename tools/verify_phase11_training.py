"""Repeat the locked train/validation recipe; never open test arrays/inference."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import CNN_LSTM
from evaluation.manifest import sha256, write_json
from evaluation.reproducibility import RECIPE, seed_everything
from evaluation.train import MODEL_CONFIG, fit, load_split


def main():
    artifacts = ROOT / "results/phase11"
    protocol = json.loads((artifacts / "protocol.json").read_text())
    assert protocol["recipe"] == RECIPE
    for name, expected in protocol["source_code_sha256"].items():
        assert sha256(ROOT / name) == expected
    seed_everything(RECIPE["seed"])
    train, validation = load_split(artifacts, "train"), load_split(artifacts, "validation")
    model = CNN_LSTM(MODEL_CONFIG).to("cpu")
    repeated = fit(model, train, validation)
    canonical = json.loads((artifacts / "training_history.json").read_text())
    checkpoint = ROOT / "models/phase11/cnn_lstm_clean_split.pt"
    reference = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_equal = all(torch.equal(tensor, reference[name]) for name, tensor in model.state_dict().items())
    history_equal = repeated["epochs"] == canonical["epochs"]
    selection_equal = all(repeated[name] == canonical[name] for name in ("best_epoch", "best_validation_macro_f1", "stopping_epoch", "stop_reason"))
    write_json(artifacts / "reproducibility_check.json", {"same_recipe": True, "loaded_splits": ["train", "validation"], "test_inference_passes": 0,
        "same_cpu_software_only": True, "selected_weights_bit_exact": state_equal, "epoch_metrics_exact": history_equal,
        "selection_exact": selection_equal, "canonical_checkpoint_sha256": sha256(checkpoint), "repeat_history": repeated})
    assert state_equal and history_equal and selection_equal
    print("Fixed train/validation repeat: weights, epoch metrics and selection exact; no test access.")


if __name__ == "__main__":
    main()
