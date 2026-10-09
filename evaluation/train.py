"""One fixed train/validation recipe; the test arrays are never loaded here."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import CNN_LSTM
from . import FEATURE_NAMES
from .manifest import sha256, write_json
from .metrics import classification_metrics
from .reproducibility import RECIPE, environment, seed_everything, seed_worker

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = {"num_features": 5, "num_classes": 3, "cnn_dropout_rate": .25, "lstm_dropout_rate": .25}


def load_split(directory, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    directory = Path(directory)
    statistics = json.loads((directory / "dataset_statistics.json").read_text())
    filename = split + ".npz"
    if sha256(directory / filename) != statistics["artifact_sha256"][filename]:
        raise ValueError("Prepared split hash changed")
    with np.load(directory / filename, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if result["windows"].shape != (len(result["labels"]), 5, 16) or not np.isfinite(result["windows"]).all():
        raise ValueError("Invalid experimental windows")
    return result


def loader(data, *, shuffle, batch_size=64, seed=0):
    dataset = TensorDataset(torch.from_numpy(data["windows"].astype(np.float32)), torch.from_numpy(data["labels"].astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, worker_init_fn=seed_worker, generator=generator)


def predict(model, data, *, batch_size=64):
    model.eval()
    probabilities, total_loss, count = [], 0.0, 0
    with torch.inference_mode():
        for inputs, labels in loader(data, shuffle=False, batch_size=batch_size):
            logits = model(inputs)
            if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model logits")
            total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            count += len(labels)
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    if not count:
        raise ValueError("No windows for evaluation")
    probabilities = np.concatenate(probabilities)
    return probabilities, total_loss/count


def fit(model, train_data, validation_data, *, recipe=RECIPE, on_best=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"])
    train_loader = loader(train_data, shuffle=True, batch_size=recipe["batch_size"], seed=recipe["seed"])
    history, best_state, best_score, best_epoch, stale = [], None, -1.0, None, 0
    for epoch in range(1, recipe["max_epochs"]+1):
        model.train()
        total_loss = correct = count = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
        probabilities, validation_loss = predict(model, validation_data, batch_size=recipe["batch_size"])
        metrics = classification_metrics(validation_data["labels"], probabilities.argmax(1))
        entry = {"epoch": epoch, "train_loss": total_loss/count, "train_accuracy": correct/count,
                 "validation_loss": validation_loss, "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"], "learning_rate": recipe["learning_rate"]}
        history.append(entry)
        score = metrics["macro_f1"]
        if score > best_score + recipe["minimum_improvement"]:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = deepcopy(model.state_dict())
            if on_best:
                on_best(best_state, entry)
        else:
            stale += 1
        print(json.dumps(entry, sort_keys=True), flush=True)
        if stale >= recipe["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"epochs": history, "best_epoch": best_epoch, "best_validation_macro_f1": best_score, "stopping_epoch": history[-1]["epoch"],
            "stop_reason": "patience" if stale >= recipe["patience"] else "max_epochs", "selection_used_test": False}


def training_plots(directory, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True)
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for axis, names, ylabel in zip(axes, (("train_loss", "validation_loss"), ("train_accuracy", "validation_accuracy"), ("validation_macro_f1",)), ("Cross-entropy", "Accuracy", "Macro F1")):
        for name in names:
            axis.plot(epochs, [item[name] for item in history], label=name.replace("_", " "))
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(directory / "training_curves.png", dpi=150)
    plt.close(fig)


def train(artifacts, models):
    artifacts, models = Path(artifacts).resolve(), Path(models).resolve()
    if ROOT / "models" not in models.parents:
        raise ValueError("Experimental checkpoint must be in a new directory under models/")
    protocol = json.loads((artifacts / "protocol.json").read_text())
    if protocol["recipe"] != RECIPE:
        raise ValueError("Fixed predeclared recipe changed")
    for name, expected in protocol["source_code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Declared source code changed: {name}")
    if not json.loads((artifacts / "integrity.json").read_text())["passed"]:
        raise ValueError("Leakage integrity gate failed")
    norm = json.loads((artifacts / "train_norm.json").read_text())
    if norm["sha256"] != sha256(artifacts / "train_norm.npy") or norm["split_sha256"] != sha256(artifacts / "splits.json"):
        raise ValueError("Normalization/split changed")
    models.mkdir(parents=True, exist_ok=False)
    seed_everything(RECIPE["seed"])
    write_json(artifacts / "training_started.json", {"recipe": RECIPE, "environment": environment(), "test_opened": False})
    train_data, validation_data = load_split(artifacts, "train"), load_split(artifacts, "validation")
    model = CNN_LSTM(MODEL_CONFIG).to("cpu")  # Fresh weights; never loads production checkpoint.
    checkpoint = models / "cnn_lstm_clean_split.pt"
    def save_best(state, entry):
        temporary = models / "best_checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(checkpoint)  # Only this newly owned experiment checkpoint.
    started = time.perf_counter()
    history = fit(model, train_data, validation_data, on_best=save_best)
    history.update(recipe=RECIPE, training_wall_seconds=time.perf_counter()-started)
    write_json(artifacts / "training_history.json", history)
    import csv
    with (artifacts / "training_history.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history["epochs"][0]))
        writer.writeheader()
        writer.writerows(history["epochs"])
    training_plots(artifacts / "plots", history["epochs"])
    final_metrics = {}
    for split, data in (("train", train_data), ("validation", validation_data)):
        probabilities, loss = predict(model, data)
        final_metrics[split] = dict(classification_metrics(data["labels"], probabilities.argmax(1)), loss=loss)
    write_json(artifacts / "selected_train_validation_metrics.json", final_metrics)
    metadata = {"architecture": "CNN_LSTM", "model_config": MODEL_CONFIG, "parameter_count": sum(p.numel() for p in model.parameters()),
                "feature_order": list(FEATURE_NAMES), "sequence_length": 16, "train_normalization_sha256": norm["sha256"],
                "split_manifest_sha256": sha256(artifacts / "splits.json"), "training_seed": RECIPE["seed"],
                "best_epoch": history["best_epoch"], "best_validation_macro_f1": history["best_validation_macro_f1"], "recipe": RECIPE,
                "checkpoint_sha256": sha256(checkpoint), "environment": environment(), "test_evaluated": False,
                "training_history_sha256": sha256(artifacts / "training_history.json")}
    write_json(models / "cnn_lstm_clean_split.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(ROOT / "results/phase11"))
    parser.add_argument("--models", default=str(ROOT / "models/phase11"))
    args = parser.parse_args()
    train(args.artifacts, args.models)


if __name__ == "__main__":
    main()
