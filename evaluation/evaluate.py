"""One-time final test inference after validation selection; never tune on test."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import CNN_LSTM
from . import CLASS_NAMES
from .manifest import sha256, write_json
from .metrics import aggregate_videos, classification_metrics
from .reproducibility import seed_everything
from .train import ROOT, load_split, predict


def save_reports(directory, window_metrics, video_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    for name, metrics in (("confusion_matrix", window_metrics), ("video_confusion_matrix", video_metrics)):
        with (directory / (name + ".csv")).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual/predicted", *CLASS_NAMES])
            writer.writerows([label, *row] for label, row in zip(CLASS_NAMES, metrics["confusion_matrix"]))
        fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        matrix = np.asarray(metrics["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
        axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                 xlabel="Predicted class", ylabel="Actual class", title="Video holdout" if name.startswith("video") else "Window holdout")
        fig.colorbar(image, ax=axis)
        fig.savefig(plots / (name + ".png"), dpi=150)
        plt.close(fig)
    with (directory / "classification_report.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("level", "class_name", "precision", "recall", "f1", "support"))
        writer.writeheader()
        for level, metrics in (("window", window_metrics), ("video", video_metrics)):
            for name, values in metrics["per_class"].items():
                writer.writerow(dict(level=level, class_name=name, **values))


def evaluate(artifacts, models):
    artifacts, models = Path(artifacts).resolve(), Path(models).resolve()
    metadata_path = models / "cnn_lstm_clean_split.json"
    metadata = json.loads(metadata_path.read_text())
    checkpoint = models / "cnn_lstm_clean_split.pt"
    if metadata["checkpoint_sha256"] != sha256(checkpoint) or metadata["test_evaluated"]:
        raise ValueError("Changed checkpoint or final test already evaluated")
    if metadata["split_manifest_sha256"] != sha256(artifacts / "splits.json") or metadata["train_normalization_sha256"] != sha256(artifacts / "train_norm.npy"):
        raise ValueError("Split/normalization changed")
    # Exclusive marker before test access prevents accidental repeated canonical evaluation.
    write_json(artifacts / "test_evaluation_started.json", {"checkpoint_sha256": metadata["checkpoint_sha256"], "selected_best_epoch": metadata["best_epoch"], "one_time_final_test": True})
    seed_everything(metadata["training_seed"])
    model = CNN_LSTM(metadata["model_config"]).to("cpu")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    data = load_split(artifacts, "test")
    probabilities, loss = predict(model, data)
    # One common inference pass supplies both evaluation levels and offline reports.
    np.savez_compressed(artifacts / "test_predictions.npz", probabilities=probabilities, labels=data["labels"], source_ids=data["source_ids"])
    window = dict(classification_metrics(data["labels"], probabilities.argmax(1)), loss=loss, final_test_evaluation=True,
                  checkpoint_sha256=metadata["checkpoint_sha256"])
    video = dict(aggregate_videos(data["labels"], probabilities, data["source_ids"]), final_test_evaluation=True,
                 checkpoint_sha256=metadata["checkpoint_sha256"])
    write_json(artifacts / "test_metrics.json", window)
    write_json(artifacts / "video_test_metrics.json", video)
    save_reports(artifacts, window, video)
    metadata.update(test_evaluated=True, test_metrics=window, video_test_metrics={key: value for key, value in video.items() if key != "videos"})
    temporary = models / "final_metadata.tmp"
    write_json(temporary, metadata)
    temporary.replace(metadata_path)
    print(json.dumps({"window": window, "video": video}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(ROOT / "results/phase11"))
    parser.add_argument("--models", default=str(ROOT / "models/phase11"))
    args = parser.parse_args()
    evaluate(args.artifacts, args.models)


if __name__ == "__main__":
    main()
