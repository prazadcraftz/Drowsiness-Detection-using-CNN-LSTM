"""Explicit three-class metrics: rows actual, columns predicted, zero division=0."""

import numpy as np

from . import CLASS_NAMES


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError("Labels/predictions must be nonempty aligned vectors")
    if not np.isin(labels, range(3)).all() or not np.isin(predictions, range(3)).all():
        raise ValueError("Unknown class ID")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall != 0)
    return {"class_order": list(CLASS_NAMES), "confusion_matrix": matrix.tolist(), "confusion_orientation": "rows actual, columns predicted",
            "accuracy": float(diagonal.sum()/support.sum()), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support)/support.sum()), "support": int(support.sum()),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(CLASS_NAMES)}}


def aggregate_videos(labels, probabilities, source_ids):
    labels, probabilities, source_ids = np.asarray(labels), np.asarray(probabilities), np.asarray(source_ids)
    if probabilities.shape != (len(labels), 3) or len(source_ids) != len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability/source shapes")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Expected normalized class probabilities")
    videos = []
    for source_id in sorted(set(source_ids.tolist())):
        select = source_ids == source_id
        classes = set(labels[select].tolist())
        if len(classes) != 1:
            raise ValueError("One source has conflicting labels")
        mean = probabilities[select].mean(axis=0)
        videos.append({"source_id": source_id, "class_id": int(next(iter(classes))), "predicted_class_id": int(mean.argmax()),
                       "mean_probabilities": mean.tolist(), "window_count": int(select.sum())})
    metrics = classification_metrics([video["class_id"] for video in videos], [video["predicted_class_id"] for video in videos])
    metrics["aggregation"] = "unweighted mean of window probabilities within source; argmax; each video one vote"
    metrics["videos"] = videos
    return metrics
