"""Record and apply the fixed experimental CPU training recipe."""

import importlib.metadata
import os
import platform
import random

import numpy as np

from . import SEED

RECIPE = {"seed": SEED, "max_epochs": 60, "patience": 10, "learning_rate": .001, "batch_size": 64,
          "optimizer": "Adam", "loss": "CrossEntropyLoss", "scheduler": None, "class_weighting": False,
          "weighted_sampling": False, "selection_metric": "validation_window_macro_f1", "minimum_improvement": 1e-12,
          "tie_policy": "earliest epoch", "num_workers": 0, "torch_threads": 8, "torch_interop_threads": 8,
          "deterministic_algorithms": True, "normalization": "train-only legacy zero-initialized extrema",
          "window_length": 16, "window_stride": 8, "padding": "repeat last valid observation; range(0,n,8)",
          "feature_extraction": "unchanged legacy every-decoded-frame valid-face observations, without runtime sampling/reset",
          "test_policy": "one final inference pass after validation checkpoint selection; no test tuning"}


def seed_everything(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    # Calling twice in one process isn't supported by torch; apply only if needed.
    if torch.get_num_interop_threads() != 8:
        torch.set_num_interop_threads(8)


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def environment():
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
            "software": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "opencv-contrib-python", "mediapipe")},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "logical_cpu_count": os.cpu_count(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "device": "cpu", "bit_identical_training_verified": False}
