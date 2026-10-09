"""Shared read-only Phase 2 data for the new engine tests."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def pytest_addoption(parser):
    parser.addoption("--phase3-replay-output", default=None,
                     help="Preserve the six-source Phase 3 comparison in a new empty directory")


@pytest.fixture(scope="session")
def frozen_phase2():
    directory = ROOT / "tests" / "fixtures"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert hashlib.sha256((directory / "legacy_baseline.npz").read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(directory / "legacy_baseline.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return manifest, arrays
