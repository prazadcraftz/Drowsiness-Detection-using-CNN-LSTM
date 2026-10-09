"""The experiment gate must reject fast but incorrect or unstable candidates."""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from tools.optimize_phase9 import ROOT, assess, canonical, digest, experiment


def evidence():
    source = "Normal/Normal_1.avi"
    def rows(value):
        return [{"source": key, "stage": stage, "distribution": {"p50_ms": value, "p95_ms": value+2},
                 "per_run_median_ms": [value]*5} for key in ("ALL_SOURCES", str((ROOT/source).resolve()))
                for stage in ("inference_pipeline", "application_processing")]
    return {"candidate": "no_blendshapes", "parity": {"exact_parity": True},
            "methodology": {"sources": [source]}, "aggregate": {"baseline": rows(30), "no_blendshapes": rows(25)},
            "runs": [{"job": {"candidate": name, "kind": "pipeline"}, "throughput": {"processing_average_cpu_percent": 200}}
                     for name in ("baseline", "no_blendshapes") for _ in range(5)]}


def test_gate_requires_exact_parity_despite_speedup():
    result = evidence(); result["parity"]["exact_parity"] = False
    assert assess(result)["decision"] == "REJECTED"


def test_gate_reports_formula_and_repeated_wins():
    result = assess(evidence())
    assert result["decision"] == "ELIGIBLE"
    assert result["paired_median_wins"] == result["pairs"] == 5
    assert result["improvements_percent"]["inference_pipeline"]["p50_ms"] == pytest.approx(100/6)


@pytest.mark.parametrize("failure", ["p95", "single_run", "cpu", "trivial"])
def test_gate_rejects_insufficient_benefit(failure):
    result = evidence(); rows = result["aggregate"]["no_blendshapes"]
    if failure == "p95":
        for row in rows: row["distribution"]["p95_ms"] = 40
    elif failure == "single_run":
        for row in rows: row["per_run_median_ms"] = [20,31,31,31,31]
    elif failure == "trivial":
        for row in rows: row["distribution"]["p50_ms"] = 29.99
    else:
        for run in result["runs"]:
            if run["job"]["candidate"] != "baseline": run["throughput"]["processing_average_cpu_percent"] = 300
    assert assess(result)["decision"] == "INCONCLUSIVE"


@pytest.mark.parametrize("runs,sources", [(0,["Normal/Normal_1.avi"]), (-1,["Normal/Normal_1.avi"]), (True,[]), (1,[]), (1,["x","x"])])
def test_bad_experiment_rejected_before_creating_evidence(runs, sources):
    with pytest.raises(ValueError): experiment("no_blendshapes", runs, sources, "headless")


def test_missing_source_rejected_before_creating_evidence():
    with pytest.raises(FileNotFoundError): experiment("no_blendshapes", 1, ["missing-phase9.avi"], "headless")


def test_policy_digest_preserves_array_values_and_enum_semantics():
    class State(Enum): ACTIVE = "active"
    @dataclass
    class Policy:
        state: State
        values: np.ndarray
    value = Policy(State.ACTIVE, np.array([.1, .2]))
    assert canonical(value) == {"state": "active", "values": [.1, .2]}
    assert digest(value) == digest({"values": [.1, .2], "state": "active"})
    changed = deepcopy(value); changed.values[0] += 1e-10
    assert digest(value) != digest(changed)
