"""Six-source production analytics with silent audio and exact frozen raw parity."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))

import numpy as np
import main
from driver_monitor.inference.decision import DecisionLayer


def characterize(output):
    output = Path(output).resolve()
    frozen = ROOT/"tests"/"fixtures"
    if output == frozen or frozen in output.parents:
        raise ValueError("Never write characterization into frozen fixtures")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty characterization directory")
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((frozen/"manifest.json").read_text())
    assert hashlib.sha256((frozen/"legacy_baseline.npz").read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(frozen/"legacy_baseline.npz",allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    previous = json.loads((ROOT/"docs"/"phase6_characterization"/"characterization.json").read_text())
    runs = []
    for source_id,source in enumerate(manifest["sources"]):
        filename = source["filename"]
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == source["sha256"]
        received = matched = 0
        class Observer(DecisionLayer):
            def update(self,raw,**kwargs):
                nonlocal received,matched
                row = source["first_frame_row"]+received
                received += 1
                obs = raw.observation
                assert obs.timestamp_seconds == arrays["timestamps_seconds"][row]
                assert obs.face_detected == bool(arrays["face_detected"][row])
                assert obs.sequence_fill == arrays["sequence_fill"][row]
                assert obs.sampled == obs.face_detected and not obs.gap_reset
                if obs.face_detected:
                    np.testing.assert_array_equal(obs.raw_features,arrays["raw_features"][row])
                    np.testing.assert_array_equal(obs.normalized_features,arrays["normalized_features"][row])
                window = int(arrays["frame_window_indices"][row])
                assert (raw.raw_class_id is not None) == (window>=0)
                if window>=0:
                    for key,actual in [("sequences",raw.sequence),("input_tensors",raw.input_tensor),("logits",raw.logits),("probabilities",raw.probabilities)]:
                        np.testing.assert_array_equal(actual,arrays[key][window])
                    assert raw.raw_class_id == arrays["predicted_class_ids"][window]
                    matched += 1
                decision = super().update(raw,**kwargs)
                assert decision.raw_result is raw
                return decision
        with patch.object(main,"_create_decision_layer",lambda config=None,*,enabled=True:Observer(config,enabled=enabled)):
            summary = main.run(str(ROOT/filename),no_display=True,no_audio=True)
        metrics = summary.session_metrics
        assert received == summary.frames_decoded == source["decoded_frame_count"]
        for old,new in [("raw_counts","raw_prediction_counts"),("smoothed_counts","smoothed_prediction_counts"),("state_frame_counts","state_frame_counts")]:
            assert previous["runs"][source_id][old] == metrics[new]
        for state,duration in previous["runs"][source_id]["state_seconds"].items():
            assert duration == metrics["state_occupancy_seconds"][state]
        assert metrics["audio"]["audible_trigger_count"] == 0
        run = {"source":filename,"application":asdict(summary),"raw_windows_matched":matched,"max_raw_numeric_difference":0.0}
        (output/f"source_{source_id}.json").write_text(json.dumps(run,indent=2)+"\n")
        runs.append(run)
        print(filename,{name:e["count"] for name,e in metrics["events"].items()},flush=True)
    assert sum(r["raw_windows_matched"] for r in runs) == 699
    report = {"status":"PASS","raw_windows_matched":699,"max_raw_numeric_difference":0.0,
              "phase6_state_and_prediction_metrics_unchanged":True,"audio_disabled":True,
              "policy_defaults":previous["defaults"],"audio_cooldown_seconds":5.0,
              "purpose":"Engineering characterization only; no event-level accuracy or medical validation", "runs":runs}
    (output/"characterization.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    characterize(parser.parse_args().output)
