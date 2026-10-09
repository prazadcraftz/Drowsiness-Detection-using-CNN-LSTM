"""Compare newly extracted raw feature caches with the six frozen legacy clips."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.manifest import read_manifest, sha256, write_json


def main():
    manifest = read_manifest(ROOT / "results/phase11/dataset_manifest.csv")
    sources = {row["relative_path"]: row for row in manifest}
    baseline = json.loads((ROOT / "tests/fixtures/manifest.json").read_text())
    assert sha256(ROOT / "tests/fixtures/legacy_baseline.npz") == baseline["fixture_sha256"]
    results = []
    with np.load(ROOT / "tests/fixtures/legacy_baseline.npz", allow_pickle=False) as frozen:
        for source in baseline["sources"]:
            rows = np.arange(source["first_frame_row"], source["last_frame_row_exclusive"])
            valid = rows[frozen["face_detected"][rows]]
            row = sources[source["filename"]]
            with np.load(ROOT / "results/phase11/raw_features" / (row["source_id"] + ".npz"), allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["frame_indices"], frozen["frame_indices"][valid])
                np.testing.assert_array_equal(cache["features"], frozen["raw_features"][valid])
                results.append({"source": source["filename"], "valid_observations": len(valid), "raw_features_exact": True, "maximum_difference": 0.0})
    write_json(ROOT / "results/phase11/legacy_feature_parity.json", {"exact": True, "sources": results,
               "total_valid_observations": sum(row["valid_observations"] for row in results), "maximum_difference": 0.0})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
