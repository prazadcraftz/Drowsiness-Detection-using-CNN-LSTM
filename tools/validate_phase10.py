"""Finite Phase 10 evidence workers, sequential fresh processes; no old rewrites."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from driver_monitor.metrics import benchmark as bench


def compare_database(path, session_id, aggregate, offline_fps=None):
    from driver_monitor.storage.session_persistence import EVENT_NAMES, metric_columns
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute("SELECT * FROM session_metrics WHERE session_id=?", (session_id,)).fetchone())
        assert json.loads(row["aggregate_json"]) == aggregate
        for key, value in metric_columns(aggregate, offline_processing_fps=offline_fps).items():
            assert row[key] == value, (key, row[key], value)
        events = [dict(r) for r in connection.execute("SELECT * FROM events WHERE session_id=? ORDER BY id", (session_id,))]
    for name, event_type in EVENT_NAMES.items():
        selected = [r for r in events if r["event_type"] == event_type]
        assert len(selected) == aggregate["events"][name]["count"]
        assert abs(sum(r["observed_duration_seconds"] for r in selected) - aggregate["events"][name]["total_duration_seconds"]) <= 1e-12
    return {"session_id": session_id, "normalized_metrics_exact": True, "event_counts_exact": True, "event_duration_sums_within_1e_12": True, "events": events}


def parity(enabled, directory):
    # Reuse frozen verification and canonical trace hashes, not the Phase 9
    # optimization dispatcher (which could overwrite its own evidence).
    from tools.optimize_phase9 import digest
    from driver_monitor.storage import SQLiteStore, SessionPersistence, session_provenance
    runtime = bench.runtime_imports()
    np = runtime["np"]
    manifest = json.loads((ROOT / "tests/fixtures/manifest.json").read_text())
    reference = json.loads((ROOT / "results/phase9_parity/production_cache_only/result.json").read_text())
    database = directory / "parity.db"
    store = SQLiteStore(database) if enabled else None
    results = []
    try:
        for source, frozen_trace in zip(manifest["sources"], reference["sources"]):
            filename = source["filename"]
            verifier = bench.FrozenVerifier(np, ROOT / filename)
            landmarks, policies, pixels = [], [], []
            differences = {key: 0.0 for key in ("raw_features", "normalized_features", "sequences", "input_tensors", "logits", "probabilities")}
            frames = classes = 0
            with runtime["DrowsinessInferenceEngine"]() as engine, runtime["VideoSource"](ROOT / filename) as video:
                layer = runtime["DecisionLayer"](engine.config)
                metrics = runtime["SessionMetrics"]()
                persistence = None
                if enabled:
                    session_id = store.start_session(source_type="file", source_name=Path(filename).name,
                        provenance=session_provenance(engine, timed=True, audio_enabled=False, alert_cooldown_seconds=5))
                    persistence = SessionPersistence(store, session_id)
                detect = engine.detector.detect
                def capture(frame):
                    value = detect(frame)
                    landmarks.append(np.full((478, 2), np.nan) if value is None else value.copy())
                    return value
                engine.detector.detect = capture
                while True:
                    packet = video.read()
                    if packet.eof:
                        break
                    raw = engine.process_frame(packet.frame, timestamp_seconds=packet.timestamp_seconds)
                    verifier.check(frames, raw)
                    row = source["first_frame_row"] + frames
                    win = int(verifier.arrays["frame_window_indices"][row])
                    for key, actual in (("raw_features", raw.observation.raw_features), ("normalized_features", raw.observation.normalized_features),
                                        ("sequences", raw.sequence), ("input_tensors", raw.input_tensor), ("logits", raw.logits), ("probabilities", raw.probabilities)):
                        index = row if key.endswith("features") else win
                        if actual is not None and index >= 0:
                            differences[key] = max(differences[key], float(np.max(np.abs(actual - verifier.arrays[key][index]))))
                    if win >= 0 and raw.raw_class_id == int(verifier.arrays["predicted_class_ids"][win]):
                        classes += 1
                    policy = layer.update(raw)
                    metrics.update(policy)
                    if persistence:
                        persistence.update(metrics, policy)
                    policies.append(digest(policy))
                    display = runtime["main"].render_result(packet.frame, packet, raw, video.source_name,
                        source_fps=video.nominal_fps, target_fps=engine.config.target_model_fps, policy=policy.driver, metrics=metrics)
                    pixels.append(bench.hashlib.sha256(display.tobytes()).hexdigest())
                    frames += 1
                metrics.finalize()
                aggregate = metrics.snapshot()
                persisted = None
                if persistence:
                    persistence.finalize(metrics)
                    persisted = compare_database(database, session_id, aggregate)
            array = np.array(landmarks)
            archive_name = filename.replace("/", "_").replace(" ", "_") + ".npz"
            np.savez_compressed(directory / archive_name, landmarks=array)
            with np.load(ROOT / "results/phase9_parity/production_cache_only" / archive_name) as archive:
                assert np.array_equal(array, archive["landmarks"], equal_nan=True)
            trace = {"policy_trace_sha256": digest(policies), "session_sha256": digest(aggregate), "pixel_trace_sha256": digest(pixels)}
            assert all(trace[key] == frozen_trace[key] for key in trace)
            assert all(value == 0 for value in differences.values())
            results.append({"source": filename, "source_sha256": bench.sha256(ROOT / filename), "frames": frames,
                            "windows_exact": verifier.matched, "classes_exact": classes, "maximum_differences": differences,
                            "landmarks_exact": True, "policy_session_pixels_exact": True, **trace, "persisted": persisted})
    finally:
        if store:
            store.close()
    assert sum(r["windows_exact"] for r in results) == 699
    return {"sqlite_enabled": enabled, "windows_exact": 699, "classes_exact": sum(r["classes_exact"] for r in results), "exact_parity": True,
            "sources": results, "threads": bench.thread_settings(runtime)}


def operations(directory):
    from driver_monitor.storage import SQLiteStore
    from driver_monitor.storage.session_persistence import metric_columns
    from driver_monitor.metrics import SessionMetrics
    samples = {name: [] for name in ("session_create", "event_insert", "event_update", "final_metrics_write")}
    path = directory / "operations.db"
    def measured(name, function):
        tick = time.perf_counter_ns()
        result = function()
        samples[name].append(time.perf_counter_ns() - tick)
        return result
    with SQLiteStore(path) as store:
        for _ in range(100):
            session_id = measured("session_create", lambda: store.start_session(source_type="file", source_name="operation-probe"))
            event_id = measured("event_insert", lambda: store.start_event(session_id, "YAWN", 0))
            measured("event_update", lambda: store.end_event(session_id, event_id, ended_at_source_seconds=1, observed_duration_seconds=1))
            # Scalar probe intentionally isolates store latency, not ML semantics.
            metrics = SessionMetrics()
            aggregate = metrics.finalize()
            measured("final_metrics_write", lambda: store.finalize_session(session_id, metrics=metric_columns(aggregate), aggregate=aggregate))
    return {"iterations": 100, "scope": "Synchronous SQLite operation including transaction commit; final metrics includes JSON serialization, excludes asset hashing/schema startup.",
            "samples_ns": samples, "distribution_ms": {name: bench.distribution_ms(ns / 1e6 for ns in values) for name, values in samples.items()},
            "database_bytes": path.stat().st_size, "sqlite_version": sqlite3.sqlite_version}


def application(job):
    import main
    summary = main.run(job["source"], no_display=True, no_audio=True, sqlite=job["sqlite"], db_path=job.get("db_path"))
    value = {"job": job, "summary": asdict(summary)}
    if job["sqlite"]:
        with sqlite3.connect(job["db_path"]) as connection:
            session_id = connection.execute("SELECT MAX(id) FROM sessions").fetchone()[0]
        value["persisted"] = compare_database(job["db_path"], session_id, summary.session_metrics, summary.processing_throughput_fps)
    return value


def suite(directory):
    directory.mkdir(parents=True, exist_ok=False)
    def run(job, label):
        output = directory / (label + ".json")
        bench.write_new_json(directory / (label + ".input.json"), job)
        with (directory / (label + ".stderr.txt")).open("x", encoding="utf-8") as stderr:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(directory / (label + ".input.json")), "--output", str(output)],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=stderr, check=True)
        print(label + " complete", flush=True)
        return json.loads(output.read_text())
    results = {}
    for enabled in (False, True):
        label = "parity_on" if enabled else "parity_off"
        subdirectory = directory / label
        subdirectory.mkdir()
        results[label] = run({"kind": "parity", "sqlite": enabled, "directory": str(subdirectory)}, label)
    videos = ("Normal/Normal_1.avi", "Yawning/Yawning_1.avi", "Yawning/Yawning_60.avi", "Eye Close/Eye Close_60.avi", "Eye Close/Eye Close_1.avi")
    results["videos"] = [run({"kind": "application", "source": source, "sqlite": True, "db_path": str(directory / "videos.db")}, f"video_{index}") for index, source in enumerate(videos)]
    # A finite, balanced order; deliberately no statistical speed claim.
    results["representative"] = [run({"kind": "application", "source": "Eye Close/Eye Close_60.avi", "sqlite": enabled,
        **({"db_path": str(directory / "optional_overhead.db")} if enabled else {})}, f"overhead_{index}_{'on' if enabled else 'off'}") for index, enabled in enumerate((False, True, True, False))]
    off = [r["summary"]["session_metrics"] for r in results["representative"] if not r["job"]["sqlite"]]
    on = [r["summary"]["session_metrics"] for r in results["representative"] if r["job"]["sqlite"]]
    assert all(value == off[0] for value in off + on)
    results["operations"] = run({"kind": "operations", "directory": str(directory)}, "operations")
    with sqlite3.connect(directory / "videos.db") as connection:
        queries = {
            "total_sessions": "SELECT COUNT(*) FROM sessions",
            "events_by_type": "SELECT event_type,COUNT(*) FROM events GROUP BY event_type ORDER BY event_type",
            "alert_sessions": "SELECT s.source_name,m.alert_events FROM sessions s JOIN session_metrics m ON m.session_id=s.id WHERE m.alert_events>0",
            "longest_eye": "SELECT s.source_name,m.longest_eye_closure_seconds FROM sessions s JOIN session_metrics m ON m.session_id=s.id ORDER BY m.longest_eye_closure_seconds DESC LIMIT 1",
            "average_drowsy_event_seconds": "SELECT AVG(observed_duration_seconds) FROM events WHERE event_type='DROWSINESS'",
        }
        results["queries"] = {name: {"sql": query, "rows": connection.execute(query).fetchall()} for name, query in queries.items()}
    assert results["queries"]["total_sessions"]["rows"] == [(5,)]
    results["database_size_bytes"] = (directory / "videos.db").stat().st_size
    results["representative_metrics_exact"] = True
    bench.write_new_json(directory / "validation.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--output")
    parser.add_argument("--suite")
    args = parser.parse_args()
    if args.worker:
        job = json.loads(Path(args.worker).read_text())
        if job["kind"] == "parity":
            result = parity(job["sqlite"], Path(job["directory"]))
        elif job["kind"] == "operations":
            result = operations(Path(job["directory"]))
        else:
            result = application(job)
        bench.write_new_json(args.output, result)
    elif args.suite:
        suite(Path(args.suite).resolve())
    else:
        parser.error("Choose --worker or --suite")


if __name__ == "__main__":
    main()
