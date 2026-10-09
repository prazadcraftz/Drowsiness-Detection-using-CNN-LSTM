"""Phase 10: schema, transactions, integrity and conservative tracker identity."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

import main
from driver_monitor.metrics import SessionMetrics
from driver_monitor.storage import SQLiteStore, SessionPersistence, StorageError
from driver_monitor.storage.session_persistence import metric_columns
from driver_monitor.storage.sqlite_store import METRIC_FIELDS, SCHEMA_VERSION
from test_session_metrics import decision

ROOT = Path(__file__).resolve().parents[1]


def rows(path, table):
    assert table in ("sessions", "events", "session_metrics", "schema_metadata")
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]


@pytest.fixture
def store(tmp_path):
    with SQLiteStore(tmp_path / "nested" / "sessions.db") as value:
        yield value


def start(store, name="sample.avi"):
    return store.start_session(source_type="file", source_name=name)


def stream(store, observations, *, finalize=True):
    metrics = SessionMetrics()
    session_id = start(store)
    persistence = SessionPersistence(store, session_id)
    for item in observations:
        d = decision(*item[:2], **(item[2] if len(item) == 3 else {}))
        metrics.update(d)
        persistence.update(metrics, d)
    if finalize:
        metrics.finalize()
        persistence.finalize(metrics)
    return metrics, persistence


def test_new_schema_version_tables_keys_and_reopen(store):
    assert store._connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 1
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert {row[0] for row in store._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"sessions", "events", "session_metrics", "schema_metadata"}
    assert rows(store.path, "schema_metadata")[0]["schema_version"] == 1
    path = store.path
    store.close()
    with SQLiteStore(path) as reopened:
        assert start(reopened) == 1


def test_session_uuid_utc_unfinished_start_and_special_characters(store):
    name = "O'Brien; DROP TABLE sessions; -- 日本語.avi"
    start(store, name)
    start(store)
    values = rows(store.path, "sessions")
    assert values[0]["source_name"] == name and values[0]["finalized"] == 0
    assert values[0]["started_at_utc"].endswith("Z") and values[0]["ended_at_utc"] is None
    assert values[0]["session_uuid"] != values[1]["session_uuid"]


def test_insert_update_same_event_multiple_and_wrong_session(store):
    session_id = start(store)
    event_id = store.start_event(session_id, "YAWN", 1, start_state="FATIGUE")
    with pytest.raises(StorageError):
        store.start_event(session_id, "YAWN", 2)
    other_id = start(store)
    with pytest.raises(StorageError):
        store.end_event(other_id, event_id, ended_at_source_seconds=2, observed_duration_seconds=1)
    store.end_event(session_id, event_id, ended_at_source_seconds=3, observed_duration_seconds=1, end_state="ATTENTIVE")
    store.start_event(session_id, "YAWN", 4)
    values = rows(store.path, "events")
    assert len(values) == 2 and values[0]["id"] == event_id
    assert values[0]["observed_duration_seconds"] == 1  # Never infer duration from the timestamps.
    assert values[1]["ended_at_source_seconds"] is None
    assert values[0]["audio_triggered"] is None and values[0]["max_confidence"] is None
    with pytest.raises(StorageError):
        store.end_event(session_id, event_id, ended_at_source_seconds=3, observed_duration_seconds=1)


@pytest.mark.parametrize("bad_type", ["bad", "YAWN'); DROP TABLE events; --", "yawn"])
def test_validated_event_types(store, bad_type):
    session_id = start(store)
    with pytest.raises(StorageError):
        store.start_event(session_id, bad_type, 0)
    assert rows(store.path, "events") == []


@pytest.mark.parametrize("overrides", [
    {"closed_reason": "invented"}, {"end_state": "invented"}, {"observed_duration_seconds": -1}, {"ended_at_source_seconds": -1},
])
def test_invalid_end_rolls_back(store, overrides):
    session_id = start(store)
    event_id = store.start_event(session_id, "YAWN", 1)
    arguments = dict(ended_at_source_seconds=2, observed_duration_seconds=1)
    arguments.update(overrides)
    with pytest.raises((StorageError, ValueError)):
        store.end_event(session_id, event_id, **arguments)
    assert rows(store.path, "events")[0]["ended_at_source_seconds"] is None


def test_finalize_exact_aggregates_and_idempotent_adapter(store):
    metrics, persistence = stream(store, [(0, "SUSPECTED_DROWSY"), (1, "DROWSY"), (2.5, "ALERT"), (3, "ALERT")])
    persisted = rows(store.path, "session_metrics")[0]
    for key, value in metric_columns(metrics.snapshot()).items():
        assert persisted[key] == value
    assert set(METRIC_FIELDS) <= set(persisted)
    assert json.loads(persisted["aggregate_json"]) == metrics.snapshot()
    assert rows(store.path, "sessions")[0]["finalized"] == 1
    assert persisted["audio_completed"] is None and persisted["offline_processing_fps"] is None
    persistence.finalize(metrics)
    assert len(rows(store.path, "session_metrics")) == 1 and persistence.open_ids == {}
    with pytest.raises(StorageError):
        store.start_event(persistence.session_id, "YAWN", 4)


def test_foreign_key_cascade_and_orphan_rejection(store):
    metrics, persistence = stream(store, [(0, "ALERT"), (1, "ALERT")])
    assert len(rows(store.path, "events")) == 3
    with store.transaction():
        store._connection.execute("DELETE FROM sessions WHERE id=?", (persistence.session_id,))
    assert rows(store.path, "events") == rows(store.path, "session_metrics") == []
    with pytest.raises(StorageError):
        with store.transaction():
            store._connection.execute("INSERT INTO events(session_id,event_type,started_at_source_seconds) VALUES (?,?,?)", (999, "YAWN", 0))


def test_multi_write_transaction_rollback(store):
    session_id = start(store)
    with pytest.raises(StorageError):
        with store.transaction():
            store.start_event(session_id, "YAWN", 0)
            store.start_event(session_id, "invalid", 0)
    assert rows(store.path, "events") == [] and rows(store.path, "sessions")[0]["finalized"] == 0


def test_caught_inner_failure_aborts_whole_transaction(store):
    session_id = start(store)
    with pytest.raises(StorageError, match="earlier write failed"):
        with store.transaction():
            store.start_event(session_id, "YAWN", 0)
            try:
                store.start_event(session_id, "invalid", 0)
            except StorageError:
                pass
    assert rows(store.path, "events") == []
    assert store.start_event(session_id, "YAWN", 1) > 0


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), -1, True, "1"])
def test_nonfinite_or_invalid_source_numbers_rejected(store, bad_number):
    session_id = start(store)
    with pytest.raises(ValueError, match="finite nonnegative"):
        store.start_event(session_id, "YAWN", bad_number)
    event_id = store.start_event(session_id, "YAWN", 0)
    with pytest.raises(ValueError, match="finite nonnegative"):
        store.end_event(session_id, event_id, ended_at_source_seconds=1, observed_duration_seconds=bad_number)
    assert rows(store.path, "events")[0]["ended_at_source_seconds"] is None


def test_atomic_finalization_failure_retains_open_events_and_unfinished_session(store):
    metrics, persistence = stream(store, [(0, "ALERT"), (1, "ALERT")], finalize=False)
    metrics.finalize()
    # Force the final aggregate INSERT to fail after all event END updates.
    with store.transaction():
        store._connection.execute("CREATE TRIGGER reject_metrics BEFORE INSERT ON session_metrics BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(StorageError, match="injected failure"):
        persistence.finalize(metrics)
    assert rows(store.path, "session_metrics") == []
    assert rows(store.path, "sessions")[0]["finalized"] == 0
    assert all(row["ended_at_source_seconds"] is None for row in rows(store.path, "events"))
    assert len(persistence.open_ids) == 3


@pytest.mark.parametrize("path_kind", ["directory", "text", "empty", "parent_file", "foreign_db"])
def test_bad_paths_and_unrelated_files_preserved(tmp_path, path_kind):
    path = tmp_path / "target.db"
    if path_kind == "directory":
        path.mkdir()
    elif path_kind in ("text", "empty"):
        path.write_bytes(b"unrelated private content" if path_kind == "text" else b"")
    elif path_kind == "parent_file":
        path.write_text("parent")
        path = path / "child.db"
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
    before = path.read_bytes() if path.is_file() else None
    with pytest.raises(StorageError):
        SQLiteStore(path)
    if before is not None:
        assert path.read_bytes() == before


def test_permission_failure_reported(tmp_path, monkeypatch):
    original = Path.open
    def deny(path, *args, **kwargs):
        if args and args[0] == "xb":
            raise PermissionError("injected denied file creation")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(StorageError, match="denied"):
        SQLiteStore(tmp_path / "denied.db")


@pytest.mark.parametrize("change", ["newer", "older", "metadata", "table"])
def test_version_and_incompatible_schema_rejection(store, change):
    path = store.path
    store.close()
    with sqlite3.connect(path) as connection:
        if change in ("newer", "older"):
            connection.execute("PRAGMA user_version = " + ("2" if change == "newer" else "0"))
        elif change == "metadata":
            connection.execute("DELETE FROM schema_metadata")
        else:
            connection.execute("ALTER TABLE sessions ADD COLUMN incompatible TEXT")
    before = path.read_bytes()
    with pytest.raises(StorageError, match="schema"):
        SQLiteStore(path)
    assert path.read_bytes() == before


def test_lock_failure_is_transparent_and_rollback(store):
    session_id = start(store)
    store._connection.execute("PRAGMA busy_timeout = 10")
    with sqlite3.connect(store.path) as locker:
        locker.execute("BEGIN IMMEDIATE")
        with pytest.raises(StorageError, match="locked"):
            store.start_event(session_id, "YAWN", 0)
    assert rows(store.path, "events") == []
    assert store.start_event(session_id, "YAWN", 0) > 0


def test_close_context_exception_and_no_auto_recovery(store):
    path = store.path
    stream(store, [(0, "FATIGUE"), (1, "FATIGUE")], finalize=False)
    store.close()
    store.close()
    with pytest.raises(StorageError, match="closed"):
        start(store)
    with pytest.raises(ValueError, match="controlled"):
        with SQLiteStore(path):
            raise ValueError("controlled")
    with SQLiteStore(path) as reopened:
        assert rows(path, "sessions")[0]["finalized"] == 0
        assert rows(path, "events")[0]["observed_duration_seconds"] is None
        assert start(reopened) == 2


def test_owner_thread_enforced(store):
    errors = []
    def write():
        try:
            start(store)
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=write)
    worker.start()
    worker.join()
    assert len(errors) == 1 and isinstance(errors[0], (StorageError, sqlite3.ProgrammingError))
    assert rows(store.path, "sessions") == []


@pytest.mark.parametrize("observations, expected_count", [
    ([(0, "FATIGUE"), (1, "FATIGUE"), (2, "FATIGUE"), (3, "ATTENTIVE")], 1),
    ([(0, "FATIGUE"), (1, "FATIGUE"), (2, "ATTENTIVE"), (3, "FATIGUE"), (5, "FATIGUE")], 2),
])
def test_yawn_episode_identity(store, observations, expected_count):
    metrics, _ = stream(store, observations)
    events = rows(store.path, "events")
    assert len(events) == expected_count == metrics.events["yawn"].count
    assert sum(row["observed_duration_seconds"] for row in events) == metrics.events["yawn"].total_duration_seconds


def test_suspected_drowsy_alert_containing_identity(store):
    stream(store, [(0, "SUSPECTED_DROWSY"), (1, "DROWSY"), (2.5, "ALERT"), (3, "ALERT"), (4, "ATTENTIVE")])
    values = {row["event_type"]: row for row in rows(store.path, "events")}
    assert set(values) == {"EYE_CLOSURE", "DROWSINESS", "ALERT"}
    assert [values[name]["observed_duration_seconds"] for name in ("EYE_CLOSURE", "DROWSINESS", "ALERT")] == [3, 2, .5]


def test_short_gap_same_eye_id_excludes_missing_time(store):
    stream(store, [(0, "SUSPECTED_DROWSY"), (.1, "SUSPECTED_DROWSY"), (.15, "NO_FACE"), (.3, "NO_FACE"), (.4, "SUSPECTED_DROWSY"), (.5, "SUSPECTED_DROWSY")])
    values = {row["event_type"]: row for row in rows(store.path, "events")}
    assert len(values) == 2 and values["EYE_CLOSURE"]["observed_duration_seconds"] == pytest.approx(.2)
    assert values["EYE_CLOSURE"]["ended_at_source_seconds"] == .5
    assert values["NO_FACE"]["observed_duration_seconds"] == pytest.approx(.25)


@pytest.mark.parametrize("state", ["NO_FACE", "WARMING_UP"])
def test_long_reset_last_valid_boundary(store, state):
    stream(store, [(0, "ALERT"), (.1, "ALERT"), (.2, "NO_FACE"), (10, state, {"reset": True}), (11, "SUSPECTED_DROWSY")])
    for row in rows(store.path, "events"):
        if row["event_type"] in ("EYE_CLOSURE", "DROWSINESS") and row["started_at_source_seconds"] == 0:
            assert row["ended_at_source_seconds"] == .1
            assert row["observed_duration_seconds"] == .1 and row["closed_reason"] == "temporal_reset"


def test_no_sql_or_history_on_10000_active_updates(store):
    metrics, persistence = stream(store, [(0, "FATIGUE")], finalize=False)
    statements = []
    store._connection.set_trace_callback(statements.append)
    for i in range(1, 10001):
        d = decision(i / 20, "FATIGUE")
        metrics.update(d)
        persistence.update(metrics, d)
    assert statements == [] and len(persistence.open_ids) == 1
    store._connection.set_trace_callback(None)
    metrics.finalize()
    persistence.finalize(metrics)
    assert rows(store.path, "events")[0]["observed_duration_seconds"] == 500


def test_compatibility_no_fabricated_events(store):
    from dataclasses import replace
    metrics = SessionMetrics()
    persistence = SessionPersistence(store, start(store))
    metrics.update(replace(decision(0), driver=None))
    persistence.update(metrics, replace(decision(0), driver=None))
    metrics.finalize()
    persistence.finalize(metrics)
    assert rows(store.path, "events") == []


def test_cli_path_requires_explicit_sqlite(tmp_path, capsys):
    path = tmp_path / "unrequested.db"
    assert main.main(["--source", "0", "--db-path", str(path), "--no-display"]) == 1
    assert "requires --sqlite" in capsys.readouterr().err and not path.exists()


def test_disabled_never_imports_storage_in_fresh_process(tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sqlite3' or fullname.startswith('driver_monitor.storage'):
            raise AssertionError('Disabled storage imported: '+fullname)
sys.meta_path.insert(0, Block())
import main
sys.path.insert(0, 'tests')
from test_main import FakeSource, FakeEngine
main.VideoSource = FakeSource
main._create_engine = lambda: FakeEngine()
main.run(0, no_display=True)
assert 'sqlite3' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("stop", ["eof", "q", "escape", "error", "storage_error", "interrupt"])
def test_main_persistence_cleanup_and_completion(tmp_path, monkeypatch, stop):
    from test_main import FakeEngine, FakeSource
    import driver_monitor.storage
    path = tmp_path / "app.db"
    engine = FakeEngine()
    monkeypatch.setattr(main, "VideoSource", FakeSource)
    monkeypatch.setattr(main, "_create_engine", lambda: engine)
    monkeypatch.setattr(driver_monitor.storage, "session_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(main.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda _: ord("q") if stop == "q" else 27)
    if stop in ("error", "interrupt"):
        def fail(frame):
            raise KeyboardInterrupt() if stop == "interrupt" else RuntimeError("controlled failure")
        monkeypatch.setattr(engine, "process_frame", fail)
    elif stop == "storage_error":
        def fail(*args):
            raise StorageError("injected storage failure")
        monkeypatch.setattr(SessionPersistence, "update", fail)
    kwargs = dict(no_display=stop not in ("q", "escape"), no_audio=True, sqlite=True, db_path=path)
    if stop in ("error", "storage_error", "interrupt"):
        with pytest.raises(KeyboardInterrupt if stop == "interrupt" else RuntimeError):
            main.run(0, **kwargs)
        assert rows(path, "sessions")[0]["finalized"] == 0 and rows(path, "session_metrics") == []
    else:
        summary = main.run(0, **kwargs)
        assert rows(path, "sessions")[0]["finalized"] == 1
        assert json.loads(rows(path, "session_metrics")[0]["aggregate_json"]) == summary.session_metrics
    assert engine.closed == 1
    with SQLiteStore(path):
        pass  # Cleanup released the connection and doesn't rewrite unfinished rows.


@pytest.mark.parametrize("filename, expected", [
    ("Normal/Normal_1.avi", {"YAWN": 0, "EYE_CLOSURE": 0, "DROWSINESS": 0, "ALERT": 0}),
    ("Yawning/Yawning_1.avi", {"YAWN": 1}),
    ("Yawning/Yawning_60.avi", {"YAWN": 1, "NO_FACE": 4}),
    ("Eye Close/Eye Close_60.avi", {"EYE_CLOSURE": 1, "DROWSINESS": 1, "ALERT": 1}),
    ("Eye Close/Eye Close_1.avi", {"EYE_CLOSURE": 0}),
])
@pytest.mark.vision
def test_real_video_sqlite_aggregates_exact(tmp_path, filename, expected):
    path = tmp_path / "real.db"
    # A fresh worker keeps CPU settings isolated from tests altering thread counts.
    result = subprocess.run([sys.executable, "-B", "main.py", "--source", filename, "--no-display", "--no-audio", "--sqlite", "--db-path", str(path)],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    aggregate = summary["session_metrics"]
    persisted = rows(path, "session_metrics")[0]
    assert json.loads(persisted["aggregate_json"]) == aggregate
    for key, value in metric_columns(aggregate, offline_processing_fps=summary["processing_throughput_fps"]).items():
        if isinstance(value, float):
            assert persisted[key] == pytest.approx(value, rel=1e-12, abs=1e-12)
        else:
            assert persisted[key] == value
    events = rows(path, "events")
    for name, count in expected.items():
        assert sum(row["event_type"] == name for row in events) == count
    for name, label in (("yawn", "YAWN"), ("eye_closure", "EYE_CLOSURE"), ("drowsiness", "DROWSINESS"), ("alert", "ALERT"), ("no_face", "NO_FACE")):
        matching = [row for row in events if row["event_type"] == label]
        assert len(matching) == aggregate["events"][name]["count"]
        assert sum(row["observed_duration_seconds"] for row in matching) == pytest.approx(aggregate["events"][name]["total_duration_seconds"], abs=1e-12)
    session = rows(path, "sessions")[0]
    assert session["source_name"] == Path(filename).name and session["parameter_count"] == 246595
    for column, asset in (("checkpoint_sha256", "saved_model.pt"), ("normalization_sha256", "norm.npy"), ("face_task_sha256", "face_landmarker.task")):
        assert session[column] == hashlib.sha256((ROOT / asset).read_bytes()).hexdigest()


def test_read_only_query_utility(store, tmp_path):
    from tools.query_sessions import query
    _, persistence = stream(store, [(0, "FATIGUE"), (1, "FATIGUE")])
    before = store.path.read_bytes()
    assert query(store.path)[0]["id"] == persistence.session_id
    assert query(store.path, session_id=persistence.session_id, events=True)[0]["event_type"] == "YAWN"
    assert query(store.path, session_id=persistence.session_id)[0]["yawn_events"] == 1
    assert store.path.read_bytes() == before
    missing = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        query(missing)
    assert not missing.exists()
