"""Versioned, main-thread SQLite event/session store using only the stdlib."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import sqlite3
from threading import get_ident
import uuid

SCHEMA_VERSION = 1
EVENT_TYPES = ("YAWN", "EYE_CLOSURE", "DROWSINESS", "ALERT", "NO_FACE")
CLOSE_REASONS = ("evidence_exit", "temporal_reset", "session_end")
STATES = ("ATTENTIVE", "FATIGUE", "SUSPECTED_DROWSY", "DROWSY", "ALERT", "NO_FACE", "WARMING_UP")

# All interpolated SQL identifiers below come from these fixed definitions.
SESSION_FIELDS = {
    "source_type": "TEXT NOT NULL CHECK(source_type IN ('file','webcam'))",
    "source_name": "TEXT NOT NULL", "started_at_utc": "TEXT NOT NULL",
    "created_at_utc": "TEXT NOT NULL", "ended_at_utc": "TEXT",
    "source_start_seconds": "REAL CHECK(source_start_seconds >= 0)",
    "source_end_seconds": "REAL CHECK(source_end_seconds >= source_start_seconds)",
    "source_duration_seconds": "REAL CHECK(source_duration_seconds >= 0)",
    **{name: "TEXT" for name in ("checkpoint_sha256", "normalization_sha256", "face_task_sha256", "model_name")},
    "parameter_count": "INTEGER CHECK(parameter_count >= 0)",
    "sequence_length": "INTEGER CHECK(sequence_length > 0)",
    "target_model_fps": "REAL CHECK(target_model_fps > 0)",
    "max_face_gap_seconds": "REAL CHECK(max_face_gap_seconds > 0)",
    "smoothing_mode": "TEXT", "prediction_history_size": "INTEGER CHECK(prediction_history_size > 0)",
    **{name: "TEXT" for name in ("application_version", "python_version", "torch_version", "opencv_version", "mediapipe_version")},
    "runtime_config_json": "TEXT NOT NULL", "finalized": "INTEGER NOT NULL DEFAULT 0 CHECK(finalized IN (0,1))",
}
COUNT_FIELDS = (
    "decoded_frames", "face_detected_frames", "no_face_frames", "sampled_observations", "model_predictions",
    "normal_raw_count", "yawning_raw_count", "eye_close_raw_count",
    "normal_smoothed_count", "yawning_smoothed_count", "eye_close_smoothed_count", "state_transitions",
    "yawn_events", "eye_closure_events", "drowsy_events", "alert_events", "no_face_events",
    "audible_alert_requests", "audio_suppressed_requests", "audio_completed", "audio_failed",
)
REAL_FIELDS = (
    "longest_yawn_seconds", "longest_eye_closure_seconds", "longest_drowsy_seconds", "longest_alert_seconds", "longest_no_face_seconds",
    "attentive_seconds", "fatigue_seconds", "suspected_drowsy_seconds", "drowsy_seconds", "alert_seconds", "no_face_seconds", "warming_up_seconds",
    "valid_observed_decision_seconds", "unobserved_interval_seconds", "eye_evidence_fraction", "effective_observation_fps", "offline_processing_fps",
)
METRIC_FIELDS = {
    **{name: f"INTEGER CHECK({name} >= 0)" for name in COUNT_FIELDS},
    **{name: f"REAL CHECK({name} >= 0)" for name in REAL_FIELDS},
    "finalized_at_utc": "TEXT NOT NULL", "aggregate_json": "TEXT NOT NULL",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError("Source timestamps/durations must be finite nonnegative numbers")


SCHEMA = (
    "CREATE TABLE schema_metadata (schema_version INTEGER PRIMARY KEY CHECK(schema_version = 1), created_at_utc TEXT NOT NULL)",
    "CREATE TABLE sessions (id INTEGER PRIMARY KEY, session_uuid TEXT UNIQUE NOT NULL, " +
    ", ".join(f"{name} {definition}" for name, definition in SESSION_FIELDS.items()) + ")",
    "CREATE TABLE events (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, "
    "event_type TEXT NOT NULL CHECK(event_type IN ('YAWN','EYE_CLOSURE','DROWSINESS','ALERT','NO_FACE')), "
    "started_at_source_seconds REAL NOT NULL CHECK(started_at_source_seconds >= 0), "
    "ended_at_source_seconds REAL CHECK(ended_at_source_seconds >= started_at_source_seconds), "
    "observed_duration_seconds REAL CHECK(observed_duration_seconds >= 0), "
    "start_state TEXT CHECK(start_state IN ('ATTENTIVE','FATIGUE','SUSPECTED_DROWSY','DROWSY','ALERT','NO_FACE','WARMING_UP')), "
    "end_state TEXT CHECK(end_state IN ('ATTENTIVE','FATIGUE','SUSPECTED_DROWSY','DROWSY','ALERT','NO_FACE','WARMING_UP')), "
    "max_confidence REAL CHECK(max_confidence BETWEEN 0 AND 1), mean_confidence REAL CHECK(mean_confidence BETWEEN 0 AND 1), "
    "audio_triggered INTEGER CHECK(audio_triggered IN (0,1)), audio_suppressed INTEGER CHECK(audio_suppressed IN (0,1)), "
    "closed_reason TEXT CHECK(closed_reason IN ('evidence_exit','temporal_reset','session_end')), "
    "CHECK((ended_at_source_seconds IS NULL AND observed_duration_seconds IS NULL AND closed_reason IS NULL) OR "
    "(ended_at_source_seconds IS NOT NULL AND observed_duration_seconds IS NOT NULL AND closed_reason IS NOT NULL)))",
    "CREATE TABLE session_metrics (session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE, " +
    ", ".join(f"{name} {definition}" for name, definition in METRIC_FIELDS.items()) + ")",
    "CREATE UNIQUE INDEX one_open_event ON events(session_id, event_type) WHERE ended_at_source_seconds IS NULL",
    "CREATE INDEX events_by_session ON events(session_id)",
)


class StorageError(RuntimeError):
    """Requested persistence could not be honored; the run must stop."""


class SQLiteStore:
    def __init__(self, path, *, timeout=5.0):
        self.path = Path(path).expanduser().resolve()
        self._connection = None
        self._depth = 0
        self._owner_thread = get_ident()
        self._rollback_only = False
        try:
            if self.path.exists():
                if not self.path.is_file():
                    raise StorageError(f"SQLite path is not a file: {self.path}")
                # Check before obtaining a writable connection: do not modify an unrelated file.
                with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=timeout)) as probe:
                    self._validate_schema(probe)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("xb"):
                    pass  # Exclusive creation; never truncate an existing file.
                self._connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=timeout)
                with self._connection:
                    self._connection.execute("BEGIN")
                    for statement in SCHEMA:
                        self._connection.execute(statement)
                    self._connection.execute("INSERT INTO schema_metadata VALUES (?, ?)", (SCHEMA_VERSION, utc_now()))
                    self._connection.execute("PRAGMA user_version = 1")
            if self._connection is None:
                self._connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=timeout)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._validate_schema(self._connection)
        except (OSError, sqlite3.Error, StorageError) as error:
            self.close()
            raise StorageError(f"Cannot initialize SQLite database {self.path}: {error}") from error

    @staticmethod
    def _validate_schema(connection):
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise StorageError(f"Unsupported SQLite schema version {version}; expected {SCHEMA_VERSION}")
        expected = {statement.split()[2 if statement.startswith("CREATE TABLE") else (3 if statement.startswith("CREATE UNIQUE") else 2)]: statement for statement in SCHEMA}
        actual = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"))
        if actual != expected:
            raise StorageError("Incompatible SQLite schema: tables, constraints or indexes differ from version 1")
        rows = connection.execute("SELECT schema_version FROM schema_metadata").fetchall()
        if rows != [(SCHEMA_VERSION,)]:
            raise StorageError("Incompatible SQLite schema metadata")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise StorageError("SQLite integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StorageError("SQLite foreign-key integrity check failed")

    def _ensure_open(self):
        if self._connection is None:
            raise StorageError("SQLite store is closed")
        if get_ident() != self._owner_thread:
            raise StorageError("SQLite store must be used on its owner thread")

    @contextmanager
    def transaction(self):
        self._ensure_open()
        outer = self._depth == 0
        self._depth += 1
        try:
            if outer:
                self._rollback_only = False
                self._connection.execute("BEGIN")
            yield
            if outer:
                if self._rollback_only:
                    raise StorageError("SQLite transaction aborted because an earlier write failed")
                self._connection.commit()
        except BaseException as error:
            self._rollback_only = True
            if outer:
                self._connection.rollback()
            if isinstance(error, sqlite3.Error):
                raise StorageError(f"SQLite transaction failed ({self.path}): {error}") from error
            raise
        finally:
            self._depth -= 1
            if outer:
                self._rollback_only = False

    def _insert(self, table, fields, values):
        # Call sites supply only fixed field sets. Reject unrecognized fields before formatting SQL.
        allowed = set(SESSION_FIELDS) | {"session_uuid"} if table == "sessions" else set(METRIC_FIELDS) | {"session_id"}
        if table not in ("sessions", "session_metrics") or set(fields) - allowed:
            raise ValueError("Unknown storage field")
        statement = f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})"
        return self._connection.execute(statement, values).lastrowid

    def start_session(self, *, source_type, source_name, provenance=None):
        now = utc_now()
        fields = dict(provenance or {})
        if set(fields) - (set(SESSION_FIELDS) - {"source_type", "source_name", "started_at_utc", "created_at_utc", "ended_at_utc", "source_start_seconds", "source_end_seconds", "source_duration_seconds", "finalized"}):
            raise ValueError("Unknown or reserved session provenance field")
        fields.update(source_type=source_type, source_name=source_name, started_at_utc=now,
                      created_at_utc=now, session_uuid=str(uuid.uuid4()))
        fields.setdefault("runtime_config_json", "{}")
        with self.transaction():
            return self._insert("sessions", tuple(fields), tuple(fields.values()))

    def _require_unfinalized(self, session_id):
        row = self._connection.execute("SELECT finalized FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row != (0,):
            raise StorageError("Session is missing or already finalized")

    def start_event(self, session_id, event_type, started_at_source_seconds, *, start_state=None):
        with self.transaction():
            _source_number(started_at_source_seconds)
            self._require_unfinalized(session_id)
            return self._connection.execute(
                "INSERT INTO events (session_id,event_type,started_at_source_seconds,start_state) VALUES (?,?,?,?)",
                (session_id, event_type, started_at_source_seconds, start_state)).lastrowid

    def end_event(self, session_id, event_id, *, ended_at_source_seconds, observed_duration_seconds, end_state=None, closed_reason="evidence_exit"):
        with self.transaction():
            _source_number(ended_at_source_seconds)
            _source_number(observed_duration_seconds)
            self._require_unfinalized(session_id)
            cursor = self._connection.execute(
                "UPDATE events SET ended_at_source_seconds=?, observed_duration_seconds=?, end_state=?, closed_reason=? "
                "WHERE id=? AND session_id=? AND ended_at_source_seconds IS NULL",
                (ended_at_source_seconds, observed_duration_seconds, end_state, closed_reason, event_id, session_id))
            if cursor.rowcount != 1:
                raise StorageError("Event is missing, belongs to another session, or is already closed")

    def finalize_session(self, session_id, *, metrics, aggregate):
        if set(metrics) - (set(METRIC_FIELDS) - {"finalized_at_utc", "aggregate_json"}):
            raise ValueError("Unknown aggregate metric field")
        if not aggregate.get("finalized"):
            raise ValueError("Only finalized SessionMetrics can complete a session")
        now = utc_now()
        with self.transaction():
            self._require_unfinalized(session_id)
            if self._connection.execute("SELECT 1 FROM events WHERE session_id=? AND ended_at_source_seconds IS NULL LIMIT 1", (session_id,)).fetchone():
                raise StorageError("Cannot finalize with open events")
            fields = dict(metrics, session_id=session_id, finalized_at_utc=now, aggregate_json=canonical_json(aggregate))
            self._insert("session_metrics", tuple(fields), tuple(fields.values()))
            self._connection.execute(
                "UPDATE sessions SET ended_at_utc=?, source_start_seconds=?, source_end_seconds=?, source_duration_seconds=?, finalized=1 WHERE id=?",
                (now, aggregate["session_start_timestamp"], aggregate["session_end_timestamp"], aggregate["session_duration_seconds"], session_id))

    def close(self):
        if self._connection is not None:
            self._connection.close()  # Any pending transaction is rolled back, never finalized.
            self._connection = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()
