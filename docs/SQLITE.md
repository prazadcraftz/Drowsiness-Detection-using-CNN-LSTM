# Optional local SQLite records

SQLite is disabled by default. `--sqlite --db-path <new-local-path>` creates a local version-1 database through Python's standard-library `sqlite3`; `--db-path` requires `--sqlite`. Its four tables are `schema_metadata`, `sessions`, `events`, and `session_metrics`. There are **no raw-frame, landmark, or per-frame prediction tables**. Session rows contain source identity, lifecycle and model provenance (including profile/checkpoint/normalizer hashes); `events` stores segmented YAWN, EYE_CLOSURE, DROWSINESS, ALERT, and NO_FACE episodes; `session_metrics` stores one final aggregate per completed session. Abnormal exits can leave a session unfinished for later inspection. The [Phase 10 schema and transaction record](PHASE10_SQLITE.md) has exact columns, constraints, and limitations.

A safe local example from the repository root uses a new scratch filename:

```powershell
python main.py --source 'path/to/video.avi' --model-profile lstm32 --no-display --no-audio --sqlite --db-path '.cache/demo_sessions_02.db'
python tools/query_sessions.py '.cache/demo_sessions_02.db'
```

`query_sessions.py` opens the database read-only. It can also use `--session-id 1` and `--session-id 1 --events`; check the returned session ID rather than assuming it is 1. For direct SQLite inspection, these queries use actual schema columns:

```sql
SELECT id, source_type, source_name, model_name, sequence_length, finalized
FROM sessions ORDER BY id DESC LIMIT 10;

SELECT event_type, COUNT(*) AS episodes,
       SUM(observed_duration_seconds) AS observed_seconds
FROM events WHERE ended_at_source_seconds IS NOT NULL
GROUP BY event_type ORDER BY event_type;

SELECT s.id, s.source_name, m.alert_events, m.longest_eye_closure_seconds
FROM sessions AS s JOIN session_metrics AS m ON m.session_id = s.id
WHERE m.alert_events > 0 ORDER BY s.id;

SELECT id, source_name, started_at_utc
FROM sessions WHERE finalized = 0 ORDER BY id;
```

The database stores local metadata, so users should handle its file and any annotated output according to their own privacy needs. The application is **designed for local processing** and requires no cloud inference; it does not make an absolute security/privacy guarantee. SQLite is a single-owner local record, not a multi-user service. Event durations count observed evidence according to the [state/event policy](STATE_MACHINE.md), not ground-truth event duration.
