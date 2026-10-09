"""Build the Phase 10 report from completed tests and preserved evidence."""

from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driver_monitor.metrics import benchmark as bench
from driver_monitor.storage.sqlite_store import SESSION_FIELDS, METRIC_FIELDS, SCHEMA


def main():
    evidence = json.loads((ROOT / "results/phase10/validation.json").read_text())
    xml = ET.parse(ROOT / "docs/phase10_tests.xml")
    cases = xml.findall(".//testcase")
    test_seconds = sum(float(suite.attrib["time"]) for suite in xml.findall(".//testsuite"))
    assert not xml.findall(".//failure") and not xml.findall(".//error") and not xml.findall(".//skipped")
    counts = Counter(case.attrib["classname"] for case in cases)
    new_count = counts["tests.test_sqlite_store"]
    assert len(cases) - new_count == 380
    before = json.loads((ROOT / "docs/phase10_before_hashes.json").read_text(encoding="utf-8-sig"))
    changed = {name: {"before": old, "after": bench.sha256(ROOT / name)} for name, old in before.items() if bench.sha256(ROOT / name) != old}
    assert set(changed) == {"main.py"}, changed
    assert not (ROOT / "data/drowsiness.db").exists(), "An unrequested default database was created"
    assert evidence["representative_metrics_exact"]
    for mode in ("parity_off", "parity_on"):
        result = evidence[mode]
        assert result["exact_parity"] and result["windows_exact"] == result["classes_exact"] == 699
    for off, on in zip(evidence["parity_off"]["sources"], evidence["parity_on"]["sources"]):
        assert all(off[key] == on[key] for key in ("policy_trace_sha256", "session_sha256", "pixel_trace_sha256", "maximum_differences"))
    expected_counts = [{"yawn": 0, "eye_closure": 0, "drowsiness": 0, "alert": 0}, {"yawn": 1}, {"yawn": 1, "no_face": 4}, {"eye_closure": 1, "drowsiness": 1, "alert": 1}, {"eye_closure": 0}]
    for video, expected in zip(evidence["videos"], expected_counts):
        assert all(video["summary"]["session_metrics"]["events"][name]["count"] == count for name, count in expected.items())
    phase_counts = json.loads((ROOT / "docs/phase9_validation.json").read_text())["tests_by_phase"]
    phase_counts["Phase 10"] = new_count
    created = ["driver_monitor/storage/__init__.py", "driver_monitor/storage/sqlite_store.py", "driver_monitor/storage/session_persistence.py",
               "tests/test_sqlite_store.py", "tools/query_sessions.py", "tools/validate_phase10.py", "tools/report_phase10.py"]
    schema_tables = []
    for title, fields in (("sessions (plus id INTEGER PRIMARY KEY, session_uuid TEXT UNIQUE NOT NULL)", SESSION_FIELDS),
                          ("session_metrics (plus session_id INTEGER PRIMARY KEY / FK)", METRIC_FIELDS)):
        schema_tables.append(f"### {title}\n\n| Field | SQLite type and constraint |\n|---|---|\n" + "\n".join(f"| `{name}` | `{kind}` |" for name, kind in fields.items()))
    events_schema = next(sql for sql in SCHEMA if sql.startswith("CREATE TABLE events"))
    video_table = "| Source | Frames | Yawn | Eye | Drowsy | Alert | No-face |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for video in evidence["videos"]:
        aggregate = video["summary"]["session_metrics"]
        video_table += "| " + " | ".join([video["job"]["source"], str(aggregate["decoded_frames"])] + [str(aggregate["events"][name]["count"]) for name in ("yawn", "eye_closure", "drowsiness", "alert", "no_face")]) + " |\n"
    query_table = "| Executed query | Result |\n|---|---|\n" + "\n".join(
        f"| {name} | `{json.dumps(value['rows'])}` |" for name, value in evidence["queries"].items())
    operation_table = "| Committed operation | n | P50 ms | P95 ms | Max ms |\n|---|---:|---:|---:|---:|\n"
    for name, value in evidence["operations"]["distribution_ms"].items():
        operation_table += f"| {name} | {value['sample_count']} | {value['p50_ms']:.4f} | {value['p95_ms']:.4f} | {value['max_ms']:.4f} |\n"
    runtime_table = "| Fresh-worker order | SQLite | Loop seconds | Offline FPS | Total run seconds |\n|---|---|---:|---:|---:|\n"
    for index, run in enumerate(evidence["representative"]):
        summary = run["summary"]
        runtime_table += f"| {index+1} | {'on' if run['job']['sqlite'] else 'off'} | {summary['loop_wall_seconds']:.6f} | {summary['processing_throughput_fps']:.3f} | {summary['total_wall_seconds']:.6f} |\n"
    medians = {enabled: statistics.median(run["summary"]["loop_wall_seconds"] for run in evidence["representative"] if run["job"]["sqlite"] == enabled) for enabled in (False, True)}
    delta = 100 * (medians[True] / medians[False] - 1)
    text = f"""# Phase 10 — optional local SQLite persistence

**PHASE 10 PASS.** {len(cases)} tests passed, including all 380 unchanged Phase 2–9 tests and {new_count} Phase 10 cases. Independently replayed SQLite off/on: 699/699 raw windows and classes exact in each mode, maximum numeric difference 0.0. Stopped after Phase 10.

## 1–2. Files created and modified

Created implementation/test/tool files: {', '.join('`'+name+'`' for name in created)}. Created this report, `docs/phase10_before_hashes.json`, `docs/phase10_validation.json`, final/initial full-suite XML (`docs/phase10_tests.xml`, `docs/phase10_tests_initial.xml`), initial/final focused-test XML (`docs/phase10_unit_initial.xml`, `docs/phase10_unit_final.xml`), the smoke DB `results/phase10_smoke.db`, and all finite worker inputs, outputs, stderr logs, landmark archives and example databases under `results/phase10/`.

Modified only `main.py`: lazy opt-in storage startup, adapter update after analytics, atomic completion on normal exit, and two CLI flags. {len(before)-1} of {len(before)} prior files remain byte-identical. This includes SessionMetrics, all old tests, canonical inference/vision/model/config/assets, dashboard cache, and Phase 8/9 reports/results/benchmark evidence. Preservation hashes are in `docs/phase10_before_hashes.json` and `docs/phase10_validation.json`. Existing unrelated working-tree changes were preserved.

## 3. Storage architecture

```mermaid
flowchart TD
  V[VideoSource] --> E[DrowsinessInferenceEngine]
  E --> D[DecisionLayer]
  D --> M[SessionMetrics]
  M --> A[AlertManager]
  A --> UI[Dashboard]
  M --> P[Optional SessionPersistence]
  P --> S[SQLiteStore]
```

The adapter reads each authoritative tracker immediately after metrics.update; it compares active status with its currently open row IDs. It retains at most five IDs and one last-state value. ACTIVE observations issue no SQL or transaction. It neither recomputes events nor retains episode/prediction/frame history. Analytics imports no sqlite3. Audio never accesses storage. sqlite3's default owner-thread restriction remains enabled.

## 4–8. Schema and all fields

Schema version **1**, recorded in both `PRAGMA user_version` and the single-row `schema_metadata(schema_version INTEGER PRIMARY KEY CHECK(schema_version=1), created_at_utc TEXT NOT NULL)`. Existing files are probed read-only before obtaining a writable connection. Unsupported versions, mismatched metadata/table/index definitions, foreign-key violations and failed quick_check are rejected. Existing unrelated/empty/text databases are rejected, never initialized or overwritten. There is no migration or automatic unfinished-session recovery.

```mermaid
erDiagram
  sessions ||--o{{ events : contains
  sessions ||--o| session_metrics : finalizes
```

sessions holds identity, wall-clock lifecycle, source-time span and provenance. events holds one row per segmented episode. session_metrics holds one final aggregate per session. schema_metadata identifies this format. There are no frame/blob/landmark/prediction tables. Source names in runtime persistence are basenames or Camera N; private absolute source/asset paths are omitted from provenance.

{chr(10).join(schema_tables)}

### events (all fields, keys and constraints)

```sql
{events_schema};
```

Open events have NULL end/duration/reason; closed events require all three. Start/end state indicates the policy state when the transition was observed, even when the authoritative end boundary precedes that observation. max_confidence and mean_confidence are NULL: Phase 7 does not collect per-episode confidence aggregates. audio_triggered/audio_suppressed are NULL: the tracker adapter has no authoritative per-episode playback attribution. Logical ALERT counts and final audio request/suppression/completion/failure counts are independent and retained truthfully. No fabricated confidence/audio values.

## 9–10. Referential integrity and transactions

Foreign keys are ON on every store connection. Deleting a session cascades to its events and aggregate. A partial unique index permits only one open row per session/event type; events_by_session indexes the FK. All variable values use parameter placeholders; formatted identifiers come only from fixed schema whitelists. Quotes, semicolons and Unicode source names are tested.

Session creation commits immediately. A frame with actual transitions uses one small transaction for all its START/END operations. Continued episodes perform no writes. Normal finalization atomically closes remaining events, inserts metrics, updates session timestamps/span and marks finalized=1. Any exception rolls back the entire current transaction. No per-frame commit. sqlite3 timeout is 5 seconds; default rollback journal/durability settings are retained, with no WAL/tuning/ORM/dependency.

## 11–13. Startup, completion and abnormal sessions

Requested storage initialization/write failures raise StorageError and fail the run transparently; main reports a clear error and releases engine/source/writer/windows/audio/storage resources. Startup commits a UUID session with finalized=0. EOF/q/Escape closes normal analytics events at available source observations, finishes audio, then persists final metrics. Wall-clock times are UTC ISO-8601 Z; model/event timing remains source/acquisition seconds. Offline processing FPS is the existing full loop frames/loop_seconds metric for files; webcam storage leaves it NULL.

Controlled exceptions/KeyboardInterrupt skip finalization. Prior committed START/END transitions survive, finalized=0, ended_at_utc and final metrics remain absent. Open events retain NULL duration/end because ACTIVE evidence was deliberately not written. This preserves safe completed evidence without pretending an incomplete event/session completed. Power loss/process kill cannot guarantee cleanup; SQLite's committed transaction durability applies. Opening again never rewrites unfinished sessions. Close is idempotent; context exit closes and rolls back pending work.

## 14–16. Event identity, short gaps and resets

START inserts; END updates the same ID with the tracker's end_timestamp and duration_seconds. Types: YAWN, EYE_CLOSURE, DROWSINESS, ALERT, NO_FACE. A sustained yawn is one row; separated yawns are separate rows. SUSPECTED_DROWSY→DROWSY→ALERT creates one containing eye episode, one drowsiness episode and one logical alert. ALERT's external transition semantics are unchanged: NO_FACE closes alert, and ALERT re-entry is a new alert episode.

Short NO_FACE pauses evidence bridges while retaining the containing eye/yawn/drowsiness identity. Stored observed duration excludes missing/recovery bridges, rather than subtracting end minus start. Deterministic short-gap test: eye spans 0–0.5 s, observed duration 0.2 s, same eye ID; NO_FACE duration 0.25 s.

Long reset/warm-up closes evidence at its last valid observation: deterministic eye/drowsiness end 0.1 s despite reset at 10 s. closed_reason is temporal_reset for evidence invalidation, evidence_exit for ordinary exits, session_end for normal EOF/keyboard closure. These labels describe existing tracker closures; they do not influence policy.

## 17. Provenance

Once per opt-in session: checkpoint/normalization/face-task SHA-256, CNN_LSTM name and 246595 parameters, sequence length 16, target 20 Hz, face gap 0.5 s, smoother/history, full scalar runtime/policy thresholds/durations, class mapping, compatibility flag, device/dropout, audio enable/cooldown, application format version, Python/Torch/OpenCV/MediaPipe distribution versions. Hashes are streamed once at startup, never per frame. Source video hashes are also retained in validation evidence, not required runtime DB fields. Validation workers retain default Torch 8/8 and OpenCV 12 CPU threads.

## 18–19. CLI and default-disabled behavior

```powershell
python main.py --source 0 --sqlite
python main.py --source "Eye Close/Eye Close_60.avi" --no-display --no-audio --sqlite --db-path data/drowsiness.db
python main.py --source "Normal/Normal_1.avi" --no-display --no-audio
```

--sqlite is explicitly required; --db-path alone raises a useful error before source/model loading. Enabled default path is data/drowsiness.db relative to cwd; parent directories are created. Ordinary runs never import the storage package/sqlite3, hash assets, open/create a DB or issue SQL. Only optional condition checks were added; algorithms/source-time contracts remain identical. The entire previous regression suite runs with default persistence disabled. No default database was created during validation.

## 20. Actual-video persistence results

{video_table}

`results/phase10/videos.db` contains these five finalized sessions. Every normalized integer/float aggregate equals its final SessionMetrics snapshot; event counts match exactly and summed observed durations match within 1e-12 s. Supplemental aggregate_json is the exact canonical final snapshot (including source/state-frame counts and audio-disabled counts); it is bounded and contains no arrays/predictions. The same five clips are covered by fresh-process integration tests. Eye Close_1 naturally has zero eye events; no policy tuning or folder-label correction was performed.

## 21. SQL and read-only inspection

```sql
SELECT COUNT(*) FROM sessions;
SELECT event_type, COUNT(*) FROM events GROUP BY event_type;
SELECT s.id,s.source_name,m.alert_events FROM sessions s
JOIN session_metrics m ON m.session_id=s.id WHERE m.alert_events>0;
SELECT COUNT(*) FROM events WHERE event_type='YAWN';
SELECT AVG(observed_duration_seconds) FROM events WHERE event_type='DROWSINESS';
SELECT session_id,observed_duration_seconds FROM events WHERE event_type='EYE_CLOSURE'
ORDER BY observed_duration_seconds DESC LIMIT 1;
SELECT s.source_name,m.longest_eye_closure_seconds FROM sessions s
JOIN session_metrics m ON m.session_id=s.id ORDER BY m.longest_eye_closure_seconds DESC LIMIT 1;
SELECT source_type,source_name,COUNT(*) FROM sessions GROUP BY source_type,source_name;
SELECT id,source_name,started_at_utc FROM sessions WHERE finalized=0;
```

Executed total/event-type/alert/longest-eye/average-drowsy queries and exact outputs are retained in `results/phase10/validation.json`:

{query_table}

Optional tool uses a read-only SQLite URI and parameterized filters, so a missing path cannot create a DB:

```powershell
python tools/query_sessions.py results/phase10/videos.db
python tools/query_sessions.py results/phase10/videos.db --session-id 4
python tools/query_sessions.py results/phase10/videos.db --session-id 4 --events
```

## 22. Storage-operation timings

{operation_table}

100 operations of each type in one dedicated fresh worker; includes synchronous commit, no removed samples/warm-up exclusions. Final metric write includes JSON serialization, excludes schema startup, provenance hashing and final event closures. This finite synthetic API probe is separate from real-video analytics. All raw nanoseconds/distributions are in results/phase10/operations.json. SQLite {evidence['operations']['sqlite_version']}, stdlib only, same local Windows machine/venv as Phase 9. Commits take roughly 10 ms here: actual transitions can add a visible single-frame latency spike, even though ACTIVE frames issue no SQL. Multiple same-frame transitions share one commit. No claim that enabled persistence has zero latency cost.

## 23. Optional runtime overhead (descriptive)

Eye Close_60, 124 frames, headless/audio-disabled; four sequential fresh workers in off/on/on/off order:

{runtime_table}

Median loop time off {medians[False]:.6f} s, on {medians[True]:.6f} s, descriptive difference {delta:+.2f}%. Two observations per mode cannot isolate IO from CPU/cache/background variation or support a full-system speed/regression claim. Loop time includes transition writes and adapter checks; total time includes startup/provenance/finalization/cleanup but excludes interpreter startup/imports before main.run. Source-time policy/final metrics were exactly equal across all four runs. No Phase 9 benchmarks or its 8.6% dashboard optimization claim were regenerated or mixed with optional persistence overhead.

## 24. Database size

Five real sessions / {sum(len(video['persisted']['events']) for video in evidence['videos'])} event rows / five aggregate rows: {evidence['database_size_bytes']} bytes ({evidence['database_size_bytes']/1024:.1f} KiB) for videos.db after connection cleanup, including schema/index pages. This is one finite local example, not a scalability/storage-growth guarantee. Parity and synthetic operations use separate DBs.

## 25–27. Regression and exact parity

All {len(cases)} tests passed, zero failures/errors/skips; XML: docs/phase10_tests.xml ({test_seconds:.3f} seconds recorded by JUnit). Phase counts: {phase_counts}. Prior test source bytes and assertions are unchanged. New tests cover schema/UTC/UUID, multiple episodes/same ID, quote safety, checks/FKs/cascade, transaction and atomic-finalization rollback (including caught inner write failure), malformed/path/permission/lock/version failures, finite source timestamps/durations, owner thread, context cleanup, unfinished reopen, 10000 ACTIVE updates with no SQL, deterministic short/long gaps, compatibility, CLI/default import isolation, EOF/q/Escape/interrupt/runtime/storage failure cleanup, read-only queries, and all five actual clips.

Fresh SQLite off/on six-source replay each matches all 699 frozen windows/classes, raw/normalized features, sequences, float32 inputs, logits and probabilities exactly (max difference 0.0). All detector x/y coordinates exposed by the runtime, complete policy/smoothing traces, final analytics snapshots and rendered pixel traces match the immutable Phase 9 production evidence and each other. Detector/engine source and task bytes are unchanged. Six landmark archives per mode plus hashes/counters are retained under results/phase10/parity_off and parity_on. Persistence-enabled final normalized metrics/counts/durations also agree with analytics.

## 28. Limitations

Local single-owner SQLite, not a multi-user backend. No migration/recovery/retention service, cloud/network/auth, raw-data storage, retraining/resplit/preprocessing/model/window/threshold changes, deployment or Phase 11 work. No live webcam/physical sound/interactive GUI repeat was claimed; q/Escape and exception flows are deterministic integration tests, videos are actual headless runs. Confidence/per-event audio attribution is intentionally NULL. File basenames can collide; UUID/id disambiguates sessions but not identically named source content. Abnormal open events contain start evidence only. Initial creation failure can leave a newly created empty/incomplete owned file requiring inspection/new filename; unrelated existing files are never truncated. File-system permission errors were injected deterministically, not measured against arbitrary ACLs. Default SQLite concurrency/durability limits apply; no arbitrary power-loss guarantee or scalability claim. Finite validation archives contain offline diagnostic landmarks, separate from runtime SQLite and bounded runtime memory.

## 29–30. Resume-safe wording and outcome

“Designed a local SQLite event/session store for timestamped fatigue, eye-closure and alert episodes, preserving aggregate driver-monitoring metrics and model provenance without storing raw frames.”

**PHASE 10 PASS. STOP after Phase 10.**

Reproduce tests with `.venv-baseline/Scripts/python.exe -B -m pytest -q`. The evidence suite uses exclusive writes; choose a new directory: `.venv-baseline/Scripts/python.exe -B tools/validate_phase10.py --suite results/phase10_new`. Do not rerun into existing evidence. Report generation asserts preservation/parity/test gates before writing new Phase 10 report/validation files.
"""
    # Confirm the clip count from evidence rather than trusting explanatory prose.
    frame_count = evidence["representative"][0]["summary"]["frames_decoded"]
    text = text.replace("Eye Close_60, 124 frames", f"Eye Close_60, {frame_count} frames")
    report_path = ROOT / "docs/PHASE10_SQLITE.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    summary = {
        "status": "PHASE10_PASS", "tests_passed": len(cases), "test_seconds_junit": test_seconds, "test_counts_by_module": dict(counts), "phase_test_counts": phase_counts,
        "sqlite_off_windows_exact": 699, "sqlite_on_windows_exact": 699, "maximum_numeric_difference": 0.0,
        "policy_session_landmarks_pixels_exact": True, "prior_files_preserved": len(before)-1, "prior_files_checked": len(before),
        "changed_prior_files": changed, "default_database_created": False,
        "implementation_sha256": {name: bench.sha256(ROOT / name) for name in created + ["main.py"]},
        "report_sha256": bench.sha256(report_path), "evidence_sha256": bench.sha256(ROOT / "results/phase10/validation.json"),
        "test_xml_sha256": bench.sha256(ROOT / "docs/phase10_tests.xml"), "optional_loop_difference_percent": delta,
        "database_size_bytes": evidence["database_size_bytes"], "phase11_started": False,
    }
    bench.write_new_json(ROOT / "docs/phase10_validation.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
