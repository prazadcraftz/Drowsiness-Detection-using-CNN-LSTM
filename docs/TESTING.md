# Correctness and regression evidence

The test suite checks contracts and integration behavior; a passing test count is **not** evidence of ML generalization or event accuracy.

| Gate | What it establishes | Evidence |
|---|---|---|
| Frozen legacy replay | **699/699** development-fixture windows and classes match the saved historical output; maximum numerical difference **0.0** across raw/normalized features, sequences, tensors, logits, and probabilities. Final Phase 16 replay also checked policy/session traces, landmarks, and rendered pixels. | [Final legacy parity](../results/phase16/legacy_parity_final/result.json) |
| Candidate direct/integrated parity | All **64/64** frozen Phase 15 development verification tensors match direct model inference versus integrated engine, with **0.0** maximum numerical difference and identical class IDs. | [Candidate parity](../results/phase16/candidate_parity.json) |
| Phase 15 repeat fit | On the recorded CPU/software, same inputs/seed/recipe reproduced weights, training history, and 64 verification logits bit-for-bit. | [Phase 15 report](PHASE15_FINAL_CANDIDATE.md) |
| Phase 16 full regression | **575 tests passed** across Phases 2–16, including source-access guards, warm-up/reset/stride, state/event behavior, dashboard and SQLite checks. | |
| Source and artifact guards | Candidate work used development sources and sealed candidate/legacy assets; Phase 16 recorded zero candidate held-out inference, cache reads, and video reads. | [Access audit](../results/phase16/test_access_audit.json) |

The offline verification set contains one repeat-last-padded window from model development; it was supplied directly to `infer_sequence`. The live runtime never manufactures missing face observations. Exact parity means numerical compatibility **on these frozen fixtures**, not correctness for every camera, lighting condition, or driver.

Run the current suite from the repository root in the pinned environment:

```powershell
.\.venv-baseline\Scripts\python.exe -B -m pytest -q
```

Or use the newly created `.venv` from the [README](../README.md) and install [test dependencies](../requirements-dev.txt). Avoid writing test XML to an existing canonical `docs/phase*_tests*.xml` file; use a fresh path if an XML artifact is needed. [Reproducibility](REPRODUCIBILITY.md) covers evidence and safe reruns.
