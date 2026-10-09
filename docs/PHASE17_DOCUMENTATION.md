# Phase 17: documentation and packaging report

**Scope:** repository documentation and focused documentation-only tests. No retraining, model/normalizer replacement, inference or threshold change, new held-out candidate evaluation, cloud deployment, or Phase 18 work.

1. **Files created:** [`ARCHITECTURE.md`](ARCHITECTURE.md), [`STATE_MACHINE.md`](STATE_MACHINE.md), [`SQLITE.md`](SQLITE.md), [`ML_EVALUATION.md`](ML_EVALUATION.md), [`PERFORMANCE.md`](PERFORMANCE.md), [`TESTING.md`](TESTING.md), [`LIMITATIONS.md`](LIMITATIONS.md), [`CLAIMS.md`](CLAIMS.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md), [`ENGINEERING_SUMMARY.md`](ENGINEERING_SUMMARY.md), this report, `phase17_before_hashes.json`, `phase17_preservation.json`, and `tests/test_phase17_documentation.py`.
2. **Files modified:** the pre-existing root `Readme.MD` was rewritten and case-normalized to [`README.md`](../README.md). Of 4,745 pre-Phase-17 files, **4,744 are byte-identical**; the README is the one expected documentation change. Earlier unrelated worktree modifications visible before this phase were preserved as-is.
3. **README structure:** description, factual highlights, Windows quickstart, model profile comparison, authentic preview/demo, repository map, and links to deeper material.
4. **Architecture:** one shared runtime, Mermaid flow, exact five-feature input, profile-specific normalization, 20 Hz target, stride-1 runtime inference, face-gap reset, and retained legacy color/pose semantics.
5. **State machine:** seven states, frozen engineering thresholds/timers, five event kinds, and the distinct ALERT-versus-evidence short-`NO_FACE` identity rule.
6. **Evaluation:** historical leakage problem, source-first split, train-only normalization, clean Phase 11 baseline test, development-only Phase 12–15 selection, and absent pristine candidate final test.
7. **Performance:** Phase 8 baseline, Phase 9 matched dashboard gain, Phase 16 matched profile CPU boundaries, and explicit offline/hardware/scope caveats.
8. **SQLite:** optional stdlib local storage, schema/provenance overview, no raw-frame/per-frame-prediction tables, and read-only SQL inspection examples.
9. **Testing:** 699-window legacy and 64-tensor candidate exact-parity gates, deterministic checks, source-access guards, and full suite context.
10. **Limitations:** clinical, dataset, subject, label, candidate test, preprocessing, threshold, detector, warm-up, privacy, and performance boundaries are prominent.
11. **Claim registry:** allowed and forbidden metric wording separates the historical runtime asset, clean baseline, and selected candidate.
12. **Reproducibility:** Windows/Python/dependency versions, deterministic training contract, artifact map, safe tests, and scratch-output guidance.
13. **Interview guide:** concise answers to temporal, feature, leakage, balancing, architecture, window, default, latency, and threshold questions.
14. **Engineering summary:** Phase 1–16 decision chain and four resume-safe candidate bullets; no resume file was edited.
15. **Demo commands:** verified legacy/candidate webcam, local-file, headless, audio-disabled, saved annotated output, SQLite, and read-only query examples. The preview is an authentic development-video frame, not a performance screenshot.
16. **Link/command validation:** **145/145** relative Markdown links in the root README and existing/new `docs/**/*.md` resolved in the Phase 17 audit. Both application and query CLI `--help` executed; focused tests parsed every documented `python main.py` example against the actual parser. Direct SQL examples were executed read-only against the existing version-1 Phase 16 database.
17. **Documentation tests:** `tests/test_phase17_documentation.py` checks required files/links, CLI examples and help, and clean-baseline/candidate wording.
18. **Full regression:** **578 passed**, 0 failed; 575 Phase 2–16 tests plus three Phase 17 documentation tests, using the pinned Python 3.12 environment. The full suite completed in 392.00 s.
19. **Preservation:** a SHA256 snapshot was taken before edits; `phase17_preservation.json` records the one expected README change and **4,744 unchanged** prior files. Runtime, model, checkpoint, normalizer, source videos, and historical evidence are included in that byte audit. Canonical Phase 16 **699/699 legacy** and **64/64 candidate** parity records retain their prior hashes.
20. **Decision:** **PHASE 17 PASS** after link/command checks, full regression, and preservation audit. Stop here; Phase 18 is not begun.

Canonical earlier evidence remains [Phase 16](PHASE16_RUNTIME_INTEGRATION.md), [Phase 15](PHASE15_FINAL_CANDIDATE.md), and [Phase 11](PHASE11_LEAKAGE_SAFE_EVALUATION.md). The candidate's **0.949829 development cross-validation macro-F1** is not a final test result. The **94.1909% / 0.941808 test metrics** are for the clean Phase 11 CNN-LSTM16 baseline only.
