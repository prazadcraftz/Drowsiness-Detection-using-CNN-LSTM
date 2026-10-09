# Phase 18: final portfolio package

**Scope:** communication artifacts only. No model fitting, new ML experiment, Phase 11 candidate access, threshold tuning, or runtime behavior change.

1. **Files created:** [portfolio brief](PORTFOLIO_BRIEF.md), [resume bullets](RESUME_BULLETS.md), [GitHub/application copy](GITHUB_DESCRIPTION.md), [interview pitches and Q&A](INTERVIEW_PITCH.md), [STAR stories](STAR_STORIES.md), this report.
2. **Files modified:** the root [README](../README.md) received one navigation paragraph pointing to the new materials. No other pre-Phase-18 file was intentionally edited.
3. **Resume versions:** two-, three-, and four-bullet variants are in [RESUME_BULLETS.md](RESUME_BULLETS.md). They distinguish the clean baseline test from candidate development CV.
4. **ATS version:** uses Python, PyTorch, OpenCV, MediaPipe, LSTM/CNN-LSTM, SQLite, pytest, cross-validation, and latency benchmarking without keyword stuffing.
5. **AI/ML version:** emphasizes leakage-safe source splitting, train-only normalization, macro-F1, ablations, window studies, fixed-seed refit, and model-selection limits.
6. **Software-engineering version:** emphasizes the reusable engine, dual profiles, bounded temporal state, CLI, SQLite/provenance, parity, and tests.
7. **GitHub description:** short description, factual topics, pinned-repo copy, five portfolio highlights, and three length-limited one-liners are in [GITHUB_DESCRIPTION.md](GITHUB_DESCRIPTION.md).
8. **Interview pitches:** 30-second, 60-second, and two-minute walkthrough versions are in [INTERVIEW_PITCH.md](INTERVIEW_PITCH.md).
9. **Interview Q&A:** the same guide covers 30 required technical questions, challenge answers, motivation without invented personal history, improvements, metrics, architecture, system design, and the ML-evaluation/software-testing distinction.
10. **STAR stories:** four actual project stories cover leakage correction, reusable-engine extraction, measured bottleneck/dashboard optimization, and controlled LSTM-only32 selection.
11. **Application summaries:** [GitHub/application copy](GITHUB_DESCRIPTION.md) provides LinkedIn copy and ≤50-, ≤100-, and ≤150-word form answers.
12. **Final tech stack:** Python, PyTorch, OpenCV, MediaPipe, NumPy, scikit-learn, Python `sqlite3`/SQLite, and pytest; SciPy and Matplotlib supported analysis and plots.
13. **Claim audit:** passed against [claims](CLAIMS.md), [ML evaluation](ML_EVALUATION.md), [performance](PERFORMANCE.md), and [limitations](LIMITATIONS.md). Every 94.19%/0.9418 test reference in the new package names the clean CNN-LSTM16 baseline, and every 0.9498 candidate reference labels **development cross-validation**. No new subject-independence, clinical-validation, or universal-real-time claim appears. Phase 9's 8.58% remains a dashboard/application figure.
14. **Preservation:** SHA256 audit found **4,759/4,760** pre-Phase-18 files byte-identical; the sole expected edit is README navigation. Runtime, evaluation code, checkpoint, normalizer, Phase 16 parity evidence, and Phase 17 records are unchanged.
15. **Regression and documentation validation:** **578 passed**, 0 failed in the full pytest suite (310.41 s); focused Phase 17 documentation tests passed **3/3**. Across **37 Markdown files**, **198/198** relative links resolved with exact casing. The portfolio brief is **714 words**; three one-liners are **93/142/223 characters**, and application answers are **43/82/136 words**, all within their limits. The new portfolio CLI example parsed against the actual application parser without executing inference.
16. **Final recommended resume bullets:**

    - Built a local CPU-only driver-monitoring pipeline with MediaPipe facial landmarks, PyTorch temporal models, 20 Hz timestamp-aware sampling, missing-face recovery, smoothed state tracking, dashboard alerts, and optional SQLite session records.
    - Rebuilt evaluation with video-level splits and train-only normalization: the **clean CNN-LSTM16 baseline** reached **94.19% window-level test accuracy and 0.9418 macro-F1** on 27 held-out videos; architecture/window studies selected a **209,667-parameter LSTM-only32 candidate** with **0.9498 mean development cross-validation macro-F1**.

17. **Decision:** **PHASE 18 PASS.** This is the final requested packaging phase; stop here.

The [Phase 17 report](PHASE17_DOCUMENTATION.md) and [claim registry](CLAIMS.md) remain the evidence boundary. The selected LSTM-only32 candidate has **no pristine external final-test estimate**.
