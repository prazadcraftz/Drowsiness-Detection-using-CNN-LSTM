# Claim registry

Use the allowed wording with its stated cohort and measurement boundary. The forbidden examples illustrate an attribution or scope that the evidence does not support.

| Claim | Evidence | Allowed wording | Forbidden or overstated wording |
|---|---|---|---|
| Clean CNN-LSTM16 test | [Phase 11](PHASE11_LEAKAGE_SAFE_EVALUATION.md): 482 windows from 27 held-out videos | “The clean CNN-LSTM16 baseline achieved 94.1909% video-independent window-level test accuracy and 0.941808 macro-F1.” | “The final LSTM32 model achieved 94.19% test accuracy.” |
| Clean baseline video result | [Phase 11](PHASE11_LEAKAGE_SAFE_EVALUATION.md): 27 videos | “The clean CNN-LSTM16 baseline classified all 27 held-out videos in this cohort.” | “The system is 100% accurate for new drivers.” |
| Candidate selection quality | [Phase 14](PHASE14_WINDOW_LENGTH.md) and [Phase 15](PHASE15_FINAL_CANDIDATE.md) | “LSTM-only32 achieved 0.949829 mean **development cross-validation** window macro-F1.” | “LSTM32 achieved 95% final test F1.” |
| Candidate external generalization | [Candidate access audit](../results/phase16/test_access_audit.json) | “The LSTM32 candidate has no pristine external final-test estimate.” | “The Phase 11 held-out metrics validate LSTM32.” |
| Dashboard optimization | [Phase 9](PHASE9_CPU_OPTIMIZATION.md): matched render-application P50 | “Dashboard optimization reduced matched render-application P50 from 34.069 to 31.146 ms (8.58%).” | “Model inference was optimized by 8.6%” or “inference was optimized by 35%.” |
| Local CPU performance | [Phase 16](PHASE16_RUNTIME_INTEGRATION.md): matched offline-video benchmark | “The local CPU-only, headless Phase 16 raw-pipeline P50 was 28.460 ms for legacy and 30.088 ms for LSTM32 on the tested machine.” | “Real-time on all devices” or “sustained webcam FPS is 33.” |
| Exact regression | [Legacy parity](../results/phase16/legacy_parity_final/result.json) and [candidate parity](../results/phase16/candidate_parity.json) | “699/699 frozen legacy outputs and 64/64 integrated candidate verification outputs matched their references exactly.” | “Exact parity proves drowsiness detection accuracy.” |
| Local storage/privacy | [Phase 10](PHASE10_SQLITE.md) | “Designed for local processing; optional SQLite stores sessions, events, aggregates, and provenance without raw frames.” | “Completely secure/private” or “no sensitive data is stored.” |
| Policy alerts | [State machine](STATE_MACHINE.md) | “Engineering thresholds convert smoothed evidence into logical states and alerts.” | “Clinically validated to detect dangerous drowsiness.” |

The historical production `legacy` checkpoint, the separately trained clean Phase 11 CNN-LSTM16 baseline, and the development-selected Phase 15/16 LSTM32 candidate are three distinct evidence objects, even where architectures match. [Limitations](LIMITATIONS.md) provides the conditions behind every claim.
