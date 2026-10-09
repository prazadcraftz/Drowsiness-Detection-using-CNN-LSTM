# Interview pitch and technical Q&A

Use the wording as a speaking guide, not a memorized script. The [claim registry](CLAIMS.md) and [limitations](LIMITATIONS.md) define what the evidence supports.

## 30-second version

“I built a local driver-monitoring prototype that uses MediaPipe facial landmarks and a temporal PyTorch model rather than judging one frame at a time. It samples face observations, smooths class probabilities, and turns them into driver states, events, a dashboard, and optional SQLite records. The key research correction was splitting raw videos before making overlapping windows and fitting normalization only on training videos. Under that video-independent protocol, the **clean CNN-LSTM16 baseline** achieved **94.19% window-level test accuracy and 0.9418 macro-F1** on 27 held-out videos. It is a prototype, not a validated road-safety device.”

## 60-second version

“The problem was to reason about eye closure and yawning over time without treating every single frame as a reliable decision. MediaPipe gives landmarks; I calculate five eye, mouth, and pose features, accept observations at a 20 Hz target, and feed a rolling sequence to a PyTorch model. Smoothed probabilities go into a deterministic seven-state policy, which tracks events and can trigger local alerts. The historical CNN-LSTM16 profile remains the default for compatibility. I also compared model architectures and window lengths on development data and selected an opt-in LSTM-only32 candidate at **0.9498 mean development cross-validation macro-F1**. Separately, I rebuilt the evaluation protocol because random overlapping-window splits and whole-corpus normalization could leak information. The **clean CNN-LSTM16 baseline**, not the new candidate, reached **94.19% window-level held-out test accuracy** on 27 videos. The candidate still needs an independent external test.”

## Two-minute technical walkthrough

1. **Input and vision.** `VideoSource` reads a webcam or local video with source timestamps. MediaPipe runs on every decoded frame. When a face is found, EAR-R, EAR-L, MAR, phi, and theta are extracted in the saved model's historical feature contract.
2. **Temporal input.** A sampler targets 20 accepted face observations/s. The profile-specific normalizer transforms each five-value vector. A bounded buffer holds 16 observations for the default CNN-LSTM or 32 for the opt-in LSTM-only model. LSTM32 needs **32 accepted observations** for its first prediction, then advances by one accepted observation per prediction. A face gap over 0.5 seconds resets the buffer; missing samples are never filled.
3. **Model and policy.** The classifier outputs Normal, Yawning, and Eye Close probabilities. A bounded five-prediction average stabilizes evidence. A separate deterministic policy moves among seven driver states using engineering thresholds and persistence timers. Session metrics track yawns, eye closure, drowsiness, alerts, and no-face periods; the dashboard shows raw/smoothed predictions and state. Optional nonblocking audio and SQLite event/aggregate storage stay local.
4. **Evaluation and proof of compatibility.** Phase 11 fixed video-level train/validation/test assignments before windows and fitted normalization on train only. Its clean CNN-LSTM16 checkpoint achieved **94.1909% window test accuracy / 0.941808 macro-F1** across 27 held-out videos. Phase 12–15 development studies selected LSTM-only32 with **0.949829 mean development CV macro-F1**, which is not a candidate final-test score. The final suite passed **578 tests**, and frozen numerical replay matched **699/699** legacy windows plus **64/64** candidate integration sequences.

## Thirty concise technical questions

1. **Why MediaPipe rather than a raw-image CNN?** Landmark extraction lets the classifier consume five compact, inspectable eye, mouth, and pose features on a local CPU. It is a design trade-off; no controlled raw-image-CNN comparison was run.
2. **Why temporal modeling?** Eye closure and yawning have duration. A sequence provides context a single frame lacks, while the policy separately enforces event persistence.
3. **Why an LSTM?** It can summarize an ordered sequence of five-dimensional observations. The development architecture study compared it with CNN-LSTM, CNN-GRU, and CNN-BiLSTM variants rather than assuming it was best.
4. **Why did LSTM-only outperform CNN-LSTM?** In this dataset and fixed recipe, the LSTM-only variant had higher development mean macro-F1. One plausible interpretation is that the input is already low-dimensional and the CNN compresses temporal resolution; this is not universal proof that convolution hurts.
5. **Why choose a 32-observation window?** In the grouped development study, LSTM-only N=32 reached **0.949829 mean macro-F1**, versus **0.932276** at N=16, with a positive paired difference in **9/9** matched runs. That is model-selection evidence from development data.
6. **What is the downside of N=32?** At 20 accepted observations/s it needs **1.60 seconds nominal accumulation** before its first prediction, versus 0.80 seconds for N=16, and its measured model-forward P50 was higher on the tested CPU.
7. **What was wrong with the original evaluation?** It randomly split overlapping windows and fitted normalization on the whole corpus. Those choices can share source information and evaluation-set extrema with training.
8. **What is data leakage here?** Near-identical windows from the same video can cross split boundaries; whole-corpus normalization also transfers test-set statistics into preprocessing. Both can make generalization look better than it is.
9. **Why split videos before windows?** All windows from one source then stay in the same split, preventing overlap from putting shared observations in both training and evaluation.
10. **Why train-only normalization?** The transform must be learned without seeing validation/test extrema. The exact train-fitted transform is then applied to those held-out splits.
11. **Why macro-F1 as well as accuracy?** Macro-F1 gives each class equal weight and exposes a class that accuracy could hide. Window counts and class recall still matter, so both metrics and supports are reported.
12. **Why reject class weighting?** The Phase 12 development study did not justify adopting it under its quality criteria; the selected candidate retained unweighted cross-entropy. It was not rejected using a final test result.
13. **Why not claim subject independence?** Reliable person IDs were unavailable. Different videos may contain the same subject or scene, even though source videos are isolated across splits.
14. **Why no candidate test score?** LSTM32 was selected from development studies after the Phase 11 cohort had already been used for the baseline. That cohort is not pristine for a new final candidate; no candidate inference on it was performed in Phases 15/16.
15. **Why keep legacy as default?** Backward compatibility. The historical CNN-LSTM16 path reproduces **699/699** frozen outputs exactly; LSTM32 is an explicit opt-in recommendation.
16. **How do you handle a missing face?** The frame is recorded as `NO_FACE`; it contributes no invented feature vector. Short gaps retain temporal context and can pause evidence events without counting missing time.
17. **What happens after a long face gap?** A gap over **0.5 seconds** clears the temporal buffer and policy evidence. LSTM32 then needs 32 fresh accepted observations before another prediction.
18. **How does smoothing work?** The default policy averages up to five recent class-probability vectors after warm-up. That is a prediction count, not a fixed five-second window.
19. **What is the state machine?** It is deterministic logic outside the network: WARMING_UP, NO_FACE, ATTENTIVE, FATIGUE, SUSPECTED_DROWSY, DROWSY, and ALERT, with hysteresis and persistence rules.
20. **How are alerts generated?** Sustained smoothed eye evidence reaches DROWSY after 1.0 second and logical ALERT after another 1.5 seconds under frozen engineering defaults. Audio requests are local and nonblocking; the thresholds have no clinical validation.
21. **What is stored in SQLite?** Optional local sessions, event episodes, final aggregate session metrics, and model/configuration provenance. There are no raw-frame or per-frame prediction tables.
22. **Why avoid raw frames in SQLite?** Session/event questions can be answered from compact records without duplicating video content in the database. This reduces stored detail but is not an absolute privacy guarantee.
23. **What dominates runtime?** MediaPipe detection was about **83% of summed raw processing time** in the measured Phase 8/9 local-video workload. The share is machine and workload specific.
24. **Why didn't a smaller model always lower latency?** Parameter count alone does not determine kernel costs, memory access, or stage scheduling. In the matched Phase 16 benchmark, 209,667-parameter LSTM32 had a **3.749 ms** model-forward P50 versus **2.142 ms** for the 246,595-parameter legacy model, while vision dominated raw processing.
25. **How was backward compatibility validated?** Frozen development fixtures replayed **699 legacy windows/classes** with exact numerical outputs, and final parity also covered policy/session and dashboard traces.
26. **Why exactly 699 windows?** They are the frozen set yielded by six development-video replay fixtures, not a sample size chosen to claim statistical confidence.
27. **How was candidate numerical parity checked?** The integrated profile replayed **64 saved Phase 15 development verification sequences** against direct candidate inference; tensors, logits, probabilities, and classes matched with maximum difference **0.0**.
28. **What would you improve next?** Gather independent subject-labeled data and event-onset annotations, validate externally, then revisit preprocessing and policy calibration in a separately retrained and evaluated line.
29. **What are the largest limitations?** Small custom data, no subject IDs or event-onset labels, retained legacy feature quirks, engineering thresholds, no pristine candidate final test, and machine-specific offline latency.
30. **How would you validate road safety?** Prospectively define drivers, environments, events, alert outcomes, and failure criteria; collect independent labeled data; measure event-level errors and latency under real camera/vehicle conditions, with appropriate safety oversight before any safety claim.

## Challenges faced: concise answers

| Challenge | Answer |
|---|---|
| Dataset leakage | “I moved the split boundary to raw videos and fitted the normalizer only on train data, then checked source/observation identities for overlap.” |
| Legacy preprocessing | “I preserved the audited color and pose semantics for saved-model parity; correcting them would require a separate retrain and evaluation.” |
| Missing faces | “I do not synthesize observations. Short gaps pause observed event duration; gaps over 0.5 seconds reset the temporal context.” |
| CPU latency | “Stage measurements showed MediaPipe dominated raw time, so I accepted only a matched, pixel-exact dashboard optimization and kept its 8.58% claim scoped to render-application P50.” |
| Backward compatibility | “I kept `legacy` as the CLI default and used 699 frozen windows plus policy/dashboard traces as exact replay gates.” |
| Missing subject IDs | “I describe the split as video-independent and leave subject-generalization claims open for new data.” |

## Why this project?

“It is a useful intersection of machine learning and systems engineering: eye and mouth cues develop over time, while a usable local application also needs reliable timestamps, missing-face behavior, bounded state, alerts, and measurable latency. Building both the model evaluation and runtime made the trade-offs visible.” This describes the project's appeal without inventing a personal history.

## What would you improve?

“I would first collect subject-labeled, independent videos with event-onset and outcome annotations across more lighting, pose, and occlusion conditions. I would run an external candidate evaluation and physical webcam/driver tests. With those labels, I could calibrate probabilities and policy thresholds. I would test corrected color/pose preprocessing only with a **separately retrained model**, preserving the current compatibility profile.” None of those steps has been completed here.

## Explain the three commonly quoted metrics

- **94.19% window accuracy:** about 94 of 100 held-out windows were classified correctly by the **clean Phase 11 CNN-LSTM16 baseline**; the actual support was 482 windows from 27 videos. Correlated windows are not 482 independent drivers.
- **0.9418 macro-F1:** the same clean baseline's held-out test combines precision and recall and averages the three class F1 values equally. It does not measure event-onset detection.
- **0.9498 mean development CV macro-F1:** the LSTM-only32 configuration's mean across grouped development folds/seeds. It supported selection, but is **not a final test** and cannot be treated as a directly comparable independent estimate against the baseline's held-out number.

## Architecture and system-design explanation

The historical CNN-LSTM16 applies **Conv1d → Conv1d → pooling → LSTM → dense classifier** to the five-channel temporal input. The selected LSTM-only32 applies **five features → two-layer LSTM → dense classifier**, allowing the recurrent block to read all 32 observation positions directly. With only five engineered channels, the CNN's temporal compression may have removed useful timing detail in this dataset; that is an interpretation of the development result, not a universal explanation.

The modular boundaries are video source, detector/features, profile-specific normalization/model, bounded sampler/buffer, deterministic smoothing/policy, session/events/audio, dashboard, and optional SQLite. Keeping policy outside the model makes thresholds and episodes testable. Fixed-size temporal and smoothing buffers avoid retaining a per-frame history. Nonblocking audio and local persistence keep the frame loop's responsibilities explicit; checkpoint/normalizer hashes in provenance and frozen parity fixtures support reproducibility and compatibility.

**Testing story:** ML evaluation asks how a selected model behaves on genuinely held-out sources; software regression asks whether the implementation preserves specified outputs and state transitions. **578 tests**, **699 exact legacy windows**, and **64 exact candidate parity tensors** support software correctness on those fixtures. They do not turn into an ML accuracy claim. [Evaluation](ML_EVALUATION.md), [testing](TESTING.md), and [performance](PERFORMANCE.md) provide the underlying records.
