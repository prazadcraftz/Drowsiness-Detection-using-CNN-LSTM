# ML evaluation and selection

The project has two different evaluation stories. The **clean Phase 11 CNN-LSTM16 baseline** has one video-independent held-out test result. The **LSTM-only32 candidate** was selected from development cross-validation and fitted on all development videos; it has **no pristine external final-test estimate**. The historical production `legacy` checkpoint is retained for runtime compatibility and should not be conflated with the separately trained Phase 11 clean baseline checkpoint.

## Why the historical result needed replacement

The older workflow split randomly after extracting overlapping windows and fitted normalization on the whole corpus. Windows from one source can share observations, so a random window split can put near-duplicates across train and evaluation. Whole-corpus normalization also lets evaluation-set extrema influence model inputs. The older workflow lacked the fixed, auditable source manifest needed to reproduce a clean split. Its displayed metrics are therefore not used as the clean generalization claim. [Phase 11](PHASE11_LEAKAGE_SAFE_EVALUATION.md) records the audit and revised protocol.

## Clean Phase 11 baseline

Phase 11 inventoried and hashed raw videos, assigned **whole videos before constructing windows**, and fixed a seeded split manifest. It fitted the normalizer on **train videos only**, then applied it to validation and test. Window source/observation identities were checked for cross-split overlap. The split had 126 train, 27 validation, and **27 test videos**. A fixed CNN-LSTM16 recipe selected its checkpoint by validation macro-F1; the test cohort was evaluated once after selection. This is **video-independent**, not subject-independent: reliable subject IDs were unavailable, so the same person or scene could occur in more than one video.

| Clean CNN-LSTM16 Phase 11 **test** metric | Result | Support |
|---|---:|---:|
| Window accuracy | **94.1909%** | 482 windows |
| Window macro-F1 | **0.941808** | 482 windows |
| Video accuracy | **100%** | 27 videos |
| Video macro-F1 | **1.000000** | 27 videos |

Video predictions averaged window probabilities within each held-out source, so each video contributed one vote. **Twenty-seven videos are a small test cohort**; 100% on it should not be projected to new drivers, cameras, or environments. Windows are correlated within videos. Labels describe whole-video actions; they do not provide event onset, duration, or frame-level truth. Thus these metrics do not validate YAWN/EYE_CLOSURE/ALERT event precision or onset latency. See the [detailed Phase 11 report](PHASE11_LEAKAGE_SAFE_EVALUATION.md) and [limitations](LIMITATIONS.md).

## Development-only model selection, Phases 12–15

Phase 12 tested class balancing and found no support for adopting it, so the final candidate uses unweighted cross-entropy. Phase 13 compared temporal architectures; Phase 14 studied window length under grouped development cross-validation. After the Phase 14 development results were observed, Phase 15 froze a quality-first rule, selected **LSTM-only, 32 observations, training stride 16**, and refitted one candidate on all **153 development videos**. The rule was fixed before Phase 15 training, **not** predeclared before seeing Phase 14 results. The full-development fit used its own development-only normalizer and 13 fixed epochs; no test loader or test-driven checkpoint selection was used.

| LSTM-only32 **development cross-validation** metric | Mean |
|---|---:|
| Window macro-F1 | **0.949829** |
| Window accuracy | **0.949755** |
| Macro recall | **0.949599** |
| Worst-class recall | **0.910781** |

These development metrics supported the candidate choice. They are **not final-test metrics** and are not directly comparable as an independent head-to-head test against the Phase 11 baseline. The 27 Phase 11 test videos were previously used to evaluate the baseline, so they no longer form a pristine untouched final cohort for a newly selected candidate. Phase 15/16 recorded **zero candidate inference passes, cache reads, or video reads** on them. No external candidate generalization estimate is claimed. See [Phase 12](PHASE12_CLASS_BALANCING.md), [Phase 13](PHASE13_ARCHITECTURE_ABLATIONS.md), [Phase 14](PHASE14_WINDOW_LENGTH.md), [Phase 15](PHASE15_FINAL_CANDIDATE.md), and the [claim registry](CLAIMS.md).
