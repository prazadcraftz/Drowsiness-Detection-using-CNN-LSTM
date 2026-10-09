# Local reproducibility guide

Run commands from the repository root on Windows. Canonical Phase 11–16 experiments used **Windows 11 x64, Python 3.12.14, CPU-only PyTorch 2.14.0+cpu**, NumPy 2.5.3, OpenCV-contrib-python 5.0.0.93, and MediaPipe 1.0.1. The pinned [runtime](../requirements.txt) and [test](../requirements-dev.txt) requirements record the full dependency set. The historical `.venv-baseline` environment is used by the commands below; a fresh Python 3.12 environment can instead install `requirements-dev.txt`.

```powershell
.\.venv-baseline\Scripts\python.exe -B main.py --help
.\.venv-baseline\Scripts\python.exe -B tools/query_sessions.py --help
.\.venv-baseline\Scripts\python.exe -B -m pytest -q
```

The final Phase 16 regression record has 575 passing tests. The Phase 17 report records the current total after documentation checks. Tests do not replace ML evaluation.

## Artifact map

| Artifact | Role |
|---|---|
| [`saved_model.pt`](../saved_model.pt), [`norm.npy`](../norm.npy) | Historical production legacy profile; frozen compatibility assets |
| [Phase 11 clean checkpoint metadata](../models/phase11/cnn_lstm_clean_split.json) | Clean CNN-LSTM16 train/validation/test provenance; separate from production asset |
| [Phase 15 LSTM-only32 checkpoint](../models/phase15/lstm_only_w32_full_development.pt) and [normalizer](../models/phase15/lstm32_dev_norm.npy) | Development refit loaded by explicit `lstm32` profile |
| [Phase 16 legacy parity](../results/phase16/legacy_parity_final/result.json) | 699-window frozen-output comparison |
| [Phase 16 candidate parity](../results/phase16/candidate_parity.json) | 64-development-tensor direct/integrated comparison |
| [Phase 16 matched benchmark](../results/phase16/benchmark_matched.json) | Local CPU stage and frame-loop samples |
| [Phase 11 split and evaluation report](PHASE11_LEAKAGE_SAFE_EVALUATION.md) | Video-independent baseline protocol and metrics |

The Phase 11 training selected a validation checkpoint from a fixed recipe without loading test arrays into the training loop. Phase 15 fixed seed **1501**, unweighted loss, CPU float32, deterministic Torch algorithms, eight intra/inter-op threads, no DataLoader workers, and **13 epochs**, the median of its nine development CV best epochs. On the recorded environment, a fresh-process repeat matched weights/history/logits bit-for-bit. This is evidence for those runs, **not a promise of bit identity on arbitrary hardware or library versions**. The [Phase 15 report](PHASE15_FINAL_CANDIDATE.md) and metadata carry hashes, source identities, and the recipe.

## Safe parity and demo practice

The existing [legacy parity record](../results/phase16/legacy_parity_final/result.json) and [candidate parity record](../results/phase16/candidate_parity.json) are the canonical production parity evidence. The full pytest command above exercises the parity assertions without replacing those records. To run the standalone parity tools, inspect `--help`/tool source first and pass a **fresh output path** wherever supported; never point them at `results/phase11` through `results/phase16` or canonical `docs/phase*_*.xml` destinations. Several historical scripts intentionally refuse an existing output, and others may overwrite a named file; a separate copy or new scratch path is appropriate for historical experiment reproduction.

For a noncanonical local-video demo, use the `--source`, `--model-profile`, `--no-display`, `--no-audio`, `--output`, and `--sqlite` examples in the [README](../README.md). Put generated video/DB files under a fresh `.cache/` name, or another scratch location. `--output` refuses an existing file; SQLite should also use a new path to avoid mixing sessions with prior demonstrations. Do not use the Phase 11 held-out videos to evaluate the candidate. The [testing guide](TESTING.md) explains what each gate verifies.
