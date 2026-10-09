# Phase 9 CPU experiments

Each serious experiment has independent six-source correctness evidence and five fresh paired timing repetitions. Rejected and inconclusive attempts remain recorded. No slow run was discarded.

Gate defined before latency measurement: exact landmark/raw/policy parity; pooled raw/application P50 improvement ≥1%; no pooled P95 regression; ≥80% paired source median wins; mean process CPU ≤110% of paired baseline. Dashboard uses overlay/application boundaries. ELIGIBLE is provisional until combined production, unchanged prior tests and boundedness pass. These are conservative engineering criteria, not statistical significance claims.

The first 217-job combined comparison failed the original gate: headless raw P50 reduction 0.27% with 14/20 median wins; render raw 0.52% with 13/20 wins and a 0.29% P95 regression. Render app improved 8.79% P50/8.22% P95, but this did not justify accepting the detector change. Complete measurements, source snapshots and original failed gates remain preserved.

After restoring detector and engine byte-for-byte, a separate cache-only protocol was declared before fresh timing: exact raw/policy/pixel parity, original engine/detector source hashes, render overlay/application P50 improvement >=1%, no target P95 regression, >=80% paired application median wins, CPU <=110% and processing/peak RSS <=110%. Untouched raw/headless controls allow at most 5% P50/P95 regression; no headless inference optimization is claimed. This protocol cannot accept the old detector candidate because it requires restoration and an explicit cache-only methodology marker.

Normal_1 and Yawning_60 are the two initial MediaPipe representatives. Thread sweeps use Normal_1 plus all 699 prebuilt tensors. Dashboard is a render-only representative experiment. Both failed combined and final cache-only comparisons use all four Phase 8 sources and both modes. All warm-up rules, hashes and raw samples are in each evidence JSON.

| Candidate | Baseline P50/P95 ms | Candidate P50/P95 ms | Parity | Baseline/candidate CPU | RSS/peak MiB baseline/candidate | Decision |
|---|---|---|---|---|---|---|
| contiguous_tensor | 29.609/30.262 | 29.357/30.400 | exact 699/699 | 307.2/307.7% | 314.7/314.5 (peak 315.6/315.9) | INCONCLUSIVE |
| dashboard_cache | 4.496/4.800 | 1.621/1.702 | exact 699/699 | 288.0/293.9% | 324.2/325.2 (peak 325.3/326.9) | ACCEPTED |
| interop1 | 30.931/32.145 | 30.906/33.862 | exact 699/699 | 256.8/249.0% | 314.6/314.4 (peak 315.3/315.5) | INCONCLUSIVE |
| interop2 | 30.064/31.534 | 30.050/31.456 | exact 699/699 | 272.1/257.8% | 314.7/314.3 (peak 315.7/316.3) | INCONCLUSIVE |
| intra1 | 29.678/30.376 | 30.741/37.660 | exact 699/699 | 297.2/99.1% | 315.2/312.6 (peak 316.3/313.2) | REJECTED |
| intra2 | 29.469/30.629 | 29.640/32.974 | exact 699/699 | 268.9/174.9% | 314.9/313.1 (peak 315.6/313.8) | REJECTED |
| intra4 | 30.401/31.288 | 30.605/31.910 | exact 699/699 | 255.9/221.6% | 314.8/314.0 (peak 316.6/314.8) | INCONCLUSIVE |
| no_blendshapes | 27.619/30.127 | 27.369/30.029 | exact 699/699 | 250.6/251.2% | 314.7/314.5 (peak 316.6/315.2) | INCONCLUSIVE |
| no_matrices | 28.221/31.826 | 27.749/29.568 | exact 699/699 | 250.4/257.5% | 314.5/313.9 (peak 317.2/314.8) | INCONCLUSIVE |
| no_unused_outputs | 29.295/31.728 | 28.909/31.132 | exact 699/699 | 279.8/284.9% | 314.7/314.1 (peak 316.0/314.7) | REJECTED |
| video | 29.382/33.288 | 25.538/27.555 | FAILED | 263.3/305.1% | 314.6/314.6 (peak 315.8/317.0) | REJECTED |
| initial_snapshot_probe | — | — | diagnostic/deferred | — | — | REJECTED |
| pose_cache | — | — | exact 699/699 | — | — | INCONCLUSIVE |
| cv1 | — | — | exact 699/699 | — | — | INCONCLUSIVE |
| cv2 | — | — | exact 699/699 | — | — | INCONCLUSIVE |
| cv4 | — | — | exact 699/699 | — | — | INCONCLUSIVE |
| combined_unused_outputs_dashboard | — | — | diagnostic/deferred | — | — | REJECTED |
| tensor_from_numpy_owned | — | — | diagnostic/deferred | — | — | INCONCLUSIVE |
| ring_buffer | — | — | diagnostic/deferred | — | — | INCONCLUSIVE |
| remove_result_copies | — | — | diagnostic/deferred | — | — | REJECTED |
| encoding | — | — | diagnostic/deferred | — | — | INCONCLUSIVE |

## MediaPipe option experiments

| Candidate | Baseline detector P50/P95 ms | Candidate detector P50/P95 ms | Decision |
|---|---|---|---|
| no_blendshapes | 23.036/25.246 | 22.763/25.072 | INCONCLUSIVE |
| no_matrices | 23.578/26.491 | 23.168/24.682 | INCONCLUSIVE |
| no_unused_outputs | 24.492/26.343 | 24.139/26.255 | REJECTED |
| video | 24.667/27.111 | 20.816/22.507 | REJECTED |

## Thread sweep

| Candidate | Intra/inter | Baseline model P50/P95/P99 | Candidate model | Candidate detector | Candidate raw | Candidate app | Decision |
|---|---|---|---|---|---|---|---|
| interop1 | 8/1 | 0.440/2.515/2.890 | 0.388/1.888/2.052 | 25.938/27.993/29.074 | 30.906/33.862/36.399 | 31.251/34.253/36.816 | INCONCLUSIVE |
| interop2 | 8/2 | 0.446/2.162/2.452 | 0.466/2.079/2.314 | 25.272/26.207/26.546 | 30.050/31.456/31.761 | 30.398/31.824/32.108 | INCONCLUSIVE |
| intra1 | 1/8 | 0.386/1.645/1.811 | 0.359/0.401/0.553 | 26.247/32.760/39.353 | 30.741/37.660/44.560 | 31.129/38.096/44.997 | REJECTED |
| intra2 | 2/8 | 0.483/1.972/2.254 | 0.370/2.063/2.480 | 25.205/28.225/31.902 | 29.640/32.974/36.905 | 30.011/33.332/37.405 | REJECTED |
| intra4 | 4/8 | 0.526/2.691/3.516 | 0.343/1.984/2.158 | 25.728/26.723/27.679 | 30.605/31.910/33.101 | 30.978/32.290/33.448 | INCONCLUSIVE |

Rows use raw pipeline latency except dashboard_cache, which uses overlay latency. Detailed MediaPipe/model/application metrics, per-source/per-run medians, mean/std, pooled samples, startup and RSS remain in the JSON; the short table is not a substitute for them.

## Individual purposes and decisions

MediaPipe output switches disable only unused facial expression/effect outputs; IMAGE, one face, CPU asset, .5 confidence defaults and legacy BGR-as-SRGB input remain. no_unused_outputs measures the two previously isolated switches together; its benefit is not attributed independently to both switches. All experiments compare the runtime’s 478×2 x/y landmarks. The combined candidate also has a separate six-source native x/y/z/visibility/presence equality check with both full archives retained. Installed API signature/docstring and file hash are in docs/phase9_mediapipe_options.json. VIDEO receives rounded monotonic source milliseconds but changes tracking/landmarks and is rejected; timing runs explicitly disable frozen verification after recording the failure, and cannot claim parity. LIVE_STREAM callbacks, confidence changes, resizing/ROI and detector skipping were not introduced.

Intra-op 1/2/4 versus 8 and inter-op 1/2 versus 8 are independent fresh-process sweeps; inter-op tests keep intra-op 8. Model-only and raw/app distributions are measured, not inferred from core count. OpenCV 1/2/4 versus 12 have complete exact parity and separate five-repeat head-pose/render probes; probe results determine whether a full pipeline thread experiment is justified. They do not imply MediaPipe thread control.

Intra-op 1 and 2 are rejected for measured full-pipeline P95 regression despite lower process CPU and favorable standalone model timings. Intra-op 4 and inter-op 1/2 remain inconclusive because they did not meet the repeated full-path acceptance gate. No cross-library scheduling or cache cause was established from these measurements.

Head-pose cache reuses only camera/distortion constants for up to four resolutions and retains all solvePnP/Rodrigues/decomposition/arithmetic quirks. Its exact six-source feature gate passed, but it is retained only if worthwhile full-path benefit is established. The fixed 3-D model points were already global. EAR/MAR arithmetic and feature ordering remain untouched.

Tensor probe compares torch.tensor with a fresh owned float32 NumPy array plus torch.from_numpy, with equal tensor values; production ownership/copy guarantees remain. Deque/ring and result-copy changes are not promoted based on tiny diagnostic differences. Snapshot copies retain independent prior-result ownership.

Dashboard cache stores a bounded background and invariant separator for frame shape/dtype, copies it each frame, then resizes the current source and draws all dynamic state/text every frame. Six-source dashboard pixels were exact; resolution/dtype/input ownership tests supplement that check. Encoding remains separate and unchanged.

## Probe distributions

| Probe | Workload | P50 ms | P95 ms | Mean run median | Std run median |
|---|---|---|---|---|---|
| baseline | head_pose | 1.467 | 1.663 | 1.353 | 0.2547 |
| baseline | overlay | 2.930 | 3.860 | 2.625 | 0.8710 |
| baseline | tensor_legacy | 0.0236 | 0.0283 | 0.0177 | 0.0085 |
| baseline | tensor_from_numpy_owned | 0.0046 | 0.0166 | 0.0092 | 0.0057 |
| baseline | snapshot_arrays | 0.0036 | 0.0070 | 0.0047 | 0.0017 |
| cv1 | head_pose | 1.463 | 1.670 | 1.349 | 0.2561 |
| cv1 | overlay | 3.581 | 4.072 | 2.954 | 0.8817 |
| cv1 | tensor_legacy | 0.0173 | 0.0263 | 0.0164 | 0.0079 |
| cv1 | tensor_from_numpy_owned | 0.0166 | 0.0174 | 0.0166 | 0.0003 |
| cv1 | snapshot_arrays | 0.0063 | 0.0087 | 0.0054 | 0.0017 |
| cv2 | head_pose | 1.501 | 1.671 | 1.502 | 0.0048 |
| cv2 | overlay | 3.557 | 4.096 | 2.941 | 0.8855 |
| cv2 | tensor_legacy | 0.0241 | 0.0254 | 0.0176 | 0.0085 |
| cv2 | tensor_from_numpy_owned | 0.0157 | 0.0166 | 0.0115 | 0.0057 |
| cv2 | snapshot_arrays | 0.0068 | 0.0073 | 0.0062 | 0.0013 |
| cv4 | head_pose | 1.501 | 1.660 | 1.501 | 0.0017 |
| cv4 | overlay | 3.585 | 3.996 | 3.165 | 0.8681 |
| cv4 | tensor_legacy | 0.0246 | 0.0254 | 0.0212 | 0.0071 |
| cv4 | tensor_from_numpy_owned | 0.0154 | 0.0166 | 0.0115 | 0.0058 |
| cv4 | snapshot_arrays | 0.0067 | 0.0072 | 0.0068 | 0.0002 |
| dashboard_cache | head_pose | 1.481 | 1.686 | 1.481 | 0.0061 |
| dashboard_cache | overlay | 1.003 | 1.158 | 0.7949 | 0.3282 |
| dashboard_cache | tensor_legacy | 0.0260 | 0.0270 | 0.0221 | 0.0078 |
| dashboard_cache | tensor_from_numpy_owned | 0.0169 | 0.0180 | 0.0122 | 0.0066 |
| dashboard_cache | snapshot_arrays | 0.0067 | 0.0073 | 0.0067 | 0.0002 |
| pose_cache | head_pose | 1.467 | 1.624 | 1.467 | 0.0012 |
| pose_cache | overlay | 3.596 | 4.066 | 3.175 | 0.8586 |
| pose_cache | tensor_legacy | 0.0244 | 0.0257 | 0.0244 | 0.0002 |
| pose_cache | tensor_from_numpy_owned | 0.0162 | 0.0169 | 0.0162 | 0.0003 |
| pose_cache | snapshot_arrays | 0.0067 | 0.0073 | 0.0068 | 0.0002 |

Probes use prepared inputs and finite histories; they include clock/call overhead and omit MediaPipe/decision/source acquisition. Counts, raw samples and per-run medians are retained. They cannot establish end-to-end speedup.

## Reproduction

Use .venv-baseline\Scripts\python.exe -B tools\optimize_phase9.py --parity baseline, then --parity <candidate>, followed by --experiment <candidate> --runs 5. Thread experiments add --sources Normal/Normal_1.avi. Dashboard adds --mode render. Allocation probes use tools\probe_phase9_allocations.py --candidate <name> --output <new-path>. Existing evidence is never overwritten; run in a separate checkout/output location for a new experiment. Final commands and production configuration are in PHASE9_CPU_OPTIMIZATION.md.
