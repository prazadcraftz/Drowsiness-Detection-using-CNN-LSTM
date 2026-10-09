"""Finite allocation/OpenCV probes; diagnostic timings do not establish app speedup."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.optimize_phase9 import ROOT, configuration
from driver_monitor.metrics.benchmark import distribution_ms, write_new_json


def probe(candidate):
    with configuration(candidate) as runtime:
        np, torch = runtime["np"], runtime["torch"]
        import driver_monitor.vision.features as features
        import draw_landmarks
        from driver_monitor.results import InferenceResult, ObservationResult
        from driver_monitor.vision.video_source import FramePacket
        # NpzFile indexing decompresses an entire member. Load ONCE outside all
        # endpoints: otherwise a 'copy' probe accidentally times archive I/O.
        with np.load(ROOT/'tests/fixtures/legacy_baseline.npz') as stored:
            archive = {name:stored[name] for name in stored.files}
        with runtime["VideoSource"](ROOT/'Normal/Normal_1.avi') as video:
            frame = video.read().frame
        with np.load(ROOT/'results/phase9_parity/baseline/Normal_Normal_1.avi.npz') as reference:
            landmarks = reference['landmarks'][:32]
        poses = [np.array([points[i] for i in [10,33,263,152,61,291]]) for points in landmarks]
        # Match production deque-window transpose strides, not archived C layout.
        sequences = [np.array(sequence.T).T for sequence in archive['sequences'][:32]]
        packets, results = [], []
        for i in range(32):
            obs = ObservationResult(True, archive['raw_features'][i], archive['normalized_features'][i], min(i+1,16), i>=15, timestamp_seconds=i/20)
            result = InferenceResult(obs)
            if i>=15:
                index = int(archive['frame_window_indices'][i])
                cls = int(archive['predicted_class_ids'][index])
                result = InferenceResult(obs,(1,5,16),archive['sequences'][index],archive['input_tensors'][index],archive['logits'][index],archive['probabilities'][index],cls,runtime['RuntimeConfig']().class_mapping[cls],float(archive['probabilities'][index][0,cls]))
            packets.append(FramePacket(frame,i,i/20,'file')); results.append(result)
        workloads = {
            'head_pose': lambda i: features.calculate_head_pose(frame, poses[i%32]),
            'overlay': lambda i: runtime['main'].render_result(frame, packets[i%32], results[i%32], 'Normal_1.avi', source_fps=20,target_fps=20),
            'tensor_legacy': lambda i: torch.tensor(sequences[i%32], dtype=torch.float32).unsqueeze(0),
            'tensor_from_numpy_owned': lambda i: torch.from_numpy(np.array(sequences[i%32],dtype=np.float32,copy=True,order='K')).unsqueeze(0),
            'snapshot_arrays': lambda i: tuple(archive[name][i%32].copy() for name in ('sequences','input_tensors','logits','probabilities')),
        }
        # Independent exact checks outside all timing endpoints.
        pose_exact = all(all(np.array_equal(a,b) for a,b in zip(features.calculate_head_pose(frame,p),draw_landmarks.calculate_head_pose(frame,p))) for p in poses)
        tensor_exact = all(np.array_equal(workloads['tensor_legacy'](i).numpy(), workloads['tensor_from_numpy_owned'](i).numpy()) for i in range(32))
        pixels = [hashlib.sha256(workloads['overlay'](i).tobytes()).hexdigest() for i in range(32)]
        timing = {name: [] for name in workloads}
        jobs = [(name,rep) for name in workloads for rep in range(5)]
        random.Random(20260926).shuffle(jobs)
        for name, rep in jobs:
            fn = workloads[name]
            for i in range(20): fn(i)
            count = 200 if name in ('head_pose','overlay') else 5000
            samples = []
            for i in range(count):
                started = time.perf_counter_ns(); fn(i); samples.append(time.perf_counter_ns()-started)
            timing[name].append({'run':rep,'raw_elapsed_ns':samples,'distribution':distribution_ms([ns/1e6 for ns in samples])})
        aggregate = {}
        for name, runs in timing.items():
            runs.sort(key=lambda r:r['run'])
            medians = [r['distribution']['p50_ms'] for r in runs]
            aggregate[name] = {'pooled':distribution_ms([ns/1e6 for run in runs for ns in run['raw_elapsed_ns']]),
                'per_run_median_ms':medians,'mean_run_median_ms':statistics.fmean(medians),'std_run_median_ms':statistics.pstdev(medians)}
        return {'schema_version':2,'archive_preloaded_outside_timers':True,'candidate':candidate,'threads':{'torch_intra':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads(),'opencv':runtime['cv2'].getNumThreads()},
            'methodology':{'runs':5,'warmup_calls':20,'seed':20260926,'prebuilt_inputs':True,'tensor_sequence_layout':'Production-like transposed deque window (Fortran contiguous)','includes_timing_overhead':True,'purpose':'Diagnostic allocation/pose/render costs; not full application speedup evidence'},
            'pose_exact':pose_exact,'tensor_exact':tensor_exact,'pixel_sha256':pixels,'aggregate':aggregate,'runs':timing}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=('baseline','pose_cache','cv1','cv2','cv4','dashboard_cache'),required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    write_new_json(args.output,probe(args.candidate))
