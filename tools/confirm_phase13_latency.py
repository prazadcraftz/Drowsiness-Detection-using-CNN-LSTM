"""Preserved, separate confirmation of all45 timing jobs after a wall-clock stall."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.architectures import verify_protocol,latency_statistics
from evaluation.balancing import guard_sources
from evaluation.manifest import sha256,write_json
from evaluation.models import build_model,NAMES
from evaluation.reproducibility import seed_everything,environment


def main():
    d=ROOT/'results/phase13'
    folds,_,_,_=verify_protocol(d)
    seed_everything(1201)
    directory=d/'latency_confirmation'
    directory.mkdir(exist_ok=False)
    original=json.loads((d/'latency_started.json').read_text())
    protocol=original['protocol']
    assert sha256(d/'latency_inputs.npz')==original['input_sha256']
    with np.load(d/'latency_inputs.npz',allow_pickle=False) as archive:
        guard_sources(archive['source_ids'],folds['folds'][0]['validation_source_ids'],folds['excluded_test_source_ids'])
        tensors=[torch.from_numpy(t[None].copy()) for t in archive['windows']]
    write_json(directory/'protocol.json',{'reason':'Primary run contains a preserved 423121.8114ms wall-clock stall; do not trim or replace primary data',
        'same_locked_design':True,'protocol':protocol,'job_order':original['job_order'],'input_sha256':original['input_sha256'],
        'environment':environment(),'primary_results_sha256':sha256(d/'latency_results.json'),
        'interpretation':'Separate confirmation only; all primary statistics and primary P50-based guidance retained', 'test_inference_passes':0})
    raw,records={},[]
    for name,fold,seed in original['job_order']:
        checkpoint=d/name/f'fold_{fold}'/f'seed_{seed}'/'best.pt'
        expected=json.loads(checkpoint.with_name('metrics.json').read_text())['checkpoint_sha256']
        assert sha256(checkpoint)==expected
        model=build_model(name).to('cpu')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        wall,cpu=[],[]
        with torch.inference_mode():
            for i in range(protocol['warmup_calls']):model(tensors[i%len(tensors)])
            for i in range(protocol['measured_calls_per_checkpoint']):
                value=tensors[i%len(tensors)]
                cpu_start=time.process_time_ns();wall_start=time.perf_counter_ns()
                logits=model(value)
                wall.append((time.perf_counter_ns()-wall_start)/1e6)
                cpu.append((time.process_time_ns()-cpu_start)/1e6)
            assert logits.shape==(1,3) and torch.isfinite(logits).all()
        key=f'{name}_fold_{fold}_seed_{seed}'
        raw[key]=np.array(wall);raw[key+'_process_cpu_ms']=np.array(cpu)
        records.append({'architecture':name,'fold':fold,'seed':seed,'checkpoint_sha256':expected,
                        **latency_statistics(wall),'maximum_wall_ms':max(wall),'process_cpu_total_ms':sum(cpu)})
        print(f'confirmation {key} p50={records[-1]["p50_ms"]:.4f}ms max={max(wall):.2f}ms',flush=True)
    np.savez_compressed(directory/'samples.npz',**raw)
    write_json(directory/'results.json',{'runs':records,'aggregate':{n:latency_statistics(np.concatenate([v for k,v in raw.items() if k.startswith(n+'_fold_') and not k.endswith('_process_cpu_ms')])) for n in NAMES},
        'samples_sha256':sha256(directory/'samples.npz'),'protocol_sha256':sha256(directory/'protocol.json'),'test_inference_passes':0,
        'process_cpu_note':'Diagnostic process CPU clock is coarse on this host and may sum across threads; wall-clock durations remain timing authority'})


if __name__=='__main__':main()
