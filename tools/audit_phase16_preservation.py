"""Hash every pre-Phase16 file and allow only the listed runtime integration edits."""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifest import sha256,write_json

ALLOWED={
    'driver_monitor/config.py','driver_monitor/inference/engine.py',
    'driver_monitor/inference/temporal_buffer.py','driver_monitor/results.py',
    'driver_monitor/ui/overlay.py','driver_monitor/storage/session_persistence.py','main.py',
    'tests/test_cpu_optimization.py',
    'tests/test_phase15_final_candidate.py',
}


def main():
    snapshot=ROOT/'docs/phase16_before_hashes.json'
    before=json.loads(snapshot.read_text())
    changed={};missing=[]
    for name,expected in before.items():
        path=ROOT/name
        if not path.is_file():missing.append(name)
        else:
            actual=sha256(path)
            if actual!=expected:changed[name]={'before':expected,'after':actual}
    unexpected=sorted(set(changed)-ALLOWED)
    if not ALLOWED<=set(changed) or unexpected or missing:
        raise AssertionError(f'Preservation violation: unexpected={unexpected}, missing={missing}, expected edits missing={sorted(ALLOWED-set(changed))}')
    critical=['saved_model.pt','norm.npy','models/phase15/lstm_only_w32_full_development.pt',
              'models/phase15/lstm32_dev_norm.npy']
    assert all(name not in changed for name in critical)
    assert all(not name.startswith(('results/phase11/','results/phase12/','results/phase13/','results/phase14/',
                                    'results/phase15/','models/phase11/','models/phase15/')) for name in changed)
    write_json(ROOT/'docs/phase16_preservation.json',{'prior_files':len(before),
        'unchanged_prior_files':len(before)-len(changed),'modified_prior_files':changed,
        'modified_file_names':sorted(changed),'missing_prior_files':missing,'unexpected_modifications':unexpected,
        'snapshot_sha256':sha256(snapshot),'source_videos_unchanged':True,
        'phase11_through_phase15_evidence_unchanged':True,'historical_and_candidate_assets_unchanged':True,
        'method':'SHA256 byte audit only; held-out files hashed without candidate decoding or inference'})
    print(len(before)-len(changed),'old files unchanged;',len(changed),'expected runtime files modified')


if __name__=='__main__':main()
