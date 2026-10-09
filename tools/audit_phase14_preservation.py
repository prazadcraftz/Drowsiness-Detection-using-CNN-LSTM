"""Byte-only prior-file preservation audit; never decode data or run a candidate.

This runs separately from candidate processes: it hashes the pre-existing files,
including sealed artifacts, solely to prove that Phase14 did not alter them.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifest import sha256,write_json
import json


def main():
    before = json.loads((ROOT/'docs/phase14_before_hashes.json').read_text())
    missing,changed = [],{}
    for name,expected in before.items():
        path = ROOT/name
        if not path.is_file():
            missing.append(name)
        else:
            actual = sha256(path)
            if actual!=expected:
                changed[name] = {'before':expected,'after':actual}
    write_json(ROOT/'docs/phase14_preservation.json',{'prior_files_preserved':len(before)-len(changed)-len(missing),
        'changed_prior_files':changed,'missing_prior_files':missing,'purpose':'byte preservation only; no decoding, metric inspection or inference',
        'snapshot_sha256':sha256(ROOT/'docs/phase14_before_hashes.json')})
    if changed or missing:
        raise AssertionError('Prior files changed/missing')
    print(f'{len(before)} prior files unchanged')


if __name__=='__main__':
    main()
