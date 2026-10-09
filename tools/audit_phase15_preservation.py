"""Byte-only preservation audit of every file present before Phase 15."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.manifest import sha256, write_json


def main():
    snapshot = ROOT / 'docs/phase15_before_hashes.json'
    before = json.loads(snapshot.read_text(encoding='utf-8'))
    missing, changed = [], {}
    for name, expected in before.items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        else:
            actual = sha256(path)
            if actual != expected:
                changed[name] = {'before': expected, 'after': actual}
    result = {'prior_files': len(before), 'prior_files_preserved': len(before)-len(changed)-len(missing),
              'changed_prior_files': changed, 'missing_prior_files': missing,
              'snapshot_sha256': sha256(snapshot),
              'method': 'byte hashes only; sealed files not decoded or used for candidate inference'}
    write_json(ROOT / 'docs/phase15_preservation.json', result)
    if changed or missing:
        raise AssertionError('Pre-Phase15 files changed or missing')
    print(f'{len(before)} pre-Phase15 files byte-identical')


if __name__ == '__main__':
    main()
