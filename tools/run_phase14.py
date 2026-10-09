"""Durable Phase14 CLI launcher with line-buffered progress and explicit exit state."""
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.windows import run,latency
from evaluation.manifest import write_json


def main():
    action = sys.argv[1]
    if action not in ('run','latency'):
        raise ValueError('Expected run or latency')
    label = 'training_retry1' if action=='run' else 'latency'
    sys.stdout = (ROOT/f'results/phase14_{label}_stdout.log').open('x',buffering=1,encoding='utf-8')
    sys.stderr = (ROOT/f'results/phase14_{label}_stderr.log').open('x',buffering=1,encoding='utf-8')
    try:
        (run if action=='run' else latency)(ROOT/'results/phase14')
    except BaseException as error:
        traceback.print_exc()
        write_json(ROOT/f'results/phase14/{action}_process_exit.json',
                   {'status':'FAILED','type':type(error).__name__,'message':str(error)})
        raise
    write_json(ROOT/f'results/phase14/{action}_process_exit.json',{'status':'COMPLETE'})


if __name__=='__main__':
    main()
