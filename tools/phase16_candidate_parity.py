"""Frozen Phase15 development tensors through direct and integrated models."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_monitor.config import RuntimeConfig
from driver_monitor.inference.engine import DrowsinessInferenceEngine
from evaluation.manifest import sha256, write_json
from evaluation.windows import build_model, create_windows, install_seal


def main():
    output = ROOT/'results/phase16'
    output.mkdir(exist_ok=True)
    folds = json.loads((ROOT/'results/phase12/cv_folds.json').read_text())
    audit = install_seal(folds)
    spec = json.loads((ROOT/'results/phase15/selection_spec.json').read_text())
    identities = json.loads((ROOT/'results/phase15/verification_inputs.json').read_text())['identities']
    with np.load(ROOT/'results/phase15/verification_inputs.npz',allow_pickle=False) as archive:
        frozen = archive['windows']
    norm = np.load(ROOT/'models/phase15/lstm32_dev_norm.npy',allow_pickle=False)
    direct = build_model('lstm_only',32)
    direct.load_state_dict(torch.load(ROOT/'models/phase15/lstm_only_w32_full_development.pt',map_location='cpu',weights_only=True))
    direct.eval()
    provenance = json.loads((ROOT/'results/phase12/cache_provenance.json').read_text())
    maximum = {k:0. for k in ('normalized_features','sequence','input_tensor','logits','probabilities')}
    class_exact = 0
    with DrowsinessInferenceEngine(RuntimeConfig.from_ini(model_profile='lstm32'),initialize_detector=False) as engine:
        for i, identity in enumerate(identities):
            source = identity['source_id']
            if source not in spec['full_development_source_ids'] or source in spec['excluded_phase11_test_source_ids']:
                raise ValueError('Sealed or unknown parity source')
            path = ROOT/'results/phase11/raw_features'/f'{source}.npz'
            if sha256(path) != provenance['cache_sha256'][source]:
                raise ValueError('Development cache changed')
            with np.load(path,allow_pickle=False) as archive:
                features,frames = archive['features'],archive['frame_indices']
            part = create_windows(features,frames,norm,32)
            where = np.flatnonzero(part['start_indices']==identity['start_index'])
            if len(where)!=1:raise ValueError('Verification start missing')
            j=where[0]
            np.testing.assert_array_equal(part['frame_indices'][j],identity['frame_indices'])
            normalized = part['windows'][j]
            np.testing.assert_array_equal(normalized.astype(np.float32),frozen[i])
            indices=part['observation_indices'][j]
            by_engine=np.stack([engine.normalizer.normalize(features[k]) for k in indices],axis=1)
            maximum['normalized_features']=max(maximum['normalized_features'],float(np.abs(normalized-by_engine).max()))
            result=engine.infer_sequence(normalized)
            with torch.inference_mode():
                tensor=torch.tensor(normalized,dtype=torch.float32).unsqueeze(0)
                logits=direct(tensor).numpy()
                probabilities=torch.softmax(torch.from_numpy(logits),dim=1).numpy()
            expected={'sequence':normalized,'input_tensor':tensor.numpy(),'logits':logits,'probabilities':probabilities}
            for key,array in expected.items():
                maximum[key]=max(maximum[key],float(np.abs(getattr(result,key)-array).max()))
            class_exact+=int(result.raw_class_id==int(probabilities.argmax(1)[0]))
    outcome={'verification_windows':len(identities),'class_ids_exact':class_exact,
        'maximum_differences':maximum,'exact_parity':class_exact==len(identities) and all(v==0 for v in maximum.values()),
        'candidate_checkpoint_sha256':sha256(ROOT/'models/phase15/lstm_only_w32_full_development.pt'),
        'normalizer_sha256':sha256(ROOT/'models/phase15/lstm32_dev_norm.npy'),
        'test_cache_reads':audit['test_cache_reads'],'test_video_reads':0,'candidate_test_inference_passes':0,
        'padded_offline_window_note':'Offline repeat-last verification windows are supplied directly; runtime process_features never fabricates missing observations.'}
    write_json(output/'candidate_parity.json',outcome)
    print(outcome)
    if not outcome['exact_parity']:raise AssertionError('Candidate direct/integrated parity changed')


if __name__=='__main__':main()
