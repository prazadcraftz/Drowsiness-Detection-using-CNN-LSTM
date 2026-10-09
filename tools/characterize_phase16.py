"""Descriptive six-development-video replay for both fixed runtime profiles."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import main
from driver_monitor.inference.decision import DecisionLayer
from evaluation.manifest import read_manifest,sha256,write_json

SOURCES=('Normal/Normal_1.avi','Normal/Normal_60.avi','Yawning/Yawning_1.avi',
         'Yawning/Yawning_60.avi','Eye Close/Eye Close_1.avi','Eye Close/Eye Close_60.avi')
NAMES=('Normal','Yawning','Eye Close')
THRESHOLDS=(.60,.60,.70)


def stats(values):
    if not len(values):return None
    return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
        'p10':float(np.percentile(values,10)),'p25':float(np.percentile(values,25)),
        'p75':float(np.percentile(values,75)),'p90':float(np.percentile(values,90)),
        'minimum':float(np.min(values)),'maximum':float(np.max(values))}


def characterize():
    out=ROOT/'results/phase16'
    out.mkdir(exist_ok=True)
    folds=json.loads((ROOT/'results/phase12/cv_folds.json').read_text())
    manifest={r['relative_path']:r for r in read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv')}
    for source in SOURCES:
        if manifest[source]['source_id'] not in folds['development_source_ids']:
            raise ValueError('Characterization source is not development')
    rows=[];probability=[]
    for profile in ('legacy','lstm32'):
        for source in SOURCES:
            observed=[];times=[];ids=[]
            class Observer(DecisionLayer):
                def update(self,raw,**kwargs):
                    if raw.probabilities is not None:
                        observed.append(raw.probabilities[0].copy())
                        times.append(raw.observation.timestamp_seconds)
                        ids.append(raw.raw_class_id)
                    return super().update(raw,**kwargs)
            with patch.object(main,'_create_decision_layer',lambda config=None,*,enabled=True:Observer(config,enabled=enabled)):
                summary=main.run(str(ROOT/source),no_display=True,no_audio=True,model_profile=profile)
            m=summary.session_metrics
            events={name:{'count':value['count'],'longest_seconds':value['longest_duration_seconds']}
                    for name,value in m['events'].items()}
            row={'profile':profile,'source':source,'source_sha256':sha256(ROOT/source),
                'decoded_frames':summary.frames_decoded,'face_visible_frames':summary.valid_observations,
                'sampled_observations':summary.sampled_valid_observations,'model_predictions':summary.predictions,
                'first_prediction_timestamp_seconds':times[0] if times else None,
                'gap_resets':summary.gap_resets,'raw_class_counts':m['raw_prediction_counts'],
                'smoothed_class_counts':m['smoothed_prediction_counts'],
                'state_frame_counts':m['state_frame_counts'],'state_occupancy_seconds':m['state_occupancy_seconds'],
                'events':events,'alert_occurrences':events['alert']['count'],
                'no_face_intervals':events['no_face']['count'],
                'longest_eye_evidence_seconds':events['eye_closure']['longest_seconds'],
                'longest_drowsy_progression_seconds':events['drowsiness']['longest_seconds'],
                'longest_alert_seconds':events['alert']['longest_seconds'],
                'eye_evidence_fraction':m['observed_eye_evidence_fraction'],
                'audio_disabled':True,'audio_snapshot':m['audio'],
                'warmup_observations_required':16 if profile=='legacy' else 32,
                'purpose':'descriptive development fixture behavior, not accuracy or event validation'}
            if len(observed)!=summary.predictions:raise ValueError('Prediction observation count differs')
            rows.append(row)
            array=np.asarray(observed).reshape(-1,3)
            per_class={name:stats(array[:,c]) for c,name in enumerate(NAMES)}
            near={name:{'threshold':THRESHOLDS[c],'band':.05,
                'within_band_count':int((np.abs(array[:,c]-THRESHOLDS[c])<=.05).sum()),
                'below_count':int((array[:,c]<THRESHOLDS[c]).sum()),
                'above_or_equal_count':int((array[:,c]>=THRESHOLDS[c]).sum())}
                for c,name in enumerate(NAMES)}
            probability.append({'profile':profile,'source':source,'prediction_count':len(array),
                'raw_class_probability':per_class,'frozen_threshold_margins':near,
                'raw_predicted_class_counts':{name:int((np.asarray(ids)==c).sum()) for c,name in enumerate(NAMES)}})
            print(profile,source,summary.predictions,'predictions',flush=True)
    write_json(out/'profile_characterization.json',{'purpose':'six development compatibility fixtures, descriptive only; no accuracy or threshold tuning',
        'sources':list(SOURCES),'profiles':['legacy','lstm32'],'rows':rows,'candidate_test_inference_passes':0})
    write_json(out/'probability_characterization.json',{'frozen_thresholds':dict(zip(NAMES,THRESHOLDS)),
        'margin_band':.05,'rows':probability,'purpose':'descriptive probability and margin characterization; no threshold selection'})


if __name__=='__main__':characterize()
