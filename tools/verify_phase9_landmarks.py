"""Supplement the runtime x/y gate with all native x/y/z/visibility/presence fields."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from driver_monitor.metrics.benchmark import sha256, write_new_json


def verify():
    output=ROOT/'results/phase9_native_landmarks.json'
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((ROOT/'tests/fixtures/manifest.json').read_text())
    all_arrays={'baseline':[],'no_unused_outputs':[]}; sources=[]
    def create(enabled):
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ROOT/'face_landmarker.task')),
            num_faces=1,output_face_blendshapes=enabled,output_facial_transformation_matrixes=enabled))
    with create(True) as baseline, create(False) as candidate:
        for source in manifest['sources']:
            assert sha256(ROOT/source['filename'])==source['sha256']
            capture=cv2.VideoCapture(str(ROOT/source['filename']))
            frames=mismatches=face_mismatches=0; max_difference=0.0
            try:
                while True:
                    ok,frame=capture.read()
                    if not ok:break
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
                    values=[]; flags=[]
                    for name,detector in (('baseline',baseline),('no_unused_outputs',candidate)):
                        result=detector.detect(image); face=bool(result.face_landmarks); flags.append(face)
                        array=np.full((478,5),np.nan) if not face else np.array([
                            [getattr(p,k) if getattr(p,k,None) is not None else np.nan for k in ('x','y','z','visibility','presence')]
                            for p in result.face_landmarks[0]],dtype=np.float64)
                        values.append(array);all_arrays[name].append(array)
                    face_mismatches+=flags[0]!=flags[1]
                    mismatches+=not np.array_equal(*values,equal_nan=True)
                    finite=np.isfinite(values[0])&np.isfinite(values[1])
                    if finite.any():max_difference=max(max_difference,float(np.max(np.abs(values[0][finite]-values[1][finite]))))
                    frames+=1
            finally:capture.release()
            assert frames==source['decoded_frame_count']
            sources.append({'source':source['filename'],'source_sha256':source['sha256'],'frames':frames,
                            'landmark_mismatch_frames':mismatches,'face_mismatch_frames':face_mismatches,'maximum_finite_difference':max_difference})
    artifacts={}
    for name,values in all_arrays.items():
        path=ROOT/f'results/phase9_native_landmarks_{name}.npz'
        with path.open('xb') as handle:np.savez_compressed(handle,landmarks=np.array(values))
        artifacts[name]={'path':str(path),'sha256':sha256(path)}
    result={'candidate':'no_unused_outputs','fields':['x','y','z','visibility','presence'],
            'legacy_color_contract':'BGR bytes passed as SRGB; unchanged','exact':all(r['landmark_mismatch_frames']==r['face_mismatch_frames']==0 for r in sources),
            'sources':sources,'artifacts':artifacts,'omitted_optional_fields':'represented as NaN in both arrays'}
    write_new_json(output,result)
    assert result['exact'],'Native coordinate parity failed: do not promote'
    print(json.dumps({'native_landmarks_exact':True,'frames':sum(r['frames'] for r in sources)}))


if __name__=='__main__':verify()
