"""Normalize scale using the visible core-to-board ratio measured in review pixels."""
import hashlib
from pathlib import Path
import numpy as np
from playlens_ml.board_v3.features import extract as native_extract,fingerprint as native_hash
from playlens_ml.board_dynamics.features import FEATURE_NAMES,GEOMETRY_NAMES,BASE_NAMES,extractor_hash as dynamic_hash

CORE_RATIO=.32


def normalize(frame):
    if not frame['valid']:return dict(frame)
    result=dict(frame);ratio=frame['coreApothemRatio'];scale=CORE_RATIO/ratio
    if not .27<ratio<.40:return {'valid':False,'reason':'core_scale_uncertain'}
    lanes=np.array(frame['vector'][:24]).reshape(6,4);lanes[:,0]*=scale
    found=lanes[:,2]<2.05;lanes[found,2]*=scale;lanes[:,2]=np.minimum(lanes[:,2],2.1)
    lanes[:,1]=np.maximum(0,lanes[:,2]-lanes[:,0])
    result.update(apothem=frame['apothem']/scale,coreApothemRatio=CORE_RATIO,heights=lanes[:,0].tolist(),
                  vector=lanes.ravel().tolist()+frame['vector'][24:],grayApothem=frame['apothem'])
    return result


def extract(image):return normalize(native_extract(image))


def extractor_hash():
    return hashlib.sha256(native_hash().encode()+dynamic_hash().encode()+Path(__file__).read_bytes()).hexdigest()
