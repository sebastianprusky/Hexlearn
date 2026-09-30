import json
import shutil
import numpy as np
from . import SCHEMA
from .features import normalize,extractor_hash,FEATURE_NAMES
from playlens_ml.board_v3.data import OUT as SOURCE
from playlens_ml.board_v2.data import build as assemble,VIDEO_LAG
from playlens_ml.board_invariant.features import transform
from playlens_ml.board_dynamics.features import motion_features
from playlens_ml.common import ROOT
OUT=ROOT/'artifacts/experiments/board-stable-core-v1'


def build():
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    for run in range(1,31):
        source=json.loads((SOURCE/'cache'/f'run-{run:02d}.json').read_text())
        payload={**source,'schema':SCHEMA,'extractorHash':extractor_hash(),'frames':[normalize(f) for f in source['frames']]}
        (OUT/'cache'/f'run-{run:02d}.json').write_text(json.dumps(payload,allow_nan=False))
    shutil.copy2(SOURCE/'alignment.json',OUT/'alignment.json')
    manifest=assemble(OUT,SCHEMA,extractor_hash)
    with np.load(OUT/'windows.npz') as source:data={k:source[k].copy() for k in source.files}
    features=np.zeros((len(data['X']),len(FEATURE_NAMES)))
    for run in range(1,31):
        cache=json.loads((OUT/'cache'/f'run-{run:02d}.json').read_text());frames=cache['frames'];times=np.arange(len(frames))/4+VIDEO_LAG
        base=np.array([f.get('vector',[0.]*54) for f in frames]);phase=np.array([f.get('phaseDegrees',90) for f in frames])
        for row in np.flatnonzero((data['run_order']==run)&data['valid']):
            index=int(np.searchsorted(times,data['elapsed_seconds'][row],side='right')-1)
            features[row]=np.r_[transform(data['X'][row]),motion_features(base[index-32:index+1],phase[index-32:index+1])]
    data.update(X=features,feature_names=np.array(FEATURE_NAMES))
    np.savez_compressed(OUT/'windows.npz',**data)
    manifest.update(featureNames=FEATURE_NAMES,scaleMethod='visible core apothem / 0.32; ratio measured from native review pixels',
                    hypothesis='Stable pixel-derived scale removes spurious motion and geometry resets without clock features.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':build()
