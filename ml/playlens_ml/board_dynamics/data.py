import hashlib
import json
import numpy as np
from . import SCHEMA
from .features import motion_features,FEATURE_NAMES,extractor_hash
from playlens_ml.board_invariant.data import OUT as SOURCE
from playlens_ml.board_v3.data import OUT as NATIVE
from playlens_ml.board_v2.data import VIDEO_LAG
from playlens_ml.common import ROOT
OUT=ROOT/'artifacts/experiments/board-dynamics-v1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'windows.npz') as source:data={k:source[k].copy() for k in source.files}
    extra=np.zeros((len(data['X']),48))
    for run in range(1,31):
        cache=json.loads((NATIVE/'cache'/f'run-{run:02d}.json').read_text());frames=cache['frames']
        base=np.array([f.get('vector',[0.]*54) for f in frames]);phase=np.array([f.get('phaseDegrees',90) for f in frames])
        for row in np.flatnonzero((data['run_order']==run)&data['valid']):
            index=int(np.searchsorted(np.arange(len(frames))/4+VIDEO_LAG,data['elapsed_seconds'][row],side='right')-1)
            if index<32 or not all(f['valid'] for f in frames[index-32:index+1]):raise ValueError('Causal native window changed')
            extra[row]=motion_features(base[index-32:index+1],phase[index-32:index+1])
    data.update(X=np.column_stack((data['X'],extra)),schema=np.array(SCHEMA),feature_names=np.array(FEATURE_NAMES))
    np.savez_compressed(OUT/'windows.npz',**data)
    manifest=json.loads((SOURCE/'manifest.json').read_text());manifest.update(schema=SCHEMA,featureNames=FEATURE_NAMES,extractorHash=extractor_hash(),
        sourceWindowsSha256=hashlib.sha256((SOURCE/'windows.npz').read_bytes()).hexdigest(),
        hypothesis='Adjacent-frame incoming-speed estimates and recent activity add difficulty information from pixels, without clock values.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':build()
