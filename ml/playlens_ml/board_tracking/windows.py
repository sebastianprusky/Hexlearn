import hashlib
import json
from pathlib import Path
import numpy as np
from . import SCHEMA
from .data import OUT
from .features import fingerprint
from .motion import motion_features
from playlens_ml.board_compact.features import FEATURE_NAMES,transform as compact,extractor_hash as compact_hash
from playlens_ml.board_invariant.features import transform as invariant
from playlens_ml.board_dynamics.features import motion_features as old_motion
from playlens_ml.board_v2.data import VIDEO_LAG


def feature_hash():
    return hashlib.sha256(fingerprint().encode()+compact_hash().encode()+b''.join(Path(p).read_bytes() for p in (__file__,Path(__file__).with_name('motion.py')))).hexdigest()


def build():
    with np.load(OUT/'windows.npz') as s:data={k:s[k].copy() for k in s.files}
    x=np.zeros((len(data['X']),80));geometry_only=x.copy()
    for run in range(1,31):
        f=json.loads((OUT/'cache'/f'run-{run:02}.json').read_text())['frames']
        times=np.arange(len(f))/4+VIDEO_LAG
        for row in np.flatnonzero((data['run_order']==run)&data['valid']):
            i=np.searchsorted(times,data['elapsed_seconds'][row],side='right')-1;window=f[i-32:i+1]
            if len(window)!=33:raise ValueError('Short window')
            board=invariant(data['X'][row]);new=motion_features(window)
            old=old_motion(np.array([v['vector'] for v in window]),np.array([v['phaseDegrees'] for v in window]))
            x[row]=compact(np.r_[board,new]);geometry_only[row]=compact(np.r_[board,old])
    data.update(X=x,geometry_motion=geometry_only,feature_names=np.array(FEATURE_NAMES),schema=np.array(SCHEMA))
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    np.savez_compressed(OUT/'compact_windows.npz',**data)
    m=json.loads((OUT/'manifest.json').read_text());m.update(compactFeatureNames=FEATURE_NAMES,featureHash=feature_hash())
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

if __name__=='__main__':build()
