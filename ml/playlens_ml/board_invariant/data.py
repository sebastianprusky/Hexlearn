import hashlib
import json
import numpy as np
from . import SCHEMA
from .features import transform,extractor_hash,FEATURE_NAMES
from playlens_ml.board_v3.data import OUT as SOURCE
from playlens_ml.common import ROOT
OUT=ROOT/'artifacts/experiments/board-invariant-v1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'windows.npz') as source:data={k:source[k].copy() for k in source.files}
    x=np.array([transform(row) for row in data['X']]);data.update(X=x,schema=np.array(SCHEMA),feature_names=np.array(FEATURE_NAMES))
    np.savez_compressed(OUT/'windows.npz',**data)
    manifest=json.loads((SOURCE/'manifest.json').read_text());manifest.update(schema=SCHEMA,featureNames=FEATURE_NAMES,extractorHash=extractor_hash(),
        nativeExtractorHash=manifest['extractorHash'],sourceWindowsSha256=hashlib.sha256((SOURCE/'windows.npz').read_bytes()).hexdigest(),
        hypothesis='Tallest-stack and crowding summaries reduce orientation-dependent sample complexity; fixed families and identical windows.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':build()
