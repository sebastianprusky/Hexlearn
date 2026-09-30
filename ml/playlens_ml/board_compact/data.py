import hashlib
import json
import numpy as np
from . import SCHEMA
from .features import transform,FEATURE_NAMES,extractor_hash
from playlens_ml.board_dynamics.data import OUT as SOURCE
from playlens_ml.common import ROOT
OUT=ROOT/'artifacts/experiments/board-compact-v1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'windows.npz') as source:data={k:source[k].copy() for k in source.files}
    data.update(X=np.array([transform(row) for row in data['X']]),schema=np.array(SCHEMA),feature_names=np.array(FEATURE_NAMES))
    np.savez_compressed(OUT/'windows.npz',**data)
    manifest=json.loads((SOURCE/'manifest.json').read_text());manifest.update(schema=SCHEMA,featureNames=FEATURE_NAMES,extractorHash=extractor_hash(),
        sourceWindowsSha256=hashlib.sha256((SOURCE/'windows.npz').read_bytes()).hexdigest(),
        hypothesis='Test whether speed alone carries the new signal, and whether compact board/activity features improve it without color and redundant temporal features.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':build()
