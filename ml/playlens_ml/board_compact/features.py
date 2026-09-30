import hashlib
from pathlib import Path
import numpy as np
from playlens_ml.board_dynamics.features import DYNAMIC_NAMES,extractor_hash as source_hash
from playlens_ml.board_invariant.features import GEOMETRY_NAMES
SPEED_INDICES=[k*12+j for k in range(4) for j in range(5)]
ACTIVITY_INDICES=[i for i in range(48) if i not in SPEED_INDICES]
SPEED_NAMES=[DYNAMIC_NAMES[i] for i in SPEED_INDICES]
BASE_NAMES=SPEED_NAMES+GEOMETRY_NAMES
FEATURE_NAMES=BASE_NAMES+[DYNAMIC_NAMES[i] for i in ACTIVITY_INDICES]


def extractor_hash():return hashlib.sha256(source_hash().encode()+Path(__file__).read_bytes()).hexdigest()


def transform(vector):
    dynamic=vector[-48:]
    return np.r_[dynamic[SPEED_INDICES],vector[:len(GEOMETRY_NAMES)],dynamic[ACTIVITY_INDICES]]
