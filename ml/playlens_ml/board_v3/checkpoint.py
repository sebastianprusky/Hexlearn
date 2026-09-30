from .data import OUT
from .features import fingerprint
from playlens_ml.board_v2.checkpoint import checkpoint as original

def checkpoint(out=OUT):
    return original(out, fingerprint)
