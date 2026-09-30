from .data import OUT
from .checkpoint import checkpoint
from .features import fingerprint
from playlens_ml.board_v2.report import generate

if __name__=="__main__":print(generate(OUT,checkpoint,fingerprint))
