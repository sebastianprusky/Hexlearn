from playlens_ml.board_v3.checkpoint import checkpoint as native_checkpoint
from playlens_ml.board_v3.data import OUT as SOURCE

def checkpoint(out=None):
    # Pixel extraction is unchanged; the exact native review remains authoritative.
    return native_checkpoint(SOURCE)
