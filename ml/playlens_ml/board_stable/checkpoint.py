import json
from .data import OUT
from .features import extractor_hash
from playlens_ml.board_v3.checkpoint import checkpoint as native_checkpoint
from playlens_ml.board_v3.data import OUT as SOURCE


def checkpoint(out=OUT):
    native=native_checkpoint(SOURCE);review=json.loads((out/'review/decisions.json').read_text())
    if review['extractorHash']!=extractor_hash() or review['reviewedFrames']!=60:raise ValueError('Stable-scale review missing or stale')
    result={**native,'stableGeometryConfirmed':review['geometryConfirmed'],'passed':native['passed'] and review['geometryConfirmed']>=54}
    (out/'checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');return result
