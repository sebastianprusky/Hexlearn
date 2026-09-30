import json
from .data import OUT
from .features import fingerprint
from .windows import feature_hash
from playlens_ml.board_v2.checkpoint import checkpoint as native_checkpoint


def checkpoint():
    base=native_checkpoint(OUT,fingerprint)
    labels=json.loads((OUT/'review/annotations.json').read_text());audit=json.loads((OUT/'tracking-audit.json').read_text())
    for artifact in (labels,audit):
        if artifact['extractorHash']!=fingerprint():raise ValueError('Stale tracking review')
    expected={p.stem for p in (OUT/'review/repairs').glob('*.png') if '-source' not in p.name}
    repairs=labels['repairs']
    if {r['id'] for r in repairs}!=expected:raise ValueError('Unreviewed geometry repairs')
    repair_accuracy=sum(r['geometryCorrect'] and r['colorsCorrect'] for r in repairs)/max(1,len(repairs))
    result={**base,'repairFrames':len(repairs),'repairAccuracy':repair_accuracy,
            'incomingColorAgreement':audit['incomingColorCorrect']/audit['incomingReviewedFrames'],
            'featureHash':feature_hash(),'motionLimitations':audit['clearTotals'],
            'passed':base['passed'] and repair_accuracy>=.9}
    (OUT/'checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
