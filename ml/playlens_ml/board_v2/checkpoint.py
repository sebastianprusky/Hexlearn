"""Require explicit reviewed decisions for the exact native extractor revision."""
import json
from .features import fingerprint
from .review import OUT
from playlens_ml.board_experiment.data import source_data


def checkpoint(out=OUT, hash_fn=fingerprint):
    selection=json.loads((out/'review/selection.json').read_text());decisions=json.loads((out/'review/decisions.json').read_text())
    clips=json.loads((out/'review/clips/selection.json').read_text());robustness=json.loads((out/'robustness.json').read_text())
    for artifact in (selection,decisions,clips,robustness):
        if artifact['extractorHash']!=hash_fn():raise ValueError('Review must match extractor revision')
    if selection['sourceDatasetSha256']!=source_data()[1]:raise ValueError('Review source changed')
    expected={f'run-{run:02d}-{part}' for run in range(1,31) for part in (1,2)}
    frames=decisions['frames']
    if len(frames)!=60 or {f['id'] for f in frames}!=expected:raise ValueError('Expected exactly 60 reviewed development frames')
    correct=sum(f['geometryCorrect'] is True and f['colorsCorrect'] is True for f in frames)
    reviewed=decisions['clips'];events={e for c in reviewed for e in c['events']}
    complete=len(reviewed)==12 and {c['id'] for c in reviewed}=={c['id'] for c in clips['clips']} and all(c['reviewed'] for c in reviewed)
    stable=all(r['geometryStable']>=54 and r['colorsStable']>=54 for r in robustness['summary'].values())
    result={'correctFrames':correct,'reviewedFrames':60,'accuracy':correct/60,'clipsReviewed':complete,
            'eventCoverage':sorted(events),'robustnessPassed':stable,
            'passed':correct>=54 and complete and {'rotation','stacking','incoming','clear'}<=events and stable,
            'reviewType':'assistant visual inspection; uncertainties count as failures; not independent ground truth'}
    (out/'checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');return result
