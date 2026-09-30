"""Reviewed evidence for the authorized subset; failures are never filtered out."""
import json
import hashlib
from pathlib import Path
from . import OUT, RUNS, SCHEMA
from playlens_ml.imminent_loss.review import digest

# Explicit judgments after reviewing all 81 board and incoming contact panels.
# Uncertain frames do not establish correct extraction. Remaining frames are not
# individually certified here: the resulting rate is an optimistic upper bound.
ISSUES = {
    'run-01-lead-10': ('uncertain', 'Bottom lane is visibly green while the outermost-color label is blue; fading edge pixels prevent confident verification.'),
    'run-08-lead-10': ('uncertain', 'Bottom stack ends in visible green while the reported outermost color is red; cannot certify the color/extent during the transition.'),
    'run-09-lead-2': ('uncertain', 'All six stack heights collapse to the core and top colors become empty despite large visible block groups during a clear; stack versus detached-group assignment is not trustworthy.'),
    'run-13-lead-10': ('uncertain', 'The large bottom block group is reported as an empty stack with an inner blue incoming boundary during a clear; visible outer extent cannot be certified.'),
    'run-15-lead-10': ('uncertain', 'Lower-left blue/yellow/red group is reported as an empty stack and is not represented as the nearest incoming group; clear-state assignment is ambiguous.'),
    'run-19-lead-10': ('incorrect', 'A separated green block above the core is visible, but lane 4 reports no incoming block and an empty stack.'),
    'run-27-lead-10': ('incorrect', 'A separated green group above the core is visible, but lane 4 reports no incoming block; the reported red stack does not account for this group.'),
    'run-28-lead-10': ('incorrect', 'A separated blue block on the upper-right world lane is visible during rotation, but incoming detection reports none in every lane.'),
    'run-28-lead-2': ('uncertain', 'The large bottom block group is reported as an empty stack and an inner yellow incoming boundary; its visible outer extent is not represented reliably during the clear.'),
    'run-29-lead-5': ('incorrect', 'A separated red block in the lower-right lane lies nearer than the outer blue block, but the reported incoming boundary is the outer blue block.'),
}


def decide(audit, issues):
    expected = {f'run-{n:02}-lead-{lead}' for n in RUNS for lead in (10, 5, 2)}
    ids = [r['id'] for r in audit['frames']]
    if len(ids) != 81 or set(ids) != expected:
        raise ValueError('Missing, duplicate, excluded, or unexpected reviewed frames')
    if set(issues) - expected:
        raise ValueError('Review contains unknown frame IDs')
    if tuple(r['run'] for r in audit['provenance']) != RUNS:
        raise ValueError('Subset or original run order changed')
    alignment = audit['alignment']
    if len(alignment) != 64 or set(c['run'] for c in alignment) != set(RUNS):
        raise ValueError('Incomplete alignment audit')
    timing_ok = all(c.get('passed') is True and not c.get('searchEdge', True)
                    and abs(c.get('offsetSeconds', 1))+.15 <= .5 for c in alignment)
    upper_bound = (len(ids)-len(issues))/len(ids)
    # Fewer documented failures cannot automatically certify every other frame.
    extraction_failed = upper_bound < .9
    return {'schema': SCHEMA, 'alignmentPassed': timing_ok,
            'alignmentChecks': len(alignment), 'reviewedFrames': len(ids),
            'documentedIncorrect': sum(v[0]=='incorrect' for v in issues.values()),
            'documentedUncertain': sum(v[0]=='uncertain' for v in issues.values()),
            'correctFractionUpperBound': upper_bound, 'requiredCorrectFraction': .9,
            'status': 'closed_extraction_failure' if timing_ok and extraction_failed else 'not_approved',
            'forecastingAllowed': False, 'modelsTrained': 0,
            'excludedRuns': [5,14,17], 'additionalExcludedRuns': [],
            'reviewLimitation': 'Assistant visual judgments, not independent ground truth; rate is an optimistic bound, not measured model accuracy.'}


def main():
    frozen = json.loads(Path(__file__).with_name('reviewed_evidence.json').read_text())
    if digest(OUT/'audit.json') != frozen['auditSha256'] or digest(OUT/'protocol.json') != frozen['protocolSha256']:
        raise ValueError('Reviewed evidence changed; prior visual judgments are stale')
    if hashlib.sha256(json.dumps(ISSUES, sort_keys=True).encode()).hexdigest() != frozen['issuesSha256']:
        raise ValueError('Review judgments changed after freezing')
    audit = json.loads((OUT/'audit.json').read_text())
    # Verify source and image provenance rather than trusting a stale review.
    from playlens_ml.common import ROOT
    from playlens_ml.board_tracking.features import fingerprint
    protocol = json.loads((OUT/'protocol.json').read_text())
    if protocol['extractorHash'] != fingerprint():
        raise ValueError('Extractor changed after review')
    for source in audit['provenance']:
        directory = ROOT/'data/sessions'/source['sessionId']
        for filename, key in [('gameplay.webm','videoSha256'),('observations.jsonl','observationsSha256')]:
            if digest(directory/filename) != source[key]: raise ValueError('Changed source')
        cache = ROOT/f"artifacts/experiments/board-tracking-v1/cache/run-{source['run']:02}.json"
        if digest(cache) != source['cacheSha256']: raise ValueError('Changed extraction cache')
    for row in audit['frames']:
        if digest(OUT/'review'/f"{row['id']}-source.png") != row['sourceSha256']:
            raise ValueError('Changed review image')
    result = decide(audit, ISSUES)
    result.update(auditSha256=digest(OUT/'audit.json'), protocolSha256=digest(OUT/'protocol.json'),
                  reviewCodeSha256=digest(Path(__file__)), issues=ISSUES)
    (OUT/'decision.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Exploratory 27-run imminent-loss study', '',
        '**Stopped at the extraction checkpoint. No forecasting models were trained.**', '',
        'The user authorized excluding runs 5, 14 and 17 after the original timing audit failed. No further runs were excluded. Historical runs 31–40 were not used.', '',
        '## Completed checks', '',
        '- All 64 video/observation alignment checks passed the existing 0.5-second allowance, including checks near endings and on both sides of recorded pauses.',
        '- The original precisely reviewed loss intervals were retained for the remaining 27 recordings.',
        '- All 81 cached extraction overlays were reviewed at approximately 10, 5 and 2 seconds before loss. Additional panels show incoming colors, radii and gaps.',
        '- The source-video hashes, observation hashes, cache hashes and extractor fingerprint were verified. No extractor repair was made.', '',
        '## Extraction result', '',
        'Four frames contain clear incoming-block errors and six have stack/color assignments that could not be certified during clears or transitions. Even if every remaining frame were correct, the rate would be only **71/81 = 87.7%**, below the required **90%**. This is an optimistic upper bound, not a measured accuracy estimate.',
        'The existing cache marks every reviewed frame valid; that flag does not establish that its measurements are correct. Judgments are assistant visual review, not independent ground truth.', '',
        '| Frame | Review | Finding |', '|---|---|---|']
    for key, (status, note) in ISSUES.items(): lines.append(f'| {key} | {status} | {note} |')
    lines += ['', '## What this establishes', '',
        'The narrower cohort passes the checked timing alignment requirement, but current board extraction does not meet the retained visual review standard. Visible detached blocks are sometimes missed or replaced by farther blocks, and clear animations can make reported stack heights/colors unreliable.',
        'The pixels are visibly informative in the documented examples. This failure concerns the current extractor; it does not demonstrate insufficient predictive information or impossibility of a 2–5-second warning.', '',
        '## Stopping point', '',
        'Dataset assembly, model comparison, warning metrics and full-timeline coverage were not run after the extraction gate failed. There are no Brier, recall or baseline-superiority claims. Small-sample predictive uncertainty remains unassessed.',
        'The original fold boundaries, four candidates, baselines and success thresholds were preserved in the protocol. No further game exclusions, model search, capture changes, new gameplay or live warning changes were made. The original 30-run experiment and historical reports remain intact.', '',
        'Reproduce with the audit, incoming_review and checkpoint modules described in README.md. Local review images remain outside Git.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(result['status'], f"upper bound {upper(result):.1f}%")


def upper(result):
    return 100*result['correctFractionUpperBound']


if __name__=='__main__':main()
