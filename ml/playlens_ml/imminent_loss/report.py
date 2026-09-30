"""Reproduce the stopped experiment report from reviewed, hashed evidence."""
import json
from pathlib import Path
from playlens_ml.common import ROOT
from .review import OUT, digest
from .gate import timing_gate


def main():
    manifest = json.loads((OUT / 'review-manifest.json').read_text())
    annotation_path = Path(__file__).with_name('reviewed_endings.json')
    annotations = json.loads(annotation_path.read_text())
    if annotations['manifestSha256'] != digest(OUT / 'review-manifest.json'):
        raise ValueError('Stale visual annotations; review the changed evidence')
    verification = json.loads((OUT / 'full-decode-verification.json').read_text())
    if [r['run'] for r in verification] != [5, 14, 17] or not all(r['sameAllTailPixels'] for r in verification):
        raise ValueError('Full-stream verification failed')
    for source in manifest['runs']:
        directory = ROOT / 'data/sessions' / source['sessionId']
        if digest(directory / 'gameplay.webm') != source['videoSha256'] or digest(directory / 'observations.jsonl') != source['observationsSha256']:
            raise ValueError('Source recordings changed')
    for row in annotations['runs']:
        for image in row['reviewedImages']:
            if digest(OUT / image['path']) != image['sha256']:
                raise ValueError('Changed review image')
    if [r['run'] for r in manifest['runs']] != list(range(1, 31)):
        raise ValueError('Development cohort changed')
    result = timing_gate(annotations['runs'])
    result.update({'schema': 'imminent-loss-v1', 'modelsTrained': 0,
                   'runsExcluded': [], 'historicalRunsUsed': [],
                   'annotationsSha256': digest(annotation_path)})
    (OUT / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Imminent loss v1: stopped at the timing checkpoint', '',
             'The recordings do not meet the predeclared outcome-timing requirements.',
             'No models were trained; no predictive results or baseline wins are claimed.', '',
             '| Run | Last clearly active video frame | First visible loss frame | Interval width |',
             '|---|---:|---:|---:|']
    for row in annotations['runs']:
        interval = row['lossIntervalVideoSeconds']
        if interval is None:
            lines.append(f"| {row['run']} | {row['lastSavedVideoSeconds']:.3f} | Not recorded | Unknown |")
        else:
            lines.append(f'| {row["run"]} | {interval[0]:.3f} | {interval[1]:.3f} | {interval[1]-interval[0]:.3f} s |')
    lines += ['', '## Interpretation', '',
        '- Run 5 has a 0.342-second gap across the transition; the limit is 0.250 seconds.',
        '- Runs 14 and 17 end with the score still visible and no recorded loss transition.',
        '- All 30 ending boundary pairs were visually reviewed. Native tail images and final-ten-second contact sheets were saved for every game.',
        '- Full-stream decoding of runs 5, 14 and 17 reproduced every tail image, confirming these are not seek or frame-rate-conversion artifacts.',
        '- Score disappearance is used only for outcome review. It is never a model input. Local renderer code corroborates this visible transition; no engine state was used as a feature.',
        '- Times above are video presentation timestamps, not validated active-time labels. End/pause alignment and the extraction review were not completed after the timing gate failed.',
        '- No games were silently removed. Runs 31–40 were not used. Production and prior experiment outputs remain unchanged.', '',
        '## Answers', '',
        '1. Timing is insufficient under the agreed rule. Near-loss extraction reliability remains unassessed.',
        '2. Whether pixels beat time and crowding at this horizon is untested.',
        '3. There is no validated basis for an independent test or live warnings.', '',
        'This is a timing limitation, not evidence that imminent loss is unpredictable. Predictive information and small-sample uncertainty cannot be assessed without passing the input gate. The experiment is closed; no expanded search or new gameplay is requested.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(result['status'])


if __name__ == '__main__':
    main()
