"""Reproducible report for the native replay development experiment."""
import json
import hashlib
from .review import OUT
from .checkpoint import checkpoint
from .features import fingerprint


def generate(out=OUT, checkpoint_fn=checkpoint, hash_fn=fingerprint):
    gate=checkpoint_fn();manifest=json.loads((out/'manifest.json').read_text())
    timing=json.loads((out/'alignment.json').read_text());robustness=json.loads((out/'robustness.json').read_text())
    results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else None
    if results and (results['extractorHash']!=hash_fn() or results['datasetSha256']!=hashlib.sha256((out/'windows.npz').read_bytes()).hexdigest()):raise ValueError('Stale results')
    total=sum(r['eligibleWindows'] for r in manifest['runs']);valid=sum(r['validWindows'] for r in manifest['runs'])
    evaluated=[r for r in manifest['runs'] if r['run']>=16];minimum=min(r['coverage'] for r in evaluated)
    lines=['# Hexlearn: native replay experiment','',
        '**Status: '+('development comparison complete' if results else 'extraction complete; evaluation pending')+'.**','',
        '## What changed','',
        '- Existing native canvas replay pixels replace the tiny saved ML images. Replays are decoded at 960×524, from 2940×1602 originals.',
        '- Geometry uses visible gray fragments and the core orientation; four block colors and six attached-stack faces remain separate.',
        '- Temporal changes compare sorted distributions so a pure rotation cannot create false growth or clears. These are estimates, not tracked physical blocks.',
        '- The experiment is offline, uses runs 1–30 only, and leaves production artifacts and historical test runs 31–40 untouched.',
        '', '## Extraction evidence','',
        f'- Conservative visual review: **{gate["correctFrames"]}/60** stills confirmed, with uncertain cases counted as failures. All twelve clips inspected, covering rotation, stacking, incoming blocks and clears.',
        '- This review was performed by the assistant on development material, not an independent annotation team. Automatic validity and transform consistency do not prove semantic accuracy.',
        '- Five unresolved stills: run-05-2, run-12-2, run-13-2, run-19-2, run-26-2. Clear animations and detached layers remain the main ambiguity.',
        '- The earlier low-resolution review used conservative judgments on much less legible images. Its 25/60 score is not an independently verified ground-truth accuracy figure and should not be used as a formal before/after accuracy benchmark.',
        '', '| Transformation | Stable geometry | Identical lane colors |','|---|---:|---:|']
    for name,r in robustness['summary'].items():lines.append(f'| {name} | {r["geometryStable"]}/60 | {r["colorsStable"]}/60 |')
    lines+=['','## Timing and coverage','',
        f'- Additional alignment checks: {len(timing["checks"])}; maximum observed absolute match offset {max(abs(r["offsetSeconds"]) for r in timing["checks"]):.3f} seconds. Predictions use video pixels with a conservative **0.5-second delay**.',
        '- Alignment includes points after recorded pauses. Image matching empirically supports the margin; exact hardware synchronization is not available.',
        f'- Supported windows: **{valid:,}/{total:,} ({valid/total:.1%})**. Minimum coverage across evaluation games: **{minimum:.1%}**; screening requires 95% in every evaluated game.',
        '- Geometry changes, unsupported geometry, pauses, and capture gaps reset the eight-second history. All excluded anchors remain in the dataset.',
        '', '| Run | Eligible | Supported | Coverage |','|---|---:|---:|---:|']
    for r in manifest['runs']:lines.append(f'| {r["run"]} | {r["eligibleWindows"]} | {r["validWindows"]} | {r["coverage"]:.1%} |')
    if results:
        lines+=['','## Controlled development comparison','',
            'Every row below uses the same supported evaluation windows and gives each game equal weight. The six visual candidates use the two fixed model families; no parameter search was performed. Fold-specific calibrators see calibration games only.',
            '', '| Method | Brier ↓ | Last-minute error ↓ | Final-10s error ↓ | Timely recall ↑ | False alerts/min ↓ | Worst ECE ↓ |','|---|---:|---:|---:|---:|---:|---:|']
        for name,r in results['models'].items():
            s=r['summary'];lines.append(f'| {name} | {s["brier"]:.4f} | {s["mae"]:.2f}s | {s["final10Mae"]:.2f}s | {s["timelyWarningRecall"]:.0%} | {s["falseAlertsPerMinute"]:.3f} | {s["worstCalibrationError"]:.3f} |')
        lines+=['','## Screening decision','',f'**Selected candidate: {results["selected"] or "none"}.**','', '| Candidate | Failed requirements |','|---|---|']
        for name,r in results['models'].items():
            if 'gates' in r:lines.append('| '+name+' | '+(', '.join(k for k,v in r['gates'].items() if not v) or 'none')+' |')
        lines+=['','Complete per-game/fold metrics, seven horizon errors, calibration errors, paired descriptive run-bootstrap intervals, alert controls and all-eligible-window baseline results are in `results.json`.']
    lines+=['','## Interpretation and next decision','',
        '- **Extraction:** higher-resolution replay images support substantially clearer inspection. Remaining clear-animation ambiguity and unsupported intervals must stay visible; a high automatic coverage figure does not certify every color assignment.',
        '- **Predictive information:** development model comparisons determine whether these particular features help. Failure of these candidates cannot establish that all pixel-only approaches are impossible.',
        '- **Small-sample uncertainty:** only 15 games are evaluated across overlapping chronological folds. Repeated development changes increase selection bias. Bootstrap intervals describe these games and are not independent validation.',
        '- A pipeline may proceed to a new prospective test only if every original screening requirement passes. Nothing from this experiment enables live forecasts.',
        '', '## Reproduce','',
        'See `ml/playlens_ml/board_v2/README.md`. The source schema, feature names, video hashes, exclusions and timing margin are retained in `manifest.json`; exact review decisions and overlays are under `review/`.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return out/'REPORT.md'

if __name__=='__main__':print(generate())
