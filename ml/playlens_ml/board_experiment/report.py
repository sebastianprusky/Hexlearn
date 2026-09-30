import json
import hashlib
from .features import extractor_hash
from .data import DEFAULT_OUT
from .review import checkpoint


def generate(out=DEFAULT_OUT):
    gate=checkpoint(out)
    manifest=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else None
    robustness=json.loads((out/'robustness.json').read_text()) if (out/'robustness.json').exists() else None
    results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() and gate['passed'] else None
    if results and (results['extractorHash']!=extractor_hash() or results['datasetSha256']!=hashlib.sha256((out/'windows.npz').read_bytes()).hexdigest()):
        raise ValueError('Evaluation results do not match the current extractor/dataset')
    decision='improve_extraction_first' if not gate['passed'] else (results['decision'] if results else 'evaluation_pending')
    lines=['# Hexlearn board-feature experiment v1','',f'**Decision: {decision.replace("_"," ")}.**','',
        'This experiment uses only the original 30 development games. Original test games 31–40, production datasets, live models, the API, and dashboard are unchanged.',
        '', '## Extraction checkpoint','',
        f'- Confirmed correct geometry and all visible lane colors: **{gate["correctFrames"]}/60 ({gate["accuracy"]:.1%})**. Required: 54/60 (90%).',
        '- Review is conservative assistant inspection of fixed contact sheets and twelve chronological clip storyboards, not an independently labeled accuracy benchmark. Uncertain assignments count as failures.',
        '- Clip coverage: '+', '.join(gate['eventCoverage'])+'. GIFs and review decisions are retained under `review/`.',
        '- In examples run-21-1 and run-28-2, the bottom lane is assigned an inner color instead of the visible outer green/red layer. Rotating blocks, JPEG aliasing, and gaps between layers make the radial connectivity heuristic unreliable.',
        '- Gray-board localization usually works, but appearance validity does not establish semantic correctness. A frame can pass the automatic geometry test while its lane colors are wrong.']
    if robustness:
        lines+=['','## Resize and letterbox checks','','| Transformation | Stable geometry | Identical lane colors |','|---|---:|---:|']
        for name,r in robustness['summary'].items(): lines.append(f'| {name} | {r["geometryStable"]}/60 | {r["identicalLaneColors"]}/60 |')
        lines+=['','Geometry stability requires center and scale within 5% of board size. Color stability compares predictions before and after transformation; it does not establish that either prediction is correct. Invalid reference detections count as failures.']
    if manifest:
        runs=manifest['runs']; total=sum(r['eligibleWindows'] for r in runs); usable=sum(r['validWindows'] for r in runs)
        lines+=['','## Dataset coverage','',f'{usable:,}/{total:,} eligible windows pass automatic extraction and continuous-history checks ({usable/total:.1%}). Lowest per-game coverage: {min(r["coverage"] for r in runs):.1%}. Required: at least 95% in every game.',
                '', '| Run | Eligible windows | Valid windows | Coverage |','|---|---:|---:|---:|']
        lines += [f'| {r["run"]} | {r["eligibleWindows"]} | {r["validWindows"]} | {r["coverage"]:.1%} |' for r in runs]
        lines+=['','Every eligible anchor remains in `windows.npz` with its validity, segment ID, and exclusion reason. No failed windows or games are silently removed from the coverage denominator. Frame hashes and feature names are in `manifest.json`; frame extraction is cached by content hash and extractor revision.']
    if results:
        lines+=['','## Development comparison','','| Model | Brier | Last-minute MAE | Final-ten-second MAE |','|---|---:|---:|---:|']
        for name,r in results['models'].items():
            s=r['summary']; lines.append(f'| {name} | {s["brier"]:.4f} | {s["mae"]:.2f} s | {s["final10Mae"]:.2f} s |')
        lines+=['','Per-fold, per-game, calibration, alert, bootstrap interval, coverage, and gate results are in `results.json`. Candidate comparisons use common supported windows; reference results on all eligible windows are reported separately.']
    else:
        lines+=['','## Model comparison','',
            ('**Real model training was not run because the extraction checkpoint failed.**' if not gate['passed'] else '**Model evaluation has not been run yet.**') + ' The six-candidate comparison, four time baselines, persistent-alert metrics, screening, and conditional candidate freeze are implemented behind that checkpoint. There are no new predictive scores to compare with the original audit.']
    lines+=['','## What this establishes','',
        ('- **Unreliable extraction:** demonstrated for this prototype. The current lane/color heuristic is not reliable enough to support a predictive experiment.' if not gate['passed'] else '- **Extraction review:** passed for this reviewed revision; per-game coverage remains a separate screening requirement.'),
        ('- **Insufficient predictive information:** not established. No model was trained on these new features after the checkpoint failure.' if not results else '- **Predictive information:** inspect each failed screening gate above and in results.json. A failed candidate does not establish that all pixel-based models must fail.'),
        '- **Small-sample uncertainty:** remains. Even a later development win across these 15 evaluation games would need a frozen pipeline and a new prospective test.',
        '', 'The saved images are 160×160; individual strips are only a few pixels thick. The review does not prove the recordings are unusable or that vision cannot beat time baselines. It shows that this extractor does not meet the agreed standard.',
        '', '## Recommendation','',
        ('Improve extraction first. Create explicit lane/color annotations on the existing review frames and handle rotation/stack connectivity before another prediction experiment. Assess whether saved replay video exposes finer detail before requesting new gameplay. Do not add a larger model, tune against runs 31–40, or collect more runs to compensate for incorrect inputs.' if not gate['passed'] else ('Freeze the selected candidate and define a new prospective test before any collection or live promotion.' if results and results['selected'] else 'Stop this experiment without expanding the candidate search or requesting additional games. Review the reported failing gates before defining another study.')),
        '', '## Reproduce','', 'From `playlens`, use `PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment COMMAND`.',
        '', 'Commands: `review`, `robustness`, `extract`, `checkpoint`, `evaluate`, `report`. Review decisions must match the current extractor and source dataset. `evaluate` refuses to train when the review fails. The README in the source package documents outputs and defaults.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'decision.json').write_text(json.dumps({'decision':decision,'extractionCheckpoint':gate,'realTrainingRun':bool(results)},indent=2)+'\n')
    return out/'REPORT.md'
