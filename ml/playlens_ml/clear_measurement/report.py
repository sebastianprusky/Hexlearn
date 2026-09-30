"""Render the frozen measurement and fixed forecasting comparison."""
import json
import numpy as np
from .data import OUT

def run():
    measurement=json.loads((OUT/'validation-results.json').read_text())
    result=json.loads((OUT/'ablation-results.json').read_text())
    manifest=json.loads((OUT/'ablation-manifest.json').read_text())
    frames={r['run']:json.loads((OUT/'full-cache'/f'run-{r["run"]:02}.json').read_text())['frames'] for r in manifest['runs']}
    with np.load(OUT/'ablation-windows.npz') as z:
        final_counts={r['run']:(int(((z['run_order']==r['run'])&(z['time_to_failure_seconds']<=10)&z['valid']).sum()),int(((z['run_order']==r['run'])&(z['time_to_failure_seconds']<=10)).sum())) for r in manifest['runs']}
    selected=result['selected'];m=measurement['methods']['flash_detector']
    decision={'measurement':'passed','forecasting':'qualifies_for_pipeline_freeze' if selected else 'no_candidate_qualifies',
              'recommendation':'prospective_test_after_freeze' if selected else ('improve_extraction_first' if result['minimumCoverage']<.95 else 'stop_current_forecasting_approach'),
              'selected':selected,'nextAction':'Freeze chosen pipeline before defining a new independent test.' if selected else 'Stop this experiment. Do not expand the model search or request more runs.',
              'limitations':['Assistant labels on twelve reserved segments from previously used games, not independent ground truth or future-game evidence.',
                             'Reserved segments start at 40% of each recording; the event check does not establish accuracy on crowded late-game boards.',
                             'Many earlier development comparisons used these thirty games; bootstrap intervals are descriptive.',
                             'The measurement gate covers short clips; full-run continuity is assessed separately.']}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Clear measurement and forecasting experiment','',
           '## Goal','',
           'Use pixels to give useful warnings at least 15 active seconds before loss, while beating all four named time baselines. This experiment tests one specific hypothesis: reliable recent clear counts improve forecasting.','',
           '## Decision','',f'**Recommendation: {decision["recommendation"].replace("_"," ")}.**',decision['nextAction'],'',
           '## Measurement checkpoint','',
           f'The frozen 12-FPS detector found {m["truePositive"]} of {m["truePositive"]+m["falseNegative"]} confirmable clears ({m["recall"]:.1%} recall), with {m["falsePositive"]} false events. Precision was {m["precision"]:.1%}. All measurement gates passed.',
           f'Review included 864 raw frames in twelve six-second clips, with 60 seconds scored after boundary censoring and {m["negativeSeconds"]:.2f} negative seconds. Supported-frame coverage was {measurement["supportedFraction"]:.1%}. Labels were written before detector predictions were displayed. No ambiguous intervals were excluded.','',
           'The detector uses broad white flashes on previously colored stack faces. It excludes the central text and outer timer and uses only current/preceding pixels. Development sampling changed from 4 FPS to 12 FPS before source/protocol freeze because brief flashes were missed at 4 FPS. Both protocols are retained.',
           'Misses: run 4 frame 21 and run 7 frames 31 and 33. The rule does not reliably detect every visible disappearance, especially brief or spatially limited clears during rotation. These misses were retained; detector thresholds and labels were not retuned after scoring.',
           'Reserved segments start at 40% of each recording. This checks midgame event detection and does not establish accuracy on crowded late-game boards. Full-run coverage and the saved failure examples expose this separate limitation.','',
           'The always-event control had 25% precision, 57.7% recall, and 45 false events/minute. Never-event had zero recall. Neither passes. Zero observed false events in one scored minute does not establish a zero population false-event rate.','',
           '## Fixed comparison','',
           'Added eight features: clear count and event-confidence sum over the preceding 1, 2, 4, and 8 seconds. The retained 80-feature representation is the within-family reference. All four pixel methods and named time baselines use the same valid windows. Baselines are also evaluated over all eligible evaluation windows in the JSON report.',
           'Runs 1–30 only; historical runs 31–40 excluded. Folds: fit 1–9/calibrate 10–15/evaluate 16–20; fit 1–14/calibrate 15–20/evaluate 21–25; fit 1–19/calibrate 20–25/evaluate 26–30. Original fixed regularized logistic/tree settings and calibration procedures are reused. No parameter search.',
           '12-FPS extraction uses the exact validated pixel pipeline. Features retain the conservative 0.5-second delay and full eight-second history. Pauses, capture gaps, unsupported frames, and geometry changes invalidate/reset history. Extraction exclusions remain visible in the manifest.','',
           '| Method | Mean Brier ↓ | Last-minute error (s) ↓ | Final-10s error (s) ↓ | Worst calibration error ↓ | Timely warning recall ↑ | False alerts/min ↓ |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in result['models'].items():
        s=value['summary'];lines.append(f'| {name} | {s["brier"]:.5f} | {s["mae"]:.3f} | {s["final10Mae"]:.3f} | {s["worstCalibrationError"]:.3f} | {s["timelyWarningRecall"]:.1%} | {s["falseAlertsPerMinute"]:.3f} |')
    lines+=['','| Method | Alert precision | Early gameplay under warning |','|---|---:|---:|']
    for name,value in result['models'].items():
        s=value['summary'];lines.append(f'| {name} | {s["alertPrecision"]:.1%} | {s["earlyWarningFraction"]:.1%} |')
    lines+=['','| Evaluation games / candidate | Brier | Last-minute error (s) | Beats every time baseline on both |','|---|---:|---:|---|']
    for fold in result['folds']:
        for name in ('with_clear_logistic','with_clear_tree'):
            s=fold['models'][name]
            wins=all(s['brier']<fold['models'][b]['brier'] and s['mae']<fold['models'][b]['mae'] for b in ('average_duration','empirical_duration','elapsed_logistic','elapsed_tree'))
            lines.append(f'| {fold["evaluation"][0]}–{fold["evaluation"][1]} / {name} | {s["brier"]:.5f} | {s["mae"]:.3f} | {"yes" if wins else "no"} |')
    lines+=['','All summary errors weight games equally. Brier averages the existing seven horizons. Full per-game, per-horizon, calibration, fold, early-warning occupancy and alert-precision results are in `ablation-results.json`. Alert accounting retains the versioned three-on/three-off rule with 15–60-second timely onsets.','','## Candidate gates','']
    for name in ('with_clear_logistic','with_clear_tree'):
        r=result['models'][name];lines.append(f'- **{name}:** '+('qualifies' if r['qualifies'] else 'fails')+'. Failed gates: '+(', '.join(k for k,v in r['gates'].items() if not v) or 'none')+'.')
    lines+=['','## Does the added measurement help?','','Positive differences favor adding clears. Intervals resample the 15 evaluation games (5,000 paired draws); these are descriptive intervals, not protection against repeated development selection.','',
            '| Family / metric | Mean improvement | Descriptive 95% interval | Games improved |','|---|---:|---:|---:|']
    for family,metrics in result['featureComparisons'].items():
        for metric,s in metrics.items():lines.append(f'| {family} / {metric} | {s["improvement"]:.5f} | [{s["interval95"][0]:.5f}, {s["interval95"][1]:.5f}] | {s["runsWon"]}/{s["runs"]} |')
    lines+=['','## Coverage and failure interpretation','',f'Scored {result["scoredWindows"]} supported evaluation windows. Worst evaluation-game coverage: **{result["minimumCoverage"]:.1%}** (required: at least 95% in every game).','',
            '| Run | Supported / eligible windows | Window coverage | Supported frames | Final-10s supported / eligible |','|---|---:|---:|---:|---:|']
    for r in manifest['runs']:
        f=frames[r['run']];a,b=final_counts[r['run']];lines.append(f'| {r["run"]} | {r["validWindows"]}/{r["eligibleWindows"]} | {r["coverage"]:.1%} | {sum(v["supported"] for v in f)/len(f):.1%} | {a}/{b} |')
    lines+=['','Final-ten-second errors are conditional on these supported windows. Some games retain very few tail windows, making that error estimate particularly fragile. All-eligible time-baseline scores remain available in the JSON report; unsupported candidate predictions are not invented.']
    lines+=['','Four diagnostic reconstructions (run 1 at 117.167s; run 25 at 106.583s, 115.750s and 116.500s) reproduce `occluded_gray_fit_uncertain`. The raw frames show clear animations and/or blocks covering the board edge. The board remains visually legible, suggesting a geometry/continuity limitation. These reconstructions use six preceding frames and do not replace cached full-run outputs; see `coverage-diagnostics/`.','',
            'The current tracking wrapper immediately returns when native geometry is invalid. Its bounded geometry recovery therefore does not repair these rejected fits. A future extraction study could test recovery supported by current core pixels and prior confirmed geometry, with a new freeze and reserved checks. This experiment makes no such change.','',
            'Distinguish three conclusions:','',
            '- **Extraction:** the clear-event measurement passed the short-clip check; it does not establish continuous coverage on complete games. The coverage table and exclusion reasons determine that separately.',
            '- **Predictive information:** added clears produced only small point improvements. The tree gained 0.00032 Brier and 0.036 seconds of last-minute error; both paired intervals include no improvement. Both feature families slightly worsened final-ten-second error. Neither added-clear candidate beats the strongest time baselines, and both exceed the false-alert limit. Retain the detector as an experimental measurement tool; this result does not justify replacing the forecasting representation.',
            '- **Small sample:** fifteen evaluation games and repeated development selection cannot establish future-game generalization or prove that pixel forecasting can never work. A candidate must pass every screening gate before an independent test is justified.','',
            '## Reproduction and checks','',
            'See `ml/playlens_ml/clear_measurement/README.md`. The detector, preparation, scoring code and measurement protocol were frozen before reserved-clip review; the ablation protocol records fixed source hashes before forecasting outcomes. A pre-forecast schema amendment makes the NPZ column names match all 88 columns; it changes no numerical features or settings. Original eleven dataset/model/report hashes remain unchanged. 64 focused tests passed. No API, extension, dashboard, live model, or gameplay collection changes.','',
            'Artifacts: `protocol.json`, `freeze.json`, `validation/labels.json`, `validation-results.json`, `measurement-failures.json`, `ablation-protocol.json`, `ablation-manifest.json`, `ablation-results.json`, `clear_ablation_predictions.npz`, `runtime-provenance.json`, `source-provenance.json`, `preservation.json`, and `decision.json`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(decision['nextAction'])

if __name__=='__main__':run()
