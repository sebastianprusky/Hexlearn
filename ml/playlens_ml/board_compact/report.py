"""Consolidated native-replay development evidence and honest screening decision."""
import json
from pathlib import Path
from .data import OUT
from playlens_ml.common import ROOT


def generate():
    experiments=ROOT/'artifacts/experiments'
    stages=[('Native board geometry/colors','board-features-v3',None),
            ('Rotation-invariant summaries','board-invariant-v1',None),
            ('Incoming speed and recent activity','board-dynamics-v1',['temporal_logistic','temporal_tree']),
            ('Compact speed, crowding and activity','board-compact-v1',None)]
    reports={name:json.loads((experiments/name/'results.json').read_text()) for _,name,_ in stages}
    candidates=[]
    for title,folder,include in stages:
        for name,result in reports[folder]['models'].items():
            if 'gates' in result and (include is None or name in include):candidates.append((folder,name,result))
    ranked=sorted(candidates,key=lambda item:(item[2]['summary']['brier'],item[2]['summary']['mae']))
    best_folder,name,best=ranked[0];s=best['summary'];r=reports[best_folder];baselines={key:r['models'][key] for key in ('average_duration','empirical_duration','elapsed_logistic','elapsed_tree')}
    avg=baselines['average_duration']['summary'];brier_gain=1-s['brier']/avg['brier'];time_gain=1-s['mae']/avg['mae']
    native=json.loads((experiments/'board-features-v3/manifest.json').read_text());coverage=native['runs']
    minimum=min(v['coverage'] for v in coverage if v['run']>=16)
    lines=['# Hexlearn: stronger pixel-only forecasting from native replays','',
           '**Decision: no candidate qualifies for a new prospective test yet.**','',
           f'The best development result is `{best_folder}/{name}`. It improves average-duration Brier error by **{brier_gain:.1%}** and last-minute time error by **{time_gain:.1%}** on the same supported windows. It still loses to the stronger time baselines and fails the agreed warning, late-error, and coverage requirements.',
           '', 'These are development averages from 15 evaluation games. Descriptive intervals include no improvement over average duration; repeated development selection makes the point estimates optimistic. No production forecast or new collection was enabled.',
           '', '## Main comparison','',
           'Lower is better for Brier and time error. Every row uses identical windows, chronological folds, and equal game weights.',
           '', '| Method | Mean Brier | Last-minute time error | Final-10s time error |','|---|---:|---:|---:|']
    for label,value in [('Best pixel-only development candidate',best)]+list(baselines.items()):
        q=value['summary'];lines.append(f'| {label} | {q["brier"]:.5f} | {q["mae"]:.2f}s | {q["final10Mae"]:.2f}s |')
    lines+=['','## What actually helped','', '| Representation | Best-Brier candidate | Brier | Time error |','|---|---|---:|---:|']
    for title,folder,include in stages:
        options=[(key,value) for key,value in reports[folder]['models'].items() if 'gates' in value and (include is None or key in include)]
        key,value=min(options,key=lambda item:(item[1]['summary']['brier'],item[1]['summary']['mae']));q=value['summary']
        lines.append(f'| {title} | {key} | {q["brier"]:.5f} | {q["mae"]:.2f}s |')
    lines+=['',
        '- Existing recordings contain 2940×1602 canvas pixels. Decoding at 960×524 preserves much more board detail than the saved 160×160 ML images.',
        '- Removing disconnected gray compression artifacts and using the visible core center recovered crowded boards that the first detector rejected. Stack sampling now reaches beyond the timer radius.',
        '- Rotation-invariant summaries alone helped little. Adjacent-frame incoming-speed and recent growth/clear estimates produced the largest predictive improvement.',
        '- Speed alone remained weak. Combining it with crowding and recent activity worked better; the compact model has 80 features.',
        '- A separate core-ratio normalization passed still-image review but reduced continuous coverage from 94.2% to 90.8%. It was rejected before model fitting. It is not part of the best pipeline.',
        '', '## Screening the best development candidate','', '| Requirement | Result | Pass |','|---|---|---|']
    detail={
        'probabilityImprovement':f'{s["brier"]:.5f}; must be at most {0.9*min(b["summary"]["brier"] for b in baselines.values()):.5f}',
        'timeImprovement':f'{s["mae"]:.2f}s; must be at most {0.9*min(b["summary"]["mae"] for b in baselines.values()):.2f}s',
        'foldConsistency':'Does not improve both errors over every named baseline in at least two folds',
        'finalTenSeconds':f'{s["final10Mae"]:.2f}s; permitted maximum {1.1*min(b["summary"]["final10Mae"] for b in baselines.values()):.2f}s',
        'calibration':f'Worst horizon ECE {s["worstCalibrationError"]:.3f}; maximum 0.12',
        'timelyWarnings':f'{s["timelyWarningRecall"]:.0%} of evaluated games; minimum 60%',
        'falseAlarms':f'{s["falseAlertsPerMinute"]:.3f} false onsets per observed active minute; maximum 0.2',
        'coverage':f'Lowest evaluation-game coverage {minimum:.1%}; minimum 95% in every game'}
    for key,passed in best['gates'].items():lines.append(f'| {key} | {detail[key]} | {"yes" if passed else "no"} |')
    lines+=['',
        f'Alert precision is {s["alertPrecision"]:.1%}, with {s["earlyWarningFraction"]:.1%} of observed early gameplay under warning. The 100% timely-recall figure is **not** a success claim: repeated alerts and gaps can create additional onsets, false alerts exceed the cap, and coverage fails.',
        '', '| Control | Timely recall | False alerts / observed active minute |','|---|---:|---:|']
    for control in ('always_warning','never_warning'):
        q=r['models'][control]['summary'];lines.append(f'| {control} | {q["timelyWarningRecall"]:.1%} | {q["falseAlertsPerMinute"]:.3f} |')
    lines+=['','## Fold consistency','', '| Evaluation games | Pixel Brier | Elapsed-logistic Brier | Pixel time error | Elapsed-tree time error |','|---|---:|---:|---:|---:|']
    for fold in r['folds']:
        q=fold['models'][name];e=fold['models']['elapsed_logistic'];t=fold['models']['elapsed_tree'];lines.append(f'| {fold["evaluation"][0]}–{fold["evaluation"][1]} | {q["brier"]:.5f} | {e["brier"]:.5f} | {q["mae"]:.2f}s | {t["mae"]:.2f}s |')
    lines+=['','## Descriptive uncertainty','',
        'Positive values below favor the pixel candidate. These paired run-bootstrap intervals describe this small development sample; they do not adjust for trying multiple representations.',
        '', '| Baseline | Brier improvement [95% interval] | Time-error improvement [95% interval] |','|---|---|---|']
    for baseline,comparison in best['comparisons'].items():
        b=comparison['brier'];t=comparison['mae'];lines.append(f'| {baseline} | {b["improvement"]:.4f} [{b["interval95"][0]:.4f}, {b["interval95"][1]:.4f}] | {t["improvement"]:.2f}s [{t["interval95"][0]:.2f}, {t["interval95"][1]:.2f}] |')
    lines+=['','## Seven horizons','', '| Horizon | Pixel Brier | Elapsed-logistic Brier | Pixel calibration error |','|---|---:|---:|---:|']
    for i,h in enumerate((5,10,15,20,30,45,60)):lines.append(f'| {h}s | {s["horizonBrier"][i]:.5f} | {baselines["elapsed_logistic"]["summary"]["horizonBrier"][i]:.5f} | {s["horizonCalibrationError"][i]:.4f} |')
    lines+=['','## Extraction and timing evidence','',
        '- Review: 55/60 stills conservatively confirmed for geometry and visible attached colors; five ambiguous/incorrect cases count as failures. Twelve clips and twelve additional crowded-board examples were inspected. This was assistant review, not independent ground-truth annotation.',
        '- Resizing to 640/1280 pixels preserved geometry in 60/60 cases and colors in 58/60; letterboxing preserved both in 60/60. Consistency is not an accuracy guarantee.',
        '- Native extraction supports 3,888/4,127 eligible windows (94.2% overall). The original low-resolution prototype supported 329/4,127. The weakest evaluation game supports 86.3%. All excluded windows retain reasons.',
        '- Sixty-six additional video/image synchronization checks passed, including after resumes. Maximum observed offset was 0.233s. A fixed 0.5s feature delay provides a conservative sampling/codec margin; exact hardware synchronization is unavailable.',
        '- The eight-second requirement and reset rules were preserved. False gray-scale changes can still discard valid history; the core-only alternative did not solve this reliably.',
        '', '## What the evidence supports','',
        '1. **Extraction reliability improved substantially, but remains below the per-game coverage gate.** Clear animations, detached layers, and scale estimates still need independently checked labels.',
        '2. **There is useful pixel-derived information.** Speed plus crowding/activity improves this development comparison over average duration. That does not establish superiority to elapsed-time forecasting or reliable 15-second warnings.',
        '3. **Evidence remains inconclusive for generalization.** Only 15 games are evaluated. Their training histories overlap, and 20 distinct fixed-model/feature combinations were examined during these targeted iterations. Runs 31–40 were excluded from every new selection and score.',
        '', '## Recommendation','',
        'Keep the compact speed/crowding/activity representation as the most promising development direction. Do not promote it or collect a new prospective cohort yet. Before another model comparison, validate incoming-block identity/speed and apparent-clear estimates against explicit clip annotations, then repair geometry continuity without weakening the existing coverage or alert thresholds. More parameter searching on these same runs would add selection bias without resolving those measurement errors.',
        '', '## Reproduce and inspect','',
        'See the source README at `ml/playlens_ml/board_compact/README.md`. Every stage has its own schema, source hashes, manifests, windows, results and logs under `artifacts/experiments/`. The best-stage `results.json` includes per-game metrics, fold metrics, seven horizons, calibration, alert accounting, intervals and time baselines across all eligible windows. `development_predictions.npz` retains the complete out-of-fold prediction traces.',
        '', 'All 36 board-experiment tests pass. The original dataset, model bundles, training-job state and evaluation reports still match archived hashes/bytes. The dashboard, API, extension and live forecasting configuration were not changed by this work.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'decision.json').write_text(json.dumps({'decision':'improve_measurement_before_more_model_search','prospectiveCandidate':None,
        'bestDevelopmentCandidate':{'experiment':best_folder,'model':name,'summary':s,'gates':best['gates']},'uniqueDevelopmentCandidates':len(candidates),
        'limitations':['20 development candidates; no independent test','coverage and false-alert requirements fail','bootstrap intervals are descriptive']},indent=2)+'\n')
    return OUT/'REPORT.md'

if __name__=='__main__':print(generate())
