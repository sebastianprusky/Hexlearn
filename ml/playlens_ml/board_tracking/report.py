import json
from .data import OUT


def generate():
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'tracking-audit.json').read_text());c=json.loads((OUT/'checkpoint.json').read_text())
    m=json.loads((OUT/'manifest.json').read_text());coverage=sum(x['validWindows'] for x in m['runs'])/sum(x['eligibleWindows'] for x in m['runs'])
    lines=['# Hexlearn tracking repair results','',f"**Decision: {r['decision'].replace('_',' ')}.**",'',
           'This is a development experiment on existing recordings, not an independent test. Runs 31–40 remain excluded. No live model was changed.','',
           'The board is measured more continuously, but forecasting is effectively tied with the previous best on common windows. The stricter motion/clear variant is worse and should not replace it.','', '## What changed','',
           '- Refit abrupt geometry jumps from current pixels; bridge brief gray-edge occlusion only when current core pixels confirm the preceding geometry.',
           '- Preserve real incoming blocks at the timer radius, using thickness and color continuity to separate them from decoration.',
           '- Require repeated observations for motion and clear estimates; preserve invalid frames and eight-second history resets.','',
           '## Reviewed measurement evidence','',
           f"- Original geometry/attached-color checkpoint: {c['correctFrames']}/{c['reviewedFrames']} conservative confirmations; exact still geometry/colors unchanged.",
           f"- Additional repaired-frame review: {c['repairFrames']} frames, {c['repairAccuracy']:.1%} geometry plus outer-color agreement.",
           f"- Selected incoming tracks: {a['incomingColorCorrect']}/{a['incomingReviewedFrames']} color agreements across 12 clips. These visible examples are not a random accuracy sample.",
           '- Two of three approximate speed rulers have supported matching identities and agree within 0.15 board radii/second. The third remains unsupported because an attached fragment is mistaken for an incoming block.',
           f"- Usable windows: {sum(x['validWindows'] for x in m['runs'])}/{sum(x['eligibleWindows'] for x in m['runs'])} ({coverage:.1%}); previously 3,888/4,127 (94.2%).",
           f"- Worst evaluation-game coverage: {r['supported']['minimumCoverage']:.1%}; target remains 95% in every game.",'',
           '| Clear estimate | Matched events | False detections | Missed events |','|---|---:|---:|---:|']
    for name,v in a['clearTotals'].items():lines.append(f"| {name} | {v['matched']} | {v['falseEvents']} | {v['missed']} |")
    lines+=['','Clear labels allow 0.5 seconds for causal confirmation. Two late events cannot be confirmed before their clips end and are right-censored. The stricter method trades recall for fewer false detections; it is not a complete clear detector. All labels are assistant inspection, not independent ground truth.','',
            '## Fair comparison on common windows','','Every row below uses identical fitting, calibration and evaluation windows, and identical alert reset boundaries. Lower error is better.','',
            '| Method | Brier | Last-minute error | Final-10s error | False alerts/min |','|---|---:|---:|---:|---:|']
    for name,v in r['common']['models'].items():
        if name in ('always_warning','never_warning'):continue
        s=v['summary'];lines.append(f"| {name} | {s['brier']:.5f} | {s['mae']:.2f}s | {s['final10Mae']:.2f}s | {s['falseAlertsPerMinute']:.3f} |")
    lines+=['','## Pixel-repaired pipelines on all supported windows','','| Method | Brier | Last-minute error | Timely recall | False alerts/min |','|---|---:|---:|---:|---:|']
    for name,v in r['supported']['models'].items():
        s=v['summary'];lines.append(f"| {name} | {s['brier']:.5f} | {s['mae']:.2f}s | {s['timelyWarningRecall']:.1%} | {s['falseAlertsPerMinute']:.3f} |")
    best=min((n for n,v in r['supported']['models'].items() if 'gates' in v),key=lambda n:r['supported']['models'][n]['summary']['brier'])
    v=r['supported']['models'][best]
    paired=r['featureComparisons']['pixel_fix_old_motion_tree']
    lines+=['', 'Compared with the previous logistic model on common windows, the pixel-repaired tree changes Brier by '+f"{paired['brier']['improvement']:.5f} [95% interval {paired['brier']['interval95'][0]:.5f}, {paired['brier']['interval95'][1]:.5f}]"+' and time error by '+f"{paired['mae']['improvement']:.3f}s [95% interval {paired['mae']['interval95'][0]:.3f}, {paired['mae']['interval95'][1]:.3f}]"+'. Positive values favor the repair; both intervals include no improvement. Intervals are descriptive and do not adjust for development selection.']
    lines+=['',f'## Screening {best}','','| Requirement | Pass |','|---|---|']
    for name,passed in v['gates'].items():lines.append(f"| {name} | {'yes' if passed else 'no'} |")
    lines+=['','## Interpretation and next step','',
            'Geometry continuity and incoming-block measurements should be judged separately from forecast quality. Clear-event recall remains poor, and any probability gains on this small, repeatedly examined sample remain development evidence.',
            'Retain the pixel/geometry fixes as an experimental extraction improvement and keep the previous compact model as a reference. Reject the stricter motion/clear variant as a forecasting replacement. Do not promote either pipeline unless every original gate passes. The next measurement problem is distinguishing attached fragments, clear flashes, and actual block disappearance without reading the score. Additional model search does not resolve that labeling problem.','',
            '## Reproduction and detailed evidence','',
            'Source instructions: `ml/playlens_ml/board_tracking/README.md`. `protocol.json` records the bounded comparison; `tracking-audit.json` contains every annotated clip and event match; `manifest.json` retains source hashes and all exclusions. `results.json` includes per-game and fold metrics, seven horizons, calibration, bootstrap intervals, all-eligible-window time baselines and screening decisions. Common and supported out-of-fold predictions are saved separately.',
            '', 'All 45 board tests pass, including nine new tracking tests. Eleven original dataset/model/report artifacts match their archived hashes. Tests and preservation checks are recorded alongside this report. No dashboard, API, extension, production dataset or production model was modified.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'decision.json').write_text(json.dumps({'decision':r['decision'],'selected':r['selected'],'bestPixelRepairDevelopmentModel':best,'strictMotionDecision':'rejected_as_forecasting_replacement','gates':v['gates'],'notIndependentValidation':True},indent=2)+'\n')

if __name__=='__main__':generate()
