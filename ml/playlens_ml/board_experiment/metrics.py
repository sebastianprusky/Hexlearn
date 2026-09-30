"""Metrics v1: game-weighted errors and persistent alert episodes."""
import numpy as np
from . import ALERT_SCHEMA
from playlens_ml.common import HORIZONS_SECONDS, monotonic_probabilities, remaining_from_probabilities


def alert_metrics(probabilities, elapsed, remaining, segments):
    order=np.argsort(elapsed,kind='stable'); active=False; above=below=0
    previous_time=None; previous_segment=None; bucket=None; onsets=[]
    early_seconds=warning_early=observed_seconds=0.
    for index in order:
        t=float(elapsed[index]); seg=int(segments[index]); current_bucket=int(np.floor(t))
        if previous_time is not None and seg==previous_segment and current_bucket==bucket: continue
        if previous_time is None or seg!=previous_segment or t-previous_time>1.6:
            active=False; above=below=0
        p=float(probabilities[index]); r=float(remaining[index])
        if not np.isfinite(p):
            active=False; above=below=0; previous_time=None; continue
        above=above+1 if p>.5 else 0
        below=below+1 if p<.5 else 0
        if not active and above>=3:
            active=True; onsets.append(r); above=0
        elif active and below>=3:
            active=False; below=0
        # Each sampled prediction contributes at most its next active second.
        duration=min(1.,max(0.,r)); observed_seconds+=duration
        if r>60:
            duration=min(duration,r-60); early_seconds+=duration
            warning_early+=duration if active else 0.
        previous_time=t; previous_segment=seg; bucket=current_bucket
    timely=sum(15<=r<=60 for r in onsets); false=sum(r>60 for r in onsets)
    return {'timelyWarningRecall':float(timely>0),'alertPrecision':timely/len(onsets) if onsets else 0.,
            'falseAlertsPerMinute':false/(observed_seconds/60) if observed_seconds else 0.,
            'earlyWarningFraction':warning_early/early_seconds if early_seconds else 0.,
            'alertCount':len(onsets),'falseAlertCount':false,'timelyAlertCount':timely,
            'observedSeconds':observed_seconds,'earlySeconds':early_seconds,'warningEarlySeconds':warning_early,
            'onsetLeadSeconds':onsets}


def weighted_calibration(y,p,runs):
    weights=np.zeros(len(runs)); unique=np.unique(runs)
    for run in unique:
        m=runs==run; weights[m]=1/(len(unique)*m.sum())
    errors=[]
    for h in range(y.shape[1]):
        error=0.
        for lo in np.arange(0,1,.1):
            m=(p[:,h]>=lo)&(p[:,h]<(lo+.1) if lo<.9 else p[:,h]<=1.)
            weight=weights[m].sum()
            if weight: error+=weight*abs(np.average(y[m,h],weights=weights[m])-np.average(p[m,h],weights=weights[m]))
        errors.append(float(error))
    return errors


def evaluate(p,data,mask):
    p=monotonic_probabilities(p)
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    y=data['y_failure'][mask]; runs=data['run_order'][mask]; rem=data['time_to_failure_seconds'][mask]
    elapsed=data['elapsed_seconds'][mask]; segments=data['segments'][mask]
    prediction=remaining_from_probabilities(p); rows=[]
    for run in np.unique(runs):
        m=runs==run; last=m&(rem<=60); final=m&(rem<=10)
        if not last.any() or not final.any(): raise ValueError(f'Run {run} lacks final-minute/ten-second evaluation coverage')
        row={'run':int(run),'brier':float(np.mean((p[m]-y[m])**2)),
             'mae':float(np.mean(abs(prediction[last]-rem[last]))),
             'final10Mae':float(np.mean(abs(prediction[final]-rem[final]))),
             'horizonBrier':np.mean((p[m]-y[m])**2,axis=0).tolist()}
        row.update(alert_metrics(p[m,-1],elapsed[m],rem[m],segments[m])); rows.append(row)
    summary={key:float(np.mean([r[key] for r in rows])) for key in
             ('brier','mae','final10Mae','timelyWarningRecall','alertPrecision','falseAlertsPerMinute','earlyWarningFraction')}
    ece=weighted_calibration(y,p,runs); summary['worstCalibrationError']=max(ece)
    summary['horizonCalibrationError']=ece
    summary['horizonBrier']=np.mean([r['horizonBrier'] for r in rows],axis=0).tolist()
    return {'summary':summary,'perRun':rows}


def screen(result,baselines,fold_results,coverage):
    s=result['summary']
    return {
      'probabilityImprovement':all(s['brier']<=.9*b['summary']['brier'] for b in baselines.values()),
      'timeImprovement':all(s['mae']<=.9*b['summary']['mae'] for b in baselines.values()),
      'foldConsistency':sum(all(c['brier']<b['brier'] and c['mae']<b['mae'] for b in references.values()) for c,references in fold_results)>=2,
      'finalTenSeconds':s['final10Mae']<=1.1*min(b['summary']['final10Mae'] for b in baselines.values()),
      'calibration':s['worstCalibrationError']<=.12,
      'timelyWarnings':s['timelyWarningRecall']>=.6,
      'falseAlarms':s['falseAlertsPerMinute']<=.2,
      'coverage':bool(coverage) and min(coverage)>=.95,
    }
