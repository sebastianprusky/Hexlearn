"""One-shot frozen measurement gate; no forecasting training on failure."""
import hashlib
import json
import math
from pathlib import Path
from .data import OUT
from .detector import Detector,fingerprint


def score_events(events,truth,n,fps,ambiguous=(),uncertainty_frames=1):
    start=math.ceil(.5*fps);stop=n-math.ceil(.5*fps);excluded=set(ambiguous)
    scored=set(range(start,stop))-excluded
    actual=[t for t in truth if t in scored]
    available=set(actual);pairs=[];false=[]
    for event in sorted(set(events)):
        if event not in scored:continue
        options=[t for t in available if t-uncertainty_frames<=event<=t+math.floor(.5*fps)]
        if options:
            t=min(options);available.remove(t);pairs.append([t,event])
        else:false.append(event)
    positive=set()
    for t in actual:positive.update(range(t-uncertainty_frames,t+math.floor(.5*fps)+1))
    return {'matches':pairs,'falseEvents':false,'missedEvents':sorted(available),'actualEvents':actual,
            'reviewedSeconds':len(scored)/fps,'negativeSeconds':len(scored-positive)/fps,
            'scoredFrames':sorted(scored),'ambiguousFrames':sorted(excluded)}


def summarize(records):
    tp=sum(len(r['matches']) for r in records);fp=sum(len(r['falseEvents']) for r in records);fn=sum(len(r['missedEvents']) for r in records)
    seconds=sum(r['reviewedSeconds'] for r in records)
    return {'truePositive':tp,'falsePositive':fp,'falseNegative':fn,'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,'falseEventsPerMinute':fp*60/seconds if seconds else None,
            'reviewedSeconds':seconds,'negativeSeconds':sum(r['negativeSeconds'] for r in records)}


def run():
    result_path=OUT/'validation-results.json'
    if result_path.exists():raise ValueError('Validation already scored; inspect saved results, do not retune or overwrite')
    frozen=json.loads((OUT/'freeze.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    if frozen['detectorHash']!=fingerprint() or frozen['protocolSha256']!=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest():raise ValueError('Freeze changed')
    if frozen['evaluationSha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Evaluation changed after freeze')
    source=json.loads((OUT/'validation/observations.json').read_text());labels_path=OUT/'validation/labels.json';labels=json.loads(labels_path.read_text())
    if source['detectorHash']!=fingerprint():raise ValueError('Stale observations')
    by_id={c['id']:c for c in labels['clips']}
    expected={f'run-{n:02}' for n in protocol['validation']['runs']}
    if set(by_id)!=expected or {c['id'] for c in source['clips']}!=expected or len(labels['clips'])!=len(expected):raise ValueError('Incomplete annotation cohort')
    methods={name:[] for name in ('flash_detector','always_event','never_event')};supported=0;reviewed=0;ambiguous_count=0;total=0;traces=[]
    for c in source['clips']:
        label=by_id[c['id']]
        if label['reviewed'] is not True:raise ValueError('Clip not reviewed')
        fps=c['fps'];n=len(c['observations'])
        if fps!=protocol['samplingHz'] or n!=6*fps:raise ValueError('Unexpected clip cadence or length')
        if any(not isinstance(i,int) or not 0<=i<n for i in label['clearFrames']):raise ValueError('Invalid label frame')
        d=Detector(fps);pred=[d.update(o,i/fps,reset=i in c['resetIndices']) for i,o in enumerate(c['observations'])]
        ambiguous=set()
        for start,end in label.get('ambiguousIntervals',[]):ambiguous.update(range(start,end+1))
        events={'flash_detector':[i for i,p in enumerate(pred) if p['event']],'always_event':list(range(math.ceil(.5*fps),n,fps)),'never_event':[]}
        for name,event in events.items():methods[name].append({'id':c['id'],**score_events(event,label['clearFrames'],n,fps,ambiguous)})
        eligible=methods['flash_detector'][-1]['scoredFrames'];supported+=sum(pred[i]['supported'] for i in eligible);reviewed+=len(eligible);ambiguous_count+=len(ambiguous);total+=n
        traces.append({'id':c['id'],'predictions':pred})
    summaries={name:summarize(values) for name,values in methods.items()};s=summaries['flash_detector'];g=protocol['measurementGate']
    gates={'enoughEvents':s['truePositive']+s['falseNegative']>=g['minimumConfirmableEvents'],
           'enoughNegativeTime':s['negativeSeconds']>=g['minimumReviewedNegativeSeconds'],
           'precision':s['precision']>=g['minimumPrecision'],'recall':s['recall']>=g['minimumRecall'],
           'falseEventRate':s['falseEventsPerMinute'] is not None and s['falseEventsPerMinute']<=g['maximumFalseEventsPerMinute'],
           'coverage':supported/max(1,reviewed)>=g['minimumSupportedFrameFraction'],
           'ambiguity':ambiguous_count/max(1,total)<=g['maximumAmbiguousFrameFraction']}
    payload={'detectorHash':fingerprint(),'protocolSha256':frozen['protocolSha256'],'labelsSha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
             'observationSha256':hashlib.sha256((OUT/'validation/observations.json').read_bytes()).hexdigest(),
             'evaluationCodeSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'methods':summaries,'perClip':methods,
             'supportedFraction':supported/max(1,reviewed),'ambiguousFraction':ambiguous_count/max(1,total),'gates':gates,
             'passed':all(gates.values()),'nextAction':'one_fixed_forecasting_ablation' if all(gates.values()) else 'stop_before_forecasting',
             'limitation':'New clip segments from previously used development games; assistant labels; not independent generalization evidence.'}
    result_path.write_text(json.dumps(payload,indent=2)+'\n');(OUT/'validation/predictions.json').write_text(json.dumps(traces)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k in ('methods','gates','passed','nextAction')},indent=2))
    return payload

if __name__=='__main__':run()
