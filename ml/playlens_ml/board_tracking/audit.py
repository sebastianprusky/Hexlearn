"""Score explicit clip labels before looking at forecasting results."""
import json
import numpy as np
from .data import OUT,SOURCE
from .features import fingerprint
from .motion import motion_features


def audit():
    labels=json.loads((OUT/'review/annotations.json').read_text());current=json.loads((OUT/'review/clips/selection.json').read_text());old=json.loads((SOURCE/'review/clips/selection.json').read_text())
    records=[];totals={name:dict(matched=0,falseEvents=0,missed=0) for name in ('old','corrected')}
    for annotation,new,previous in zip(labels['clips'],current['clips'],old['clips']):
        if annotation['id']!=new['id'] or new['id']!=previous['id']:raise ValueError('Clip mismatch')
        fs=new['frames'];target=annotation['incoming'];lane=target['lane'];start=target['startFrame'];end=target['endFrame']
        colors=[f.get('incomingColors',[-1]*6)[lane] for f in fs[start:end+1]]
        radii=[f['vector'][lane*4+2] for f in fs[start:end+1]]
        incoming={'colorCorrect':sum(c==target['color'] for c in colors),'reviewedFrames':len(colors),'colors':colors,'radii':radii,
                  'inwardPairs':sum(b<a for a,b in zip(radii,radii[1:])),'possiblePairs':len(radii)-1}
        # The last two frames cannot confirm an onset; they are right-censored.
        truth=[f for f in annotation['clearFrames'] if 1<=f<=13];detected={}
        old_heights=np.array([f['heights'] for f in previous['frames']])
        detected['old']=(np.flatnonzero(np.diff(old_heights.sum(axis=1))<-.06)+1).tolist()
        corrected=[]
        for i in range(1,len(fs)):
            prefix=[fs[0]]*(33-i-1)+fs[:i+1]
            # One-second count difference is not an onset; inspect the newest
            # confirmed transition by truncating a second copy before this frame.
            now=motion_features(prefix).reshape(4,12)[-1,10]
            before=motion_features([fs[0]]*(33-i)+fs[:i]).reshape(4,12)[-1,10]
            if now>before:corrected.append(i)
        detected['corrected']=corrected
        scores={}
        for method,events in detected.items():
            available=set(truth);matched=[];false=[]
            for event in events:
                options=[t for t in available if 0<=event-t<=2]
                if options:
                    t=min(options);available.remove(t);matched.append([t,event])
                elif event<=13:false.append(event)
            scores[method]={'matched':matched,'falseEvents':false,'missed':sorted(available),'detectedFrames':events}
            totals[method]['matched']+=len(matched);totals[method]['falseEvents']+=len(false);totals[method]['missed']+=len(available)
        records.append({'id':annotation['id'],'incoming':incoming,'clearScores':scores})
    rulers=[]
    for r in labels['speedRulers']:
        c=next(c for c in current['clips'] if c['id']==r['clip']);a=c['frames'][r['startFrame']]['vector'][r['lane']*4+2];b=c['frames'][r['endFrame']]['vector'][r['lane']*4+2]
        seconds=(r['endFrame']-r['startFrame'])/4;estimated=(a-b)/seconds;manual=(r['startRadius']-r['endRadius'])/seconds
        ca=c['frames'][r['startFrame']]['incomingColors'][r['lane']];cb=c['frames'][r['endFrame']]['incomingColors'][r['lane']]
        identity_ok=ca>=0 and ca==cb
        rulers.append({**r,'identitySupported':identity_ok,'estimatedSpeed':estimated if identity_ok else None,'approximateManualSpeed':manual,'absoluteDifference':abs(estimated-manual) if identity_ok else None,'withinTolerance':identity_ok and abs(estimated-manual)<=.15})
    payload={'extractorHash':fingerprint(),'clipCount':12,'incomingColorCorrect':sum(r['incoming']['colorCorrect'] for r in records),
             'incomingReviewedFrames':sum(r['incoming']['reviewedFrames'] for r in records),'clearTotals':totals,'clips':records,'speedRulers':rulers,
             'limitations':['Assistant annotations from 4-FPS clips; one-to-one matching permits 0.5s causal confirmation delay.',
                            'Last two frames are right-censored for confirmed clears; diagnostic prefixes repeat the first frame and are never training data.',
                            'Known incoming tracks were manually chosen for visibility; not an unbiased detector accuracy estimate.']}
    (OUT/'tracking-audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print({k:v for k,v in payload.items() if k not in ('clips','limitations','extractorHash')})

if __name__=='__main__':audit()
