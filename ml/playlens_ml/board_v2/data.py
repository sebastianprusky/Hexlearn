"""Native replay extraction and conservative causal windows, development runs only."""
from __future__ import annotations
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from . import SCHEMA
from .features import extract, fingerprint
from .review import OUT, decoder
from playlens_ml.common import ROOT
from playlens_ml.build_dataset import personal_runs, read_jsonl
from playlens_ml.board_experiment.data import source_data
from playlens_ml.board_experiment.features import BASE_NAMES

# Conservative transport/codec margin; all reviewed alignments must fit within it.
VIDEO_LAG=.5
TEMPORAL_NAMES=[f'{seconds}s_rank_{rank}_{name}' for seconds in (1,2,4,8) for rank in range(6)
                for name in ('height_change','apparent_clear','incoming_motion','confidence')]
FEATURE_NAMES=BASE_NAMES+TEMPORAL_NAMES


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def extract_run(order):
    if not 1<=order<=30:raise ValueError('Only development runs 1-30 are allowed')
    run=personal_runs(ROOT/'data')[order-1];video=ROOT/'data/sessions'/run['id']/'gameplay.webm'
    folder=OUT/'cache';folder.mkdir(parents=True,exist_ok=True);dest=folder/f'run-{order:02d}.json'
    video_hash=digest(video)
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['videoSha256']==video_hash and old['extractorHash']==fingerprint():return old
    stream=decoder().read_frames(str(video),output_params=['-vf','fps=4,scale=960:-2'])
    meta=next(stream);frames=[]
    try:
        for i,raw in enumerate(stream):
            result=extract(Image.frombytes('RGB',meta['size'],raw));result['videoSeconds']=i/4
            frames.append(result)
            if i%200==0:print('run',order,'frame',i,flush=True)
    finally:stream.close()
    payload={'schema':SCHEMA,'run':order,'sessionId':run['id'],'videoSha256':video_hash,'extractorHash':fingerprint(),
             'sourceSize':list(meta['source_size']),'decodedSize':list(meta['size']),'fps':4,'frames':frames}
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(payload,allow_nan=False));temp.replace(dest)
    print('run',order,'complete',len(frames),flush=True)
    return payload


def temporal(times,vectors,index):
    """Ranked distributions avoid claiming that a rotated lane was cleared.

    These are changes in stack-height and incoming-distance distributions, not
    physical block tracks. Motion/clear confidence is deliberately conservative.
    """
    current=vectors[index,:24].reshape(6,4);out=[]
    for seconds in (1,2,4,8):
        target=times[index]-seconds;j=np.searchsorted(times[:index+1],target,side='right')-1
        if j<0 or target-times[j]>.3:raise ValueError('Insufficient preceding context')
        previous=vectors[j,:24].reshape(6,4)
        growth=np.sort(current[:,0])-np.sort(previous[:,0])
        movement=(np.sort(previous[:,2])-np.sort(current[:,2]))/seconds
        confidence=float(min(current[:,3].min(),previous[:,3].min()))*.5
        for k in range(6):out.extend((growth[k],max(0.,-growth[k]),movement[k],confidence))
    return np.r_[vectors[index],out]


def build(out=OUT, schema=SCHEMA, hash_fn=fingerprint):
    data,source_hash=source_data();selection=json.loads((out/'review/selection.json').read_text())
    timing=json.loads((out/'alignment.json').read_text())
    if not timing['passed']:raise ValueError('Additional video synchronization checks failed')
    if any(abs(f['alignment']['offsetSeconds'])+.15>VIDEO_LAG or f['alignment']['searchEdge'] for f in selection['frames']):
        raise ValueError('Video timing is outside the conservative margin')
    n=len(data['run_order']);x=np.zeros((n,len(FEATURE_NAMES)));valid=np.zeros(n,bool)
    reasons=np.full(n,'unprocessed',dtype='<U64');segments=np.zeros(n,int);manifest=[]
    for order,run in enumerate(personal_runs(ROOT/'data')[:30],1):
        path=out/'cache'/f'run-{order:02d}.json';cache=json.loads(path.read_text())
        if cache['schema']!=schema or cache['extractorHash']!=hash_fn() or cache['sessionId']!=run['id']:raise ValueError('Stale cache')
        frames=cache['frames'];times=np.arange(len(frames))/4+VIDEO_LAG
        ok=np.array([f['valid'] for f in frames]);base=np.array([f.get('vector',[0.]*54) for f in frames])
        observations=read_jsonl(ROOT/'data/sessions'/run['id']/'observations.jsonl')
        obs=sorted(observations,key=lambda v:v['timestampMs']);resets=list(np.array(run['resumePoints'])/1000)
        for a,b in zip(obs,obs[1:]):
            if (b['timestampMs']-a['timestampMs'])>800 or b['activeElapsedMs']-a['activeElapsedMs']>600:
                resets.append(b['activeElapsedMs']/1000)
        seg=np.ones(len(frames),int)
        for i in range(1,len(frames)):
            previous,current=frames[i-1],frames[i]
            changed=not(ok[i-1] and ok[i])
            if not changed:
                changed=abs(current['apothem']/previous['apothem']-1)>.08 or np.linalg.norm(np.array(current['center'])-previous['center'])>.08*previous['apothem']
            # Reset after capture gaps/pauses, including the half-second safety buffer.
            reset=changed or any(times[i-1]-VIDEO_LAG<p<=times[i]-VIDEO_LAG for p in resets)
            seg[i]=seg[i-1]+int(reset)
        rows=np.flatnonzero(data['run_order']==order)
        for row in rows:
            t=float(data['elapsed_seconds'][row]);i=int(np.searchsorted(times,t,side='right')-1)
            if i<0 or i>=len(frames) or t-times[i]>.3:reasons[row]='missing_video_anchor';continue
            segments[row]=order*100000+seg[i];j=i-32
            if not ok[i]:reasons[row]=frames[i]['reason']
            elif j<0 or seg[j]!=seg[i]:reasons[row]='insufficient_continuous_context'
            elif not ok[j:i+1].all():reasons[row]='invalid_frame_in_context'
            else:
                x[row]=temporal(times[j:i+1],base[j:i+1],32);valid[row]=True;reasons[row]='ok'
        manifest.append({'run':order,'sessionId':run['id'],'videoSha256':cache['videoSha256'],'frameCount':len(frames),
                         'eligibleWindows':len(rows),'validWindows':int(valid[rows].sum()),'coverage':float(valid[rows].mean()),
                         'exclusions':{str(r):int((reasons[rows]==r).sum()) for r in np.unique(reasons[rows]) if r!='ok'},
                         'frameFailures':{f['reason']:sum(v['reason']==f['reason'] for v in frames) for f in frames if not f['valid']}})
        print('windows',order,valid[rows].sum(),'/',len(rows),flush=True)
    np.savez_compressed(out/'windows.npz',**data,X=x,valid=valid,reasons=reasons,segments=segments,schema=np.array(schema),feature_names=np.array(FEATURE_NAMES))
    payload={'schema':schema,'extractorHash':hash_fn(),'sourceDatasetSha256':source_hash,'featureNames':FEATURE_NAMES,
             'videoLagSeconds':VIDEO_LAG,'temporalMethod':'sorted distributions; estimates, not tracked blocks','runs':manifest}
    (out/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n');return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=int);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.run is not None:extract_run(args.run)
    elif args.build:build()
    else:parser.error('Choose --run 1..30 or --build')
