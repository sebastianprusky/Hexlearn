"""One fixed clear-feature ablation after the frozen measurement gate passes."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from .data import OUT, SOURCE, resets
from .detector import Detector, observation, fingerprint
from playlens_ml.common import ROOT
from playlens_ml.board_tracking.features import extract
from playlens_ml.board_v2.review import decoder
from playlens_ml.board_v2.data import digest, VIDEO_LAG

FEATURES=[f'clear_{kind}_{seconds}s' for seconds in (1,2,4,8) for kind in ('count','confidence_sum')]

def ready():
    result=json.loads((OUT/'validation-results.json').read_text())
    if not result['passed'] or result['detectorHash']!=fingerprint():
        raise ValueError('A passed, unchanged measurement is required')

def extract_run(run):
    if not 1<=run<=30:raise ValueError('Only development runs 1-30')
    ready();folder=OUT/'full-cache';folder.mkdir(exist_ok=True)
    path=folder/f'run-{run:02}.json'
    old=json.loads((SOURCE/'cache'/path.name).read_text())
    video=ROOT/'data/sessions'/old['sessionId']/'gameplay.webm'
    if digest(video)!=old['videoSha256']:raise ValueError('Changed source video')
    if path.exists():
        result=json.loads(path.read_text())
        if result['detectorHash']!=fingerprint() or result['videoSha256']!=old['videoSha256']:raise ValueError('Stale cache')
        return run
    stream=decoder().read_frames(str(video),output_params=['-vf','fps=12,scale=960:-2','-threads','1'])
    meta=next(stream);previous=None;detector=Detector(12);boundaries=resets(run);frames=[];segment=0
    try:
        for i,raw in enumerate(stream):
            reset=any((i-1)/12<p<=i/12 for p in boundaries)
            if reset:previous=None;segment+=1
            image=Image.frombytes('RGB',meta['size'],raw)
            geometry=extract(image,previous);previous=geometry
            obs=observation(image,geometry)
            prior_geometry=detector.previous_geometry
            changed=obs['valid'] and prior_geometry is not None and (abs(obs['apothem']/prior_geometry[0]-1)>.08 or np.linalg.norm(np.array(obs['center'])-prior_geometry[1])>.08*prior_geometry[0])
            if changed or not obs['valid']:segment+=1
            prediction=detector.update(obs,i/12,reset=reset)
            frames.append({**prediction,'segment':segment,'videoSeconds':i/12})
    finally:stream.close()
    result={'run':run,'sessionId':old['sessionId'],'videoSha256':old['videoSha256'],'detectorHash':fingerprint(),'fps':12,'frames':frames}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,allow_nan=False));temp.replace(path)
    print(f'run {run:02}: {len(frames)} frames, {sum(f["event"] for f in frames)} clears',flush=True)
    return run

def temporal(frames,elapsed):
    times=np.arange(len(frames))/12+VIDEO_LAG
    end=np.searchsorted(times,elapsed,side='right')-1
    if end<96:return None,'short_history'
    window=frames[end-96:end+1]
    if not all(f['supported'] for f in window):return None,'unsupported_geometry'
    if len({f['segment'] for f in window})!=1:return None,'reset_within_window'
    values=[]
    for seconds in (1,2,4,8):
        rows=frames[end-seconds*12+1:end+1]
        events=[f for f in rows if f['event']]
        values.extend([len(events),sum(f['confidence'] for f in events)])
    return values,None

def build():
    ready()
    with np.load(SOURCE/'compact_windows.npz') as z:data={k:z[k].copy() for k in z.files}
    if np.any((data['run_order']<1)|(data['run_order']>30)):raise ValueError('Future run forbidden')
    extra=np.zeros((len(data['run_order']),len(FEATURES)));valid=np.zeros(len(extra),bool);reasons=[];runs=[];segments=data['segments'].copy()
    for run in range(1,31):
        path=OUT/'full-cache'/f'run-{run:02}.json';cached=json.loads(path.read_text())
        if cached['detectorHash']!=fingerprint() or cached['fps']!=12 or cached['run']!=run:raise ValueError('Stale detector or cache schema')
        frames=cached['frames'];excluded={}
        for row in np.flatnonzero(data['run_order']==run):
            values,reason=temporal(frames,float(data['elapsed_seconds'][row]))
            if not data['valid'][row]:reason='original_extraction_invalid'
            if reason:excluded[reason]=excluded.get(reason,0)+1;reasons.append({'row':int(row),'run':run,'reason':reason});continue
            extra[row]=values;valid[row]=True
            end=np.searchsorted(np.arange(len(frames))/12+VIDEO_LAG,data['elapsed_seconds'][row],side='right')-1
            segments[row]=segments[row]*1000000+frames[end]['segment']
        rows=data['run_order']==run
        runs.append({'run':run,'eligibleWindows':int(rows.sum()),'validWindows':int(valid[rows].sum()),'coverage':float(valid[rows].mean()),'excluded':excluded,'cacheSha256':digest(path),'videoSha256':cached['videoSha256']})
    baseline=data['geometry_motion'];base_names=data['feature_names'].tolist()
    data.update(X=np.c_[baseline,extra],without_clear=baseline,valid=valid,segments=segments,base_feature_names=np.array(base_names),feature_names=np.array(base_names+FEATURES),clear_feature_names=np.array(FEATURES),schema=np.array('clear-ablation-v1'))
    if not np.isfinite(data['X']).all():raise ValueError('Nonfinite features')
    np.savez_compressed(OUT/'ablation-windows.npz',**data)
    (OUT/'ablation-manifest.json').write_text(json.dumps({'schema':'clear-ablation-v1','detectorHash':fingerprint(),'sourceDatasetSha256':digest(SOURCE/'compact_windows.npz'),'featureNames':data['feature_names'].tolist(),'runs':runs,'excludedWindows':reasons},indent=2)+'\n')

def evaluate():
    ready();path=OUT/'ablation-results.json'
    if path.exists():raise ValueError('Fixed comparison already completed')
    from playlens_ml.board_tracking import evaluate as comparison_module
    from playlens_ml.development_audit import paired_intervals
    with np.load(OUT/'ablation-windows.npz') as z:data={k:z[k].copy() for k in z.files}
    manifest=json.loads((OUT/'ablation-manifest.json').read_text())
    if str(data['schema'])!='clear-ablation-v1' or data['clear_feature_names'].tolist()!=FEATURES or manifest['detectorHash']!=fingerprint():raise ValueError('Incompatible feature schema')
    if data['feature_names'].tolist()!=manifest['featureNames'] or len(data['feature_names'])!=data['X'].shape[1]:raise ValueError('Feature columns do not match schema')
    if manifest['sourceDatasetSha256']!=digest(SOURCE/'compact_windows.npz'):raise ValueError('Original comparison dataset changed')
    if np.any((data['run_order']<1)|(data['run_order']>30)) or not np.isfinite(data['X']).all():raise ValueError('Invalid development data')
    comparison_module.OUT=OUT
    result=comparison_module.comparison(data,data['valid'],{'without_clear':data['without_clear'],'with_clear':data['X']},'clear_ablation')
    result['featureComparisons']={family:paired_intervals(result['models'][f'with_clear_{family}']['perRun'],result['models'][f'without_clear_{family}']['perRun']) for family in ('logistic','tree')}
    candidates=[name for name in ('with_clear_logistic','with_clear_tree') if result['models'][name]['qualifies']]
    result['selected']=min(candidates,key=lambda name:(result['models'][name]['summary']['brier'],result['models'][name]['summary']['mae'])) if candidates else None
    result['datasetSha256']=digest(OUT/'ablation-windows.npz')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Selected:',result['selected'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','build','evaluate']);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.action=='extract':
        ready()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(extract_run,range(1,31)))
    else:globals()[args.action]()
