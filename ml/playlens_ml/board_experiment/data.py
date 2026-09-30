from __future__ import annotations
import hashlib
import io
import json
import sqlite3
from pathlib import Path
import numpy as np
from . import SCHEMA
from .features import BoardFrame, extract, extractor_hash, geometry_changed, temporal_vector, FEATURE_NAMES
from playlens_ml.common import ROOT, DATASET_DIR
from playlens_ml.build_dataset import personal_runs, read_jsonl

DEFAULT_OUT=ROOT/'artifacts/experiments/board-features-v1'


def source_data():
    path=DATASET_DIR/'personal_windows.npz'
    with np.load(path) as source:
        keep=source['run_order']<=30
        keys=('run_order','session_ids','timestamps_ms','active_elapsed_ms','elapsed_seconds','time_to_failure_seconds','y_failure')
        data={k:source[k][keep].copy() for k in keys}
    if set(data['run_order'].tolist())!=set(range(1,31)):
        raise ValueError('Expected the original 30 development games')
    for run in range(1,31):
        m=data['run_order']==run
        if len(set(data['session_ids'][m].tolist()))!=1 or np.any(np.diff(data['active_elapsed_ms'][m])<=0):
            raise ValueError('Run IDs or chronological anchors are inconsistent')
    return data,hashlib.sha256(path.read_bytes()).hexdigest()


class FrameCache:
    def __init__(self,out):
        out.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(out/'frame_cache.sqlite3')
        self.connection.execute('CREATE TABLE IF NOT EXISTS frames (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.code_hash=extractor_hash()

    def get(self,path):
        raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest(); key=self.code_hash+':'+digest
        found=self.connection.execute('SELECT payload FROM frames WHERE key=?',(key,)).fetchone()
        if found:
            value=json.loads(found[0])
            value['vector']=np.array(value['vector']); value['center']=tuple(value['center']); value['colors']=np.empty((0,0))
            return BoardFrame(**value),digest
        frame=extract(raw)
        value={k:v for k,v in vars(frame).items() if k!='colors'}; value['vector']=frame.vector.tolist()
        self.connection.execute('INSERT INTO frames VALUES (?,?)',(key,json.dumps(value)))
        return frame,digest

    def close(self):
        self.connection.commit(); self.connection.close()


def split_segments(times, walls, frames, resumes):
    segments=[]; segment=0
    for i in range(len(times)):
        reset=i==0
        if i:
            reset=(times[i]<=times[i-1] or times[i]-times[i-1]>.6 or walls[i]-walls[i-1]>.8 or
                   any(times[i-1]<point<=times[i] for point in resumes) or geometry_changed(frames[i-1],frames[i]))
        if reset: segment+=1
        segments.append(segment)
    return np.array(segments)


def build(out=DEFAULT_OUT):
    data,dataset_hash=source_data(); cache=FrameCache(out)
    vectors=np.zeros((len(data['run_order']),len(FEATURE_NAMES)))
    valid=np.zeros(len(vectors),dtype=bool); reasons=np.full(len(vectors),'unprocessed',dtype='<U64')
    segment_ids=np.zeros(len(vectors),dtype=int); manifest=[]
    runs=personal_runs(ROOT/'data')[:30]
    try:
        for order,run in enumerate(runs,1):
            directory=ROOT/'data/sessions'/run['id']; observations=read_jsonl(directory/'observations.jsonl')
            entries=sorted({int(item['timestampMs']):item for item in observations}.items())
            paths=[]; frames=[]; times=[]; walls=[]; hashes=[]
            for wall,item in entries:
                path=directory/'frames'/f'{wall}.jpg'
                if not path.exists(): continue
                frame,digest=cache.get(path); paths.append(path); frames.append(frame)
                times.append(float(item['activeElapsedMs'])/1000); walls.append(wall/1000); hashes.append(digest)
            times=np.array(times); walls=np.array(walls)
            seg=split_segments(times,walls,frames,np.asarray(run['resumePoints'])/1000)
            by_wall={int(round(t*1000)):i for i,t in enumerate(walls)}
            base=np.array([f.vector for f in frames]); rows=np.flatnonzero(data['run_order']==order)
            if set(data['session_ids'][rows].tolist())!={run['id']}:
                raise ValueError('Original cohort order changed')
            for row in rows:
                i=by_wall.get(int(data['timestamps_ms'][row])); reason='missing_anchor'
                if i is not None:
                    segment_ids[row]=order*100000+seg[i]
                    start=i
                    while start>0 and seg[start-1]==seg[i] and times[i]-times[start-1]<=8.6: start-=1
                    if not frames[i].valid: reason=frames[i].reason
                    elif times[i]-times[start]<8 or i-start+1<28: reason='insufficient_continuous_context'
                    elif not all(f.valid for f in frames[start:i+1]): reason='invalid_frame_in_context'
                    else:
                        try:
                            vectors[row]=temporal_vector(times[start:i+1],base[start:i+1],i-start)
                            valid[row]=np.isfinite(vectors[row]).all(); reason='ok' if valid[row] else 'nonfinite_features'
                        except ValueError: reason='insufficient_continuous_context'
                reasons[row]=reason
            manifest.append({'run':order,'sessionId':run['id'],'frameCount':len(frames),
                'eligibleWindows':len(rows),'validWindows':int(valid[rows].sum()),
                'coverage':float(valid[rows].mean()),
                'exclusions':{str(r):int(np.sum(reasons[rows]==r)) for r in np.unique(reasons[rows]) if r!='ok'},
                'frames':[{'timestampMs':int(round(t*1000)),'sha256':h} for t,h in zip(walls,hashes)]})
            cache.connection.commit()
            print(f'run {order}: {valid[rows].sum()}/{len(rows)} valid windows',flush=True)
    finally: cache.close()
    np.savez_compressed(out/'windows.npz',**data,X=vectors,valid=valid,reasons=reasons,segments=segment_ids,
                        schema=np.array(SCHEMA),feature_names=np.array(FEATURE_NAMES))
    payload={'schema':SCHEMA,'extractorHash':extractor_hash(),'sourceDatasetSha256':dataset_hash,
             'featureNames':FEATURE_NAMES,'runs':manifest}
    (out/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload
