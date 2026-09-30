"""One pixel pass, with cached source provenance and strict development boundaries."""
import argparse
import json
import shutil
import numpy as np
from PIL import Image
from . import SCHEMA
from .features import extract,fingerprint
from playlens_ml.common import ROOT
from playlens_ml.board_v3.data import OUT as SOURCE
from playlens_ml.board_v2.data import build as assemble,digest
from playlens_ml.board_v2.review import decoder,overlay
from playlens_ml.build_dataset import personal_runs,read_jsonl
OUT=ROOT/'artifacts/experiments/board-tracking-v1'


def extract_run(order):
    if not 1<=order<=30:raise ValueError('Only development runs 1-30')
    path=SOURCE/'cache'/f'run-{order:02}.json';old=json.loads(path.read_text())
    dest=OUT/'cache'/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
    source_hash=digest(path)
    if dest.exists():
        cached=json.loads(dest.read_text())
        if cached['extractorHash']==fingerprint() and cached['sourceCacheSha256']==source_hash:return cached
    video=ROOT/'data/sessions'/old['sessionId']/'gameplay.webm'
    if digest(video)!=old['videoSha256']:raise ValueError('Changed video')
    run=personal_runs(ROOT/'data')[order-1];resets=list(np.array(run['resumePoints'])/1000)
    obs=sorted(read_jsonl(ROOT/'data/sessions'/run['id']/'observations.jsonl'),key=lambda x:x['timestampMs'])
    resets += [b['activeElapsedMs']/1000 for a,b in zip(obs,obs[1:]) if b['timestampMs']-a['timestampMs']>800 or b['activeElapsedMs']-a['activeElapsedMs']>600]
    stream=decoder().read_frames(str(video),output_params=['-vf','fps=4,scale=960:-2']);meta=next(stream)
    frames=[];previous=None;repair_folder=OUT/'review/repairs';repair_folder.mkdir(parents=True,exist_ok=True)
    try:
        for i,raw in enumerate(stream):
            if any((i-1)/4<p<=i/4 for p in resets):previous=None
            image=Image.frombytes('RGB',meta['size'],raw);r=extract(image,previous,old['frames'][i]);r['videoSeconds']=i/4
            if r.get('geometryHeld') or r.get('geometryRefit'):
                image.save(repair_folder/f'run-{order:02}-{i}-source.png')
                overlay(image,r,f'run{order} frame{i} {i/4}s').save(repair_folder/f'run-{order:02}-{i}.png')
            frames.append(r);previous=r
    finally:stream.close()
    if len(frames)!=len(old['frames']):raise ValueError('Frame count changed')
    payload={**old,'schema':SCHEMA,'extractorHash':fingerprint(),'sourceCacheSha256':source_hash,'frames':frames}
    dest.write_text(json.dumps(payload,allow_nan=False));print('run',order,'frames',len(frames),'held',sum(f.get('geometryHeld',False) for f in frames),flush=True)
    return payload


def build():
    return assemble(out=OUT,schema=SCHEMA,hash_fn=fingerprint)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int);p.add_argument('--all',action='store_true');p.add_argument('--build',action='store_true');a=p.parse_args()
    if a.run:extract_run(a.run)
    elif a.all:
        for order in range(1,31):extract_run(order)
    elif a.build:build()
    else:p.error('Choose --all, --run or --build')
