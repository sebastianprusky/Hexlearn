"""Reuse exact v2 frames; decode pixels only to repair unsupported/tall stacks."""
import argparse
import json
import numpy as np
from PIL import Image
from . import SCHEMA
from .features import needs_repair,fingerprint
from playlens_ml.board_v2 import fallback
from playlens_ml.board_v2.features import fingerprint as primary_hash
from playlens_ml.board_v2.data import FEATURE_NAMES,build as assemble,digest
from playlens_ml.board_v2.review import OUT as PREVIOUS,decoder
from playlens_ml.common import ROOT
OUT=ROOT/'artifacts/experiments/board-features-v3'


def extract_run(order):
    if not 1<=order<=30:raise ValueError('Only development runs 1-30 are allowed')
    old=json.loads((PREVIOUS/'cache'/f'run-{order:02d}.json').read_text())
    if old['extractorHash']!=primary_hash():raise ValueError('Stale primary extraction')
    dest=OUT/'cache'/f'run-{order:02d}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        cached=json.loads(dest.read_text())
        if cached['extractorHash']==fingerprint() and cached['videoSha256']==old['videoSha256']:return cached
    video=ROOT/'data/sessions'/old['sessionId']/'gameplay.webm'
    if digest(video)!=old['videoSha256']:raise ValueError('Video changed')
    frames=old['frames'];wanted={i for i,f in enumerate(frames) if needs_repair(f)}
    # Keep exactly the same full-stream FPS conversion as v2, avoiding seek offsets.
    stream=decoder().read_frames(str(video),output_params=['-vf','fps=4,scale=960:-2']);meta=next(stream);seen=0
    try:
        for i,raw in enumerate(stream):
            seen+=1
            if i in wanted:
                result=fallback.extract(Image.frombytes('RGB',meta['size'],raw));result['videoSeconds']=i/4;result['usedFallback']=True;frames[i]=result
    finally:stream.close()
    if seen!=len(frames):raise ValueError('Decode frame count changed')
    payload={**old,'schema':SCHEMA,'extractorHash':fingerprint(),'primaryExtractorHash':primary_hash(),'reprocessedFrames':len(wanted),'frames':frames}
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(payload,allow_nan=False));tmp.replace(dest)
    print('repaired',order,len(wanted),'frames',flush=True);return payload


def build():return assemble(out=OUT,schema=SCHEMA,hash_fn=fingerprint)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int);p.add_argument('--build',action='store_true');a=p.parse_args()
    if a.run is not None:extract_run(a.run)
    elif a.build:build()
    else:p.error('Choose --run 1..30 or --build')
