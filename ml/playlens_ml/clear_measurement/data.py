"""Deterministic clip preparation. Validation is inaccessible until source freeze."""
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw
from .detector import observation,fingerprint,Detector
from playlens_ml.common import ROOT
from playlens_ml.board_v2.review import decoder
from playlens_ml.board_tracking.data import OUT as SOURCE
from playlens_ml.board_tracking.features import extract
from playlens_ml.build_dataset import personal_runs,read_jsonl
OUT=ROOT/'artifacts/experiments/clear-measurement-v1'


def resets(run):
    r=personal_runs(ROOT/'data')[run-1];points=[p/1000 for p in r['resumePoints']]
    obs=sorted(read_jsonl(ROOT/'data/sessions'/r['id']/'observations.jsonl'),key=lambda x:x['timestampMs'])
    points += [b['activeElapsedMs']/1000 for a,b in zip(obs,obs[1:]) if b['timestampMs']-a['timestampMs']>800 or b['activeElapsedMs']-a['activeElapsedMs']>600]
    return points


def sheet(images,path,title,offset=0):
    im=Image.new('RGB',(1800,((len(images)+5)//6)*300),'white')
    for i,source in enumerate(images):
        side=min(source.size);left=(source.width-side)//2;top=(source.height-side)//2
        crop=source.crop((left,top,left+side,top+side)).resize((300,300));d=ImageDraw.Draw(crop)
        d.rectangle((0,0,300,20),fill='white');d.text((5,4),f'{title} frame {i+offset}',fill='black');im.paste(crop,((i%6)*300,(i//6)*300))
    im.save(path)


def development(fps=12):
    if (OUT/'freeze.json').exists():raise ValueError('Development is closed after freeze')
    clips=json.loads((SOURCE/'review/clips/selection.json').read_text())['clips'];folder=OUT/'development';folder.mkdir(exist_ok=True)
    records=[]
    for clip in clips:
        run=clip['run'];cache=json.loads((SOURCE/'cache'/f'run-{run:02}.json').read_text());video=ROOT/'data/sessions'/cache['sessionId']/'gameplay.webm'
        stream=decoder().read_frames(str(video),input_params=['-ss',str(clip['videoStartSeconds'])],output_params=['-vf',f'fps={fps},scale=960:-2','-frames:v',str(fps*4)]);meta=next(stream)
        obs=[];detector=Detector(fps);pred=[];previous=None;boundaries=resets(run);reset_indices=[]
        try:
            for i,raw in enumerate(stream):
                t=clip['videoStartSeconds']+i/fps;reset=any(t-1/fps<p<=t for p in boundaries)
                if reset:previous=None;reset_indices.append(i)
                image=Image.frombytes('RGB',meta['size'],raw)
                g=clip['frames'][i] if fps==4 else extract(image,previous);previous=g
                o=observation(image,g);obs.append(o);pred.append(detector.update(o,i/fps,reset=reset))
        finally:stream.close()
        records.append({'id':clip['id'],'run':run,'fps':fps,'observations':obs,'predictions':pred,'resetIndices':reset_indices,'videoSha256':cache['videoSha256']})
        print(clip['id'],[round(i/fps,3) for i,p in enumerate(pred) if p['event']],flush=True)
    (folder/f'observations-{fps}fps.json').write_text(json.dumps({'detectorHash':fingerprint(),'clips':records})+'\n')


def freeze():
    path=OUT/'freeze.json'
    if path.exists():raise ValueError('Already frozen; do not overwrite validation commitment')
    path.write_text(json.dumps({'detectorHash':fingerprint(),'protocolSha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                               'preparationSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'evaluationSha256':hashlib.sha256(Path(__file__).with_name('evaluate.py').read_bytes()).hexdigest()},indent=2)+'\n')


def validation():
    frozen=json.loads((OUT/'freeze.json').read_text())
    if frozen['detectorHash']!=fingerprint() or frozen['protocolSha256']!=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest():raise ValueError('Detector/protocol changed after freeze')
    if frozen['preparationSha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Preparation changed after freeze')
    protocol=json.loads((OUT/'protocol.json').read_text());fps=protocol['samplingHz'];folder=OUT/'validation';folder.mkdir(exist_ok=True);records=[]
    for run in protocol['validation']['runs']:
        if not 1<=run<=30:raise ValueError('Future run forbidden')
        cache=json.loads((SOURCE/'cache'/f'run-{run:02}.json').read_text());start=int(.4*len(cache['frames'])/4)*fps
        video=ROOT/'data/sessions'/cache['sessionId']/'gameplay.webm';images=[];obs=[];reset_indices=[];previous=None;boundaries=resets(run)
        stream=decoder().read_frames(str(video),output_params=['-vf',f'fps={fps},scale=960:-2']);meta=next(stream)
        try:
            for i,raw in enumerate(stream):
                if start<=i<start+6*fps:
                    reset=any((i-1)/fps<p<=i/fps for p in boundaries)
                    if reset:previous=None;reset_indices.append(i-start)
                    image=Image.frombytes('RGB',meta['size'],raw);image.save(folder/f'run-{run:02}-{i-start:02}.png');images.append(image)
                    geometry=extract(image,previous);previous=geometry;obs.append(observation(image,geometry))
                if i>=start+6*fps-1:break
        finally:stream.close()
        for n in range(0,len(images),36):sheet(images[n:n+36],folder/f'run-{run:02}-sheet-{n//36+1}.png',f'run {run} {start/fps}s',offset=n)
        records.append({'id':f'run-{run:02}','run':run,'startFrame':start,'fps':fps,'observations':obs,'resetIndices':reset_indices,'videoSha256':cache['videoSha256']})
        print('prepared',run,flush=True)
    # Predictions are deliberately absent while annotating the validation clips.
    (folder/'observations.json').write_text(json.dumps({'detectorHash':fingerprint(),'clips':records})+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('development','freeze','validation'));p.add_argument('--fps',type=int,default=12);args=p.parse_args()
    if args.action=='development':development(args.fps)
    else:globals()[args.action]()
