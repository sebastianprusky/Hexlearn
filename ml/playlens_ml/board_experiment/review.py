"""Deterministic extraction checkpoint. Review decisions are explicit, never inferred from validity flags."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .data import source_data, FrameCache, DEFAULT_OUT
from .features import NORMALS, COLOR_NAMES, extractor_hash, geometry_changed
from playlens_ml.common import ROOT


def overlay(path,frame,label):
    image=Image.open(path).convert('RGB').resize((320,320),Image.Resampling.NEAREST)
    original=Image.open(path); sx=320/original.width; sy=320/original.height
    draw=ImageDraw.Draw(image); cx,cy=frame.center; a=frame.apothem
    if frame.valid:
        vertices=[((cx+a/np.cos(np.pi/6)*np.cos(k*np.pi/3))*sx,(cy+a/np.cos(np.pi/6)*np.sin(k*np.pi/3))*sy) for k in range(6)]
        draw.line(vertices+[vertices[0]],fill='magenta',width=2)
        for lane,n in enumerate(NORMALS):
            height=frame.vector[lane*4]; end=((cx+n[0]*height*a)*sx,(cy+n[1]*height*a)*sy)
            draw.line(((cx+n[0]*.36*a)*sx,(cy+n[1]*.36*a)*sy,*end),fill='black',width=2)
            color='-' if frame.top_colors[lane]<0 else COLOR_NAMES[frame.top_colors[lane]][0].upper()
            draw.text(((cx+n[0]*1.35*a)*sx,(cy+n[1]*1.35*a)*sy),f'{lane}:{color}',fill='purple',stroke_width=1,stroke_fill='white')
    draw.rectangle((0,0,320,25),fill='white'); draw.text((4,4),label,fill='black')
    return image


def prepare(out=DEFAULT_OUT):
    data,digest=source_data(); out.mkdir(parents=True,exist_ok=True); folder=out/'review'; folder.mkdir(exist_ok=True)
    cache=FrameCache(out); records=[]; thumbnails=[]; clips=[]
    try:
        for run in range(1,31):
            rows=np.flatnonzero(data['run_order']==run)
            for part,fraction in enumerate((1/3,2/3),1):
                row=rows[int((len(rows)-1)*fraction)]; sid=str(data['session_ids'][row]); wall=int(data['timestamps_ms'][row])
                path=ROOT/'data/sessions'/sid/'frames'/f'{wall}.jpg'; frame,sha=cache.get(path)
                key=f'run-{run:02d}-{part}'
                im=overlay(path,frame,key); im.save(folder/f'{key}.png'); thumbnails.append(im)
                records.append({'id':key,'run':run,'timestampMs':wall,'sourceSha256':sha,'sourcePath':str(path),
                    'valid':frame.valid,'reason':frame.reason,'center':list(frame.center),'apothem':frame.apothem,
                    'topColors':frame.top_colors,'geometryCorrect':None,'colorsCorrect':None,'notes':''})
        # Fixed runs and positions; clips are never chosen from prediction outcomes.
        for n,run in enumerate((1,3,6,9,12,15,18,21,24,26,28,30)):
            rows=np.flatnonzero(data['run_order']==run); row=rows[len(rows)//2]
            sid=str(data['session_ids'][row]); wall=int(data['timestamps_ms'][row]); directory=ROOT/'data/sessions'/sid/'frames'
            paths=sorted((p for p in directory.glob('*.jpg') if wall-2000<=int(p.stem)<=wall+2000),key=lambda p:int(p.stem))
            ims=[]
            for p in paths:
                frame,_=cache.get(p); ims.append(overlay(p,frame,f'clip {n+1:02d} run {run} {int(p.stem)-wall:+}ms'))
            if ims:
                ims[0].save(folder/f'clip-{n+1:02d}.gif',save_all=True,append_images=ims[1:],duration=250,loop=0)
                selected=np.linspace(0,len(ims)-1,min(8,len(ims))).round().astype(int)
                sheet=Image.new('RGB',(320*4,320*2),'white')
                for i,j in enumerate(selected): sheet.paste(ims[j],((i%4)*320,(i//4)*320))
                sheet.save(folder/f'clip-{n+1:02d}.png')
            clips.append({'id':f'clip-{n+1:02d}','run':run,'frames':len(paths),'reviewed':False,'events':[],'notes':''})
        for page in range(6):
            sheet=Image.new('RGB',(320*5,320*2),'white')
            for i,im in enumerate(thumbnails[page*10:(page+1)*10]): sheet.paste(im,((i%5)*320,(i//5)*320))
            sheet.save(folder/f'contact-{page+1}.png')
    finally: cache.close()
    payload={'extractorHash':extractor_hash(),'sourceDatasetSha256':digest,'frames':records,'clips':clips}
    (folder/'selection.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(folder)


def checkpoint(out=DEFAULT_OUT):
    selection=json.loads((out/'review/selection.json').read_text())
    decisions=json.loads((out/'review/decisions.json').read_text())
    if selection['extractorHash']!=extractor_hash() or decisions['extractorHash']!=extractor_hash():
        raise ValueError('Review must match the exact extractor revision')
    if selection['sourceDatasetSha256']!=source_data()[1]: raise ValueError('Review dataset changed')
    by_id={item['id']:item for item in decisions['frames']}
    if set(by_id)!={item['id'] for item in selection['frames']} or len(by_id)!=60 or len(decisions['frames'])!=60: raise ValueError('Expected 60 unique reviewed frames')
    correct=sum(item.get('geometryCorrect') is True and item.get('colorsCorrect') is True for item in by_id.values())
    clips=decisions.get('clips',[])
    reviewed=(len(clips)==12 and {c['id'] for c in clips}=={c['id'] for c in selection['clips']} and all(c.get('reviewed') is True for c in clips))
    events=set(e for c in clips for e in c.get('events',[]))
    covered={'rotation','stacking','incoming','clear'}<=events
    result={'correctFrames':correct,'reviewedFrames':60,'accuracy':correct/60,'clipsReviewed':reviewed,
            'eventCoverage':sorted(events),'passed':correct>=54 and reviewed and covered}
    (out/'checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
