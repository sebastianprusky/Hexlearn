from __future__ import annotations
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from playlens_ml.common import ROOT
from playlens_ml.board_experiment.data import source_data
from .features import extract,fingerprint

OUT=ROOT/'artifacts/experiments/board-features-v2'

def decoder():
    try:import imageio_ffmpeg
    except ImportError:
        sys.path.insert(0,str(OUT/'deps'));import imageio_ffmpeg
    return imageio_ffmpeg


def low_resolution(image,size=(160,160)):
    image=image.copy();image.thumbnail(size,Image.Resampling.BILINEAR)
    result=Image.new('RGB',size,'black');result.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return np.asarray(result,dtype=float)/255


def aligned_frame(video,target,jpeg):
    start=max(0,target-.5)
    stream=decoder().read_frames(str(video),input_params=['-ss',str(start)],output_params=['-vf','fps=15,scale=960:-2','-frames:v','16'])
    meta=next(stream);truth=np.asarray(Image.open(jpeg).convert('RGB'),dtype=float)/255
    best=None;errors=[]
    try:
        for index,raw in enumerate(stream):
            image=Image.frombytes('RGB',meta['size'],raw);thumbnail=low_resolution(image)
            # Board-centered match includes colors/motion; exclude mostly static black padding.
            error=float(np.mean((thumbnail[45:115,45:115]-truth[45:115,45:115])**2))
            errors.append(error)
            if best is None or error<best[0]:best=(error,index,image)
    finally:stream.close()
    if best is None:raise ValueError('Video yielded no frames')
    error,index,image=best
    return image,{'mse':error,'offsetSeconds':start+index/15-target,'searchIndex':index,'candidateCount':len(errors),
                  'sourceSize':list(meta['source_size']),'searchEdge':index in (0,len(errors)-1)}


def overlay(image,result,label):
    # Crop a generous board region for review; preserve the original source separately.
    if result['valid']:
        cx,cy=result['center'];a=result['apothem'];box=(int(cx-1.8*a),int(cy-1.8*a),int(cx+1.8*a),int(cy+1.8*a))
    else:
        side=min(image.width,image.height);left=(image.width-side)//2;top=(image.height-side)//2
        box=(left,top,left+side,top+side)
    crop=image.crop(box);sx=400/crop.width;sy=400/crop.height;crop=crop.resize((400,400))
    d=ImageDraw.Draw(crop)
    if result['valid']:
        def xy(x,y):return ((x-box[0])*sx,(y-box[1])*sy)
        vertices=[xy(cx+a/np.cos(np.pi/6)*np.cos(k*np.pi/3),cy+a/np.cos(np.pi/6)*np.sin(k*np.pi/3)) for k in range(6)]
        d.line(vertices+[vertices[0]],fill='magenta',width=1)
        for k in range(6):
            angle=np.deg2rad(30+60*k+result['phaseDegrees']);height=result['heights'][k]
            point=xy(cx+height*a*np.cos(angle),cy+height*a*np.sin(angle))
            d.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),outline='black',width=1)
            c=result['topColors'][k];text='-' if c<0 else ('R','Y','B','G')[c]
            d.text(xy(cx+1.35*a*np.cos(angle),cy+1.35*a*np.sin(angle)),f'{k}:{text}',fill='purple',stroke_width=1,stroke_fill='white')
    d.rectangle((0,0,400,20),fill='white');d.text((5,4),label,fill='black')
    return crop


def prepare():
    data,source_hash=source_data();folder=OUT/'review';folder.mkdir(parents=True,exist_ok=True)
    records=[];images=[]
    for run in range(1,31):
        rows=np.flatnonzero(data['run_order']==run)
        for part,fraction in enumerate((1/3,2/3),1):
            row=rows[int((len(rows)-1)*fraction)];sid=str(data['session_ids'][row]);timestamp=int(data['timestamps_ms'][row]);t=float(data['elapsed_seconds'][row])
            directory=ROOT/'data/sessions'/sid;key=f'run-{run:02d}-{part}';dest=folder/f'{key}-source.png';alignment_file=folder/f'{key}-alignment.json'
            if dest.exists() and alignment_file.exists():
                image=Image.open(dest).convert('RGB');alignment=json.loads(alignment_file.read_text())
            else:
                image,alignment=aligned_frame(directory/'gameplay.webm',t,directory/'frames'/f'{timestamp}.jpg')
                image.save(dest);alignment_file.write_text(json.dumps(alignment,indent=2)+'\n')
            result=extract(image);overlay_image=overlay(image,result,key);overlay_image.save(folder/f'{key}.png');images.append(overlay_image)
            records.append({'id':key,'run':run,'sessionId':sid,'activeSeconds':t,'alignment':alignment,'extraction':result})
            print(key,'valid',result['valid'],'alignment',round(alignment['mse'],5),'phase',result.get('phaseDegrees'),flush=True)
    for page in range(6):
        sheet=Image.new('RGB',(2000,800),'white')
        for i,im in enumerate(images[page*10:(page+1)*10]):sheet.paste(im,((i%5)*400,(i//5)*400))
        sheet.save(folder/f'contact-{page+1}.png')
    (folder/'selection.json').write_text(json.dumps({'extractorHash':fingerprint(),'sourceDatasetSha256':source_hash,'frames':records},indent=2)+'\n')

if __name__=='__main__':prepare()
