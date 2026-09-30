"""Pixel-only diagnostics; consistency is not a substitute for reviewed accuracy."""
from __future__ import annotations
import json
import numpy as np
from PIL import Image
from .features import extract, fingerprint
from .review import OUT, decoder, overlay
from playlens_ml.common import ROOT
from playlens_ml.board_experiment.data import source_data


def robustness(out=OUT, extract_fn=extract, hash_fn=fingerprint):
    selection=json.loads((out/'review/selection.json').read_text()); records=[]
    for frame in selection['frames']:
        source=Image.open(out/'review'/f"{frame['id']}-source.png").convert('RGB'); base=extract_fn(source)
        variants={'width640':source.resize((640,round(source.height*640/source.width)),Image.Resampling.LANCZOS),
                  'width1280':source.resize((1280,round(source.height*1280/source.width)),Image.Resampling.LANCZOS)}
        letterbox=Image.new('RGB',(source.width+240,source.height+180));letterbox.paste(source,(120,90));variants['letterbox']=letterbox
        for name,image in variants.items():
            result=extract_fn(image);scale=image.width/source.width if name!='letterbox' else 1
            center=np.array(base['center'])*scale+(np.array([120,90]) if name=='letterbox' else 0)
            geometry=result['valid'] and np.linalg.norm(np.array(result['center'])-center)<.025*base['apothem']*scale and abs(result['apothem']/(base['apothem']*scale)-1)<.025
            records.append({'id':frame['id'],'variant':name,'geometryStable':bool(geometry),
                            'colorsStable':bool(result['valid'] and result['topColors']==base['topColors'])})
    summary={name:{'geometryStable':sum(r['geometryStable'] for r in records if r['variant']==name),
                   'colorsStable':sum(r['colorsStable'] for r in records if r['variant']==name),'total':60} for name in variants}
    (out/'robustness.json').write_text(json.dumps({'extractorHash':hash_fn(),'summary':summary,'frames':records},indent=2)+'\n')
    print(summary,flush=True)


def clips(out=OUT, extract_fn=extract, hash_fn=fingerprint):
    data,_=source_data(); folder=out/'review/clips';folder.mkdir(parents=True,exist_ok=True);records=[]
    for n,run in enumerate((1,3,6,9,12,15,18,21,24,26,28,30),1):
        rows=np.flatnonzero(data['run_order']==run);row=rows[len(rows)//2]
        target=float(data['elapsed_seconds'][row]);sid=str(data['session_ids'][row]);start=max(0,target-2)
        stream=decoder().read_frames(str(ROOT/'data/sessions'/sid/'gameplay.webm'),input_params=['-ss',str(start)],output_params=['-vf','fps=4,scale=960:-2','-frames:v','16'])
        meta=next(stream);images=[];results=[]
        try:
            for i,raw in enumerate(stream):
                image=Image.frombytes('RGB',meta['size'],raw);result=extract_fn(image)
                images.append(overlay(image,result,f'clip {n:02d} run {run} video {start+i/4:.2f}s'));results.append(result)
        finally:stream.close()
        sheet=Image.new('RGB',(1600,1600),'white')
        for i,image in enumerate(images):sheet.paste(image,((i%4)*400,(i//4)*400))
        sheet.save(folder/f'clip-{n:02d}.png')
        if images:images[0].save(folder/f'clip-{n:02d}.gif',save_all=True,append_images=images[1:],duration=250,loop=0)
        records.append({'id':f'clip-{n:02d}','run':run,'videoStartSeconds':start,'frames':results})
        print('clip',n,'done',flush=True)
    (folder/'selection.json').write_text(json.dumps({'extractorHash':hash_fn(),'clips':records},indent=2)+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('robustness','clips'));args=parser.parse_args()
    globals()[args.action]()
