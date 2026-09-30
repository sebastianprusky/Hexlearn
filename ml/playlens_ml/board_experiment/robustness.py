"""Image transformations for the explicitly requested extractor invariance check."""
import io
import json
import numpy as np
from PIL import Image
from .features import extract, extractor_hash
from .data import DEFAULT_OUT


def check(out=DEFAULT_OUT):
    selection=json.loads((out/'review/selection.json').read_text()); rows=[]
    if selection['extractorHash']!=extractor_hash(): raise ValueError('Review extractor changed')
    for item in selection['frames']:
        original=Image.open(item['sourcePath']).convert('RGB'); raw=io.BytesIO(); original.save(raw,format='PNG')
        reference=extract(raw.getvalue())
        variants={}
        for name,scale,pad in [('upscaled',2.,0),('downscaled',.75,0),('letterboxed',1.,24)]:
            im=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.BILINEAR)
            if pad:
                canvas=Image.new('RGB',(im.width+2*pad,im.height+2*pad),'black'); canvas.paste(im,(pad,pad)); im=canvas
            raw=io.BytesIO(); im.save(raw,format='PNG'); candidate=extract(raw.getvalue())
            comparable=reference.valid and candidate.valid
            error=float(np.linalg.norm((np.array(candidate.center)-pad)/scale-np.array(reference.center))/reference.apothem) if comparable else None
            radius_error=float(abs(candidate.apothem/scale/reference.apothem-1)) if comparable else None
            variants[name]={'valid':candidate.valid,'geometryStable':bool(comparable and error<=.05 and radius_error<=.05),
                'centerErrorInBoardUnits':error,'relativeScaleError':radius_error,
                'identicalLaneColors':bool(comparable and candidate.top_colors==reference.top_colors)}
        rows.append({'id':item['id'],'referenceValid':reference.valid,'variants':variants})
    result={'extractorHash':extractor_hash(),'frames':rows,'summary':{name:{
        'geometryStable':sum(r['variants'][name]['geometryStable'] for r in rows),
        'identicalLaneColors':sum(r['variants'][name]['identicalLaneColors'] for r in rows), 'total':len(rows)}
        for name in ('upscaled','downscaled','letterboxed')}}
    (out/'robustness.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
