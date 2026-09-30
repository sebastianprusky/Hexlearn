import json
import shutil
from PIL import Image
from .data import OUT,SOURCE
from .features import extract,fingerprint
from playlens_ml.board_v2.review import overlay
from playlens_ml.board_v2.audit import robustness,clips


def prepare():
    folder=OUT/'review';folder.mkdir(parents=True,exist_ok=True)
    old=json.loads((SOURCE/'review/selection.json').read_text());records=[]
    for f in old['frames']:
        path=SOURCE/'review'/f"{f['id']}-source.png";shutil.copy2(path,folder/path.name)
        im=Image.open(path).convert('RGB');r=extract(im);records.append({**f,'extraction':r})
        overlay(im,r,f['id']).save(folder/f"{f['id']}.png")
        if r['topColors']!=f['extraction']['topColors'] or r['center']!=f['extraction']['center'] or r['apothem']!=f['extraction']['apothem']:
            raise ValueError('Still geometry/colors changed; new visual review required')
    (folder/'selection.json').write_text(json.dumps({**old,'extractorHash':fingerprint(),'frames':records},indent=2)+'\n')
    shutil.copy2(SOURCE/'alignment.json',OUT/'alignment.json')
    # Pixel-equivalent still geometry and attached colors retain prior judgments;
    # movement and repaired geometry get separate explicit reviews.
    decisions=json.loads((SOURCE/'review/decisions.json').read_text());decisions['extractorHash']=fingerprint()
    decisions['inheritedEvidence']='All 60 still centers, scales and attached colors exactly match v3; incoming and repaired geometry reviewed separately.'
    (folder/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    robustness(OUT,extract,fingerprint);clips(OUT,extract,fingerprint)

if __name__=='__main__':prepare()
