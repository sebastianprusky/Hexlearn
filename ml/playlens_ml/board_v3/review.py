import json
import shutil
from PIL import Image
from .data import OUT,PREVIOUS
from .features import extract,fingerprint
from playlens_ml.board_v2.review import overlay
from playlens_ml.board_v2.audit import robustness,clips


def prepare():
    folder=OUT/'review';folder.mkdir(parents=True,exist_ok=True)
    original=json.loads((PREVIOUS/'review/selection.json').read_text());records=[];images=[]
    for frame in original['frames']:
        source=PREVIOUS/'review'/f"{frame['id']}-source.png";shutil.copy2(source,folder/source.name)
        image=Image.open(source).convert('RGB');result=extract(image);im=overlay(image,result,frame['id']);images.append(im)
        im.save(folder/f"{frame['id']}.png");records.append({**frame,'extraction':result})
    for p in range(6):
        sheet=Image.new('RGB',(2000,800),'white')
        for i,im in enumerate(images[p*10:p*10+10]):sheet.paste(im,((i%5)*400,(i//5)*400))
        sheet.save(folder/f'contact-{p+1}.png')
    (folder/'selection.json').write_text(json.dumps({**original,'extractorHash':fingerprint(),'frames':records},indent=2)+'\n')
    shutil.copy2(PREVIOUS/'alignment.json',OUT/'alignment.json')
    robustness(OUT,extract,fingerprint);clips(OUT,extract,fingerprint)

if __name__=='__main__':prepare()
