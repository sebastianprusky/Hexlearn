import json
from PIL import Image
from .data import OUT,SOURCE
from .features import normalize,extractor_hash
from playlens_ml.board_v2.review import overlay


def prepare():
    folder=OUT/'review';folder.mkdir(parents=True,exist_ok=True);original=json.loads((SOURCE/'review/selection.json').read_text());records=[];images=[]
    for frame in original['frames']:
        result=normalize(frame['extraction']);source=SOURCE/'review'/f"{frame['id']}-source.png";image=Image.open(source).convert('RGB')
        im=overlay(image,result,frame['id']);images.append(im);im.save(folder/f"{frame['id']}.png");records.append({**frame,'extraction':result})
    for p in range(6):
        sheet=Image.new('RGB',(2000,800),'white')
        for i,im in enumerate(images[p*10:p*10+10]):sheet.paste(im,((i%5)*400,(i//5)*400))
        sheet.save(folder/f'contact-{p+1}.png')
    (folder/'selection.json').write_text(json.dumps({**original,'extractorHash':extractor_hash(),'frames':records},indent=2)+'\n')

if __name__=='__main__':prepare()
