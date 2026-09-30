"""Display cached incoming measurements; does not alter extraction."""
import json
import math
from PIL import Image, ImageDraw
from . import OUT


def main():
    audit = json.loads((OUT/'audit.json').read_text()); images=[]
    for row in audit['frames']:
        f=row['extraction']; image=Image.open(OUT/'review'/f"{row['id']}-source.png")
        cx,cy=f['center']; a=f['apothem']; box=(int(cx-2.2*a),int(cy-2.2*a),int(cx+2.2*a),int(cy+2.2*a))
        shown=Image.new('RGB',(440,510),'white');shown.paste(image.crop(box).resize((440,440)),(0,0));d=ImageDraw.Draw(shown)
        d.rectangle((0,0,440,20),fill='white');d.text((4,4),row['id'],fill='black')
        for lane in range(6):
            height,gap,radius,confidence=f['vector'][lane*4:lane*4+4]
            color=f['incomingColors'][lane];label='none' if color<0 else 'RYBG'[color]
            angle=math.radians(30+lane*60)
            x=(cx+radius*a*math.cos(angle)-box[0])*440/(box[2]-box[0]); y=(cy+radius*a*math.sin(angle)-box[1])*440/(box[3]-box[1])
            if color>=0:
                d.line((x-6,y,x+6,y),fill='magenta',width=2);d.line((x,y-6,x,y+6),fill='magenta',width=2)
                d.text((x+6,y),str(lane),fill='black')
            d.text((4+(lane%2)*220,444+(lane//2)*20),f'{lane}: incoming {label} r{radius:.2f} gap{gap:.2f}',fill='black')
        shown.save(OUT/'review'/f"{row['id']}-incoming.png");images.append(shown)
    for page in range(9):
        sheet=Image.new('RGB',(1320,1530),'white')
        for i,im in enumerate(images[page*9:(page+1)*9]):sheet.paste(im,((i%3)*440,(i//3)*510))
        sheet.save(OUT/'review'/f'incoming-{page+1}.jpg')


if __name__=='__main__':main()
