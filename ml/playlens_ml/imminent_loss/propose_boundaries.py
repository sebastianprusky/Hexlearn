"""Suggest visual outcome boundaries for review, never approve them automatically.

The white center-score pixels are outcome-annotation aids only. This module
never extracts model inputs, and no score value is read.
"""
import json
import numpy as np
from PIL import Image, ImageDraw
from .review import OUT


def main():
    manifest = json.loads((OUT / 'review-manifest.json').read_text())
    rows, images = [], []
    for run in manifest['runs']:
        order = run['run']
        paths = sorted((OUT / f'review/run-{order:02}-frames').glob('*.png'))
        times = run['decoderMetadata']['frameTimesSeconds']
        if len(paths) != len(times) or not paths:
            raise ValueError('Missing or inconsistent review frames')
        counts = []
        for path in paths:
            array = np.asarray(Image.open(path))
            height, width = array.shape[:2]
            center = array[height//2-7:height//2+8, width//2-23:width//2+24].astype(float)
            counts.append(int(((center.min(2) > 150) &
                               (center.max(2) - center.min(2) < 50)).sum()))
        transitions = [i for i, count in enumerate(counts)
                       if i > 0 and count < 5 and counts[i-1] >= 5]
        index = transitions[-1] if transitions else len(paths)-1
        rows.append({'run': order, 'index': index,
                     'times': times[max(0, index-1):index+1],
                     'counts': counts[max(0, index-2):index+3],
                     'found': bool(transitions)})
        pair = Image.new('RGB', (600, 180), 'white')
        for column, frame in enumerate((max(0, index-1), index)):
            im = Image.open(paths[frame]); im.thumbnail((300, 165))
            ImageDraw.Draw(im).text((2, 2),
                f'run{order} {times[frame]:.3f} count{counts[frame]}', fill='red')
            pair.paste(im, (column*300, 0))
        images.append(pair)
    for page in range((len(images)+9)//10):
        sheet = Image.new('RGB', (1200, 900), 'white')
        for index, im in enumerate(images[page*10:(page+1)*10]):
            sheet.paste(im, ((index % 2)*600, (index//2)*180))
        sheet.save(OUT / f'review/boundaries-{page+1}.jpg')
    (OUT / 'boundary-proposals.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Prepared {len(rows)} boundary proposals; visual approval required')


if __name__ == '__main__':
    main()
