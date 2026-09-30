"""Prepare outcome-only review evidence before any forecasting is permitted."""
import hashlib
import json
import re
import subprocess
from PIL import Image, ImageDraw
from playlens_ml.common import ROOT
from playlens_ml.build_dataset import personal_runs
from playlens_ml.board_v2.review import decoder
from . import SCHEMA

OUT = ROOT / 'artifacts/experiments' / SCHEMA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_runs():
    runs = personal_runs(ROOT / 'data')[:30]
    if len(runs) != 30:
        raise ValueError('Exactly 30 development recordings are required')
    return runs


def prepare():
    folder = OUT / 'review'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    endings = []
    for order, run in enumerate(development_runs(), 1):
        directory = ROOT / 'data/sessions' / run['id']
        video = directory / 'gameplay.webm'
        start = max(0, run['activeDurationMs'] / 1000 - 10)
        frame_dir = folder / f'run-{order:02}-frames'
        frame_dir.mkdir(exist_ok=True)
        command = [decoder().get_ffmpeg_exe(), '-y', '-ss', str(start), '-i', str(video),
                   '-vf', 'scale=960:-2,showinfo', '-vsync', '0',
                   str(frame_dir / '%04d.png')]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        log = proc.stderr.decode()
        (folder / f'run-{order:02}-decode.log').write_text(log)
        times = [start + float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.e+-]+)', log)]
        paths = sorted(frame_dir.glob('*.png'))
        if len(paths) != len(times) or not paths:
            raise ValueError(f'Run {order}: inconsistent decoded frames/timestamps')
        # Contact sheets sample real timestamps. No synthetic duplicate frames.
        selected = []
        next_time = start
        for path, time in zip(paths, times):
            if time >= next_time:
                selected.append(path)
                next_time = time + .24
        sampled = [Image.open(p).convert('RGB') for p in selected]
        tail = [Image.open(p).convert('RGB') for p in paths[-12:]]
        count = len(paths)
        meta = {'timestampSource': 'ffmpeg showinfo PTS after accurate input seek',
                'frameTimesSeconds': times, 'seekAlignmentVerified': False}
        last = tail[-1]
        last.save(folder / f'run-{order:02}-last.png')
        for name, frames in [('ten-seconds', sampled), ('native-tail', list(tail))]:
            sheet = Image.new('RGB', (1600, 180 * ((len(frames) + 4) // 5)), 'white')
            for i, im in enumerate(frames):
                im = im.copy(); im.thumbnail((320, 175))
                ImageDraw.Draw(im).text((4, 4), f'run {order} {name} frame {i}', fill='red')
                sheet.paste(im, ((i % 5) * 320, (i // 5) * 180))
            sheet.save(folder / f'run-{order:02}-{name}.jpg')
        thumb = last.copy(); thumb.thumbnail((320, 175))
        ImageDraw.Draw(thumb).text((4, 4), f'RUN {order} LAST SAVED FRAME', fill='red')
        endings.append(thumb)
        records.append({'run': order, 'sessionId': run['id'], 'videoSha256': digest(video),
                        'observationsSha256': digest(directory / 'observations.jsonl'),
                        'metadataActiveDurationSeconds': run['activeDurationMs'] / 1000,
                        'resumePointsMs': run['resumePoints'], 'seekSeconds': start,
                        'decoderMetadata': meta, 'decodedTailFrames': count,
                        'alignmentStatus': 'not_verified', 'lossInterval': None})
        (OUT / 'review-manifest.json').write_text(json.dumps({'schema': SCHEMA,
            'reviewCodeSha256': digest(__file_path()), 'runs': records}, indent=2) + '\n')
        print(f'Prepared run {order}', flush=True)
    sheet = Image.new('RGB', (1600, 1080), 'white')
    for i, im in enumerate(endings):
        sheet.paste(im, ((i % 5) * 320, (i // 5) * 180))
    sheet.save(folder / 'all-last-frames.jpg')
    (OUT / 'review-manifest.json').write_text(json.dumps({'schema': SCHEMA,
        'reviewCodeSha256': digest(__file_path()), 'runs': records}, indent=2) + '\n')


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__ == '__main__':
    prepare()
