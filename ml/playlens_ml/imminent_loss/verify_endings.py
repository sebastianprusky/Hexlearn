"""Check failed endings without seeking or synthetic frame-rate conversion."""
import json
import re
import subprocess
from playlens_ml.imminent_loss.review import OUT, digest
from playlens_ml.board_v2.review import decoder
from playlens_ml.common import ROOT


def main():
    manifest = json.loads((OUT / 'review-manifest.json').read_text())
    results = []
    for order in (5, 14, 17):
        run = manifest['runs'][order - 1]
        folder = OUT / f'review/run-{order:02}-full-decode'
        folder.mkdir(exist_ok=True)
        video = ROOT / 'data/sessions' / run['sessionId'] / 'gameplay.webm'
        proc = subprocess.run([
            decoder().get_ffmpeg_exe(), '-y', '-i', str(video), '-vf',
            f"select='gte(t,{run['seekSeconds']})',scale=960:-2,showinfo",
            '-vsync', '0', str(folder / '%04d.png')], capture_output=True, check=True)
        log = proc.stderr.decode()
        (folder / 'decode.log').write_text(log)
        times = [float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.e+-]+)', log)]
        full = sorted(folder.glob('*.png'))
        sought = sorted((OUT / f'review/run-{order:02}-frames').glob('*.png'))
        equal = bool(full) and len(full) == len(sought) == len(times)
        equal = equal and all(digest(a) == digest(b) for a, b in zip(full, sought))
        results.append({'run': order, 'sameAllTailPixels': equal,
                        'count': len(full), 'frameTimesSeconds': times})
        print(order, equal, len(full), flush=True)
    (OUT / 'full-decode-verification.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
