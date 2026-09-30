"""Review alignment and near-loss extraction before any forecast evaluation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import OUT, RUNS, EXCLUDED, SCHEMA, FOLDS
from playlens_ml.common import ROOT
from playlens_ml.build_dataset import personal_runs, read_jsonl
from playlens_ml.imminent_loss import review as original
from playlens_ml.board_v2.review import aligned_frame, overlay, decoder
from playlens_ml.board_tracking.features import fingerprint


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / 'review'; folder.mkdir(exist_ok=True)
    annotations = json.loads(Path(original.__file__).with_name('reviewed_endings.json').read_text())
    sources = personal_runs(ROOT / 'data')
    protocol = {'schema': SCHEMA, 'runs': RUNS, 'excludedRuns': EXCLUDED, 'folds': FOLDS,
                'authorization': 'User authorized explicit 27-run exploratory subset',
                'primaryHorizonSeconds': 5, 'secondaryHorizonSeconds': 3,
                'originalProtocolSha256': original.digest(Path(original.__file__).with_name('README.md')),
                'changes': 'Only exclude runs 5,14,17; retain original numerical run splits and thresholds',
                'extractorHash': fingerprint(), 'forecastResultsViewed': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    checks, reviewed, provenance, images = [], [], [], []
    for n in RUNS:
        run = sources[n-1]; annotation = annotations['runs'][n-1]
        directory = ROOT / 'data/sessions' / run['id']; video = directory / 'gameplay.webm'
        cache_path = ROOT / f'artifacts/experiments/board-tracking-v1/cache/run-{n:02}.json'
        cache = json.loads(cache_path.read_text())
        video_hash = original.digest(video)
        if video_hash != annotation['videoSha256'] or video_hash != cache['videoSha256'] or cache['extractorHash'] != fingerprint():
            raise ValueError('Source or extractor changed')
        obs = sorted(read_jsonl(directory / 'observations.jsonl'), key=lambda o:o['activeElapsedMs'])
        times = np.array([o['activeElapsedMs']/1000 for o in obs])
        loss = sum(annotation['lossIntervalVideoSeconds'])/2
        targets = [('ending', loss-8), ('ending', loss-3)]
        for point in run['resumePoints']:
            if point > 0:
                targets.extend([('before_pause', point/1000-1), ('after_resume', point/1000+1)])
        for j, (kind, target) in enumerate(targets):
            i = int(np.argmin(abs(times-target))); item = obs[i]
            jpeg = directory / 'frames' / f"{item['timestampMs']}.jpg"
            if not jpeg.exists():
                checks.append({'run': n, 'kind': kind, 'passed': False, 'reason': 'missing_observation_image'}); continue
            image, match = aligned_frame(video, float(times[i]), jpeg)
            passed = not match['searchEdge'] and abs(match['offsetSeconds'])+.15 <= .5
            image.save(folder / f'alignment-{n:02}-{j}.png')
            checks.append({'run': n, 'kind': kind, 'activeSeconds': float(times[i]), 'passed': passed, **match})
        # Decode the SAME fps-filtered pixels used by the existing cache.
        selected = {int((loss-lead)*4): lead for lead in (10, 5, 2)}
        stream = decoder().read_frames(str(video), output_params=['-vf', 'fps=4,scale=960:-2'])
        meta = next(stream)
        try:
            for index, raw in enumerate(stream):
                if index not in selected: continue
                image = Image.frombytes('RGB', meta['size'], raw)
                frame = cache['frames'][index]; lead = selected[index]
                key = f'run-{n:02}-lead-{lead}'
                image.save(folder / f'{key}-source.png')
                shown = overlay(image, frame, key); shown.save(folder / f'{key}.png'); images.append(shown)
                reviewed.append({'id': key, 'run': n, 'leadTarget': lead, 'videoSeconds': index/4,
                                 'extraction': frame, 'sourceSha256': original.digest(folder/f'{key}-source.png')})
                if index == max(selected): break
        finally: stream.close()
        provenance.append({'run': n, 'sessionId': run['id'], 'videoSha256': video_hash,
                           'observationsSha256': original.digest(directory/'observations.jsonl'),
                           'cacheSha256': original.digest(cache_path), 'lossIntervalVideoSeconds': annotation['lossIntervalVideoSeconds']})
        print('Audited run', n, flush=True)
    for page in range((len(images)+8)//9):
        sheet = Image.new('RGB', (1200, 1200), 'white')
        for j, im in enumerate(images[page*9:(page+1)*9]): sheet.paste(im, ((j%3)*400, (j//3)*400))
        sheet.save(folder/f'contact-{page+1}.jpg')
    result = {'schema': SCHEMA, 'provenance': provenance, 'alignment': checks,
              'alignmentPassed': all(c['passed'] for c in checks), 'frames': reviewed,
              'visualReviewStatus': 'pending', 'forecastingAllowed': False}
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Alignment passed:', result['alignmentPassed'], flush=True)


if __name__ == '__main__':
    prepare()
