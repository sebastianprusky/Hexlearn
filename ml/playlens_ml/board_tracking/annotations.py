"""Explicit assistant clip annotations, separate from prediction targets."""
import json
from .data import OUT
from .features import fingerprint

# Zero-based video-frame indices in the deterministic 16-frame contact sheets.
# A clear means visible block disappearance/clear flash; score text is not used.
CLIPS=[
 (1, [15], (3,0,1,6), 'Red upper-left block approaches; yellow pair stacks; visible white clear at final frame.'),
 (2, [], (1,2,0,3), 'Bottom blue approaches and stacks; rotation with no visible clear.'),
 (3, [7], (1,1,0,3), 'Bottom yellow approaches; right/bottom yellow disappearance at frame 7.'),
 (4, [], (3,3,1,5), 'Upper-left green approaches and stacks; rotations without clear.'),
 (5, [9], (4,2,0,4), 'Top blue approaches; right-side green stack disappears at frame 9.'),
 (6, [3,9,14], (5,2,1,6), 'Three visible flashes/disappearances; blue approaches upper-right.'),
 (7, [3], (1,1,3,7), 'Left yellow clear at frame 3; later bottom yellow approaches.'),
 (8, [9], (4,2,2,6), 'Top/right blue pair approaches; right green clear around frame 9.'),
 (9, [4,8], (4,3,6,9), 'Left/bottom yellow and right blue clear; top green approaches.'),
 (10,[3,13],(4,3,1,5), 'Blue left/top clear then green clear; top green approach.'),
 (11,[3,8], (5,2,5,10),'Left blue and yellow clear; upper-right blue approaches.'),
 (12,[12], (3,0,2,6), 'Red upper-left approaches; blue right clears after rotation at frame 12.')]


REVIEWED_REPAIRS = ['run-01-83', 'run-02-603', 'run-04-627', 'run-07-481', 'run-08-36', 'run-10-130', 'run-10-574', 'run-11-239', 'run-12-63', 'run-13-631', 'run-16-362', 'run-16-518', 'run-18-459', 'run-19-202', 'run-20-126', 'run-20-458', 'run-20-465', 'run-21-186', 'run-21-269', 'run-22-273', 'run-22-626', 'run-23-48', 'run-24-241', 'run-24-477', 'run-24-571', 'run-24-684', 'run-27-142', 'run-27-261', 'run-27-700', 'run-28-261', 'run-28-400', 'run-30-459', 'run-30-50']
UNCERTAIN_COLORS = ['run-10-130', 'run-22-626', 'run-30-50']

def write():
    repairs=sorted(p.stem for p in (OUT/'review/repairs').glob('*.png') if '-source' not in p.name)
    if repairs != REVIEWED_REPAIRS:raise ValueError('New repaired frames require explicit visual review')
    payload={'extractorHash':fingerprint(),'reviewType':'assistant visual annotation; not independent ground truth',
       'clearTimingToleranceFrames':2,'clearDefinition':'visible block removal or clearing flash; no score-derived model features',
       'clips':[dict(id=f'clip-{n:02}',clearFrames=clears,incoming=dict(lane=track[0],color=track[1],startFrame=track[2],endFrame=track[3],direction='inward'),notes=note) for n,clears,track,note in CLIPS],
       'repairs':[dict(id=name,geometryCorrect=True,colorsCorrect=name not in UNCERTAIN_COLORS) for name in repairs],
       'repairNotes':'Reviewed held and refitted geometry against visible board edges. Uncertain outer-color judgments count as failures.',
       'speedRulers':[dict(clip='clip-02',lane=1,startFrame=0,endFrame=3,startRadius=1.21,endRadius=.72),
                       dict(clip='clip-03',lane=1,startFrame=0,endFrame=3,startRadius=.80,endRadius=.50),
                       dict(clip='clip-07',lane=1,startFrame=3,endFrame=7,startRadius=1.49,endRadius=.82)],
       'speedRulerNotes':'Approximate hand-read radii from contact sheets, in gray-board apothems; tolerance 0.15 apothems per second. Diagnostic examples, not a speed-accuracy benchmark.'}
    (OUT/'review/annotations.json').write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__':write()
