"""Additional synchronization checks near pauses and throughout each run."""
import json
import numpy as np
from .review import OUT, aligned_frame
from .data import VIDEO_LAG
from playlens_ml.common import ROOT
from playlens_ml.board_experiment.data import source_data
from playlens_ml.build_dataset import personal_runs, read_jsonl


def audit():
    data,_=source_data();records=[]
    for number,run in enumerate(personal_runs(ROOT/'data')[:30],1):
        directory=ROOT/'data/sessions'/run['id'];observations=read_jsonl(directory/'observations.jsonl')
        observations=sorted(observations,key=lambda o:o['activeElapsedMs']);times=np.array([o['activeElapsedMs']/1000 for o in observations])
        targets=[float(times[-1]*f) for f in (.1,.9)]+[p/1000+1 for p in run['resumePoints'] if p>0]
        for n,t in enumerate(targets):
            index=int(np.argmin(abs(times-t)));item=observations[index];path=directory/'frames'/f"{item['timestampMs']}.jpg"
            if not path.exists():continue
            _,alignment=aligned_frame(directory/'gameplay.webm',float(times[index]),path)
            records.append({'run':number,'activeSeconds':float(times[index]),'check':'distribution' if n<2 else 'after_resume',**alignment})
        print('alignment run',number,flush=True)
    passed=all(not r['searchEdge'] and abs(r['offsetSeconds'])+.15<=VIDEO_LAG for r in records)
    result={'passed':passed,'videoLagSeconds':VIDEO_LAG,'checks':records,
            'limitation':'Image matching empirically supports the clock margin; it is not a hardware timestamp guarantee.'}
    (OUT/'alignment.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':audit()
