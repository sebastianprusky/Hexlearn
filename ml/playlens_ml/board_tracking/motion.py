"""Conservative causal motion estimates; physical identities remain uncertain."""
import numpy as np


def motion_features(frames):
    if len(frames)!=33 or not all(f['valid'] for f in frames):raise ValueError('33 valid preceding frames required')
    base=np.array([f['vector'] for f in frames]);lanes=base[:,:24].reshape(33,6,4)
    height=lanes[:,:,0];radius=lanes[:,:,2];color=np.array([f['incomingColors'] for f in frames]);conf=np.array([f['incomingConfidence'] for f in frames])
    phase=np.array([f['phaseDegrees'] for f in frames]);scale=np.array([f['apothem'] for f in frames])
    centers=np.array([f['center'] for f in frames]);step=radius[:-1]-radius[1:]
    stable=(abs(phase[:-1])<=5)&(abs(phase[1:])<=5)&(abs(scale[1:]/scale[:-1]-1)<.03)
    stable&=np.linalg.norm(np.diff(centers,axis=0),axis=1)<.03*scale[:-1]
    same=(color[:-1]>=0)&(color[:-1]==color[1:])&(conf[:-1]>=.7)&(conf[1:]>=.7)
    matched=same&(step>.004)&(step<.25)&stable[:,None]
    # Two consecutive inward steps with the same color reject one-frame identities.
    verified=matched.copy();verified[0]=False
    verified[1:] &= matched[:-1]&(color[2:]==color[:-2])
    appearances=(color[:-1]<0)&(color[1:]>=0)&stable[:,None]
    landings=(radius[:-1]-height[:-1]<.2)&(color[:-1]>=0)&(radius[1:]-radius[:-1]>.15)&stable[:,None]&(height[1:]-height[:-1]>.025)
    # Accept a new activity state after two consistent observations. No future
    # frame is read: confirmation is attributed to the second observation.
    accepted=None;accepted_at=-99;activity=np.zeros(32);observed=np.zeros(32,bool)
    sorted_heights=np.sort(height,axis=1)
    for i in range(1,33):
        consistent=stable[i-1] and np.max(abs(sorted_heights[i]-sorted_heights[i-1]))<.035
        if not consistent:continue
        current=(sorted_heights[i]+sorted_heights[i-1])/2
        if accepted is not None and i-accepted_at<=3:
            delta=float(current.sum()-accepted.sum())
            if abs(delta)>.06:activity[i-1]=delta
            observed[i-1]=True
        accepted=current;accepted_at=i
    out=[]
    for seconds in (1,2,4,8):
        n=seconds*4;mask=verified[-n:];speeds=step[-n:][mask]*4
        q=np.quantile(speeds,[.5,.25,.75,1]).tolist() if len(speeds) else [0.]*4
        g=activity[-n:]
        out.extend(q+[float(mask.mean()),float(appearances[-n:].sum()),float(landings[-n:].sum()),
                      float(np.maximum(g,0).sum()),float(np.maximum(-g,0).sum()),float((g>.06).sum()),float((g<-.06).sum()),float(g.sum())])
    return np.array(out)
