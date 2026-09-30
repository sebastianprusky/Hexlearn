"""Causal incoming-speed and activity estimates from adjacent native frames."""
import hashlib
from pathlib import Path
import numpy as np
from playlens_ml.board_invariant.features import GEOMETRY_NAMES,BASE_NAMES,FEATURE_NAMES as INVARIANT_NAMES,extractor_hash as invariant_hash
DYNAMIC_NAMES=[f'{seconds}s_{name}' for seconds in (1,2,4,8) for name in (
    'incoming_speed_median','incoming_speed_p25','incoming_speed_p75','incoming_speed_max','tracking_pair_fraction',
    'incoming_appearances','possible_landings','total_growth','total_decrease','growth_events','clear_events','net_growth')]
FEATURE_NAMES=INVARIANT_NAMES+DYNAMIC_NAMES


def extractor_hash():return hashlib.sha256(invariant_hash().encode()+Path(__file__).read_bytes()).hexdigest()


def motion_features(base,phase):
    if len(base)!=33:raise ValueError('Eight seconds of preceding four-FPS frames required')
    lanes=np.asarray(base)[:,:24].reshape(33,6,4);radius=lanes[:,:,2];height=lanes[:,:,0]
    step=radius[:-1]-radius[1:]
    stable=(abs(np.asarray(phase[:-1]))<=5)&(abs(np.asarray(phase[1:]))<=5)
    matched=(radius[:-1]<2.05)&(radius[1:]<2.05)&(radius[:-1]>height[:-1]+.1)&(radius[1:]>height[1:]+.1)
    matched&=(step>.004)&(step<.25)&stable[:,None]
    appearances=(radius[:-1]>=2.05)&(radius[1:]<2.05)&stable[:,None]
    landings=(radius[:-1]-height[:-1]<.2)&(radius[:-1]<2.05)&(radius[1:]-radius[:-1]>.15)&stable[:,None]
    growth=np.diff(height.sum(axis=1));out=[]
    for seconds in (1,2,4,8):
        n=seconds*4;mask=matched[-n:];speeds=step[-n:][mask]*4
        quantiles=np.quantile(speeds,[.5,.25,.75,1]).tolist() if len(speeds) else [0.]*4
        g=growth[-n:];out.extend(quantiles+[float(mask.mean()),float(appearances[-n:].sum()),float(landings[-n:].sum()),
                         float(np.maximum(g,0).sum()),float(np.maximum(-g,0).sum()),float((g>.06).sum()),float((g<-.06).sum()),float(g.sum())])
    return np.array(out)
