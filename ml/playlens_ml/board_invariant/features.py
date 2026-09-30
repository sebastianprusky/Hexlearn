"""Rotation-invariant summaries of the already extracted native board features."""
import hashlib
from pathlib import Path
import numpy as np
from playlens_ml.board_v3.features import fingerprint as native_hash
GEOMETRY_NAMES=[f'sorted_{name}_{rank}' for name in ('height','gap','incoming') for rank in range(6)]+[
    'height_mean','height_std','height_range','height_sum','lanes_above_050','lanes_above_065','lanes_above_080','lanes_above_095',
    'gap_mean','gap_std','gap_at_tallest','gap_min_tall_lanes','confidence_min','confidence_mean']
COLOR_NAMES=[f'outer_color_count_{c}' for c in range(4)]+['adjacent_match_count','outer_color_entropy']+[
    f'canonical_lane_{k}_color_{c}' for k in range(6) for c in range(4)]
BASE_NAMES=GEOMETRY_NAMES+COLOR_NAMES
TEMPORAL_NAMES=[f'{seconds}s_{name}_{rank}' for seconds in (1,2,4,8) for name in ('height_change','apparent_clear','incoming_motion','confidence') for rank in range(6)]
TEMPORAL_SUMMARIES=[f'{seconds}s_{name}' for seconds in (1,2,4,8) for name in ('net_growth','largest_growth','largest_decline','total_apparent_clear','motion_mean','motion_max')]
FEATURE_NAMES=BASE_NAMES+TEMPORAL_NAMES+TEMPORAL_SUMMARIES


def extractor_hash():
    return hashlib.sha256(native_hash().encode()+Path(__file__).read_bytes()).hexdigest()


def transform(vector):
    lanes=vector[:24].reshape(6,4);heights,gaps,incoming,confidence=lanes.T
    geometry=np.r_[np.sort(heights),np.sort(gaps),np.sort(incoming),heights.mean(),heights.std(),np.ptp(heights),heights.sum(),
                    [(heights>v).sum() for v in (.5,.65,.8,.95)],gaps.mean(),gaps.std(),gaps[heights==heights.max()].min(),
                    gaps[heights>=np.median(heights)].min(),confidence.min(),confidence.mean()]
    colors=vector[24:48].reshape(6,4);counts=colors.sum(axis=0);distribution=counts/max(1,counts.sum())
    entropy=-np.sum(distribution[distribution>0]*np.log(distribution[distribution>0]))
    # Select the lexicographically maximal full ring, including color ties.
    # A cyclic lane permutation therefore produces exactly the same representation.
    rings=[np.roll(np.column_stack((heights,colors)),-k,axis=0) for k in range(6)]
    canonical=max(rings,key=lambda r:tuple(r.ravel()))[:,1:].ravel()
    color=np.r_[counts,vector[48:54].sum(),entropy,canonical]
    changes=vector[54:].reshape(4,6,4);ordered=changes.transpose(0,2,1).ravel();summary=[]
    for block in changes:
        growth=block[:,0];motion=block[:,2]
        summary.extend((growth.sum(),growth.max(),growth.min(),block[:,1].sum(),motion.mean(),motion.max()))
    result=np.r_[geometry,color,ordered,summary]
    if len(result)!=len(FEATURE_NAMES) or not np.isfinite(result).all():raise ValueError('Invalid invariant features')
    return result
