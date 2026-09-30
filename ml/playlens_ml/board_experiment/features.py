from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
import numpy as np
from PIL import Image
from scipy import ndimage

from . import SCHEMA

PALETTE = np.array([[231,76,60],[241,196,15],[52,152,219],[46,204,113]], dtype=float)/255
COLOR_NAMES = ('red','yellow','blue','green')
ANGLES = np.arange(6)*np.pi/3 + np.pi/6
NORMALS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
GEOMETRY_NAMES = [f'lane_{lane}_{name}' for lane in range(6)
                  for name in ('height','incoming_gap','incoming_radius','confidence')]
COLOR_FEATURE_NAMES = [f'lane_{lane}_top_{color}' for lane in range(6) for color in COLOR_NAMES] + [f'adjacent_match_{i}' for i in range(6)]
BASE_NAMES = GEOMETRY_NAMES + COLOR_FEATURE_NAMES
TEMPORAL_NAMES = [f'{seconds}s_lane_{lane}_{name}' for seconds in (1,2,4,8) for lane in range(6)
                  for name in ('growth','apparent_clear','incoming_motion','confidence')]
FEATURE_NAMES = BASE_NAMES + TEMPORAL_NAMES


@dataclass
class BoardFrame:
    valid: bool
    reason: str
    center: tuple[float,float]
    apothem: float
    geometry_confidence: float
    vector: np.ndarray
    top_colors: list[int]
    # Pixel maps are diagnostics, never model input.
    colors: np.ndarray


def classify_colors(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=float)
    span = np.ptp(values, axis=-1)
    normalized = (values-values.min(axis=-1,keepdims=True))/np.maximum(span[...,None],1e-8)
    palette = (PALETTE-PALETTE.min(axis=-1,keepdims=True))/np.ptp(PALETTE,axis=-1)[:,None]
    distances = np.linalg.norm(normalized[...,None,:]-palette,axis=-1)
    nearest = distances.argmin(axis=-1)
    return np.where((span>.16)&(distances.min(axis=-1)<.42),nearest,-1).astype(np.int8)


def invalid_frame(shape, reason):
    return BoardFrame(False,reason,(0.,0.),0.,0.,np.zeros(len(BASE_NAMES)),[-1]*6,np.full(shape,-1,dtype=np.int8))


def extract(jpeg: bytes) -> BoardFrame:
    try:
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'),dtype=float)/255
    except (OSError,ValueError):
        return invalid_frame((1,1),'unreadable_image')
    h,w=rgb.shape[:2]
    gray = (np.max(np.abs(rgb-np.array([189,195,199])/255),axis=2)<.085)&(np.ptp(rgb,axis=2)<.13)
    components,count = ndimage.label(gray, structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel()); sizes[0]=0
    if not count or sizes.max()<max(30,.008*h*w):
        return invalid_frame((h,w),'gray_hexagon_not_found')
    mask=components==sizes.argmax(); yy,xx=np.where(mask)
    # Gray board is the largest gray connected region, unlike letterbox/background edges.
    points=np.column_stack((xx,yy)); supports=np.quantile(np.sum(points[:,None,:]*NORMALS[None,:,:],axis=2),.995,axis=0)
    design=np.column_stack((NORMALS,np.ones(6)))
    cx,cy,apothem=np.linalg.lstsq(design,supports,rcond=None)[0]
    residual=float(np.sqrt(np.mean((design@np.array([cx,cy,apothem])-supports)**2)))
    coverage=len(xx)/max(1,2*np.sqrt(3)*apothem**2)
    confidence=float(np.clip(1-residual/max(apothem,1)*8,0,1)*min(1,coverage/.35))
    if apothem<6 or apothem>min(h,w)*.45 or not (0<=cx<w and 0<=cy<h) or confidence<.65:
        return invalid_frame((h,w),'uncertain_hexagon_geometry')
    y,x=np.indices((h,w)); dx=x-cx; dy=y-cy
    radius=np.hypot(dx,dy)
    angle=np.arctan2(dy,dx)
    lane=np.argmin(np.abs(np.angle(np.exp(1j*(angle[...,None]-ANGLES)))),axis=-1)
    deviation=np.min(np.abs(np.angle(np.exp(1j*(angle[...,None]-ANGLES)))),axis=-1)
    radial=(dx*NORMALS[lane,0]+dy*NORMALS[lane,1])/apothem
    colors=classify_colors(rgb)
    # Exclude central score and timer at the outer hexagon. Fit no labels/clock values.
    allowed=(radial>=.36)&(radial<=2.1)&~((radial>=.94)&(radial<=1.09))&(deviation<np.deg2rad(15))
    colors=np.where(allowed,colors,-1)
    edges=np.linspace(.36,.94,21)
    heights=[]; tops=[]; geometry=[]
    for k in range(6):
        profile=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            area=(lane==k)&(radial>=lo)&(radial<hi)&(deviation<np.deg2rad(15))
            v=colors[area]; counts=np.bincount(v[v>=0],minlength=4)
            profile.append(int(counts.argmax()) if counts.max()>=max(2,.30*len(v)) else -1)
        # Stack must connect to the core; permit one empty bin for JPEG aliasing.
        occupied=np.flatnonzero(np.array(profile)>=0)
        connected=[]
        for index in occupied:
            if not connected:
                if index>4: break
            elif index-connected[-1]>2: break
            connected.append(int(index))
        top=int(profile[connected[-1]]) if connected else -1
        height=float(edges[connected[-1]+1]) if connected else .36
        moving=(lane==k)&(colors>=0)&(radial>max(height+.08,.48))
        incoming=float(np.quantile(radial[moving],.1)) if moving.sum()>=3 else 2.1
        gap=max(0.,incoming-height)
        lane_conf=confidence*(1. if connected else .75)
        heights.append(height); tops.append(top)
        geometry.extend((height,gap,incoming,lane_conf))
    color_vector=[float(tops[k]==c) for k in range(6) for c in range(4)]
    adjacent=[float(tops[k]>=0 and tops[k]==tops[(k+1)%6]) for k in range(6)]
    vector=np.asarray(geometry+color_vector+adjacent,dtype=np.float64)
    return BoardFrame(True,'ok',(float(cx),float(cy)),float(apothem),confidence,vector,tops,colors)


def geometry_changed(previous: BoardFrame, current: BoardFrame) -> bool:
    return (not previous.valid or not current.valid or
            abs(current.apothem/previous.apothem-1)>.08 or
            np.linalg.norm(np.array(current.center)-previous.center)>.08*previous.apothem)


def temporal_vector(times: np.ndarray, vectors: np.ndarray, index: int) -> np.ndarray:
    """Caller supplies one continuous segment only. No future frame is consulted."""
    current=vectors[index,:24].reshape(6,4); out=[]
    for seconds in (1,2,4,8):
        target=times[index]-seconds
        j=int(np.searchsorted(times[:index+1],target,side='right')-1)
        if j<0 or target-times[j]>.6:
            raise ValueError('Insufficient past temporal context')
        previous=vectors[j,:24].reshape(6,4)
        growth=current[:,0]-previous[:,0]
        for k in range(6):
            movement=(previous[k,2]-current[k,2])/seconds
            certainty=min(current[k,3],previous[k,3])
            out.extend((growth[k],max(0.,-growth[k]),movement,certainty))
    return np.r_[vectors[index],out]


def extractor_hash():
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
