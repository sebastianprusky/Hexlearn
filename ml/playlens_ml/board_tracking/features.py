"""Causal geometry recovery and explicit colored incoming-block observations."""
import copy
import hashlib
from pathlib import Path
import numpy as np
from playlens_ml.board_experiment.features import classify_colors
from playlens_ml.board_v3.features import extract as native_extract, fingerprint as native_hash
from . import sampling


def fingerprint():
    return hashlib.sha256(native_hash().encode()+Path(__file__).read_bytes()+Path(sampling.__file__).read_bytes()).hexdigest()


def extract(image, previous=None, base=None):
    result=copy.deepcopy(native_extract(image) if base is None else base)
    result['geometryHeld']=False
    if not result['valid']:return result
    result['geometryRefit']=False
    if previous and previous['valid']:
        shift=np.linalg.norm(np.array(result['center'])-previous['center'])/previous['apothem']
        if shift>.075:
            refit=sampling.extract(image)
            if refit['valid']:
                result=refit;result['geometryHeld']=False;result['geometryRefit']=True
    # A complete incoming ring hides the gray support edge. Hold geometry for at
    # most two frames, only when current core pixels confirm unchanged scale/center.
    if result['coreApothemRatio']>.34 and previous and previous['valid']:
        old_core=previous['apothem']*previous['coreApothemRatio']
        core=result['apothem']*result['coreApothemRatio']
        centered=np.linalg.norm(np.array(result['center'])-previous['center'])<.04*previous['apothem']
        count=previous.get('geometryHoldFrames',0)
        if abs(core/old_core-1)<.025 and centered and count<2 and previous['coreApothemRatio']<.34:
            fixed=sampling.extract(image,geometry=(previous['center'],previous['apothem']))
            if fixed['valid'] and abs(fixed['coreApothemRatio']/.325-1)<.055:
                result=fixed;result['geometryHeld']=True;result['geometryRefit']=False;result['geometryHoldFrames']=count+1
    if not result['geometryHeld']:result['geometryHoldFrames']=0
    rgb=np.asarray(image.convert('RGB'),dtype=float)/255;h,w=rgb.shape[:2]
    cx,cy=result['center'];a=result['apothem'];rr=np.arange(.35,2.1,.006);offsets=np.linspace(-.06,.06,9)
    incoming=[];colors=[];confidence=[]
    for lane in range(6):
        angle=np.deg2rad(30+lane*60)
        x=np.rint(cx+a*(rr[:,None]*np.cos(angle)-offsets*np.sin(angle))).astype(int)
        y=np.rint(cy+a*(rr[:,None]*np.sin(angle)+offsets*np.cos(angle))).astype(int)
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        pixels=rgb[y.clip(0,h-1),x.clip(0,w-1)];labels=classify_colors(pixels)
        labels[(~inside)|(pixels.max(axis=2)<.52)]=-1
        votes=np.stack([(labels==c).sum(axis=1) for c in range(4)],axis=1)
        line=np.where(votes.max(axis=1)>=6,votes.argmax(axis=1),-1)
        # The decorative timer is too thin to meet the minimum radial thickness.
        # Unlike v3, do not erase the entire radius .94..1.08 annulus.
        height=result['heights'][lane] if abs(result['phaseDegrees'])<=5 else max(result['heights'])
        near=2.1;color=-1;conf=0.
        starts=np.r_[0,np.flatnonzero(line[1:]!=line[:-1])+1];ends=np.r_[starts[1:],len(line)]
        for start,end in zip(starts,ends):
            if line[start]>=0 and end-start>=7 and rr[start]>max(.48,height+.07):
                near=float(rr[start]);color=int(line[start]);conf=float(votes[start:end,color].mean()/9);break
        incoming.append(near);colors.append(color);confidence.append(conf)
    vector=np.array(result['vector']);lanes=vector[:24].reshape(6,4)
    lanes[:,2]=incoming;lanes[:,1]=np.maximum(0,np.array(incoming)-lanes[:,0])
    result.update(vector=vector.tolist(),incomingColors=colors,incomingConfidence=confidence)
    return result
