"""Causal clear-flash evidence on previously colored stack faces; no OCR."""
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from playlens_ml.board_experiment.features import classify_colors

RADII=np.arange(.37,.93,.012)
OFFSETS=np.linspace(-.19,.19,21)


def fingerprint():
    from playlens_ml.board_experiment import features
    from playlens_ml.board_tracking.features import fingerprint as geometry_hash
    return hashlib.sha256(Path(__file__).read_bytes()+Path(features.__file__).read_bytes()+geometry_hash().encode()).hexdigest()


def observation(image,geometry):
    if not geometry['valid']:return {'valid':False,'reason':'unsupported_geometry'}
    rgb=np.asarray(image.convert('RGB'),dtype=float)/255;h,w=rgb.shape[:2]
    cx,cy=geometry['center'];a=geometry['apothem'];white=[];colored=[]
    for lane in range(6):
        angle=np.deg2rad(30+60*lane+geometry['phaseDegrees'])
        x=np.rint(cx+a*(RADII[:,None]*np.cos(angle)-OFFSETS*np.sin(angle))).astype(int)
        y=np.rint(cy+a*(RADII[:,None]*np.sin(angle)+OFFSETS*np.cos(angle))).astype(int)
        if np.any((x<0)|(x>=w)|(y<0)|(y>=h)):return {'valid':False,'reason':'board_crop_incomplete'}
        pixels=rgb[y,x];white.append(((pixels.min(axis=2)>.84)&(np.ptp(pixels,axis=2)<.065)).mean(axis=1))
        colored.append((classify_colors(pixels)>=0).mean(axis=1))
    return {'valid':True,'center':geometry['center'],'apothem':a,'white':np.array(white).tolist(),'colored':np.array(colored).tolist()}


class Detector:
    def __init__(self, sample_hz=4):
        if sample_hz not in (4,12):raise ValueError("Unsupported sampling rate")
        self.sample_hz=sample_hz;self.reset()
    def reset(self):
        self.history=[];self.previous_time=None;self.previous_geometry=None;self.active=False
    def update(self,obs,time,reset=False):
        changed=False
        if obs['valid'] and self.previous_geometry:
            a,c=self.previous_geometry
            changed=abs(obs['apothem']/a-1)>.08 or np.linalg.norm(np.array(obs['center'])-c)>.08*a
        if reset or not obs['valid'] or changed or (self.previous_time is not None and (time<=self.previous_time or time-self.previous_time>1.2/self.sample_hz)):
            self.reset()
        if not obs['valid']:return {'supported':False,'event':False,'confidence':0.,'reason':obs.get('reason','unsupported')}
        white=np.asarray(obs['white']);colored=np.asarray(obs['colored']);evidence=0.
        if self.history:
            # Sixfold core symmetry aliases a full lane rotation. Use previous
            # radial color support across the ring, without claiming lane identity.
            prior=np.max(np.stack(self.history),axis=(0,1))
            for lane in range(6):
                mask=(white[lane]>=.57)&(prior>=.57)
                labels,count=ndimage.label(mask)
                for label in range(1,count+1):
                    rows=np.flatnonzero(labels==label)
                    if len(rows)>=2:evidence=max(evidence,float(white[lane,rows].mean()))
        broad_white=any(np.any(np.convolve((row>=.57).astype(int),np.ones(2,dtype=int),mode='valid')>=2) for row in white)
        active=evidence>0 or (self.active and broad_white);event=active and not self.active
        self.history=(self.history+[colored])[-2:];self.previous_time=float(time);self.previous_geometry=(obs['apothem'],np.array(obs['center']));self.active=active
        return {'supported':True,'event':bool(event),'confidence':evidence,'reason':'clear_flash' if active else 'no_confirmed_flash'}
