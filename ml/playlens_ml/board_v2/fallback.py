from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
from playlens_ml.board_experiment.features import classify_colors, NORMALS, COLOR_NAMES


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def extract(image):
    rgb=np.asarray(image.convert('RGB'),dtype=float)/255
    h,w=rgb.shape[:2]; yy,xx=np.indices((h,w)); gray=np.max(abs(rgb-np.array([189,195,199])/255),axis=2)<.06
    # Union of visible gray fragments: stacked/rotating blocks must not split the board.
    gray &= (abs(xx-w/2)<w*.26)&(abs(yy-h/2)<w*.26)&(np.ptp(rgb,axis=2)<.10)
    gray_components,_=ndimage.label(gray)
    gray_sizes=np.bincount(gray_components.ravel());gray_sizes[0]=0
    gray &= gray_sizes[gray_components]>=h*w*.0006
    y,x=np.where(gray)
    if len(x)<max(30,h*w*.008):return {'valid':False,'reason':'gray_not_found'}
    # Core center is independent of which outer gray faces a tall stack occludes.
    core_mask=(rgb[:,:,0]<.28)&(rgb[:,:,1]<.36)&(rgb[:,:,2]<.43)&(abs(xx-w/2)<w*.12)&(abs(yy-h/2)<h*.2)
    components,count=ndimage.label(core_mask)
    if not count:return {'valid':False,'reason':'core_center_missing'}
    sizes=np.bincount(components.ravel());sizes[0]=0;component=components==sizes.argmax()
    core_y,core_x=np.where(component)
    if len(core_x)<100:return {'valid':False,'reason':'core_center_missing'}
    cx=(core_x.min()+core_x.max())/2;cy=(core_y.min()+core_y.max())/2
    points=np.column_stack((x-cx,y-cy))
    supports=np.max(np.sum(points[:,None,:]*NORMALS[None,:,:],axis=2),axis=0)
    a=float(np.median(supports))
    errors=abs(supports-a)/max(1,a);residual=float(np.mean(np.sort(errors)[:3]))
    if a<8 or np.sum(errors<.035)<3 or np.mean(np.max(np.sum(points[:,None,:]*NORMALS[None,:,:],axis=2),axis=1)>a*1.04)>.02:
        return {'valid':False,'reason':'occluded_gray_fit_uncertain'}
    dx=xx-cx;dy=yy-cy;rad=np.hypot(dx,dy)/a;theta=np.arctan2(dy,dx)
    dark=(rgb[:,:,0]<.28)&(rgb[:,:,1]<.36)&(rgb[:,:,2]<.43)&(rad<.46)
    # The visible central hexagon provides orientation. Score lettering creates holes,
    # so measure each ray's last dark pixel rather than region mean/color mass.
    angles=np.linspace(-np.pi,np.pi,180,endpoint=False); radii=np.linspace(.20,.45,101)
    sx=np.rint(cx+np.cos(angles[:,None])*radii*a).astype(int).clip(0,w-1)
    sy=np.rint(cy+np.sin(angles[:,None])*radii*a).astype(int).clip(0,h-1)
    rays=dark[sy,sx]; outer=np.max(np.where(rays,radii,0),axis=1); usable=outer>.22
    if usable.mean()<.8:return {'valid':False,'reason':'central_hexagon_uncertain'}
    scores=[]
    for deg in range(60):
        normals=np.deg2rad(np.arange(6)*60+30+deg)
        projection=np.max(np.cos(angles[:,None]-normals),axis=1)
        apothems=outer[usable]*projection[usable]
        scores.append(float(np.median(abs(apothems-np.median(apothems)))))
    phase=int(np.argmin(scores)); phase=phase if phase<30 else phase-60; axis=np.deg2rad(np.arange(6)*60+30+phase)
    core_projection=np.max(np.cos(angles[:,None]-axis),axis=1)
    core=float(np.median(outer[usable]*core_projection[usable]))
    colors=classify_colors(rgb)
    colors[rgb.max(axis=2)<.52]=-1
    for color in range(4):
        components,_=ndimage.label(colors==color)
        sizes=np.bincount(components.ravel())
        small=sizes[components]<max(8,a*a*.004)
        colors[(colors==color)&small]=-1
    tops=[];heights=[];sequences=[];incoming=[];confidences=[]
    for lane,angle in enumerate(axis):
        # Sample a center strip along each face normal in board coordinates.
        rr=np.arange(core+.008,1.14,.012)
        offsets=np.linspace(-.085,.085,15)
        xq=np.rint(cx+(rr[:,None]*np.cos(angle)-offsets*np.sin(angle))*a).astype(int).clip(0,w-1)
        yq=np.rint(cy+(rr[:,None]*np.sin(angle)+offsets*np.cos(angle))*a).astype(int).clip(0,h-1)
        samples=colors[yq,xq]; counts=np.stack([(samples==c).sum(axis=1) for c in range(4)],axis=1)
        labels=np.where(counts.max(axis=1)>=7,counts.argmax(axis=1),-1)
        occupied=labels>=0
        # Only bridge tiny image/compression gaps. A detached block remains incoming.
        connected=ndimage.binary_closing(occupied,structure=np.ones(3),border_value=0)
        connected|=occupied
        positions=np.flatnonzero(connected); accepted=[]
        for j in positions:
            if not accepted and rr[j]>core+.055:break
            if accepted and j-accepted[-1]>1:break
            accepted.append(int(j))
        actual=[j for j in accepted if labels[j]>=0]
        if actual:
            top=int(labels[actual[-1]]);height=float(rr[actual[-1]]+.006)
            sequence=[]
            for j in actual:
                c=int(labels[j])
                if not sequence or c!=sequence[-1]:sequence.append(c)
        else:top=-1;height=core;sequence=[]
        # Incoming lanes stay fixed in screen space while the stack rotates.
        fixed=np.deg2rad(30+60*lane)
        projected=(dx*np.cos(fixed)+dy*np.sin(fixed))/a
        across=abs(-dx*np.sin(fixed)+dy*np.cos(fixed))/a
        moving=(across<np.tan(np.deg2rad(12))*projected)&(projected>max(.48,height+.06))&(projected<2.1)&(colors>=0)&~((projected>.94)&(projected<1.08))
        if moving.sum()>max(4,a*.05):near=float(np.quantile(projected[moving],.03))
        else:near=2.1
        tops.append(top);heights.append(height);sequences.append(sequence);incoming.append(near)
        confidences.append(float(np.clip(1-residual*10-scores[phase]*10,0,1)))
    geometry=[v for k in range(6) for v in (heights[k],max(0,incoming[k]-heights[k]),incoming[k],confidences[k])]
    color_features=[float(tops[k]==c) for k in range(6) for c in range(4)]
    adjacency=[float(tops[k]>=0 and tops[k]==tops[(k+1)%6]) for k in range(6)]
    return {'valid':True,'reason':'ok','center':[float(cx),float(cy)],'apothem':float(a),
            'phaseDegrees':phase,'coreApothemRatio':core,'orientationError':scores[phase],
            'topColors':tops,'heights':heights,'colorSequences':sequences,
            'vector':geometry+color_features+adjacency}
