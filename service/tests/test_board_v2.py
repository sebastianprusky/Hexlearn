from __future__ import annotations
import unittest
import numpy as np
from PIL import Image,ImageDraw
from playlens_ml.board_v2.features import extract
from playlens_ml.board_v2.data import temporal,extract_run,FEATURE_NAMES

PALETTE=((231,76,60),(241,196,15),(52,152,219),(46,204,113))

def board(colors=(-1,)*6,phase=0,detached=False):
    image=Image.new('RGB',(960,524),(236,240,241));draw=ImageDraw.Draw(image);cx,cy,a=480,262,100
    def hexagon(apothem,angle=0):
        return [(cx+apothem/np.cos(np.pi/6)*np.cos(np.deg2rad(k*60+angle)),cy+apothem/np.cos(np.pi/6)*np.sin(np.deg2rad(k*60+angle))) for k in range(6)]
    draw.polygon(hexagon(a),fill=(189,195,199))
    for k,c in enumerate(colors):
        if c<0:continue
        angle=np.deg2rad(30+60*k+phase);normal=np.array([np.cos(angle),np.sin(angle)]);side=np.array([-normal[1],normal[0]])
        inner=.32 if not detached else .70;outer=inner+.085
        points=[tuple(np.array([cx,cy])+a*(r*normal+s*r*np.tan(np.pi/6)*side)) for r,s in ((inner,-1),(outer,-1),(outer,1),(inner,1))]
        draw.polygon(points,fill=PALETTE[c])
    draw.polygon(hexagon(a*.32,phase),fill=(44,62,80));draw.text((470,257),'9999',fill='white')
    return image

class NativeBoardTests(unittest.TestCase):
    def test_score_and_dark_core_are_not_blue_blocks(self):
        result=extract(board());self.assertTrue(result['valid']);self.assertEqual(result['topColors'],[-1]*6)
    def test_six_lanes_preserve_four_colors(self):
        colors=[0,1,2,3,0,1];result=extract(board(colors));self.assertTrue(result['valid']);self.assertEqual(result['topColors'],colors)
    def test_rotated_lanes(self):
        colors=[0,1,2,3,0,1];result=extract(board(colors,phase=20));self.assertTrue(result['valid']);self.assertEqual(result['topColors'],colors);self.assertLess(abs(result['phaseDegrees']-20),3)
    def test_detached_blocks_do_not_inflate_stack(self):
        result=extract(board([0,1,2,3,0,1],detached=True));self.assertEqual(result['topColors'],[-1]*6)
    def test_resize_and_letterbox(self):
        source=board([0,1,2,3,0,1]);base=extract(source)
        boxed=Image.new('RGB',(1200,704));boxed.paste(source,(120,90))
        for image,scale,offset in ((source.resize((640,349)),640/960,np.zeros(2)),(boxed,1,np.array([120,90]))):
            result=extract(image);self.assertTrue(result['valid']);self.assertEqual(result['topColors'],base['topColors']);self.assertLess(np.linalg.norm(np.array(result['center'])-np.array(base['center'])*scale-offset),2)
    def test_blank_image_invalid(self):self.assertFalse(extract(Image.new('RGB',(960,524)))['valid'])
    def test_sorted_temporal_is_rotation_invariant_and_causal(self):
        times=np.arange(40)/4;vectors=np.zeros((40,54));lanes=np.arange(24).reshape(6,4)/24
        vectors[:,:24]=lanes.ravel();vectors[32,:24]=np.roll(lanes,2,axis=0).ravel()
        result=temporal(times,vectors,32);self.assertEqual(len(result),len(FEATURE_NAMES));self.assertTrue(np.allclose(result[54:].reshape(4,6,4)[:,:,:3],0))
        vectors[33:]=100;self.assertTrue(np.array_equal(result,temporal(times,vectors,32)))
        with self.assertRaises(ValueError):temporal(times,vectors,20)
    def test_historical_and_future_test_runs_excluded(self):
        for run in (0,31,40,41):
            with self.assertRaises(ValueError):extract_run(run)

if __name__=='__main__':unittest.main()
