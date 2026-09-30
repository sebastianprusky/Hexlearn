import copy
import unittest
import numpy as np
from PIL import ImageDraw
from test_board_v2 import board,PALETTE
from playlens_ml.board_tracking.features import extract
from playlens_ml.board_tracking.motion import motion_features
from playlens_ml.board_tracking.data import extract_run


def frames():
    result=[]
    for i in range(33):
        lanes=np.zeros((6,4));lanes[:,0]=.4;lanes[:,2]=2.1;lanes[0,2]=1.9-i*.025
        result.append(dict(valid=True,vector=lanes.ravel().tolist()+[0.]*30,incomingColors=[1,-1,-1,-1,-1,-1],
                           incomingConfidence=[1.]*6,phaseDegrees=0,apothem=100,center=[480,262]))
    return result


class TrackingTests(unittest.TestCase):
    def test_color_identity_switch_does_not_create_speed_pair(self):
        fs=frames()
        for i,f in enumerate(fs):f['incomingColors'][0]=i%2
        self.assertTrue(np.all(motion_features(fs).reshape(4,12)[:,:5]==0))
    def test_constant_speed_survives_identity_confirmation(self):
        out=motion_features(frames()).reshape(4,12)
        self.assertTrue(np.allclose(out[:,:4],.1))
        self.assertAlmostEqual(out[-1,4],31/(32*6))
    def test_single_frame_clear_flash_is_not_clear(self):
        fs=frames();fs[29]['vector'][0]=.1
        self.assertTrue(np.all(motion_features(fs).reshape(4,12)[:,8:11]==0))
    def test_persistent_clear_confirmed_after_second_observation(self):
        fs=frames()
        for f in fs[28:]:f['vector'][0]=.25
        out=motion_features(fs).reshape(4,12)
        self.assertEqual(out[-1,10],1);self.assertAlmostEqual(out[-1,8],.15)
    def test_rotation_and_scale_jumps_disable_motion(self):
        fs=frames()
        for i,f in enumerate(fs):f['phaseDegrees']=20
        self.assertTrue(np.all(motion_features(fs).reshape(4,12)[:,:5]==0))
    def test_short_or_invalid_history_rejected(self):
        with self.assertRaises(ValueError):motion_features(frames()[:-1])
        fs=frames();fs[5]['valid']=False
        with self.assertRaises(ValueError):motion_features(fs)
    def test_incoming_block_at_timer_radius_is_retained(self):
        image=board([0,1,2,3,0,1]);draw=ImageDraw.Draw(image)
        # Horizontal incoming block on top face, spanning former excluded annulus.
        draw.rectangle((435,159,525,165),fill=PALETTE[2])
        r=extract(image);self.assertTrue(r['valid']);self.assertEqual(r['incomingColors'][4],2)
        self.assertLess(abs(r['vector'][4*4+2]-1),.08)
    def test_geometry_hold_is_bounded_and_requires_current_core_evidence(self):
        image=board([0,1,2,3,0,1]);previous=extract(image)
        occluded=copy.deepcopy(previous)
        occluded['apothem']*=.9;occluded['coreApothemRatio']/=.9
        corrected=extract(image,previous,occluded)
        self.assertTrue(corrected['geometryHeld'])
        self.assertAlmostEqual(corrected['apothem'],previous['apothem'])
        second=extract(image,corrected,occluded)
        third=extract(image,second,occluded)
        self.assertTrue(second['geometryHeld']);self.assertFalse(third['geometryHeld'])
        # No history after a pause, or genuinely changed core scale: no hold.
        self.assertFalse(extract(image,None,occluded)['geometryHeld'])
        occluded['coreApothemRatio']*=1.2
        self.assertFalse(extract(image,previous,occluded)['geometryHeld'])
    def test_future_runs_excluded(self):
        with self.assertRaises(ValueError):extract_run(31)

if __name__=='__main__':unittest.main()
