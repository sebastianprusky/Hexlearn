import unittest
import numpy as np
from playlens_ml.board_stable.features import normalize


def frame(apothem,core_pixels=48):
    scale=150/apothem;lanes=np.ones((6,4));lanes[:,0]=.6*scale;lanes[:,1]=.8*scale;lanes[:,2]=1.4*scale
    return {'valid':True,'reason':'ok','apothem':apothem,'center':[480,262],'coreApothemRatio':core_pixels/apothem,
            'heights':lanes[:,0].tolist(),'vector':lanes.ravel().tolist()+[0.]*30}

class StableScaleTests(unittest.TestCase):
    def test_gray_scale_noise_cancels_when_core_pixels_are_unchanged(self):
        a=normalize(frame(135));b=normalize(frame(150));self.assertAlmostEqual(a['apothem'],b['apothem'])
        self.assertTrue(np.allclose(a['vector'][:24],b['vector'][:24]))
    def test_actual_image_resize_remains_a_geometry_change(self):
        a=normalize(frame(135,48));b=normalize(frame(270,96));self.assertAlmostEqual(b['apothem']/a['apothem'],2)
    def test_absent_incoming_block_stays_absent(self):
        f=frame(140);f['vector'][2]=2.1;result=normalize(f);self.assertEqual(result['vector'][2],2.1)
    def test_unreliable_core_ratio_remains_visible(self):
        f=frame(150);f['coreApothemRatio']=.5;self.assertFalse(normalize(f)['valid'])

if __name__=='__main__':unittest.main()
