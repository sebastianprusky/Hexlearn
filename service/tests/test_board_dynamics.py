import unittest
import numpy as np
from playlens_ml.board_dynamics.features import motion_features

class DynamicsTests(unittest.TestCase):
    def test_known_incoming_speed_and_missing_track_indicator(self):
        base=np.zeros((33,54));lanes=base[:,:24].reshape(33,6,4);lanes[:,:,0]=.32;lanes[:,:,2]=2.1
        lanes[:,0,2]=1.9-np.arange(33)*.025
        result=motion_features(base,np.zeros(33)).reshape(4,12)
        self.assertTrue(np.allclose(result[:,:4],.1));self.assertTrue(np.allclose(result[:,4],1/6))
        missing=motion_features(base,np.full(33,20)).reshape(4,12);self.assertTrue(np.all(missing[:,:5]==0))
    def test_rotation_alone_does_not_create_clear_activity(self):
        base=np.zeros((33,54));lanes=base[:,:24].reshape(33,6,4);lanes[:,:,2]=2.1
        for i in range(33):lanes[i,:,0]=np.roll([.4,.5,.6,.7,.8,.9],i%6)
        result=motion_features(base,np.zeros(33)).reshape(4,12);self.assertTrue(np.allclose(result[:,7:],0,atol=1e-12))
    def test_incomplete_window_rejected(self):
        with self.assertRaises(ValueError):motion_features(np.zeros((32,54)),np.zeros(32))

if __name__=='__main__':unittest.main()
