import unittest
import numpy as np
from playlens_ml.board_invariant.features import transform,FEATURE_NAMES

class InvariantTests(unittest.TestCase):
    def test_full_cyclic_permutation_including_height_ties(self):
        rng=np.random.default_rng(10);vector=rng.random(150);lanes=vector[:24].reshape(6,4);lanes[:,0]=[.8,.8,.5,.4,.6,.5]
        colors=vector[24:48].reshape(6,4);matches=vector[48:54].copy();expected=transform(vector)
        for shift in range(6):
            other=vector.copy();other[:24]=np.roll(lanes,shift,axis=0).ravel();other[24:48]=np.roll(colors,shift,axis=0).ravel();other[48:54]=np.roll(matches,shift)
            self.assertTrue(np.allclose(transform(other),expected))
    def test_invalid_zero_rows_are_finite_and_schema_sized(self):
        result=transform(np.zeros(150));self.assertEqual(len(result),len(FEATURE_NAMES));self.assertTrue(np.isfinite(result).all())

if __name__=='__main__':unittest.main()
