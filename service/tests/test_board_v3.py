import unittest
import numpy as np
from PIL import ImageDraw
from test_board_v2 import board,PALETTE
from playlens_ml.board_v2.fallback import extract as robust_extract
from playlens_ml.board_v3.features import extract,needs_repair
from playlens_ml.board_v3.data import extract_run


def crowded():
    image=board([0,1,2,3,0,1]);draw=ImageDraw.Draw(image)
    for k,c in ((2,1),(5,2)):
        angle=np.deg2rad(30+k*60);normal=np.array([np.cos(angle),np.sin(angle)]);side=np.array([-normal[1],normal[0]])
        points=[tuple(np.array([480,262])+100*(r*normal+s*r*np.tan(np.pi/6)*side)) for r,s in ((.40,-1),(1.06,-1),(1.06,1),(.40,1))]
        draw.polygon(points,fill=PALETTE[c])
    # Small disconnected gray compression-like patches must not enlarge the board.
    for x,y in ((675,330),(290,180),(650,150)):draw.rectangle((x,y,x+5,y+5),fill=(189,195,199))
    return image


class OcclusionTests(unittest.TestCase):
    def test_crowded_geometry_ignores_small_gray_outliers(self):
        result=robust_extract(crowded());self.assertTrue(result['valid'],result['reason'])
        self.assertLess(np.linalg.norm(np.array(result['center'])-[480,262]),2)
        self.assertLess(abs(result['apothem']-100),3)
    def test_tall_stacks_preserve_outer_color_past_timer_radius(self):
        result=robust_extract(crowded());self.assertEqual(result['topColors'][2],1);self.assertEqual(result['topColors'][5],2)
        self.assertGreater(result['heights'][2],1.0)
    def test_primary_route_is_preserved_for_ordinary_boards(self):
        from playlens_ml.board_v2.features import extract as primary
        image=board([0,1,2,3,0,1]);result=primary(image);self.assertFalse(needs_repair(result));self.assertEqual(extract(image),result)
    def test_future_runs_still_excluded(self):
        with self.assertRaises(ValueError):extract_run(31)

if __name__=='__main__':unittest.main()
