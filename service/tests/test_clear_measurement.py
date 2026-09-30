import copy
import unittest
import numpy as np
from PIL import Image,ImageDraw
from playlens_ml.clear_measurement.detector import Detector,observation,RADII


def obs(white=False,colored=True):
    w=np.zeros((6,len(RADII)));c=w.copy()
    if colored:c[2,10:15]=1
    if white:w[2,10:15]=1
    return {'valid':True,'center':[200,200],'apothem':100,'white':w.tolist(),'colored':c.tolist()}


class ClearMeasurementTests(unittest.TestCase):
    def test_clear_requires_prior_colored_support(self):
        d=Detector();self.assertFalse(d.update(obs(True,False),0)['event'])
        self.assertFalse(d.update(obs(True,False),.25)['event'])
        d.reset();d.update(obs(),0)
        self.assertTrue(d.update(obs(True,False),.25)['event'])
    def test_persistent_flash_counted_once(self):
        d=Detector();d.update(obs(),0)
        self.assertTrue(d.update(obs(True),.25)['event'])
        self.assertFalse(d.update(obs(True),.5)['event'])
    def test_rotation_without_flash_never_clears(self):
        d=Detector()
        for i in range(10):
            o=obs();o['colored']=np.roll(o['colored'],i,axis=0).tolist()
            self.assertFalse(d.update(o,i/4)['event'])
    def test_pause_gap_and_geometry_change_reset_history(self):
        for reason in ('pause','gap','geometry','invalid'):
            d=Detector();d.update(obs(),0);o=obs(True,False);t=.25
            if reason=='gap':t=1
            if reason=='geometry':o['apothem']=125
            if reason=='invalid':d.update({'valid':False},.1)
            self.assertFalse(d.update(o,t,reset=reason=='pause')['event'])
    def test_thin_or_narrow_bright_marks_rejected(self):
        for white in (np.eye(6,len(RADII)),np.full((6,len(RADII)),.3)):
            d=Detector();d.update(obs(),0);o=obs(False);o['white']=white.tolist()
            self.assertFalse(d.update(o,.25)['event'])
    def test_central_score_pixels_do_not_enter_observation(self):
        g={'valid':True,'center':[200,200],'apothem':100,'phaseDegrees':0}
        a=Image.new('RGB',(400,400),(189,195,199));b=a.copy();ImageDraw.Draw(b).ellipse((170,170,230,230),fill='white')
        self.assertEqual(observation(a,g),observation(b,g))
    def test_white_persistence_does_not_require_repeated_color_history(self):
        d=Detector(12);d.update(obs(),0)
        flags=[d.update(obs(True,False),(i+1)/12)['event'] for i in range(8)]
        self.assertEqual(sum(flags),1)
    def test_missing_frame_at_twelve_fps_resets(self):
        d=Detector(12);d.update(obs(),0)
        self.assertFalse(d.update(obs(True,False),2/12)['event'])
    def test_resize_and_letterbox_preserve_flash_detection(self):
        geometry={'valid':True,'center':[200,200],'apothem':100,'phaseDegrees':0}
        before=Image.new('RGB',(400,400),(189,195,199));after=before.copy()
        ImageDraw.Draw(before).rectangle((178,140,222,146),fill=(231,76,60))
        ImageDraw.Draw(after).rectangle((178,140,222,146),fill='white')
        for scale,padding in ((1,0),(.8,0),(1,40)):
            ims=[]
            for im in (before,after):
                im=im.resize((round(400*scale),round(400*scale)))
                box=Image.new('RGB',(im.width+padding*2,im.height+padding*2));box.paste(im,(padding,padding));ims.append(box)
            g={**geometry,'center':[200*scale+padding]*2,'apothem':100*scale}
            d=Detector();d.update(observation(ims[0],g),0)
            self.assertTrue(d.update(observation(ims[1],g),.25)['event'])
    def test_future_frames_cannot_change_prefix_output(self):
        sequence=[obs(),obs(True),obs(False),obs(True)]
        d=Detector();prefix=[d.update(o,i/4) for i,o in enumerate(sequence[:2])]
        d=Detector();full=[d.update(o,i/4) for i,o in enumerate(sequence)]
        self.assertEqual(prefix,full[:2])

if __name__=='__main__':unittest.main()

class ClearAccountingTests(unittest.TestCase):
    def test_one_to_one_duplicate_and_early_detection(self):
        from playlens_ml.clear_measurement.evaluate import score_events
        r=score_events([10,11,12,20],[11,24],72,12)
        self.assertEqual(r['matches'],[[11,10]])
        self.assertEqual(r['falseEvents'],[11,12,20]);self.assertEqual(r['missedEvents'],[24])
    def test_censoring_and_ambiguity_are_explicit(self):
        from playlens_ml.clear_measurement.evaluate import score_events
        r=score_events([0,10,70],[0,10,70],72,12,ambiguous=[10])
        self.assertEqual(r['matches'],[]);self.assertEqual(r['falseEvents'],[]);self.assertEqual(r['missedEvents'],[])
        self.assertAlmostEqual(r['reviewedSeconds'],59/12)
    def test_never_event_cannot_pass_precision_or_recall(self):
        from playlens_ml.clear_measurement.evaluate import score_events,summarize
        s=summarize([score_events([],[12,24],72,12)])
        self.assertEqual(s['recall'],0);self.assertEqual(s['precision'],0)
