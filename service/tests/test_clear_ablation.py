import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from playlens_ml.clear_measurement.ablation import temporal,extract_run

class ClearAblationTests(unittest.TestCase):
    def frames(self):
        return [dict(supported=True,event=i==96,confidence=.8,segment=0,videoSeconds=i/12) for i in range(121)]
    def test_future_frames_do_not_affect_window(self):
        frames=self.frames();a=temporal(frames,8.5)
        for f in frames[97:]:f.update(event=True,confidence=1,segment=7,supported=False)
        self.assertEqual(a,temporal(frames,8.5));self.assertEqual(a[0],[1,.8]*4)
    def test_delay_and_eight_second_history(self):
        self.assertEqual(temporal(self.frames(),8.49),(None,'short_history'))
        self.assertIsNone(temporal(self.frames(),8.5)[1])
    def test_gap_and_pause_remain_exclusions(self):
        f=self.frames();f[60]['supported']=False
        self.assertEqual(temporal(f,8.5),(None,'unsupported_geometry'))
        f=self.frames();f[60]['segment']=1
        self.assertEqual(temporal(f,8.5),(None,'reset_within_window'))
    def test_history_outside_window_has_no_effect(self):
        f=self.frames();f[0]['supported']=False
        self.assertIsNone(temporal(f,10.5)[1])
    def test_future_runs_refused_before_read(self):
        for run in (0,31,40):
            with self.assertRaises(ValueError):extract_run(run)
    def test_incompatible_saved_columns_rejected_before_fit(self):
        from playlens_ml.clear_measurement import ablation
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            np.savez(folder/'ablation-windows.npz',schema=np.array('clear-ablation-v1'),clear_feature_names=np.array(ablation.FEATURES),feature_names=np.array(['one']),X=np.zeros((2,2)))
            (folder/'ablation-manifest.json').write_text(json.dumps({'detectorHash':'frozen','featureNames':['one']}))
            with patch.object(ablation,'OUT',folder),patch.object(ablation,'ready'),patch.object(ablation,'fingerprint',return_value='frozen'):
                with self.assertRaisesRegex(ValueError,'columns do not match'):
                    ablation.evaluate()
