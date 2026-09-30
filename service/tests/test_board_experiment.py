from __future__ import annotations
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from playlens_ml.board_experiment.features import (extract,classify_colors,PALETTE,BASE_NAMES,FEATURE_NAMES,temporal_vector,BoardFrame)
from playlens_ml.board_experiment.data import FrameCache,split_segments
from playlens_ml.board_experiment.metrics import alert_metrics,weighted_calibration,screen
from playlens_ml.board_experiment.evaluate import load_data,run,fit
from playlens_ml.board_experiment import SCHEMA


def fixture(scale=1,pad=0):
    image=Image.new('RGB',(160,160),(236,240,241)); d=ImageDraw.Draw(image)
    points=[(80+30*np.cos(k*np.pi/3),80+30*np.sin(k*np.pi/3)) for k in range(6)]
    d.polygon(points,fill=(189,195,199))
    d.polygon([(80+8*np.cos(k*np.pi/3),80+8*np.sin(k*np.pi/3)) for k in range(6)],fill=(44,62,80))
    if scale!=1:image=image.resize((int(160*scale),int(160*scale)))
    if pad:
        other=Image.new('RGB',(image.width+2*pad,image.height+2*pad),'black');other.paste(image,(pad,pad));image=other
    output=io.BytesIO();image.save(output,format='PNG');return output.getvalue()


def frame():
    return BoardFrame(True,'ok',(80.,80.),26.,1.,np.zeros(len(BASE_NAMES)),[-1]*6,np.zeros((1,1)))


class BoardExperimentTests(unittest.TestCase):
    def test_geometry_normalizes_resize_and_letterbox(self):
        original=extract(fixture());self.assertTrue(original.valid,original.reason)
        for scale,pad in ((2,0),(1,24)):
            transformed=extract(fixture(scale,pad));self.assertTrue(transformed.valid,transformed.reason)
            self.assertLess(np.linalg.norm((np.array(transformed.center)-pad)/scale-original.center),2)
            self.assertLess(abs(transformed.apothem/scale/original.apothem-1),.05)

    def test_palette_separates_four_colors_and_rejects_gray(self):
        values=np.vstack((PALETTE,np.array([[.75,.75,.75],[.1,.1,.1],[.94,.94,.94]])))
        self.assertEqual(classify_colors(values).tolist(),[0,1,2,3,-1,-1,-1])

    def test_background_and_center_do_not_create_stack_colors(self):
        f=extract(fixture());self.assertTrue(f.valid);self.assertEqual(f.top_colors,[-1]*6)

    def test_invalid_image_fails_explicitly(self):
        f=extract(b'bad image');self.assertFalse(f.valid);self.assertEqual(f.reason,'unreadable_image')

    def test_temporal_features_do_not_read_future_frames(self):
        times=np.arange(0,10,.25);vectors=np.zeros((len(times),len(BASE_NAMES)))
        a=temporal_vector(times,vectors,32);vectors[33:]=100
        self.assertTrue(np.array_equal(a,temporal_vector(times,vectors,32)))
        self.assertEqual(len(a),len(FEATURE_NAMES))
        with self.assertRaises(ValueError):temporal_vector(times,vectors,20)

    def test_pause_gap_and_geometry_change_reset_history(self):
        frames=[frame() for _ in range(5)];frames[-1].apothem=40
        times=np.array([0,.25,.5,1.5,1.75]);walls=np.array([100,100.25,105,106,106.25])
        actual=split_segments(times,walls,frames,[.5])
        self.assertEqual(actual.tolist(),[1,1,2,3,4])

    def test_content_cache_extracts_identical_frame_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'image.png';p.write_bytes(fixture());cache=FrameCache(root)
            with patch('playlens_ml.board_experiment.data.extract',wraps=extract) as call:
                a,h=cache.get(p);b,hh=cache.get(p)
                self.assertEqual(call.call_count,1);self.assertEqual(h,hh)
                self.assertTrue(np.array_equal(a.vector,b.vector))
            cache.close()

    def test_always_and_never_warning_controls(self):
        t=np.arange(0,100.);remaining=100-t;segments=np.ones(100)
        always=alert_metrics(np.ones(100),t,remaining,segments)
        self.assertEqual(always['timelyWarningRecall'],0);self.assertEqual(always['falseAlertCount'],1)
        never=alert_metrics(np.zeros(100),t,remaining,segments)
        self.assertEqual(never['timelyWarningRecall'],0);self.assertEqual(never['alertCount'],0)

    def test_alert_requires_three_seconds_and_resets_at_gap(self):
        t=np.array([0,1,2,3,4,5,6,7,8.]);r=40-t
        p=np.array([1,1,0,1,1,1,0,0,0.]);actual=alert_metrics(p,t,r,np.ones(9))
        self.assertEqual(actual['onsetLeadSeconds'],[35.]);self.assertEqual(actual['timelyWarningRecall'],1)
        actual=alert_metrics(np.ones(4),np.array([0,1,3,4.]),np.array([40,39,37,36.]),np.ones(4))
        self.assertEqual(actual['alertCount'],0)
        actual=alert_metrics(np.ones(4),np.arange(4.),40-np.arange(4.),np.array([1,1,2,2]))
        self.assertEqual(actual['alertCount'],0)

    def test_calibration_weights_games_equally(self):
        labels=np.array([[0.],[1.],[1.],[1.]]);p=np.full((4,1),.5)
        self.assertAlmostEqual(weighted_calibration(labels,p,np.array([1,2,2,2]))[0],0.)

    def test_failed_review_prevents_any_model_fitting(self):
        with patch('playlens_ml.board_experiment.evaluate.checkpoint',return_value={'passed':False}),patch('playlens_ml.board_experiment.evaluate.fit') as train:
            with self.assertRaisesRegex(ValueError,'checkpoint failed'):run(Path('/tmp/not-used'))
            train.assert_not_called()

    def test_bad_schema_and_future_runs_are_rejected(self):
        from playlens_ml.board_experiment.features import extractor_hash
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({'extractorHash':extractor_hash(),'sourceDatasetSha256':'hash'}))
            for schema,runs in [('wrong',[1]),(SCHEMA,[31])]:
                np.savez(root/'windows.npz',schema=np.array(schema),feature_names=np.array(FEATURE_NAMES),run_order=np.array(runs),X=np.zeros((1,len(FEATURE_NAMES))))
                with patch('playlens_ml.board_experiment.evaluate.source_data',return_value=({},'hash')):
                    with self.assertRaises(ValueError):load_data(root)

    def test_screen_requires_coverage_and_both_baseline_improvements(self):
        summary={'brier':.08,'mae':8.,'final10Mae':8.,'worstCalibrationError':.1,
                 'timelyWarningRecall':.7,'falseAlertsPerMinute':.1}
        refs={'a':{'summary':{'brier':.1,'mae':10.,'final10Mae':8.}}}
        folds=[(summary,{'a':refs['a']['summary']})]*3
        gates=screen({'summary':summary},refs,folds,[1.]*30);self.assertTrue(all(gates.values()))
        gates=screen({'summary':summary},refs,folds,[1.,.9]);self.assertFalse(gates['coverage'])

    def test_calibrated_models_emit_finite_monotone_probabilities(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(100,4));y=np.array([(np.arange(7)>i%7).astype(int) for i in range(100)])
        ids=np.arange(100);train=ids<60;cal=(ids>=60)&(ids<80);test=ids>=80
        for family in ('logistic','tree'):
            p,_=fit(x,y,train,cal,test,family)
            self.assertTrue(np.isfinite(p).all());self.assertTrue((np.diff(p,axis=1)>=0).all())

    def test_synthetic_full_comparison_keeps_folds_and_outputs_isolated(self):
        from playlens_ml.board_experiment.features import extractor_hash
        from playlens_ml.board_experiment.evaluate import CANDIDATES, BASELINES
        rng=np.random.default_rng(18)
        runs=np.repeat(np.arange(1,31),16)
        elapsed=np.tile(np.arange(16)*8.+10.,30)
        remaining=135.-elapsed
        labels=(remaining[:,None]<=np.array([5,10,15,20,30,45,60])).astype(int)
        x=rng.normal(size=(len(runs),len(FEATURE_NAMES)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            np.savez(root/'windows.npz',schema=np.array(SCHEMA),feature_names=np.array(FEATURE_NAMES),
                     run_order=runs,session_ids=np.array([f'run-{i}' for i in runs]),
                     X=x,elapsed_seconds=elapsed,time_to_failure_seconds=remaining,y_failure=labels,
                     segments=runs,valid=np.ones(len(runs),dtype=bool))
            manifest={'extractorHash':extractor_hash(),'sourceDatasetSha256':'synthetic',
                      'runs':[{'run':i,'coverage':1.,'eligibleWindows':16,'validWindows':16,'exclusions':{}} for i in range(1,31)]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            with patch('playlens_ml.board_experiment.evaluate.source_data',return_value=({},'synthetic')),patch('playlens_ml.board_experiment.evaluate.checkpoint',return_value={'passed':True}):
                result=run(root)
            self.assertEqual(set(result['models']),set(CANDIDATES+BASELINES+('always_warning','never_warning')))
            self.assertEqual(len(result['folds']),3)
            for item in result['models'].values():
                self.assertEqual([r['run'] for r in item['perRun']],list(range(16,31)))
            self.assertTrue((root/'results.json').exists())
            self.assertIsNone(result['selected'])

if __name__=='__main__':unittest.main()
