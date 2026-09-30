import copy
import unittest
from playlens_ml.imminent_loss_27 import RUNS, FOLDS
from playlens_ml.imminent_loss_27.checkpoint import decide, ISSUES


class ImminentSubsetTests(unittest.TestCase):
    def audit(self):
        checks = [{'run': n, 'passed': True, 'searchEdge': False, 'offsetSeconds': 0}
                  for n in RUNS for _ in range(2)]
        checks += [dict(checks[0]) for _ in range(10)]
        return {'frames': [{'id': f'run-{n:02}-lead-{lead}'}
                           for n in RUNS for lead in (10,5,2)],
                'provenance': [{'run': n} for n in RUNS], 'alignment': checks}

    def test_fixed_subset_and_fold_boundaries(self):
        self.assertEqual(len(RUNS), 27)
        self.assertEqual(set(range(1,31))-set(RUNS), {5,14,17})
        counts = []
        for fit,cal,end in FOLDS:
            groups = [set(n for n in RUNS if n<=fit),
                      set(n for n in RUNS if fit<n<=cal),
                      set(n for n in RUNS if cal<n<=end)]
            self.assertFalse(groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
            self.assertLess(max(groups[0]),min(groups[1]))
            self.assertLess(max(groups[1]),min(groups[2]))
            counts.append(tuple(map(len,groups)))
        self.assertEqual(counts, [(8,5,4),(12,5,5),(16,6,5)])

    def test_uncertain_frames_cannot_count_as_correct(self):
        result = decide(self.audit(), ISSUES)
        self.assertEqual(result['documentedIncorrect'],4)
        self.assertEqual(result['documentedUncertain'],6)
        self.assertAlmostEqual(result['correctFractionUpperBound'],71/81)
        self.assertEqual(result['status'],'closed_extraction_failure')
        self.assertFalse(result['forecastingAllowed'])

    def test_missing_or_duplicate_frame_cannot_improve_denominator(self):
        audit = self.audit(); audit['frames'].pop()
        with self.assertRaises(ValueError): decide(audit,ISSUES)
        audit = self.audit(); audit['frames'][0]=audit['frames'][1]
        with self.assertRaises(ValueError): decide(audit,ISSUES)

    def test_future_and_excluded_runs_refused(self):
        for n in (5,14,17,31,40):
            audit = self.audit(); audit['provenance'][0]['run']=n
            with self.assertRaises(ValueError): decide(audit,ISSUES)

    def test_alignment_flag_cannot_override_margin(self):
        audit = self.audit(); audit['alignment'][0]['offsetSeconds']=.351
        self.assertFalse(decide(audit,ISSUES)['alignmentPassed'])
        audit['alignment'][0]['offsetSeconds']=.35
        self.assertTrue(decide(audit,ISSUES)['alignmentPassed'])
        audit['alignment'][0]['searchEdge']=True
        self.assertFalse(decide(audit,ISSUES)['alignmentPassed'])

    def test_incomplete_alignment_refused(self):
        audit = self.audit();audit['alignment'].pop()
        with self.assertRaises(ValueError):decide(audit,ISSUES)

    def test_report_rejects_changed_visual_evidence(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from playlens_ml.imminent_loss_27 import checkpoint
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'audit.json').write_text('{}')
            (folder/'protocol.json').write_text('{}')
            with patch.object(checkpoint, 'OUT', folder):
                with self.assertRaisesRegex(ValueError, 'stale'):
                    checkpoint.main()

    def test_unknown_review_id_and_unreviewed_frames_never_pass(self):
        issues=copy.deepcopy(ISSUES);issues['run-31-lead-2']=('incorrect','future run')
        with self.assertRaises(ValueError):decide(self.audit(),issues)
        # Removing failures is not equivalent to certifying the other frames.
        self.assertFalse(decide(self.audit(),{})['forecastingAllowed'])


if __name__=='__main__':unittest.main()
