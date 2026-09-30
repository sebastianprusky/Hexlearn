"""Guard the offline audit against future-test leakage and misleading warnings."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from playlens_ml.development_audit import development_data, per_run_metrics, summarize, FOLDS


class DevelopmentAuditTests(unittest.TestCase):
    def test_future_runs_are_removed_before_any_fit_or_score(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.npz'
            data = {k: np.array([1., 2., 999.]) for k in
                    ('X', 'sequences', 'y_failure', 'session_ids', 'elapsed_seconds', 'time_to_failure_seconds')}
            np.savez(path, **data, run_order=np.array([1, 30, 31]))
            actual = development_data(path)
            self.assertEqual(actual['run_order'].tolist(), [1, 30])
            self.assertNotIn(999., actual['X'])

    def test_chronological_folds_do_not_overlap_within_a_fold(self):
        evaluated = []
        for train_end, calibration_end, test_end in FOLDS:
            self.assertLess(train_end, calibration_end)
            self.assertLess(calibration_end, test_end)
            self.assertLessEqual(test_end, 30)
            evaluated.extend(range(calibration_end + 1, test_end + 1))
        self.assertEqual(len(evaluated), len(set(evaluated)))

    def test_always_warning_does_not_earn_useful_first_warning(self):
        remaining = np.array([90., 50., 20., 5.])
        data = {'run_order': np.ones(4), 'time_to_failure_seconds': remaining,
                'elapsed_seconds': 100-remaining, 'y_failure': np.zeros((4,7))}
        mask = np.ones(4, dtype=bool)
        row = per_run_metrics(np.ones((4,7)), data, mask)[0]
        self.assertFalse(row['firstWarningUseful'])
        self.assertEqual(row['falseWarningFraction'], 1.)
        self.assertEqual(row['firstWarningLead'], 90.)
        p = np.ones((4,7)); p[0] = 0
        self.assertTrue(per_run_metrics(p, data, mask)[0]['firstWarningUseful'])

    def test_summary_weights_games_equally(self):
        rows = [{key: value for key in ('brier','mae','final10Mae','falseWarningFraction','firstWarningUseful')}
                for value in (0., 1.)]
        self.assertEqual(summarize(rows)['brier'], .5)

if __name__ == '__main__':
    unittest.main()
