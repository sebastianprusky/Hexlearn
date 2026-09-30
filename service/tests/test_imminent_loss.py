import json
import unittest
from pathlib import Path
from playlens_ml.imminent_loss.gate import timing_gate, require_forecasting_gate


class ImminentTimingGateTests(unittest.TestCase):
    def rows(self):
        return [{'run': n, 'visuallyReviewed': True,
                 'lossIntervalVideoSeconds': [100., 100.25],
                 'alignmentVerified': False} for n in range(1, 31)]

    def test_quarter_second_boundary_and_missing_alignment(self):
        result = timing_gate(self.rows())
        self.assertEqual(result['status'], 'pending_alignment')
        self.assertFalse(result['forecastingAllowed'])

    def test_wider_interval_stops_without_dropping_game(self):
        rows = self.rows(); rows[4]['lossIntervalVideoSeconds'] = [161.426, 161.768]
        result = timing_gate(rows)
        self.assertEqual(result['status'], 'closed_timing_failure')
        self.assertEqual(result['failures'][0]['run'], 5)

    def test_missing_transition_cannot_use_metadata_end(self):
        rows = self.rows(); rows[13]['lossIntervalVideoSeconds'] = None
        rows[13]['metadataActiveDurationSeconds'] = 165.196
        self.assertEqual(timing_gate(rows)['failures'][0]['reason'], 'visible_loss_transition_missing')

    def test_exact_cohort_excludes_future_and_missing_games(self):
        for orders in (list(range(1, 30)), list(range(2, 32)), [1] * 30):
            rows = self.rows()[:len(orders)]
            for row, order in zip(rows, orders): row['run'] = order
            with self.assertRaises(ValueError): timing_gate(rows)

    def test_finite_ordered_reviewed_intervals(self):
        for interval in ([float('nan'), 101], [101, 100], [100, 100]):
            rows = self.rows(); rows[0]['lossIntervalVideoSeconds'] = interval
            self.assertEqual(timing_gate(rows)['status'], 'closed_timing_failure')
        rows = self.rows(); rows[0]['visuallyReviewed'] = False
        self.assertEqual(timing_gate(rows)['status'], 'closed_timing_failure')

    def test_alignment_alone_never_approves_extraction_or_models(self):
        rows = self.rows()
        for row in rows:
            row.update(alignmentVerified=True, alignmentBoundSeconds=.5)
        result = timing_gate(rows)
        self.assertEqual(result['status'], 'timing_passed')
        with self.assertRaises(ValueError): require_forecasting_gate(result)
        rows[0]['alignmentBoundSeconds'] = .501
        self.assertEqual(timing_gate(rows)['status'], 'pending_alignment')

    def test_frozen_review_has_three_failures_and_no_forecasting(self):
        import playlens_ml.imminent_loss as package
        path = Path(package.__file__).with_name('reviewed_endings.json')
        data = json.loads(path.read_text())
        result = timing_gate(data['runs'])
        self.assertEqual([x['run'] for x in result['failures']], [5, 14, 17])
        self.assertEqual(data['schema'], 'imminent-loss-v1')
        with self.assertRaises(ValueError): require_forecasting_gate(result)


if __name__ == '__main__':
    unittest.main()
