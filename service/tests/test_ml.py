from __future__ import annotations

import unittest

import numpy as np

from playlens_ml.build_dataset import failure_targets
from playlens_ml.common import (
    HORIZONS_SECONDS,
    eligible,
    evaluate_survival,
    monotonic_probabilities,
    promotion_gates,
    session_split,
)


class MlContractTests(unittest.TestCase):
    def test_confirmed_failure_generates_all_future_horizon_labels(self) -> None:
        self.assertEqual(failure_targets(0), [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(failure_targets(17), [0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(failure_targets(75), [0, 0, 0, 0, 0, 0, 0])

    def test_fifty_run_split_locks_last_ten_complete_runs(self) -> None:
        session_ids = np.asarray([f"session-{index:02d}" for index in range(50) for _ in range(4)])
        split = session_split(session_ids)
        self.assertFalse(np.any(split.train & split.validation))
        self.assertFalse(np.any(split.train & split.test))
        self.assertFalse(np.any(split.validation & split.test))
        self.assertEqual(len(set(session_ids[split.train])), 32)
        self.assertEqual(len(set(session_ids[split.validation])), 8)
        self.assertEqual(len(set(session_ids[split.test])), 10)
        self.assertEqual(set(session_ids[split.test]), {f"session-{index:02d}" for index in range(40, 50)})

    def test_seven_horizon_probabilities_are_monotonic(self) -> None:
        values = monotonic_probabilities(
            np.asarray([[0.4, 0.2, 0.5, 0.45, 0.8, 0.7, 0.9]])
        )
        self.assertEqual(values.shape[1], len(HORIZONS_SECONDS))
        self.assertTrue(np.all(values[:, :-1] <= values[:, 1:]))

    def test_live_gate_requires_locked_test_and_all_shortcut_improvements(self) -> None:
        metrics = {
            "evaluatedRuns": 10,
            "brierImprovements": {"elapsed_time_only": 0.2, "average_duration": 0.15},
            "timeToFailureImprovements": {"elapsed_time_only": 0.2, "average_duration": 0.12},
            "usefulWarningRateAt15s": 0.7,
            "worstCalibrationError": 0.08,
            "final10Mae": 4.0,
            "elapsedFinal10Mae": 4.0,
            "latencyMsP95": 40.0,
        }
        self.assertTrue(eligible(metrics, 50))
        self.assertFalse(eligible(metrics, 49))
        metrics["brierImprovements"]["elapsed_time_only"] = 0.05
        self.assertFalse(promotion_gates(metrics, 50)["brierVsElapsed"])

    def test_survival_evaluation_compares_time_and_probability_baselines(self) -> None:
        remaining = np.asarray([70.0, 50.0, 25.0, 15.0, 8.0, 3.0])
        labels = np.asarray([[int(value <= horizon) for horizon in HORIZONS_SECONDS] for value in remaining])
        probabilities = np.clip(labels * 0.9 + (1 - labels) * 0.1, 0, 1)
        references = {
            "elapsed_time_only": np.full_like(probabilities, 0.5, dtype=np.float64),
            "average_duration": np.full_like(probabilities, 0.45, dtype=np.float64),
        }
        metrics = evaluate_survival(
            labels,
            probabilities,
            references,
            np.asarray(["run-a"] * 3 + ["run-b"] * 3),
            remaining,
            12.0,
        )
        self.assertIn("60", metrics["horizons"])
        self.assertGreater(metrics["brierImprovements"]["elapsed_time_only"], 0)
        self.assertLess(metrics["timeToFailureMae"], 20)


if __name__ == "__main__":
    unittest.main()
