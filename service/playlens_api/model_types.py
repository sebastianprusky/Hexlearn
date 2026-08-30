from __future__ import annotations

import numpy as np


class ConstantProbabilityModel:
    """Joblib-safe fallback when a run-level split contains one target class."""

    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = np.full(len(values), self.probability, dtype=np.float64)
        return np.column_stack((1 - positive, positive))


def stable_linear_scores(base_model, values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(base_model[:-1].transform(values), dtype=np.float64)
    linear_model = base_model.steps[-1][1]
    coefficients = np.asarray(linear_model.coef_[0], dtype=np.float64)
    return np.sum(transformed * coefficients, axis=1) + float(linear_model.intercept_[0])


class CalibratedFailureModel:
    """Small joblib-safe adapter around a classifier and held-out sigmoid fit."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scores = stable_linear_scores(self.base_model, values).reshape(-1, 1)
        positive = self.calibrator.predict_proba(scores)[:, 1]
        return np.column_stack((1 - positive, positive))
