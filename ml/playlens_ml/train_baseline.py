from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from playlens_api.model_types import CalibratedFailureModel, ConstantProbabilityModel, stable_linear_scores
from playlens_api.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

from .common import (
    DATASET_DIR,
    HORIZONS_SECONDS,
    MODEL_DIR,
    eligible,
    evaluate_survival,
    monotonic_probabilities,
    promotion_gates,
    remaining_from_probabilities,
    session_split,
    write_metrics,
)


def classifier() -> object:
    return make_pipeline(
        VarianceThreshold(threshold=1e-10),
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced", solver="liblinear"),
    )


def fit_calibrated(values: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, validation: np.ndarray):
    if len(np.unique(labels[train_indices])) < 2:
        return ConstantProbabilityModel(float(labels[train_indices].mean()))
    base = classifier()
    base.fit(values[train_indices], labels[train_indices])
    if len(np.unique(labels[validation])) < 2:
        return base
    calibrator = LogisticRegression(max_iter=1_000, solver="liblinear")
    calibrator.fit(stable_linear_scores(base, values[validation]).reshape(-1, 1), labels[validation])
    return CalibratedFailureModel(base, calibrator)


def fit_reference(values: np.ndarray, labels: np.ndarray, train: np.ndarray):
    indices = np.flatnonzero(train) if train.dtype == bool else np.asarray(train)
    if len(np.unique(labels[indices])) < 2:
        return ConstantProbabilityModel(float(labels[indices].mean()))
    model = classifier()
    model.fit(values[indices], labels[indices])
    return model


def bootstrap_indices(session_ids: np.ndarray, train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_runs = list(dict.fromkeys(session_ids[train].tolist()))
    sampled = rng.choice(train_runs, size=len(train_runs), replace=True)
    return np.concatenate([np.flatnonzero(session_ids == session_id) for session_id in sampled])


def average_duration_probabilities(
    elapsed_seconds: np.ndarray,
    time_to_failure_seconds: np.ndarray,
    train: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    duration_by_run = {}
    for session_id in set(target.tolist()):
        mask = (target == session_id) & train
        if mask.any():
            duration_by_run[session_id] = float(np.max(elapsed_seconds[mask] + time_to_failure_seconds[mask]))
    average_duration = float(np.mean(list(duration_by_run.values()))) if duration_by_run else 60.0
    remaining = average_duration - elapsed_seconds
    return monotonic_probabilities(
        np.column_stack(
            [1.0 / (1.0 + np.exp(np.clip((remaining - horizon) / 5.0, -50, 50))) for horizon in HORIZONS_SECONDS]
        )
    )


def certainty_contract(member_probabilities: np.ndarray, truth: np.ndarray) -> tuple[dict, dict]:
    disagreement = np.mean(np.std(member_probabilities, axis=0), axis=1)
    median = np.median(member_probabilities, axis=0)
    errors = np.abs(remaining_from_probabilities(median) - np.minimum(truth, 60.0))
    high_threshold = float(np.quantile(disagreement, 0.33))
    medium_threshold = float(np.quantile(disagreement, 0.67))
    high = disagreement <= high_threshold
    medium = (disagreement > high_threshold) & (disagreement <= medium_threshold)
    high_mae = float(np.mean(errors[high])) if high.any() else float("inf")
    medium_mae = float(np.mean(errors[medium])) if medium.any() else float("inf")
    high_enabled = high_mae <= 7.0
    medium_enabled = medium_mae <= 15.0 and (not high_enabled or high_mae < medium_mae)
    return (
        {"high": high_threshold, "medium": medium_threshold},
        {"high": high_enabled, "medium": medium_enabled, "highMae": high_mae, "mediumMae": medium_mae},
    )


def main() -> None:
    dataset = np.load(DATASET_DIR / "personal_windows.npz")
    features = dataset["X"].astype(np.float64)
    occupancy = dataset["occupancy_X"].astype(np.float64)
    elapsed = dataset["elapsed_seconds"].astype(np.float64)
    labels = dataset["y_failure"].astype(np.int64)
    session_ids = dataset["session_ids"]
    time_to_failure = dataset["time_to_failure_seconds"].astype(np.float64)
    split = session_split(session_ids)
    total_runs = len(set(session_ids.tolist()))

    ensemble_models = []
    test_members = []
    validation_members = []
    for member in range(5):
        train_indices = bootstrap_indices(session_ids, split.train, 41 + member)
        models = []
        test_probabilities = []
        validation_probabilities = []
        for index, _horizon in enumerate(HORIZONS_SECONDS):
            model = fit_calibrated(features, labels[:, index], train_indices, split.validation)
            models.append(model)
            test_probabilities.append(model.predict_proba(features[split.test])[:, 1])
            validation_probabilities.append(model.predict_proba(features[split.validation])[:, 1])
        ensemble_models.append(models)
        test_members.append(monotonic_probabilities(np.column_stack(test_probabilities)))
        validation_members.append(monotonic_probabilities(np.column_stack(validation_probabilities)))

    probabilities = monotonic_probabilities(np.median(np.asarray(test_members), axis=0))
    elapsed_values = elapsed.reshape(-1, 1)
    elapsed_probabilities = []
    occupancy_probabilities = []
    constant_probabilities = []
    for index, _horizon in enumerate(HORIZONS_SECONDS):
        target = labels[:, index]
        elapsed_model = fit_reference(elapsed_values, target, split.train)
        occupancy_model = fit_reference(occupancy, target, split.train)
        elapsed_probabilities.append(elapsed_model.predict_proba(elapsed_values[split.test])[:, 1])
        occupancy_probabilities.append(occupancy_model.predict_proba(occupancy[split.test])[:, 1])
        constant_probabilities.append(np.full(int(split.test.sum()), float(target[split.train].mean())))
    average_duration = average_duration_probabilities(elapsed, time_to_failure, split.train, session_ids)
    references = {
        "constant_rate": np.column_stack(constant_probabilities),
        "elapsed_time_only": np.column_stack(elapsed_probabilities),
        "average_duration": average_duration[split.test],
        "board_occupancy_only": np.column_stack(occupancy_probabilities),
    }

    latency_samples = []
    sample = features[split.test][:1]
    for _ in range(30):
        started = time.perf_counter()
        for models in ensemble_models:
            monotonic_probabilities(
                np.asarray([[model.predict_proba(sample)[0, 1] for model in models]])
            )
        latency_samples.append((time.perf_counter() - started) * 1_000)
    latency_p95 = float(np.quantile(latency_samples, 0.95))
    metrics = evaluate_survival(
        labels[split.test],
        probabilities,
        references,
        session_ids[split.test],
        time_to_failure[split.test],
        latency_p95,
    )
    thresholds, certainty = certainty_contract(
        np.asarray(validation_members), time_to_failure[split.validation]
    )
    gates = promotion_gates(metrics, total_runs)
    is_eligible = eligible(metrics, total_runs)
    split_label = "32 development train / 8 development calibration / 10 locked future test" if total_runs >= 50 else "provisional chronological run split"
    payload = {
        "model": "scikit-learn personal visual survival ensemble",
        "split": split_label,
        "objective": "failure within 5/10/15/20/30/45/60 active seconds",
        "references": list(references),
        "metrics": metrics,
        "gates": gates,
        "certainty": certainty,
        "eligibleForLive": is_eligible,
    }
    write_metrics("baseline_v3_metrics", payload)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind": "sklearn_personal_survival",
        "version": f"personal-v3-r{total_runs}",
        "ensemble_models": ensemble_models,
        "horizons_seconds": HORIZONS_SECONDS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "certainty_thresholds": thresholds,
        "certainty_enabled": certainty,
        "calibrated": float(metrics["worstCalibrationError"]) <= 0.12,
        "eligible_for_live": is_eligible,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_DIR / "baseline_v3_candidate.joblib")
    if is_eligible:
        joblib.dump(artifact, MODEL_DIR / "live_model.joblib")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
