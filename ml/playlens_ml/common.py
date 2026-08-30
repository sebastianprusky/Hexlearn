from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = Path(os.environ.get("PLAYLENS_ARTIFACTS_DIR", ROOT / "artifacts"))
DATASET_DIR = ARTIFACTS_ROOT / "dataset"
MODEL_DIR = ARTIFACTS_ROOT / "models"
EVALUATION_DIR = ARTIFACTS_ROOT / "evaluation"
HORIZONS_SECONDS = (5, 10, 15, 20, 30, 45, 60)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def session_split(session_ids: np.ndarray) -> Split:
    unique = list(dict.fromkeys(session_ids.tolist()))
    if len(unique) < 7:
        raise ValueError("At least seven complete sessions are required for leakage-safe splits")
    if len(unique) >= 50:
        train_sessions = set(unique[:32])
        validation_sessions = set(unique[32:40])
        test_sessions = set(unique[40:50])
    else:
        train_end = max(1, int(len(unique) * 0.70))
        validation_end = min(max(train_end + 1, int(len(unique) * 0.85)), len(unique) - 1)
        train_sessions = set(unique[:train_end])
        validation_sessions = set(unique[train_end:validation_end])
        test_sessions = set(unique[validation_end:])
    return Split(
        train=np.asarray([value in train_sessions for value in session_ids]),
        validation=np.asarray([value in validation_sessions for value in session_ids]),
        test=np.asarray([value in test_sessions for value in session_ids]),
    )


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result


def monotonic_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(HORIZONS_SECONDS):
        raise ValueError("Expected one cumulative probability per failure horizon")
    return np.maximum.accumulate(np.clip(values, 0.0, 1.0), axis=1)


def remaining_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = monotonic_probabilities(probabilities)
    times = np.asarray((0, *HORIZONS_SECONDS), dtype=np.float64)
    survival = np.column_stack((np.ones(len(values)), 1.0 - values))
    return np.trapezoid(survival, times, axis=1)


def window_indices(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values = monotonic_probabilities(probabilities)
    crossed = values >= threshold
    first = np.argmax(crossed, axis=1)
    return np.where(crossed.any(axis=1), first, len(HORIZONS_SECONDS))


def evaluate_survival(
    labels: np.ndarray,
    probabilities: np.ndarray,
    reference_probabilities: dict[str, np.ndarray],
    session_ids: np.ndarray,
    time_to_failure_seconds: np.ndarray,
    latency_ms_p95: float,
) -> dict[str, object]:
    labels = np.asarray(labels)
    probabilities = monotonic_probabilities(probabilities)
    references = {name: monotonic_probabilities(values) for name, values in reference_probabilities.items()}
    per_horizon: dict[str, dict[str, float | str]] = {}
    briers = []
    strongest_improvements = []
    for index, horizon in enumerate(HORIZONS_SECONDS):
        target = labels[:, index]
        predicted = probabilities[:, index]
        brier = float(brier_score_loss(target, predicted))
        baseline_briers = {
            name: float(brier_score_loss(target, values[:, index])) for name, values in references.items()
        }
        strongest_name, strongest_brier = min(baseline_briers.items(), key=lambda item: item[1])
        improvement = (strongest_brier - brier) / strongest_brier if strongest_brier else 0.0
        per_horizon[str(horizon)] = {
            "auroc": float(roc_auc_score(target, predicted)) if len(np.unique(target)) > 1 else 0.5,
            "brier": brier,
            "ece": expected_calibration_error(target, predicted),
            "strongestBaseline": strongest_name,
            "strongestBaselineBrier": strongest_brier,
            "brierImprovement": improvement,
        }
        briers.append(brier)
        strongest_improvements.append(improvement)

    mean_brier = float(np.mean(briers))
    baseline_mean_briers = {
        name: float(np.mean([brier_score_loss(labels[:, index], values[:, index]) for index in range(len(HORIZONS_SECONDS))]))
        for name, values in references.items()
    }
    brier_improvements = {
        name: (value - mean_brier) / value if value else 0.0 for name, value in baseline_mean_briers.items()
    }
    predicted_remaining = remaining_from_probabilities(probabilities)
    true_capped = np.minimum(np.asarray(time_to_failure_seconds, dtype=np.float64), 60.0)
    evaluation_mask = np.asarray(time_to_failure_seconds) <= 60.0
    time_mae = float(np.mean(np.abs(predicted_remaining[evaluation_mask] - true_capped[evaluation_mask]))) if evaluation_mask.any() else 60.0
    baseline_time_mae = {
        name: float(np.mean(np.abs(remaining_from_probabilities(values)[evaluation_mask] - true_capped[evaluation_mask])))
        if evaluation_mask.any() else 60.0
        for name, values in references.items()
    }
    time_improvements = {
        name: (value - time_mae) / value if value else 0.0 for name, value in baseline_time_mae.items()
    }
    final_10 = np.asarray(time_to_failure_seconds) <= 10.0
    final_30 = np.asarray(time_to_failure_seconds) <= 30.0
    final_10_mae = float(np.mean(np.abs(predicted_remaining[final_10] - true_capped[final_10]))) if final_10.any() else 60.0
    final_30_mae = float(np.mean(np.abs(predicted_remaining[final_30] - true_capped[final_30]))) if final_30.any() else 60.0
    elapsed_final_10 = (
        float(np.mean(np.abs(remaining_from_probabilities(references["elapsed_time_only"])[final_10] - true_capped[final_10])))
        if final_10.any() and "elapsed_time_only" in references else 60.0
    )

    horizon_windows = np.asarray([0, 0, 1, 1, 2, 3, 4, 5])
    predicted_windows = horizon_windows[window_indices(probabilities)]
    remaining_values = np.asarray(time_to_failure_seconds)
    actual_windows = np.select(
        [remaining_values <= 10, remaining_values <= 20, remaining_values <= 30, remaining_values <= 45, remaining_values <= 60],
        [0, 1, 2, 3, 4],
        default=5,
    )
    window_accuracy = float(np.mean(predicted_windows == actual_windows))
    warning_leads = []
    false_early = 0
    for session_id in list(dict.fromkeys(session_ids.tolist())):
        indices = np.flatnonzero(session_ids == session_id)
        ordered = indices[np.argsort(-np.asarray(time_to_failure_seconds)[indices])]
        signaled = ordered[predicted_windows[ordered] < 5]
        false_early += int(np.sum(np.asarray(time_to_failure_seconds)[signaled] > 60.0))
        valid = signaled[np.asarray(time_to_failure_seconds)[signaled] <= 60.0]
        warning_leads.append(float(np.asarray(time_to_failure_seconds)[valid[0]]) if len(valid) else 0.0)
    useful_warning_rate = float(np.mean(np.asarray(warning_leads) >= 15.0)) if warning_leads else 0.0

    return {
        "horizons": per_horizon,
        "meanBrier": mean_brier,
        "meanBrierImprovement": float(np.mean(strongest_improvements)),
        "brierImprovements": brier_improvements,
        "worstCalibrationError": float(max(float(item["ece"]) for item in per_horizon.values())),
        "timeToFailureMae": time_mae,
        "baselineTimeToFailureMae": baseline_time_mae,
        "timeToFailureImprovements": time_improvements,
        "windowAccuracy": window_accuracy,
        "usefulWarningRateAt15s": useful_warning_rate,
        "medianWarningLeadSeconds": float(np.median(warning_leads)) if warning_leads else 0.0,
        "falseEarlyWarnings": false_early,
        "final10Mae": final_10_mae,
        "final30Mae": final_30_mae,
        "elapsedFinal10Mae": elapsed_final_10,
        "evaluatedRuns": len(set(session_ids.tolist())),
        "latencyMsP95": latency_ms_p95,
    }


def promotion_gates(metrics: dict[str, object], total_runs: int) -> dict[str, bool]:
    brier = metrics.get("brierImprovements", {})
    time_improvements = metrics.get("timeToFailureImprovements", {})
    return {
        "lockedTestComplete": total_runs >= 50 and int(metrics.get("evaluatedRuns", 0)) >= 10,
        "brierVsElapsed": float(brier.get("elapsed_time_only", 0.0)) >= 0.10,
        "brierVsAverageDuration": float(brier.get("average_duration", 0.0)) >= 0.10,
        "timeMaeVsElapsed": float(time_improvements.get("elapsed_time_only", 0.0)) >= 0.10,
        "timeMaeVsAverageDuration": float(time_improvements.get("average_duration", 0.0)) >= 0.10,
        "usefulWarning": float(metrics.get("usefulWarningRateAt15s", 0.0)) >= 0.50,
        "calibration": float(metrics.get("worstCalibrationError", 1.0)) <= 0.12,
        "finalTenSeconds": float(metrics.get("final10Mae", 60.0)) <= 1.10 * float(metrics.get("elapsedFinal10Mae", 60.0)),
        "latency": float(metrics.get("latencyMsP95", float("inf"))) < 500.0,
    }


def eligible(metrics: dict[str, object], total_runs: int = 50) -> bool:
    return all(promotion_gates(metrics, total_runs).values())


def write_metrics(name: str, payload: dict) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
