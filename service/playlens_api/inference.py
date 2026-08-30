from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, aggregate_window


HORIZONS_SECONDS = (5, 10, 15, 20, 30, 45, 60)
LOSS_WINDOW_LABELS = {
    "none": "NO FAILURE SIGNAL",
    "45_to_60s": "LOSS LIKELY IN 45-60s",
    "30_to_45s": "LOSS LIKELY IN 30-45s",
    "20_to_30s": "LOSS LIKELY IN 20-30s",
    "10_to_20s": "LOSS LIKELY IN 10-20s",
    "under_10s": "LOSS IMMINENT - <10s",
}


def monotonic_probabilities(values: list[float] | np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if probabilities.shape != (len(HORIZONS_SECONDS),):
        raise ValueError("PlayLens expects seven cumulative failure probabilities")
    return np.maximum.accumulate(probabilities)


def loss_window(values: list[float] | np.ndarray, threshold: float = 0.5) -> str:
    probabilities = monotonic_probabilities(values)
    crossings = np.flatnonzero(probabilities >= threshold)
    if not len(crossings):
        return "none"
    horizon = HORIZONS_SECONDS[int(crossings[0])]
    if horizon <= 10:
        return "under_10s"
    if horizon <= 20:
        return "10_to_20s"
    if horizon <= 30:
        return "20_to_30s"
    if horizon <= 45:
        return "30_to_45s"
    return "45_to_60s"


def estimated_remaining_seconds(values: list[float] | np.ndarray) -> float:
    probabilities = monotonic_probabilities(values)
    times = np.asarray((0, *HORIZONS_SECONDS), dtype=np.float64)
    survival = np.asarray((1.0, *(1.0 - probabilities)), dtype=np.float64)
    return float(np.trapezoid(survival, times))


@dataclass
class RuntimeSession:
    started_timestamp_ms: int
    collection_run_number: int
    active_elapsed_ms: int = 0
    frames: deque[tuple[int, np.ndarray]] = field(default_factory=lambda: deque(maxlen=36))
    previous_gray: np.ndarray | None = None
    probability_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=3))
    last_prediction_active_ms: int = 0
    last_saved_active_ms: int = -250
    last_prediction: dict[str, Any] | None = None
    valid_frames: int = 0
    paused: bool = False

    def suspend(self, active_elapsed_ms: int) -> None:
        self.active_elapsed_ms = max(self.active_elapsed_ms, active_elapsed_ms)
        self.paused = True

    def resume(self, active_elapsed_ms: int) -> None:
        self.active_elapsed_ms = max(self.active_elapsed_ms, active_elapsed_ms)
        self.paused = False
        self.frames.clear()
        self.probability_history.clear()
        self.previous_gray = None
        self.last_prediction = None
        self.last_prediction_active_ms = active_elapsed_ms


class Predictor:
    """Loads only a promoted personal survival model; otherwise capture stays collection-only."""

    def __init__(self, artifact_path: Path, _personalization_path: Path | None = None):
        self.artifact_path = artifact_path
        self.artifact: dict[str, Any] | None = None
        self.torch_models: list[Any] = []
        self.reload()

    def reload(self) -> None:
        self.artifact = None
        self.torch_models = []
        if not self.artifact_path.exists():
            return
        candidate = joblib.load(self.artifact_path)
        if candidate.get("eligible_for_live") is not True:
            return
        if tuple(candidate.get("horizons_seconds", ())) != HORIZONS_SECONDS:
            return
        if candidate.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            return
        if int(candidate.get("feature_count", -1)) != len(FEATURE_NAMES):
            return
        kind = candidate.get("kind")
        if kind == "torchscript_personal_survival":
            try:
                import torch

                for path in candidate.get("torchscript_paths", []):
                    model = torch.jit.load(path)
                    model.eval()
                    self.torch_models.append(model)
            except (ImportError, OSError, RuntimeError):
                self.torch_models = []
                return
        elif kind != "sklearn_personal_survival":
            return
        self.artifact = candidate

    @property
    def model_source(self) -> str:
        return "validated_personal_model" if self.artifact is not None else "collection_only"

    @property
    def calibrated(self) -> bool:
        return bool(self.artifact and self.artifact.get("calibrated"))

    @property
    def model_version(self) -> str:
        return str(self.artifact.get("version", "v3")) if self.artifact else "collection-v3"

    def _member_probabilities(self, vectors: list[np.ndarray]) -> np.ndarray:
        if self.artifact is None:
            raise RuntimeError("No promoted personal survival model is loaded")
        if self.artifact.get("kind") == "torchscript_personal_survival":
            import torch

            indices = np.linspace(0, len(vectors) - 1, 32).round().astype(int)
            sequence = np.stack([vectors[index] for index in indices])[None, :, :]
            normalized = (sequence - self.artifact["feature_mean"]) / self.artifact["feature_std"]
            probabilities = []
            with torch.no_grad():
                for member_index, model in enumerate(self.torch_models):
                    logits = model(torch.from_numpy(normalized.astype(np.float32)))
                    raw_logits = logits[0].numpy()
                    member_calibrators = self.artifact.get("calibrators", [])[member_index]
                    probabilities.append(
                        [
                            calibrator.predict_proba([[float(raw_logits[index])]])[0, 1]
                            for index, calibrator in enumerate(member_calibrators)
                        ]
                    )
            return np.asarray([monotonic_probabilities(values) for values in probabilities])

        aggregate = aggregate_window(vectors).reshape(1, -1)
        members = []
        for models in self.artifact["ensemble_models"]:
            members.append(monotonic_probabilities([model.predict_proba(aggregate)[0, 1] for model in models]))
        return np.vstack(members)

    def prediction(self, runtime: RuntimeSession, timestamp_ms: int, active_elapsed_ms: int) -> dict[str, Any]:
        runtime.active_elapsed_ms = max(runtime.active_elapsed_ms, active_elapsed_ms)
        if self.artifact is None:
            return {
                "timestampMs": timestamp_ms,
                "activeElapsedMs": active_elapsed_ms,
                "status": "collecting",
                "collectionRunNumber": runtime.collection_run_number,
                "collectionTarget": 50,
                "modelSource": self.model_source,
                "calibrated": False,
            }
        if runtime.paused:
            return {
                "timestampMs": timestamp_ms,
                "activeElapsedMs": active_elapsed_ms,
                "status": "paused",
                "modelSource": self.model_source,
                "calibrated": self.calibrated,
            }
        coverage_ms = runtime.frames[-1][0] - runtime.frames[0][0] if len(runtime.frames) > 1 else 0
        if coverage_ms < 7_750 or len(runtime.frames) < 28:
            return {
                "timestampMs": timestamp_ms,
                "activeElapsedMs": active_elapsed_ms,
                "status": "warming_up",
                "modelSource": self.model_source,
                "calibrated": self.calibrated,
            }
        if runtime.last_prediction and active_elapsed_ms - runtime.last_prediction_active_ms < 900:
            return runtime.last_prediction

        start = time.perf_counter()
        members = self._member_probabilities([vector for _, vector in runtime.frames])
        raw_probabilities = monotonic_probabilities(np.median(members, axis=0))
        runtime.probability_history.append(raw_probabilities)
        probabilities = monotonic_probabilities(np.mean(np.vstack(runtime.probability_history), axis=0))
        member_remaining = np.asarray([estimated_remaining_seconds(member) for member in members])
        disagreement = float(np.mean(np.std(members, axis=0)))
        thresholds = self.artifact.get("certainty_thresholds", {"high": 0.04, "medium": 0.10})
        enabled = self.artifact.get("certainty_enabled", {"high": True, "medium": True})
        if enabled.get("high") and disagreement <= float(thresholds.get("high", 0.04)):
            certainty = "high"
        elif enabled.get("medium") and disagreement <= float(thresholds.get("medium", 0.10)):
            certainty = "medium"
        else:
            certainty = "low"
        window = loss_window(probabilities)
        prediction = {
            "timestampMs": timestamp_ms,
            "activeElapsedMs": active_elapsed_ms,
            "status": "ready",
            "failureProbabilities": {
                str(horizon): round(float(probabilities[index]), 5)
                for index, horizon in enumerate(HORIZONS_SECONDS)
            },
            "lossWindow": window,
            "certainty": certainty,
            "estimatedRemainingSeconds": round(estimated_remaining_seconds(probabilities), 2),
            "uncertaintyLowSeconds": round(float(np.quantile(member_remaining, 0.1)), 2),
            "uncertaintyHighSeconds": round(float(np.quantile(member_remaining, 0.9)), 2),
            "displayedValue": LOSS_WINDOW_LABELS[window],
            "modelSource": self.model_source,
            "modelVersion": self.model_version,
            "calibrated": self.calibrated,
            "latencyMs": round((time.perf_counter() - start) * 1_000, 2),
        }
        runtime.last_prediction_active_ms = active_elapsed_ms
        runtime.last_prediction = prediction
        return prediction
