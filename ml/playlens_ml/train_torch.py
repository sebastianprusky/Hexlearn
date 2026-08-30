from __future__ import annotations

import json
import time

import joblib
import numpy as np

from playlens_api.model_types import ConstantProbabilityModel
from playlens_api.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

from .common import (
    DATASET_DIR,
    HORIZONS_SECONDS,
    MODEL_DIR,
    eligible,
    evaluate_survival,
    monotonic_probabilities,
    promotion_gates,
    session_split,
    write_metrics,
)
from .train_baseline import (
    average_duration_probabilities,
    bootstrap_indices,
    certainty_contract,
    fit_reference,
)


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Install ml/requirements-ml.txt to train the PyTorch model") from exc
    from sklearn.linear_model import LogisticRegression

    dataset = np.load(DATASET_DIR / "personal_windows.npz")
    sequences = dataset["sequences"].astype(np.float32)
    labels = dataset["y_failure"].astype(np.float32)
    session_ids = dataset["session_ids"]
    split = session_split(session_ids)
    total_runs = len(set(session_ids.tolist()))
    feature_mean = sequences[split.train].mean(axis=(0, 1), keepdims=True)
    feature_std = sequences[split.train].std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (sequences - feature_mean) / feature_std

    class TemporalModel(nn.Module):
        def __init__(self, feature_count: int):
            super().__init__()
            self.gru = nn.GRU(feature_count, 48, num_layers=1, batch_first=True)
            self.failure_head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 7))

        def forward(self, values):
            encoded, _ = self.gru(values)
            return self.failure_head(encoded[:, -1])

    models = []
    calibrators = []
    test_members = []
    validation_members = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for member in range(5):
        torch.manual_seed(107 + member)
        model = TemporalModel(normalized.shape[2])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_function = nn.BCEWithLogitsLoss()
        train_indices = bootstrap_indices(session_ids, split.train, 107 + member)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(normalized[train_indices]), torch.from_numpy(labels[train_indices])),
            batch_size=64,
            shuffle=True,
        )
        model.train()
        for _epoch in range(35):
            for values, targets in loader:
                optimizer.zero_grad()
                logits = model(values)
                loss = loss_function(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(torch.from_numpy(normalized[split.validation])).numpy()
            test_logits = model(torch.from_numpy(normalized[split.test])).numpy()
        member_calibrators = []
        validation_probabilities = []
        test_probabilities = []
        for horizon_index in range(len(HORIZONS_SECONDS)):
            target = labels[split.validation, horizon_index].astype(np.int64)
            if len(np.unique(target)) < 2:
                calibrator = ConstantProbabilityModel(float(target.mean()))
            else:
                calibrator = LogisticRegression(max_iter=1_000, solver="liblinear")
                calibrator.fit(validation_logits[:, horizon_index].reshape(-1, 1), target)
            member_calibrators.append(calibrator)
            validation_probabilities.append(
                calibrator.predict_proba(validation_logits[:, horizon_index].reshape(-1, 1))[:, 1]
            )
            test_probabilities.append(
                calibrator.predict_proba(test_logits[:, horizon_index].reshape(-1, 1))[:, 1]
            )
        calibrators.append(member_calibrators)
        validation_members.append(monotonic_probabilities(np.column_stack(validation_probabilities)))
        test_members.append(monotonic_probabilities(np.column_stack(test_probabilities)))
        scripted_path = MODEL_DIR / f"temporal_v3_member_{member}.pt"
        torch.jit.script(model).save(str(scripted_path))
        models.append(str(scripted_path))

    probabilities = monotonic_probabilities(np.median(np.asarray(test_members), axis=0))
    elapsed = dataset["elapsed_seconds"].astype(np.float64)
    time_to_failure = dataset["time_to_failure_seconds"].astype(np.float64)
    occupancy = dataset["occupancy_X"].astype(np.float64)
    elapsed_probabilities = []
    occupancy_probabilities = []
    constant_probabilities = []
    for index in range(len(HORIZONS_SECONDS)):
        target = labels[:, index].astype(np.int64)
        elapsed_model = fit_reference(elapsed.reshape(-1, 1), target, split.train)
        occupancy_model = fit_reference(occupancy, target, split.train)
        elapsed_probabilities.append(elapsed_model.predict_proba(elapsed[split.test].reshape(-1, 1))[:, 1])
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
    sample = torch.from_numpy(normalized[split.test][:1])
    runtime_models = [torch.jit.load(path) for path in models]
    for _ in range(30):
        started = time.perf_counter()
        with torch.no_grad():
            for runtime_model in runtime_models:
                runtime_model(sample)
        latency_samples.append((time.perf_counter() - started) * 1_000)
    metrics = evaluate_survival(
        labels[split.test],
        probabilities,
        references,
        session_ids[split.test],
        time_to_failure[split.test],
        float(np.quantile(latency_samples, 0.95)),
    )
    thresholds, certainty = certainty_contract(np.asarray(validation_members), time_to_failure[split.validation])
    gates = promotion_gates(metrics, total_runs)
    is_eligible = eligible(metrics, total_runs)
    metadata = {
        "model": "PyTorch personal temporal survival ensemble",
        "split": "32 development train / 8 development calibration / 10 locked future test" if total_runs >= 50 else "provisional chronological run split",
        "objective": "failure within 5/10/15/20/30/45/60 active seconds",
        "references": list(references),
        "metrics": metrics,
        "gates": gates,
        "certainty": certainty,
        "eligibleForLive": is_eligible,
    }
    write_metrics("torch_v3_metrics", metadata)
    artifact = {
        "kind": "torchscript_personal_survival",
        "version": f"personal-torch-v3-r{total_runs}",
        "torchscript_paths": models,
        "calibrators": calibrators,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "horizons_seconds": HORIZONS_SECONDS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "certainty_thresholds": thresholds,
        "certainty_enabled": certainty,
        "calibrated": float(metrics["worstCalibrationError"]) <= 0.12,
        "eligible_for_live": is_eligible,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_DIR / "torch_v3_candidate.joblib")
    if is_eligible:
        current = MODEL_DIR / "live_model.joblib"
        existing = joblib.load(current) if current.exists() else None
        if existing is None or float(metrics["meanBrier"]) < float(existing.get("metrics", {}).get("meanBrier", float("inf"))):
            joblib.dump(artifact, current)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
