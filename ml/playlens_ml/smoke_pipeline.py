from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


HORIZONS = np.asarray([5, 10, 15, 20, 30, 45, 60], dtype=np.int64)


def write_synthetic_dataset(artifacts_dir: Path) -> None:
    rng = np.random.default_rng(20260829)
    remaining_template = np.asarray([70, 55, 40, 28, 18, 12, 7, 3], dtype=np.float32)
    run_count = 10
    remaining = np.tile(remaining_template, run_count)
    session_ids = np.repeat([f"smoke-run-{index:02d}" for index in range(run_count)], len(remaining_template))
    elapsed = np.tile(80 - remaining_template, run_count)
    labels = (remaining[:, None] <= HORIZONS[None, :]).astype(np.int64)
    feature_count = 16
    signal = (60 - np.minimum(remaining, 60))[:, None] / 60
    features = rng.normal(0, 0.35, size=(len(remaining), feature_count)) + signal
    sequences = np.repeat(features[:, None, :], 32, axis=1)
    sequences += rng.normal(0, 0.05, size=sequences.shape)
    dataset_dir = artifacts_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_dir / "personal_windows.npz",
        X=features.astype(np.float64),
        sequences=sequences.astype(np.float32),
        y_failure=labels,
        session_ids=session_ids,
        elapsed_seconds=elapsed.astype(np.float32),
        time_to_failure_seconds=remaining,
        timestamps_ms=np.arange(len(remaining), dtype=np.int64) * 1_000,
        active_elapsed_ms=(elapsed * 1_000).astype(np.int64),
        run_order=np.repeat(np.arange(1, run_count + 1), len(remaining_template)),
        occupancy_X=features[:, :6].astype(np.float64),
        horizons_seconds=HORIZONS,
    )


def run_step(module: str, artifacts_dir: Path, root: Path) -> None:
    environment = os.environ.copy()
    environment["PLAYLENS_ARTIFACTS_DIR"] = str(artifacts_dir)
    python_path = os.pathsep.join([str(root / "service"), str(root / "ml")])
    environment["PYTHONPATH"] = os.pathsep.join([python_path, environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed:\n{result.stdout}\n{result.stderr}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="playlens-ml-smoke-") as directory:
        artifacts_dir = Path(directory) / "artifacts"
        write_synthetic_dataset(artifacts_dir)
        run_step("playlens_ml.train_baseline", artifacts_dir, root)
        run_step("playlens_ml.train_torch", artifacts_dir, root)
        required = [
            artifacts_dir / "models" / "baseline_v3_candidate.joblib",
            artifacts_dir / "models" / "torch_v3_candidate.joblib",
            artifacts_dir / "evaluation" / "baseline_v3_metrics.json",
            artifacts_dir / "evaluation" / "torch_v3_metrics.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Smoke pipeline did not create: {missing}")
        if (artifacts_dir / "models" / "live_model.joblib").exists():
            raise RuntimeError("A provisional smoke model was incorrectly promoted")
        print(json.dumps({"status": "ok", "models": 2, "isolated": True}, indent=2))


if __name__ == "__main__":
    main()
