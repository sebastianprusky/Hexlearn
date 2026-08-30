from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .checkpoints import CHECKPOINT_MILESTONES, create_collection_checkpoint


TRAINING_MILESTONES = {20, 25, 30, 35, 40, 50}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingCoordinator:
    def __init__(
        self,
        playlens_root: Path,
        data_dir: Path,
        artifacts_dir: Path,
        reload_model: Callable[[], None],
    ):
        self.playlens_root = playlens_root
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.reload_model = reload_model
        self.path = artifacts_dir / "training_job.json"
        self.lock = threading.Lock()
        self.running = False

    def recover_interrupted(self, usable_runs: int) -> bool:
        current = {}
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if current.get("status") != "running":
            return False
        interrupted_runs = int(current.get("usableRuns", 0))
        self._write(
            {
                **current,
                "status": "failed",
                "failedAt": _now(),
                "error": "Local service stopped during training; retrying automatically.",
            }
        )
        return self.schedule(usable_runs) if usable_runs == interrupted_runs else False

    def _write(self, payload: dict) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def schedule(self, usable_runs: int) -> bool:
        if usable_runs not in TRAINING_MILESTONES:
            return False
        with self.lock:
            if self.running:
                return False
            current = {}
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            if current.get("usableRuns") == usable_runs and current.get("status") in {"running", "complete"}:
                return False
            self.running = True
        thread = threading.Thread(target=self._run, args=(usable_runs,), daemon=True)
        thread.start()
        return True

    def _run(self, usable_runs: int) -> None:
        steps = ["playlens_ml.build_dataset", "playlens_ml.train_baseline"]
        try:
            import torch  # noqa: F401

            steps.append("playlens_ml.train_torch")
        except ImportError:
            pass
        self._write(
            {
                "status": "running",
                "usableRuns": usable_runs,
                "startedAt": _now(),
                "step": steps[0],
                "completedSteps": [],
            }
        )
        completed = []
        output_tail = ""
        try:
            if usable_runs in CHECKPOINT_MILESTONES:
                self._write(
                    {
                        "status": "running",
                        "usableRuns": usable_runs,
                        "startedAt": _now(),
                        "step": "collection_checkpoint",
                        "completedSteps": completed,
                    }
                )
                create_collection_checkpoint(self.data_dir, self.artifacts_dir, usable_runs)
                completed.append("collection_checkpoint")
            environment = os.environ.copy()
            python_path = os.pathsep.join(
                [str(self.playlens_root / "service"), str(self.playlens_root / "ml")]
            )
            environment["PYTHONPATH"] = os.pathsep.join(
                [python_path, environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            for step in steps:
                self._write(
                    {
                        "status": "running",
                        "usableRuns": usable_runs,
                        "startedAt": _now(),
                        "step": step,
                        "completedSteps": completed,
                    }
                )
                result = subprocess.run(
                    [sys.executable, "-m", step],
                    cwd=self.playlens_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=1_800,
                    check=False,
                )
                output_tail = (result.stdout + "\n" + result.stderr)[-4_000:]
                if result.returncode != 0:
                    raise RuntimeError(f"{step} exited with {result.returncode}")
                completed.append(step)
            self.reload_model()
            self._write(
                {
                    "status": "complete",
                    "usableRuns": usable_runs,
                    "completedAt": _now(),
                    "completedSteps": completed,
                    "outputTail": output_tail,
                }
            )
        except Exception as exc:  # Local job state must survive model/dependency failures.
            self._write(
                {
                    "status": "failed",
                    "usableRuns": usable_runs,
                    "failedAt": _now(),
                    "completedSteps": completed,
                    "error": str(exc),
                    "outputTail": output_tail,
                }
            )
        finally:
            with self.lock:
                self.running = False
