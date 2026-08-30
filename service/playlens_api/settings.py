from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PLAYLENS_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifact_path: Path
    artifacts_dir: Path = PLAYLENS_ROOT / "artifacts"
    personalization_path: Path = PLAYLENS_ROOT / "artifacts" / "personalization" / "live_calibrator.joblib"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("PLAYLENS_DATA_DIR", PLAYLENS_ROOT / "data")),
            artifact_path=Path(
                os.environ.get(
                    "PLAYLENS_MODEL_ARTIFACT",
                    PLAYLENS_ROOT / "artifacts" / "models" / "live_model.joblib",
                )
            ),
            artifacts_dir=Path(
                os.environ.get("PLAYLENS_ARTIFACTS_DIR", PLAYLENS_ROOT / "artifacts")
            ),
            personalization_path=Path(
                os.environ.get(
                    "PLAYLENS_PERSONALIZATION_ARTIFACT",
                    PLAYLENS_ROOT / "artifacts" / "personalization" / "live_calibrator.joblib",
                )
            ),
        )
