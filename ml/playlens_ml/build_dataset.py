from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from playlens_api.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    aggregate_feature_names,
    aggregate_window,
    extract_jpeg_features,
)
from playlens_api.training_status import legacy_run_quality
from playlens_api.collection_policy import PERSONAL_TARGET, DEVELOPMENT_RUNS, LOCKED_TEST_RUNS

from .common import DATASET_DIR, HORIZONS_SECONDS, ROOT


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


def failure_targets(remaining_seconds: float) -> list[int]:
    return [int(max(0.0, remaining_seconds) <= horizon) for horizon in HORIZONS_SECONDS]


def select_history(
    frame_paths: list[Path],
    timestamp_ms: int,
    active_by_timestamp: dict[int, int] | None = None,
    minimum_active_ms: int = 0,
) -> list[Path]:
    active_by_timestamp = active_by_timestamp or {int(path.stem): int(path.stem) for path in frame_paths}
    anchor_active = active_by_timestamp.get(timestamp_ms)
    if anchor_active is None:
        return []
    eligible = [
        path
        for path in frame_paths
        if minimum_active_ms <= active_by_timestamp.get(int(path.stem), -1) <= anchor_active
        and anchor_active - active_by_timestamp.get(int(path.stem), -1) <= 8_000
    ]
    if len(eligible) < 28 or anchor_active - active_by_timestamp[int(eligible[0].stem)] < 7_500:
        return []
    indices = np.linspace(0, len(eligible) - 1, 32).round().astype(int)
    return [eligible[index] for index in indices]


def personal_runs(data_dir: Path) -> list[dict]:
    database_path = data_dir / "playlens.sqlite3"
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT id, started_at, end_reason, active_duration_ms, usable_for_training,
            quality_reason, model_version
            FROM sessions WHERE status = 'complete' ORDER BY started_at"""
        ).fetchall()
        lifecycle_rows = connection.execute(
            "SELECT session_id, active_elapsed_ms, state FROM lifecycle_events ORDER BY timestamp_ms"
        ).fetchall()
    resume_points: dict[str, list[int]] = {}
    for event in lifecycle_rows:
        if event["state"] == "active":
            resume_points.setdefault(str(event["session_id"]), []).append(int(event["active_elapsed_ms"]))
    runs = []
    for row in rows:
        session_id = str(row["id"])
        legacy = str(row["quality_reason"] or "").startswith("legacy:") or str(row["quality_reason"] or "") == "usable legacy run"
        if row["end_reason"] != "game_over" or row["model_version"] != "collection-v3":
            continue
        if int(row["active_duration_ms"] or 0) == 0:
            usable, _reason = legacy_run_quality(
                data_dir / "sessions" / session_id / "observations.jsonl"
            )
        else:
            usable = bool(row["usable_for_training"])
        if usable:
            runs.append(
                {
                    "id": session_id,
                    "legacy": legacy,
                    "activeDurationMs": int(row["active_duration_ms"] or 0),
                    "resumePoints": resume_points.get(session_id, []),
                }
            )
    return runs[:PERSONAL_TARGET]


def main() -> None:
    data_dir = Path(os.environ.get("PLAYLENS_DATA_DIR", ROOT / "data"))
    windows: list[dict] = []
    aggregates: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    failure_labels: list[list[int]] = []
    session_ids: list[str] = []
    elapsed_seconds: list[float] = []
    time_to_failure_seconds: list[float] = []
    timestamps_ms: list[int] = []
    active_elapsed_ms_values: list[int] = []
    run_order_values: list[int] = []
    aggregate_names = aggregate_feature_names()
    occupancy_indices = np.asarray(
        [
            aggregate_names.index("mean_board_occupancy"),
            aggregate_names.index("mean_near_center_occupancy"),
            aggregate_names.index("mean_outer_stack_occupancy"),
            aggregate_names.index("mean_radial_pressure"),
            aggregate_names.index("mean_mean_stack_reach"),
            aggregate_names.index("mean_max_stack_reach"),
            aggregate_names.index("mean_sector_imbalance"),
            aggregate_names.index("delta_board_occupancy"),
            aggregate_names.index("delta_near_center_occupancy"),
            aggregate_names.index("delta_outer_stack_occupancy"),
            aggregate_names.index("delta_max_stack_reach"),
        ]
    )

    runs = personal_runs(data_dir)
    for run_order, run in enumerate(runs, start=1):
        directory = data_dir / "sessions" / run["id"]
        observations = read_jsonl(directory / "observations.jsonl")
        if len(observations) < 32:
            continue
        first_wall = int(observations[0]["timestampMs"])
        active_by_timestamp = {
            int(item["timestampMs"]): int(
                item.get("activeElapsedMs", int(item["timestampMs"]) - first_wall)
            )
            for item in observations
        }
        motion_by_timestamp = {
            int(item["timestampMs"]): float(item.get("observedMotion", 0)) for item in observations
        }
        frame_paths = sorted((directory / "frames").glob("*.jpg"), key=lambda path: int(path.stem))
        available_active = [active_by_timestamp[int(path.stem)] for path in frame_paths if int(path.stem) in active_by_timestamp]
        if not available_active:
            continue
        active_duration_ms = int(run["activeDurationMs"] or max(available_active))
        resume_points = sorted(int(value) for value in run["resumePoints"])
        for anchor_path in frame_paths[32::4]:
            timestamp = int(anchor_path.stem)
            if timestamp not in active_by_timestamp:
                continue
            active_elapsed = active_by_timestamp[timestamp]
            last_resume = max((point for point in resume_points if point <= active_elapsed), default=0)
            selected = select_history(frame_paths, timestamp, active_by_timestamp, last_resume)
            if len(selected) != 32:
                continue
            previous_gray = None
            vectors = []
            for frame_path in selected:
                frame_timestamp = int(frame_path.stem)
                features = extract_jpeg_features(
                    frame_path.read_bytes(), previous_gray, motion_by_timestamp.get(frame_timestamp, 0)
                )
                previous_gray = features.gray
                vectors.append(features.vector)
            remaining_seconds = max(0.0, (active_duration_ms - active_elapsed) / 1_000)
            failures = failure_targets(remaining_seconds)
            aggregates.append(aggregate_window(vectors))
            sequences.append(np.vstack(vectors))
            failure_labels.append(failures)
            session_ids.append(run["id"])
            elapsed_seconds.append(active_elapsed / 1_000)
            time_to_failure_seconds.append(remaining_seconds)
            timestamps_ms.append(timestamp)
            active_elapsed_ms_values.append(active_elapsed)
            run_order_values.append(run_order)
            windows.append(
                {
                    "sessionId": run["id"],
                    "runOrder": run_order,
                    "timestampMs": timestamp,
                    "activeElapsedMs": active_elapsed,
                    "contextSeconds": 8,
                    "failureWithinSeconds": {
                        str(horizon): failures[index] for index, horizon in enumerate(HORIZONS_SECONDS)
                    },
                    "timeToFailureSeconds": remaining_seconds,
                }
            )

    if not windows:
        raise SystemExit(f"No usable personal sessions found under {data_dir / 'sessions'}")
    if len(set(session_ids)) != len(runs):
        raise SystemExit("A cohort run has no usable windows; refusing to shift the chronological split")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    feature_matrix = np.vstack(aggregates)
    np.savez_compressed(
        DATASET_DIR / "personal_windows.npz",
        X=feature_matrix,
        sequences=np.stack(sequences),
        y_failure=np.asarray(failure_labels, dtype=np.int64),
        session_ids=np.asarray(session_ids),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float32),
        time_to_failure_seconds=np.asarray(time_to_failure_seconds, dtype=np.float32),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
        active_elapsed_ms=np.asarray(active_elapsed_ms_values, dtype=np.int64),
        run_order=np.asarray(run_order_values, dtype=np.int64),
        occupancy_X=feature_matrix[:, occupancy_indices],
        horizons_seconds=np.asarray(HORIZONS_SECONDS, dtype=np.int64),
        feature_schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    with (DATASET_DIR / "personal_windows.jsonl").open("w", encoding="utf-8") as stream:
        for window in windows:
            stream.write(json.dumps(window) + "\n")
    summary = {
        "sessions": len(set(session_ids)),
        "windows": len(windows),
        "contextSeconds": 8,
        "horizonsSeconds": list(HORIZONS_SECONDS),
        "failureRates": {
            str(horizon): float(np.asarray(failure_labels, dtype=np.float32)[:, index].mean())
            for index, horizon in enumerate(HORIZONS_SECONDS)
        },
        "source": "personal_clean_runs_only",
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureCount": len(FEATURE_NAMES),
        "developmentRuns": min(DEVELOPMENT_RUNS, len(set(session_ids))),
        "lockedTestRuns": max(0, min(LOCKED_TEST_RUNS, len(set(session_ids)) - DEVELOPMENT_RUNS)),
    }
    (DATASET_DIR / "personal_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
