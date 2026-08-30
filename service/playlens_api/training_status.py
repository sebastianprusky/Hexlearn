from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


PERSONAL_TARGET = 50
DEVELOPMENT_RUNS = 40
LOCKED_TEST_RUNS = 10
HORIZONS_SECONDS = [5, 10, 15, 20, 30, 45, 60]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_observations(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    values: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
            values.append(item)
        except (json.JSONDecodeError, TypeError):
            continue
    return values


def legacy_run_quality(path: Path) -> tuple[bool, str]:
    observations = _read_observations(path)
    if len(observations) < 32:
        return False, "legacy: insufficient frames"
    timestamps = [int(item.get("timestampMs", 0)) for item in observations]
    if timestamps[-1] - timestamps[0] < 8_000:
        return False, "legacy: fewer than eight seconds"
    if any(current - previous > 1_000 for previous, current in zip(timestamps, timestamps[1:])):
        return False, "legacy: capture gap"
    motions = [float(item.get("observedMotion", 0)) for item in observations]
    low_streak = 0
    for index, motion in enumerate(motions[:-12]):
        low_streak = low_streak + 1 if motion <= 0.6 else 0
        if low_streak >= 8:
            return False, "legacy: possible pause"
    return True, "usable legacy run"


def _instrumented_summary(data_dir: Path) -> dict[str, Any]:
    sessions = []
    for directory in (data_dir / "training").glob("training-*"):
        metadata = _read_json(directory / "metadata.json") or {}
        telemetry = _read_observations(directory / "telemetry.jsonl")
        timestamps = [int(item.get("timestampMs", 0)) for item in telemetry]
        sessions.append(
            {
                "id": directory.name,
                "automated": bool(metadata.get("automated", False)),
                "complete": (directory / "complete").exists(),
                "startedAt": metadata.get("startedAt"),
                "policy": metadata.get("policy"),
                "seed": metadata.get("seed"),
                "frames": len(telemetry),
                "durationMs": max(timestamps) - min(timestamps) if timestamps else 0,
            }
        )
    complete = [session for session in sessions if session["complete"]]
    return {
        "totalSessions": len(sessions),
        "completeSessions": len(complete),
        "activeSessions": len(sessions) - len(complete),
        "automatedSessions": sum(session["automated"] for session in complete),
        "manualSessions": sum(not session["automated"] for session in complete),
        "totalFrames": sum(session["frames"] for session in complete),
        "totalDurationMs": sum(session["durationMs"] for session in complete),
        "recentSessions": sorted(sessions, key=lambda item: item.get("startedAt") or "", reverse=True)[:8],
        "trainingUse": "smoke_tests_only",
    }


def _checkpoint_summary(artifacts_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for directory in sorted((artifacts_dir / "checkpoints").glob("run-*")):
        if not (directory / "complete").exists():
            continue
        manifest = _read_json(directory / "manifest.json") or {}
        checkpoints.append(
            {
                "usableRuns": int(manifest.get("usableRuns", 0)),
                "createdAt": manifest.get("createdAt"),
                "sessionCount": len(manifest.get("sessions", [])),
                "path": str(directory),
            }
        )
    return checkpoints


def build_training_status(
    data_dir: Path,
    artifacts_dir: Path,
    model_source: str,
    calibrated: bool,
) -> dict[str, Any]:
    personal_runs: list[dict[str, Any]] = []
    database_path = data_dir / "playlens.sqlite3"
    if database_path.exists():
        try:
            with sqlite3.connect(database_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    """SELECT id, started_at, end_reason, active_duration_ms,
                    usable_for_training, quality_reason, collection_index, model_version,
                    valid_frame_count, expected_frame_count, capture_coverage,
                    recording_path,
                    (SELECT COUNT(*) FROM lifecycle_events
                     WHERE lifecycle_events.session_id = sessions.id
                     AND lifecycle_events.state = 'paused') AS pause_count
                    FROM sessions WHERE status = 'complete' ORDER BY started_at"""
                ).fetchall()
            for row in rows:
                legacy = str(row["quality_reason"] or "").startswith("legacy:") or str(row["quality_reason"] or "") == "usable legacy run"
                if int(row["active_duration_ms"] or 0) == 0 and row["end_reason"] == "game_over":
                    usable, reason = legacy_run_quality(
                        data_dir / "sessions" / str(row["id"]) / "observations.jsonl"
                    )
                else:
                    usable = bool(row["usable_for_training"]) and row["model_version"] == "collection-v3"
                    reason = str(row["quality_reason"] or "excluded: incomplete quality metadata")
                personal_runs.append(
                    {
                        "id": str(row["id"]),
                        "startedAt": row["started_at"],
                        "usable": usable,
                        "reason": reason,
                        "legacy": legacy,
                        "officialV3": row["model_version"] == "collection-v3",
                        "collectionIndex": row["collection_index"],
                        "activeDurationMs": int(row["active_duration_ms"] or 0),
                        "validFrameCount": int(row["valid_frame_count"] or 0),
                        "expectedFrameCount": int(row["expected_frame_count"] or 0),
                        "captureCoverage": float(row["capture_coverage"] or 0),
                        "pauseCount": int(row["pause_count"] or 0),
                        "hasRecording": bool(row["recording_path"]),
                    }
                )
        except sqlite3.Error:
            personal_runs = []

    usable_runs = [run for run in personal_runs if run["usable"]]
    excluded = [run for run in personal_runs if run["officialV3"] and not run["usable"]]
    usable_count = len(usable_runs)
    phase = "development" if usable_count < DEVELOPMENT_RUNS else "locked_test" if usable_count < PERSONAL_TARGET else "evaluation"
    dataset = _read_json(artifacts_dir / "dataset" / "personal_summary.json")
    job = _read_json(artifacts_dir / "training_job.json") or {"status": "idle"}
    evaluations: list[dict[str, Any]] = []
    for path in sorted((artifacts_dir / "evaluation").glob("*v3*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        evaluations.append(
            {
                "id": path.stem,
                "model": payload.get("model", path.stem),
                "split": payload.get("split"),
                "eligibleForLive": bool(payload.get("eligibleForLive", False)),
                "metrics": payload.get("metrics", {}),
                "gates": payload.get("gates", {}),
                "updatedAt": path.stat().st_mtime_ns // 1_000_000,
            }
        )

    failure_rates = dataset.get("failureRates", {}) if dataset else {}
    return {
        "collector": _instrumented_summary(data_dir),
        "dataset": {
            "ready": dataset is not None,
            "sessions": int(dataset.get("sessions", 0)) if dataset else 0,
            "windows": int(dataset.get("windows", 0)) if dataset else 0,
            "failureRates": failure_rates,
            "horizonsSeconds": HORIZONS_SECONDS,
            "contextSeconds": 8,
            "minimumSessions": 20,
            "readyForTraining": usable_count >= 20,
            "source": "personal_clean_runs_only",
        },
        "evaluations": evaluations,
        "personalCollection": {
            "usableRuns": usable_count,
            "targetRuns": PERSONAL_TARGET,
            "remainingRuns": max(0, PERSONAL_TARGET - usable_count),
            "developmentRuns": min(usable_count, DEVELOPMENT_RUNS),
            "developmentTarget": DEVELOPMENT_RUNS,
            "lockedTestRuns": min(max(0, usable_count - DEVELOPMENT_RUNS), LOCKED_TEST_RUNS),
            "lockedTestTarget": LOCKED_TEST_RUNS,
            "phase": phase,
            "excludedRuns": len(excluded),
            "exclusionReasons": dict(Counter(run["reason"] for run in excluded)),
            "archivedLegacyRuns": sum(run["legacy"] for run in personal_runs),
            "cohortVersion": "collection-v3",
            "recentRuns": list(reversed(personal_runs[-8:])),
        },
        "trainingJob": job,
        "checkpoints": _checkpoint_summary(artifacts_dir),
        "liveModel": {
            "source": model_source,
            "calibrated": calibrated,
            "artifactPresent": model_source == "validated_personal_model",
            "collectionOnly": model_source != "validated_personal_model",
        },
    }
