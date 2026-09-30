from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path


from .collection_policy import CHECKPOINT_MILESTONES


def create_collection_checkpoint(data_dir: Path, artifacts_dir: Path, usable_runs: int) -> Path | None:
    """Create an atomic, self-contained archive of the official v3 cohort."""
    if usable_runs not in CHECKPOINT_MILESTONES:
        return None
    database_path = data_dir / "playlens.sqlite3"
    if not database_path.exists():
        raise FileNotFoundError("Hexlearn database is missing")

    checkpoint_root = artifacts_dir / "checkpoints"
    destination = checkpoint_root / f"run-{usable_runs:03d}"
    if (destination / "complete").exists():
        return destination
    temporary = checkpoint_root / f".run-{usable_runs:03d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)

    snapshot_path = temporary / "playlens.sqlite3"
    with sqlite3.connect(database_path) as source, sqlite3.connect(snapshot_path) as target:
        source.backup(target)
        source.row_factory = sqlite3.Row
        rows = source.execute(
            """SELECT id, started_at, active_duration_ms, valid_frame_count,
            expected_frame_count, capture_coverage, recording_path
            FROM sessions WHERE usable_for_training = 1
            AND model_version = 'collection-v3'
            ORDER BY started_at LIMIT ?""",
            (usable_runs,),
        ).fetchall()
    if len(rows) != usable_runs:
        shutil.rmtree(temporary)
        raise RuntimeError(f"Expected {usable_runs} official v3 sessions, found {len(rows)}")

    sessions = [
        {
            "id": str(row["id"]),
            "startedAt": row["started_at"],
            "activeDurationMs": int(row["active_duration_ms"] or 0),
            "validFrameCount": int(row["valid_frame_count"] or 0),
            "expectedFrameCount": int(row["expected_frame_count"] or 0),
            "captureCoverage": float(row["capture_coverage"] or 0),
            "hasRecording": bool(row["recording_path"]),
        }
        for row in rows
    ]
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "cohortVersion": "collection-v3",
        "usableRuns": usable_runs,
        "sessions": sessions,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with tarfile.open(temporary / "session-data.tar", mode="w") as archive:
        for session in sessions:
            session_path = data_dir / "sessions" / session["id"]
            if not session_path.is_dir():
                raise FileNotFoundError(f"Session data is missing for {session['id']}")
            archive.add(session_path, arcname=f"sessions/{session['id']}", recursive=True)
    (temporary / "complete").write_text("ok\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination
