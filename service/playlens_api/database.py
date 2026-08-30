from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  game TEXT NOT NULL,
  source_url TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  status TEXT NOT NULL,
  canvas_width INTEGER NOT NULL,
  canvas_height INTEGER NOT NULL,
  overlay_expanded_initial INTEGER NOT NULL,
  end_reason TEXT,
  recording_path TEXT,
  model_source TEXT NOT NULL,
  active_duration_ms INTEGER NOT NULL DEFAULT 0,
  usable_for_training INTEGER NOT NULL DEFAULT 0,
  quality_reason TEXT,
  collection_index INTEGER,
  model_version TEXT,
  valid_frame_count INTEGER NOT NULL DEFAULT 0,
  expected_frame_count INTEGER NOT NULL DEFAULT 0,
  capture_coverage REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS predictions (
  session_id TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  failure_risk REAL,
  failure_risk_20 REAL,
  failure_risk_30 REAL,
  warning_state TEXT,
  failure_window TEXT,
  risk_band TEXT,
  expected_progress REAL,
  risk_trend TEXT,
  model_source TEXT NOT NULL,
  calibrated INTEGER NOT NULL,
  latency_ms REAL,
  active_elapsed_ms INTEGER,
  failure_probabilities TEXT,
  loss_window TEXT,
  certainty TEXT,
  estimated_remaining_seconds REAL,
  uncertainty_low_seconds REAL,
  uncertainty_high_seconds REAL,
  displayed_value TEXT,
  PRIMARY KEY (session_id, timestamp_ms),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS overlay_events (
  session_id TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  expanded INTEGER NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS lifecycle_events (
  session_id TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  active_elapsed_ms INTEGER NOT NULL,
  state TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            prediction_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
            }
            prediction_additions = {
                "failure_risk_20": "REAL",
                "failure_risk_30": "REAL",
                "warning_state": "TEXT",
                "failure_window": "TEXT",
                "active_elapsed_ms": "INTEGER",
                "failure_probabilities": "TEXT",
                "loss_window": "TEXT",
                "certainty": "TEXT",
                "estimated_remaining_seconds": "REAL",
                "uncertainty_low_seconds": "REAL",
                "uncertainty_high_seconds": "REAL",
                "displayed_value": "TEXT",
            }
            for column, column_type in prediction_additions.items():
                if column not in prediction_columns:
                    connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {column_type}")
            session_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            session_additions = {
                "active_duration_ms": "INTEGER NOT NULL DEFAULT 0",
                "usable_for_training": "INTEGER NOT NULL DEFAULT 0",
                "quality_reason": "TEXT",
                "collection_index": "INTEGER",
                "model_version": "TEXT",
                "valid_frame_count": "INTEGER NOT NULL DEFAULT 0",
                "expected_frame_count": "INTEGER NOT NULL DEFAULT 0",
                "capture_coverage": "REAL NOT NULL DEFAULT 0",
            }
            for column, column_type in session_additions.items():
                if column not in session_columns:
                    connection.execute(f"ALTER TABLE sessions ADD COLUMN {column} {column_type}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_session(self, values: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO sessions
                (id, game, source_url, started_at, status, canvas_width, canvas_height,
                 overlay_expanded_initial, model_source, collection_index, model_version)
                VALUES (:id, :game, :source_url, :started_at, :status, :canvas_width,
                        :canvas_height, :overlay_expanded_initial, :model_source,
                        :collection_index, :model_version)""",
                values,
            )

    def save_prediction(self, session_id: str, prediction: dict[str, Any]) -> None:
        if prediction["status"] != "ready":
            return
        with self.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO predictions
                (session_id, timestamp_ms, status, failure_risk, failure_risk_20,
                 failure_risk_30, warning_state, failure_window, risk_band,
                 expected_progress, risk_trend, model_source, calibrated, latency_ms,
                 active_elapsed_ms, failure_probabilities, loss_window, certainty,
                 estimated_remaining_seconds, uncertainty_low_seconds,
                 uncertainty_high_seconds, displayed_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    prediction["timestampMs"],
                    prediction["status"],
                    prediction.get("failureRisk10s"),
                    prediction.get("failureRisk20s"),
                    prediction.get("failureRisk30s"),
                    prediction.get("warningState"),
                    prediction.get("failureWindow"),
                    prediction.get("riskBand"),
                    prediction.get("expectedScoreGain10s"),
                    prediction.get("riskTrend"),
                    prediction["modelSource"],
                    int(prediction["calibrated"]),
                    prediction.get("latencyMs"),
                    prediction.get("activeElapsedMs"),
                    json.dumps(prediction.get("failureProbabilities", {}), sort_keys=True),
                    prediction.get("lossWindow"),
                    prediction.get("certainty"),
                    prediction.get("estimatedRemainingSeconds"),
                    prediction.get("uncertaintyLowSeconds"),
                    prediction.get("uncertaintyHighSeconds"),
                    prediction.get("displayedValue"),
                ),
            )

    def finish_session(
        self,
        session_id: str,
        ended_at: str,
        reason: str,
        active_duration_ms: int,
        usable: bool,
        quality_reason: str,
        valid_frame_count: int,
        expected_frame_count: int,
        capture_coverage: float,
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE sessions SET ended_at = ?, end_reason = ?, status = 'complete',
                active_duration_ms = ?, usable_for_training = ?, quality_reason = ?,
                valid_frame_count = ?, expected_frame_count = ?, capture_coverage = ?
                WHERE id = ?""",
                (
                    ended_at,
                    reason,
                    active_duration_ms,
                    int(usable),
                    quality_reason,
                    valid_frame_count,
                    expected_frame_count,
                    capture_coverage,
                    session_id,
                ),
            )
            return cursor.rowcount > 0

    def add_lifecycle_event(
        self, session_id: str, timestamp_ms: int, active_elapsed_ms: int, state: str
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO lifecycle_events
                (session_id, timestamp_ms, active_elapsed_ms, state) VALUES (?, ?, ?, ?)""",
                (session_id, timestamp_ms, active_elapsed_ms, state),
            )

    def usable_run_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT COUNT(*) FROM sessions
                WHERE usable_for_training = 1 AND model_version = 'collection-v3'"""
            ).fetchone()
        return int(row[0]) if row else 0

    def legacy_candidates(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT id, started_at FROM sessions
                WHERE status = 'complete' AND end_reason = 'game_over'
                AND active_duration_ms = 0 AND quality_reason IS NULL
                ORDER BY started_at"""
            ).fetchall()
        return [{"id": str(row["id"]), "startedAt": row["started_at"]} for row in rows]

    def backfill_legacy_quality(
        self,
        session_id: str,
        active_duration_ms: int,
        usable: bool,
        reason: str,
        collection_index: int,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE sessions SET active_duration_ms = ?, usable_for_training = ?,
                quality_reason = ?, collection_index = COALESCE(collection_index, ?),
                model_version = COALESCE(model_version, 'legacy-v2') WHERE id = ?""",
                (active_duration_ms, int(usable), reason, collection_index, session_id),
            )

    def exclude_legacy_from_official_cohort(self) -> None:
        """Keep legacy captures, but never let them occupy one of the official v3 slots."""
        with self.connection() as connection:
            connection.execute(
                """UPDATE sessions SET usable_for_training = 0,
                quality_reason = 'legacy: excluded from official v3 cohort'
                WHERE model_version = 'legacy-v2' AND usable_for_training = 1"""
            )

    def exclude_duplicate_controller_runs(self) -> list[str]:
        """Quarantine simultaneous duplicate captures while preserving their files and rows."""
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT id, started_at, active_duration_ms, valid_frame_count
                FROM sessions WHERE model_version = 'collection-v3'
                AND usable_for_training = 1 AND status = 'complete'
                ORDER BY started_at"""
            ).fetchall()
            duplicate_ids: set[str] = set()
            for index, left in enumerate(rows):
                left_started = datetime.fromisoformat(str(left["started_at"]))
                for right in rows[index + 1 :]:
                    right_started = datetime.fromisoformat(str(right["started_at"]))
                    start_delta = abs((right_started - left_started).total_seconds())
                    if start_delta > 1.0:
                        break
                    duration_delta = abs(
                        int(right["active_duration_ms"]) - int(left["active_duration_ms"])
                    )
                    frame_delta = abs(
                        int(right["valid_frame_count"]) - int(left["valid_frame_count"])
                    )
                    frame_tolerance = max(
                        4,
                        round(
                            max(
                                int(right["valid_frame_count"]),
                                int(left["valid_frame_count"]),
                            )
                            * 0.02
                        ),
                    )
                    if duration_delta <= 2_000 and frame_delta <= frame_tolerance:
                        duplicate_ids.update((str(left["id"]), str(right["id"])))
            if duplicate_ids:
                placeholders = ",".join("?" for _ in duplicate_ids)
                connection.execute(
                    f"""UPDATE sessions SET usable_for_training = 0,
                    quality_reason = 'excluded: duplicate-controller capture'
                    WHERE id IN ({placeholders})""",
                    tuple(sorted(duplicate_ids)),
                )
        return sorted(duplicate_ids)

    def set_recording(self, session_id: str, path: str) -> None:
        with self.connection() as connection:
            connection.execute("UPDATE sessions SET recording_path = ? WHERE id = ?", (path, session_id))

    def add_overlay_event(self, session_id: str, timestamp_ms: int, expanded: bool) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO overlay_events (session_id, timestamp_ms, expanded) VALUES (?, ?, ?)",
                (session_id, timestamp_ms, int(expanded)),
            )

    @staticmethod
    def _session_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "game": row["game"],
            "sourceUrl": row["source_url"],
            "startedAt": row["started_at"],
            "endedAt": row["ended_at"],
            "status": row["status"],
            "canvasWidth": row["canvas_width"],
            "canvasHeight": row["canvas_height"],
            "overlayExpandedInitial": bool(row["overlay_expanded_initial"]),
            "endReason": row["end_reason"],
            "hasRecording": bool(row["recording_path"]),
            "modelSource": row["model_source"],
            "activeDurationMs": row["active_duration_ms"],
            "usableForTraining": bool(row["usable_for_training"]),
            "qualityReason": row["quality_reason"],
            "collectionIndex": row["collection_index"],
            "modelVersion": row["model_version"],
            "validFrameCount": row["valid_frame_count"],
            "expectedFrameCount": row["expected_frame_count"],
            "captureCoverage": row["capture_coverage"],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()
        return [self._session_dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            predictions = connection.execute(
                "SELECT * FROM predictions WHERE session_id = ? ORDER BY timestamp_ms", (session_id,)
            ).fetchall()
            overlay = connection.execute(
                "SELECT timestamp_ms, expanded FROM overlay_events WHERE session_id = ? ORDER BY timestamp_ms",
                (session_id,),
            ).fetchall()
            lifecycle = connection.execute(
                """SELECT timestamp_ms, active_elapsed_ms, state FROM lifecycle_events
                WHERE session_id = ? ORDER BY timestamp_ms""",
                (session_id,),
            ).fetchall()
            baseline_row = connection.execute(
                """SELECT AVG(active_duration_ms) FROM sessions
                WHERE usable_for_training = 1 AND started_at < ?""",
                (row["started_at"],),
            ).fetchone()
        result = self._session_dict(row)
        result["predictions"] = [
            {
                "timestampMs": item["timestamp_ms"],
                "status": item["status"],
                "failureRisk10s": item["failure_risk"],
                "failureRisk20s": item["failure_risk_20"],
                "failureRisk30s": item["failure_risk_30"],
                "warningState": item["warning_state"],
                "failureWindow": item["failure_window"],
                "riskBand": item["risk_band"],
                "expectedScoreGain10s": item["expected_progress"],
                "riskTrend": item["risk_trend"],
                "modelSource": item["model_source"],
                "calibrated": bool(item["calibrated"]),
                "latencyMs": item["latency_ms"],
                "activeElapsedMs": item["active_elapsed_ms"],
                "failureProbabilities": json.loads(item["failure_probabilities"] or "{}"),
                "lossWindow": item["loss_window"],
                "certainty": item["certainty"],
                "estimatedRemainingSeconds": item["estimated_remaining_seconds"],
                "uncertaintyLowSeconds": item["uncertainty_low_seconds"],
                "uncertaintyHighSeconds": item["uncertainty_high_seconds"],
                "displayedValue": item["displayed_value"],
            }
            for item in predictions
        ]
        result["overlayEvents"] = [
            {"timestampMs": item["timestamp_ms"], "expanded": bool(item["expanded"])} for item in overlay
        ]
        result["lifecycleEvents"] = [
            {
                "timestampMs": item["timestamp_ms"],
                "activeElapsedMs": item["active_elapsed_ms"],
                "state": item["state"],
            }
            for item in lifecycle
        ]
        result["baselineDurationSeconds"] = (
            float(baseline_row[0]) / 1_000 if baseline_row and baseline_row[0] is not None else None
        )
        return result

    def recording_path(self, session_id: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute("SELECT recording_path FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else row["recording_path"]

    def delete_session(self, session_id: str) -> bool:
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0
