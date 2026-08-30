from __future__ import annotations

import base64
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from playlens_api.app import create_app
from playlens_api.checkpoints import create_collection_checkpoint
from playlens_api.database import Database
from playlens_api.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_jpeg_features
from playlens_api.inference import Predictor
from playlens_api.inference import estimated_remaining_seconds, loss_window, monotonic_probabilities
from playlens_api.local_game import inject_playlens_loader
from playlens_api.settings import Settings
from playlens_api.training_jobs import TrainingCoordinator


def image_data_url(step: int) -> str:
    image = Image.new("RGB", (160, 160), "#101414")
    draw = ImageDraw.Draw(image)
    draw.regular_polygon((80, 80, 30 + step % 25), n_sides=6, fill="#5dd39e")
    draw.rectangle((step % 120, 20, step % 120 + 25, 45), fill="#f0bd61")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class ProductLogicTests(unittest.TestCase):
    def test_local_game_loader_is_injected_once(self) -> None:
        original = "<html><body><canvas></canvas></body></html>"
        injected = inject_playlens_loader(original)
        self.assertIn('src="/playlens/core.js?v=064"', injected)
        self.assertIn('src="/playlens/local-bootstrap.js?v=064"', injected)
        self.assertNotIn('src="/playlens/content.js"', injected)
        self.assertEqual(inject_playlens_loader(injected), injected)

    def test_loss_window_uses_first_median_crossing(self) -> None:
        self.assertEqual(loss_window([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]), "10_to_20s")
        self.assertEqual(loss_window([0.1] * 7), "none")
        self.assertEqual(loss_window([0.6] * 7), "under_10s")

    def test_survival_probabilities_are_monotonic_and_convert_to_time(self) -> None:
        values = monotonic_probabilities([0.4, 0.2, 0.5, 0.45, 0.7, 0.8, 0.9])
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertGreaterEqual(estimated_remaining_seconds(values), 0)
        self.assertLessEqual(estimated_remaining_seconds(values), 60)

    def test_radial_pressure_increases_toward_stack_failure_boundary(self) -> None:
        def frame_with_block(radius_pixels: int) -> bytes:
            image = Image.new("RGB", (80, 80), "#b9bec1")
            draw = ImageDraw.Draw(image)
            center_x = 40 + radius_pixels
            draw.rectangle((center_x - 3, 37, center_x + 3, 43), fill="#e74c3c")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95)
            return output.getvalue()

        inner = extract_jpeg_features(frame_with_block(7), None, 0).vector
        outer = extract_jpeg_features(frame_with_block(15), None, 0).vector
        pressure = FEATURE_NAMES.index("radial_pressure")
        outer_occupancy = FEATURE_NAMES.index("outer_stack_occupancy")
        self.assertGreater(outer[pressure], inner[pressure])
        self.assertGreater(outer[outer_occupancy], inner[outer_occupancy])

    def test_predictor_rejects_a_stale_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "live.joblib"
            joblib.dump(
                {
                    "eligible_for_live": True,
                    "horizons_seconds": (5, 10, 15, 20, 30, 45, 60),
                    "feature_schema_version": "stale-schema",
                    "feature_count": len(FEATURE_NAMES),
                    "kind": "sklearn_personal_survival",
                },
                artifact_path,
            )
            predictor = Predictor(artifact_path)
            self.assertIsNone(predictor.artifact)
            self.assertEqual(FEATURE_SCHEMA_VERSION, "hextris-stack-v2")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_directory.name)
        self.client = TestClient(
            create_app(
                Settings(
                    data_dir=data_dir,
                    artifact_path=data_dir / "missing.joblib",
                    artifacts_dir=data_dir / "artifacts",
                )
            )
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def start_session(self) -> str:
        response = self.client.post(
            "/api/v1/sessions",
            json={
                "game": "hextris",
                "sourceUrl": "http://127.0.0.1:8000/",
                "canvasWidth": 800,
                "canvasHeight": 800,
                "overlayExpanded": True,
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["mode"], "collection")
        self.assertEqual(response.json()["collectionRunNumber"], 1)
        return response.json()["id"]

    def send_frame(self, session_id: str, timestamp: int, active_elapsed: int, step: int):
        return self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={
                "timestampMs": timestamp,
                "activeElapsedMs": active_elapsed,
                "imageDataUrl": image_data_url(step),
                "observedMotion": 3 + step % 4,
            },
        )

    def test_collection_lifecycle_pauses_without_finishing_or_advancing(self) -> None:
        self.assertEqual(self.client.get("/health").json()["modelSource"], "collection_only")
        session_id = self.start_session()
        base = int(time.time() * 1_000)
        for index in range(20):
            response = self.send_frame(session_id, base + index * 250, index * 250, index)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "collecting")

        paused = self.client.post(
            f"/api/v1/sessions/{session_id}/lifecycle",
            json={"state": "paused", "timestampMs": base + 5_000, "activeElapsedMs": 4_750},
        )
        self.assertEqual(paused.status_code, 204)
        ignored = self.send_frame(session_id, base + 10_000, 9_750, 21)
        self.assertEqual(ignored.json()["status"], "paused")
        self.assertEqual(ignored.json()["activeElapsedMs"], 4_750)
        resumed = self.client.post(
            f"/api/v1/sessions/{session_id}/lifecycle",
            json={"state": "active", "timestampMs": base + 10_250, "activeElapsedMs": 4_750},
        )
        self.assertEqual(resumed.status_code, 204)
        for index in range(20, 40):
            active = 4_750 + (index - 19) * 250
            response = self.send_frame(session_id, base + 10_250 + (index - 20) * 250, active, index)
            self.assertEqual(response.status_code, 200)

        finished = self.client.post(
            f"/api/v1/sessions/{session_id}/finish",
            json={"reason": "game_over", "timestampMs": base + 15_250, "activeElapsedMs": 9_750},
        )
        self.assertEqual(finished.status_code, 200)
        self.assertTrue(finished.json()["usableForTraining"])
        self.assertEqual(finished.json()["usableRuns"], 1)
        self.assertGreaterEqual(finished.json()["captureCoverage"], 0.75)
        detail = self.client.get(f"/api/v1/sessions/{session_id}").json()
        self.assertEqual(detail["activeDurationMs"], 9_750)
        self.assertEqual([event["state"] for event in detail["lifecycleEvents"]], ["paused", "active"])
        self.assertEqual(detail["predictions"], [])
        status = self.client.get("/api/v1/ml/status").json()
        self.assertEqual(status["personalCollection"]["usableRuns"], 1)
        self.assertEqual(status["personalCollection"]["remainingRuns"], 49)
        self.assertEqual(status["personalCollection"]["cohortVersion"], "collection-v3")
        self.assertEqual(status["personalCollection"]["recentRuns"][0]["validFrameCount"], 40)

    def test_non_game_over_and_short_runs_are_excluded(self) -> None:
        session_id = self.start_session()
        base = int(time.time() * 1_000)
        for index in range(8):
            self.send_frame(session_id, base + index * 250, index * 250, index)
        finished = self.client.post(
            f"/api/v1/sessions/{session_id}/finish",
            json={"reason": "disarmed", "timestampMs": base + 2_000, "activeElapsedMs": 2_000},
        )
        self.assertFalse(finished.json()["usableForTraining"])
        self.assertIn("disarmed", finished.json()["qualityReason"])

    def test_instrumented_runs_are_smoke_test_data_only(self) -> None:
        started = self.client.post(
            "/api/v1/training/sessions",
            json={"source": "instrumented_hextris", "automated": True, "policy": "survivor", "seed": 7},
        )
        training_id = started.json()["id"]
        self.client.post(
            f"/api/v1/training/sessions/{training_id}/frames",
            json={
                "timestampMs": int(time.time() * 1_000),
                "imageDataUrl": image_data_url(1),
                "gameState": 1,
                "score": 30,
            },
        )
        self.client.post(f"/api/v1/training/sessions/{training_id}/finish")
        status = self.client.get("/api/v1/ml/status").json()
        self.assertEqual(status["collector"]["trainingUse"], "smoke_tests_only")
        self.assertEqual(status["personalCollection"]["usableRuns"], 0)

    def test_v3_prediction_contract_round_trips_through_sqlite(self) -> None:
        data_dir = Path(self.temp_directory.name)
        database = Database(data_dir / "prediction-contract.sqlite3")
        database.create_session(
            {
                "id": "session-v3",
                "game": "hextris",
                "source_url": "http://127.0.0.1:8000/",
                "started_at": "2026-08-29T00:00:00+00:00",
                "status": "active",
                "canvas_width": 800,
                "canvas_height": 800,
                "overlay_expanded_initial": 1,
                "model_source": "validated_personal_model",
                "collection_index": 51,
                "model_version": "personal-v3-r50",
            }
        )
        database.save_prediction(
            "session-v3",
            {
                "timestampMs": 1_000,
                "activeElapsedMs": 9_000,
                "status": "ready",
                "failureProbabilities": {str(value): 0.6 for value in (5, 10, 15, 20, 30, 45, 60)},
                "lossWindow": "under_10s",
                "certainty": "high",
                "estimatedRemainingSeconds": 6.0,
                "uncertaintyLowSeconds": 4.0,
                "uncertaintyHighSeconds": 9.0,
                "displayedValue": "LOSS IMMINENT - <10s",
                "modelSource": "validated_personal_model",
                "calibrated": True,
                "latencyMs": 12.0,
            },
        )
        prediction = database.get_session("session-v3")["predictions"][0]
        self.assertEqual(prediction["lossWindow"], "under_10s")
        self.assertEqual(prediction["certainty"], "high")
        self.assertEqual(prediction["failureProbabilities"]["60"], 0.6)

    def test_legacy_runs_are_preserved_but_do_not_count_toward_official_fifty(self) -> None:
        data_dir = Path(self.temp_directory.name)
        database = Database(data_dir / "legacy.sqlite3")
        database.create_session(
            {
                "id": "legacy-session",
                "game": "hextris",
                "source_url": "http://127.0.0.1:8000/",
                "started_at": "2026-08-29T00:00:00+00:00",
                "status": "active",
                "canvas_width": 800,
                "canvas_height": 800,
                "overlay_expanded_initial": 1,
                "model_source": "collection_only",
                "collection_index": 1,
                "model_version": "legacy-v2",
            }
        )
        database.finish_session(
            "legacy-session", "2026-08-29T00:02:00+00:00", "game_over", 120_000,
            True, "usable legacy run", 480, 480, 1.0,
        )
        database.exclude_legacy_from_official_cohort()
        self.assertEqual(database.usable_run_count(), 0)
        detail = database.get_session("legacy-session")
        self.assertFalse(detail["usableForTraining"])
        self.assertIn("official v3 cohort", detail["qualityReason"])

    def test_simultaneous_duplicate_controller_runs_are_both_quarantined(self) -> None:
        data_dir = Path(self.temp_directory.name)
        database = Database(data_dir / "duplicates.sqlite3")
        starts = {
            "clean": "2026-08-29T00:00:00+00:00",
            "duplicate-a": "2026-08-29T00:03:00.100000+00:00",
            "duplicate-b": "2026-08-29T00:03:00.230000+00:00",
        }
        for session_id, started_at in starts.items():
            database.create_session(
                {
                    "id": session_id,
                    "game": "hextris",
                    "source_url": "http://127.0.0.1:8000/",
                    "started_at": started_at,
                    "status": "active",
                    "canvas_width": 800,
                    "canvas_height": 800,
                    "overlay_expanded_initial": 1,
                    "model_source": "collection_only",
                    "collection_index": 1,
                    "model_version": "collection-v3",
                }
            )
            duration = 100_000 if session_id == "clean" else 101_000
            frames = 400 if session_id == "clean" else (402 if session_id.endswith("a") else 403)
            database.finish_session(
                session_id, "2026-08-29T00:05:00+00:00", "game_over",
                duration, True, "usable personal run", frames, 404, frames / 404,
            )

        self.assertEqual(database.exclude_duplicate_controller_runs(), ["duplicate-a", "duplicate-b"])
        self.assertEqual(database.usable_run_count(), 1)
        self.assertTrue(database.get_session("clean")["usableForTraining"])
        for session_id in ("duplicate-a", "duplicate-b"):
            detail = database.get_session(session_id)
            self.assertFalse(detail["usableForTraining"])
            self.assertEqual(detail["qualityReason"], "excluded: duplicate-controller capture")


class CheckpointTests(unittest.TestCase):
    def test_twenty_run_checkpoint_contains_database_manifest_and_session_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            artifacts_dir = Path(directory) / "artifacts"
            database = Database(data_dir / "playlens.sqlite3")
            for index in range(20):
                session_id = f"v3-{index:02d}"
                database.create_session(
                    {
                        "id": session_id,
                        "game": "hextris",
                        "source_url": "http://127.0.0.1:8000/",
                        "started_at": f"2026-08-29T00:{index:02d}:00+00:00",
                        "status": "active",
                        "canvas_width": 800,
                        "canvas_height": 800,
                        "overlay_expanded_initial": 1,
                        "model_source": "collection_only",
                        "collection_index": index + 1,
                        "model_version": "collection-v3",
                    }
                )
                database.finish_session(
                    session_id, f"2026-08-29T00:{index:02d}:30+00:00", "game_over",
                    30_000, True, "usable personal run", 120, 120, 1.0,
                )
                session_path = data_dir / "sessions" / session_id
                session_path.mkdir(parents=True)
                (session_path / "observations.jsonl").write_text("{}\n", encoding="utf-8")
            checkpoint = create_collection_checkpoint(data_dir, artifacts_dir, 20)
            self.assertIsNotNone(checkpoint)
            manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["usableRuns"], 20)
            self.assertEqual(len(manifest["sessions"]), 20)
            self.assertTrue((checkpoint / "playlens.sqlite3").exists())
            self.assertTrue((checkpoint / "session-data.tar").exists())

    def test_interrupted_milestone_job_is_rescheduled_on_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = TrainingCoordinator(root, root / "data", root / "artifacts", lambda: None)
            coordinator._write({"status": "running", "usableRuns": 20, "step": "playlens_ml.train_torch"})
            with patch.object(coordinator, "schedule", return_value=True) as schedule:
                self.assertTrue(coordinator.recover_interrupted(20))
                schedule.assert_called_once_with(20)
            job = json.loads(coordinator.path.read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "failed")
            self.assertIn("retrying automatically", job["error"])


if __name__ == "__main__":
    unittest.main()
