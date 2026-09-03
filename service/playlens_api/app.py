from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

from .database import Database
from .features import InvalidFrame, extract_frame_features
from .inference import Predictor, RuntimeSession
from .insights import derive_insights
from .settings import PLAYLENS_ROOT, Settings
from .training_jobs import TrainingCoordinator
from .training_status import build_training_status, legacy_run_quality


class SessionStart(BaseModel):
    game: Literal["hextris"]
    sourceUrl: HttpUrl
    canvasWidth: int = Field(gt=0, le=8192)
    canvasHeight: int = Field(gt=0, le=8192)
    overlayExpanded: bool


class FrameInput(BaseModel):
    timestampMs: int = Field(gt=0)
    activeElapsedMs: int = Field(ge=0)
    imageDataUrl: str
    observedMotion: float = Field(ge=0, le=255)


class FinishInput(BaseModel):
    reason: Literal["game_over", "disarmed", "navigation", "capture_error"]
    timestampMs: int = Field(gt=0)
    activeElapsedMs: int = Field(ge=0)


class LifecycleInput(BaseModel):
    state: Literal["active", "paused"]
    timestampMs: int = Field(gt=0)
    activeElapsedMs: int = Field(ge=0)


class OverlayInput(BaseModel):
    expanded: bool
    timestampMs: int = Field(gt=0)


class TrainingSessionStart(BaseModel):
    source: Literal["instrumented_hextris"]
    automated: bool = False
    policy: Optional[Literal["survivor", "balanced", "explorer", "adversarial"]] = None
    seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)


class TrainingFrameInput(BaseModel):
    timestampMs: int = Field(gt=0)
    imageDataUrl: str
    gameState: int
    score: int = Field(ge=0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.data_dir / "playlens.sqlite3")
    for collection_index, candidate in enumerate(database.legacy_candidates(), start=1):
        observations_path = settings.data_dir / "sessions" / candidate["id"] / "observations.jsonl"
        legacy_usable, legacy_reason = legacy_run_quality(observations_path)
        timestamps = []
        if observations_path.exists():
            for line in observations_path.read_text(encoding="utf-8").splitlines():
                try:
                    timestamps.append(int(json.loads(line)["timestampMs"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        legacy_duration = max(timestamps) - min(timestamps) if timestamps else 0
        database.backfill_legacy_quality(
            candidate["id"], legacy_duration, legacy_usable, legacy_reason, collection_index
        )
    database.exclude_legacy_from_official_cohort()
    database.exclude_duplicate_controller_runs()
    predictor = Predictor(settings.artifact_path, settings.personalization_path)
    training_coordinator = TrainingCoordinator(
        PLAYLENS_ROOT, settings.data_dir, settings.artifacts_dir, predictor.reload
    )
    training_coordinator.recover_interrupted(database.usable_run_count())
    runtimes: dict[str, RuntimeSession] = {}

    app = FastAPI(title="Hexlearn local API", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.database = database
    app.state.predictor = predictor
    app.state.runtimes = runtimes
    app.state.training_coordinator = training_coordinator

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "modelSource": predictor.model_source,
            "calibrated": predictor.calibrated,
        }

    @app.get("/api/v1/ml/status")
    def ml_status() -> dict:
        return build_training_status(
            settings.data_dir,
            settings.artifacts_dir,
            predictor.model_source,
            predictor.calibrated,
        )

    @app.post("/api/v1/sessions", status_code=201)
    def start_session(payload: SessionStart) -> dict:
        session_id = str(uuid.uuid4())
        collection_run_number = database.usable_run_count() + 1
        database.create_session(
            {
                "id": session_id,
                "game": payload.game,
                "source_url": str(payload.sourceUrl),
                "started_at": utc_now(),
                "status": "active",
                "canvas_width": payload.canvasWidth,
                "canvas_height": payload.canvasHeight,
                "overlay_expanded_initial": int(payload.overlayExpanded),
                "model_source": predictor.model_source,
                "collection_index": collection_run_number,
                "model_version": predictor.model_version,
            }
        )
        runtimes[session_id] = RuntimeSession(
            started_timestamp_ms=int(time.time() * 1_000),
            collection_run_number=collection_run_number,
        )
        (settings.data_dir / "sessions" / session_id / "frames").mkdir(parents=True, exist_ok=True)
        return {
            "id": session_id,
            "status": "active",
            "mode": "live" if predictor.artifact is not None else "collection",
            "collectionRunNumber": collection_run_number,
            "collectionTarget": 50,
            "modelVersion": predictor.model_version,
        }

    @app.post("/api/v1/sessions/{session_id}/frames")
    def add_frame(session_id: str, payload: FrameInput) -> dict:
        runtime = runtimes.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Active session not found")
        if runtime.paused:
            return {
                "timestampMs": payload.timestampMs,
                "activeElapsedMs": runtime.active_elapsed_ms,
                "status": "paused",
                "modelSource": predictor.model_source,
                "calibrated": predictor.calibrated,
            }
        try:
            frame = extract_frame_features(payload.imageDataUrl, runtime.previous_gray, payload.observedMotion)
        except InvalidFrame as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        runtime.previous_gray = frame.gray
        runtime.active_elapsed_ms = max(runtime.active_elapsed_ms, payload.activeElapsedMs)
        runtime.frames.append((payload.activeElapsedMs, frame.vector))
        runtime.valid_frames += 1
        if payload.activeElapsedMs - runtime.last_saved_active_ms >= 240:
            session_path = settings.data_dir / "sessions" / session_id
            frame_path = session_path / "frames" / f"{payload.timestampMs}.jpg"
            frame_path.write_bytes(frame.jpeg_bytes)
            with (session_path / "observations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "timestampMs": payload.timestampMs,
                            "activeElapsedMs": payload.activeElapsedMs,
                            "observedMotion": payload.observedMotion,
                        }
                    )
                    + "\n"
                )
            runtime.last_saved_active_ms = payload.activeElapsedMs
        prediction = predictor.prediction(runtime, payload.timestampMs, payload.activeElapsedMs)
        database.save_prediction(session_id, prediction)
        return prediction

    @app.post("/api/v1/sessions/{session_id}/lifecycle", status_code=204)
    def lifecycle_event(session_id: str, payload: LifecycleInput) -> Response:
        runtime = runtimes.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Active session not found")
        if payload.state == "paused":
            runtime.suspend(payload.activeElapsedMs)
        else:
            runtime.resume(payload.activeElapsedMs)
        database.add_lifecycle_event(
            session_id, payload.timestampMs, payload.activeElapsedMs, payload.state
        )
        return Response(status_code=204)

    @app.post("/api/v1/sessions/{session_id}/finish")
    def finish_session(session_id: str, payload: FinishInput) -> dict:
        runtime = runtimes.pop(session_id, None)
        active_duration_ms = max(payload.activeElapsedMs, runtime.active_elapsed_ms if runtime else 0)
        valid_frames = runtime.valid_frames if runtime else 0
        expected_frames = max(1, round(active_duration_ms / 250))
        coverage = valid_frames / expected_frames
        if payload.reason != "game_over":
            quality_reason = f"excluded: {payload.reason}"
        elif active_duration_ms < 8_000:
            quality_reason = "excluded: fewer than eight active seconds"
        elif valid_frames < 28:
            quality_reason = "excluded: insufficient valid frames"
        elif coverage < 0.75:
            quality_reason = "excluded: capture coverage below 75%"
        else:
            quality_reason = "usable personal run"
        usable = quality_reason == "usable personal run"
        if not database.finish_session(
            session_id,
            utc_now(),
            payload.reason,
            active_duration_ms,
            usable,
            quality_reason,
            valid_frames,
            expected_frames,
            coverage,
        ):
            raise HTTPException(status_code=404, detail="Session not found")
        final = runtime.last_prediction if runtime and runtime.last_prediction else {
            "timestampMs": int(time.time() * 1_000),
            "modelSource": predictor.model_source,
            "calibrated": predictor.calibrated,
            "activeElapsedMs": active_duration_ms,
        }
        usable_runs = database.usable_run_count()
        if usable:
            training_coordinator.schedule(usable_runs)
        return {
            **final,
            "status": "game_over",
            "sessionId": session_id,
            "usableForTraining": usable,
            "qualityReason": quality_reason,
            "usableRuns": usable_runs,
            "collectionTarget": 50,
            "validFrameCount": valid_frames,
            "expectedFrameCount": expected_frames,
            "captureCoverage": coverage,
        }

    @app.post("/api/v1/sessions/{session_id}/overlay", status_code=204)
    def overlay_event(session_id: str, payload: OverlayInput) -> Response:
        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        database.add_overlay_event(session_id, payload.timestampMs, payload.expanded)
        return Response(status_code=204)

    @app.post("/api/v1/sessions/{session_id}/recording", status_code=204)
    async def save_recording(session_id: str, request: Request) -> Response:
        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        body = await request.body()
        if len(body) > 250 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Recording exceeds 250 MB")
        recording_path = settings.data_dir / "sessions" / session_id / "gameplay.webm"
        recording_path.write_bytes(body)
        database.set_recording(session_id, str(recording_path))
        return Response(status_code=204)

    @app.get("/api/v1/sessions")
    def list_sessions() -> list[dict]:
        return database.list_sessions()

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session["insights"] = derive_insights(session["predictions"])
        return session

    @app.get("/api/v1/sessions/{session_id}/recording")
    def get_recording(session_id: str) -> FileResponse:
        value = database.recording_path(session_id)
        if value is None or not Path(value).exists():
            raise HTTPException(status_code=404, detail="Recording not found")
        return FileResponse(value, media_type="video/webm", filename=f"playlens-{session_id}.webm")

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        runtimes.pop(session_id, None)
        session_dir = settings.data_dir / "sessions" / session_id
        if not database.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        if session_dir.exists():
            shutil.rmtree(session_dir)
        return Response(status_code=204)

    @app.post("/api/v1/training/sessions", status_code=201)
    def start_training_session(payload: TrainingSessionStart) -> dict[str, str]:
        training_id = f"training-{uuid.uuid4()}"
        directory = settings.data_dir / "training" / training_id
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "id": training_id,
                    "startedAt": utc_now(),
                    "automated": payload.automated,
                    "policy": payload.policy,
                    "seed": payload.seed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"id": training_id, "status": "active"}

    @app.post("/api/v1/training/sessions/{training_id}/frames", status_code=204)
    def add_training_frame(training_id: str, payload: TrainingFrameInput) -> Response:
        directory = settings.data_dir / "training" / training_id
        if not directory.exists() or not training_id.startswith("training-"):
            raise HTTPException(status_code=404, detail="Training session not found")
        try:
            frame = extract_frame_features(payload.imageDataUrl, None, 0)
        except InvalidFrame as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        (directory / "frames" / f"{payload.timestampMs}.jpg").write_bytes(frame.jpeg_bytes)
        with (directory / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                f'{{"timestampMs":{payload.timestampMs},"gameState":{payload.gameState},"score":{payload.score}}}\n'
            )
        return Response(status_code=204)

    @app.post("/api/v1/training/sessions/{training_id}/finish", status_code=204)
    def finish_training_session(training_id: str) -> Response:
        directory = settings.data_dir / "training" / training_id
        if not directory.exists() or not training_id.startswith("training-"):
            raise HTTPException(status_code=404, detail="Training session not found")
        (directory / "complete").touch()
        return Response(status_code=204)

    return app


app = create_app()
