# Hexlearn

Hexlearn contains PlayLens, a local-first Hextris playtesting prototype that learns a personal
loss-window forecast from canvas pixels. It captures clean runs first, trains
scikit-learn and PyTorch survival models locally, and only shows forecasts after
the visual model beats elapsed-time and average-duration shortcuts on ten locked
future runs.

The live states are intentionally simple:

- `CAPTURING RUN 12 / 50` while building the first personal dataset.
- `PAUSED` while Hextris or the browser tab is paused.
- `RUN ACCEPTED · 12 / 50 usable` or a specific exclusion reason at game over.
- `NO FAILURE SIGNAL` or `LOSS LIKELY IN …` after a model is promoted.

PlayLens never claims to detect boredom, emotion, attention, or player intent.
Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) before changing product or ML
direction; it is the living source of truth.

## Project map

```text
hexlearn/
  extension/        Manifest V3 capture, lifecycle detection, and overlay
  game/             Clean local Hextris playtest target
  service/          FastAPI capture, inference, sessions, and training jobs
  ml/               Personal dataset, baselines, PyTorch, and evaluation
  dashboard/        Next.js session reports and Model Lab
  instrumentation/  Bot telemetry retained for pipeline smoke tests only
```

## Start the game and local API

```bash
cd playlens
python3 -m venv .venv
.venv/bin/python -m pip install -r service/requirements.txt
PYTHONPATH=service .venv/bin/uvicorn playlens_api.combined:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The local server injects and arms the PlayLens
capture controller directly, so collection works even when Chrome does not
activate the unpacked extension. Press Hextris's Play button and play normally.
If the ignored clean game copy is missing, follow `game/README.md` once.

## Reload the extension

The extension remains supported but is optional for the clean local game. The
local page reserves controller ownership before Chrome's cached extension can
claim it, and a shared DOM lock prevents two capture sessions.

1. Visit `chrome://extensions`.
2. Enable Developer mode.
3. Load `playlens/extension`, or click **Reload** if it is already installed.
4. Return to `http://127.0.0.1:8000` and refresh the page.
5. Click **Enable PlayLens**, then play normally.

Version `0.6.4` uses the Play-button interaction and visible Hextris play controls as its primary start and
game-over signals and canvas motion only as a start fallback. It does not treat
stillness as game over. P, Space, the Hextris
pause control, and a hidden tab suspend the active clock and monitoring while
keeping the same session open. The visible Hextris game-over screen is required
to complete a training run.

## Start the dashboard

```bash
cd playlens/dashboard
./node_modules/.bin/next dev --hostname 127.0.0.1 --port 3001
```

Open `http://localhost:3001`; Model Lab is at `/model`. It shows usable and
excluded runs, the 40/10 split, local training jobs, model gates, and why a
candidate remains held back.

## Personal training workflow

- Only extension 0.6.4 `collection-v3` runs count toward the official fifty.
  Legacy sessions remain visible but are archived outside the cohort.
- Runs must end at confirmed game over, contain at least eight active seconds,
  retain at least 75% expected frame coverage, and have enough valid frames.
- Runs 1–40 form the development set. Runs 41–50 are the locked future test.
- Automatic local jobs run at 20, 25, 30, 35, 40, and 50 usable runs.
- Complete local checkpoints containing the cohort database, manifest, frames,
  and recordings are created at runs 20, 40, and 50. Interrupted milestone jobs
  retry automatically when the API restarts.
- The main visual models never receive score, elapsed time, average duration, or
  Hextris state. Those values exist only in shortcut baselines and labels.
- Bot runs never enter personal fitting, selection, or evaluation.
- ML frames remain at 4 FPS. Session replay recording uses 15 FPS at 900 Kbps
  and reusable capture canvases to minimize gameplay overhead.

To rebuild the current personal candidate manually:

```bash
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.build_dataset
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.train_baseline
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.train_torch
```

The extension stays in collection mode unless the locked-test promotion gates
all pass. A failed candidate is evidence, not a reason to expose an inaccurate
forecast.

Before bulk collection, complete three pilot runs: one normal run, one paused
inside Hextris, and one paused by hiding the tab. Model Lab should mark each run
accepted and show its frame coverage, pause count, and recording status.

To verify both training implementations without touching personal data:

```bash
make smoke-ml
```

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/playlens-pycache PYTHONPATH=service:ml .venv/bin/python -m unittest discover -s service/tests
node --test extension/core.test.js
node --check extension/background.js
node --check extension/content.js
(cd dashboard && ./node_modules/.bin/eslint .)
(cd dashboard && ./node_modules/.bin/next build)
```

All captures, recordings, model artifacts, and reports stay under local ignored
directories until deleted by the user.
