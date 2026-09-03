# Instrumented local Hextris copy

This copy exists only to exercise the capture and training pipeline. It may read
Hextris's `gameState` and `score`; the production extension on the clean local
copy does not.

## Setup

```bash
cd playlens/instrumentation
git clone --branch gh-pages --depth 1 https://github.com/Hextris/hextris.git hextris
python3 ../game/prepare_clean_copy.py hextris
python3 install_bridge.py hextris
cd hextris
python3 -m http.server 8001 --bind 127.0.0.1
```

With the combined Hexlearn server running on port `8000`, open
`http://127.0.0.1:8001`. Each run creates a
local smoke-test session containing 4 FPS canvas frames and matching objective
telemetry. Add `?playlensBot=1&runs=25&seed=7` to make the local policy bot
start, play, save, and restart until it has collected 25 complete runs. It
cycles reproducibly through survivor, balanced, explorer, and adversarial
policies so the dataset contains varied survival times and board trajectories.
Bot data is never used to fit, select, personalize, or evaluate the v3 model.
It exists only to verify that frame capture and labeling code still run.

Automated runs remain tagged and stored separately. Only quality-gated
clean-copy personal runs enter the 40-development/10-locked-test pipeline.
