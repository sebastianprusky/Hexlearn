# Clean local Hextris target

The upstream Hextris repository still exists, but its hosted game redirects to
an unavailable custom domain. PlayLens therefore uses a clean, local copy for
repeatable playtests.

This directory and the instrumented training copy are intentionally separate:

- `game/hextris` on port `8000`: clean gameplay target; the extension is allowed.
- `instrumentation/hextris` on port `8001`: objective-label collection; the
  extension is not allowed.

## First-time setup

From `playlens/game`:

```bash
git clone --branch gh-pages --depth 1 https://github.com/Hextris/hextris.git hextris
python3 prepare_clean_copy.py hextris
```

The preparation step removes legacy analytics, advertising, remote-score, and
other external runtime requests. It does not add telemetry or change gameplay
logic. Its local-only content policy permits `unsafe-eval` because Hextris's
bundled JSONfn dependency requires it to restore the initial game state; all
scripts and other runtime assets remain restricted to the local origin.

## Run

```bash
cd ../..
PYTHONPATH=service .venv/bin/uvicorn playlens_api.combined:app --host 127.0.0.1 --port 8000
```

This serves the game and PlayLens API together. Then open
`http://127.0.0.1:8000` in Chrome.
