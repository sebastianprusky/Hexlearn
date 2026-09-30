# Native replay experiment with crowded-board recovery

This revision keeps the original native extractor for ordinary boards. For
unsupported geometry or stacks taller than 0.84 board radii, it uses the visible
dark core center, removes small disconnected gray artifacts, and fits the
remaining outer gray edges. Stack sampling extends past the timer radius so tall
outer layers are not truncated.

All inputs remain pixels. Core score text is not read. No API, extension, or live
forecast behavior changes. Runs 31–40 remain excluded.

## Reproduce

First run native extraction and alignment as described in `../board_v2/README.md`.
Then, with `PYTHONPATH=service:ml`:

```sh
.venv/bin/python -m playlens_ml.board_v3.review
.venv/bin/python -m playlens_ml.board_v3.data --run 1
# Repeat --run for 2 through 30; independent runs can execute concurrently.
.venv/bin/python -m playlens_ml.board_v3.data --build
.venv/bin/python -m playlens_ml.board_v3.evaluate
.venv/bin/python -m playlens_ml.board_v3.report
```

Explicit review decisions must match this exact extractor revision. Unchanged
images/outputs retain their prior visual review; changed stills and clip frames
are reinspected. Twelve additional crowded-board examples check the fallback.
None of these development reviews is an independent accuracy benchmark.

The cache reuses primary results only where the routing predicate chooses the
primary extractor. Other frames are decoded with the same full-stream four-FPS
filter as the original cache; seeking is avoided. Source-video hashes, source
extractor hashes, repair counts, and schema versions are retained. The new
feature history uses only current and preceding frames and the same conservative
half-second timing delay.

The model comparison remains the six fixed feature-group/model-family pairs,
with separate calibration games and four time baselines. A failure of coverage
or any other original screening requirement prevents candidate selection.
