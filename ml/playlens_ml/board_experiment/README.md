# Offline board-feature experiment v1

This package is an experimental reader of existing development captures. It never
modifies production features, models, recordings, the API, or dashboard. Runs
31–40 never enter feature selection, fitting, calibration, or scoring.

From `playlens`:

```sh
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment review
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment robustness
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment extract
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment checkpoint
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment evaluate
PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.board_experiment report
```

Default output: `artifacts/experiments/board-features-v1`. `--out` can name another
folder inside `artifacts/experiments`; production dataset/model paths are refused.

## Review gate

`review` creates 60 fixed image overlays (one-third/two-thirds anchors of each
run), six contact sheets, and twelve fixed four-second clips with sampled
storyboards. It writes `review/selection.json`, but never invents review decisions.
The reviewer writes `review/decisions.json` containing the exact `extractorHash`,
60 `{id, geometryCorrect, colorsCorrect, notes}` frame records, and twelve
`{id, reviewed, events, notes}` clip records. Unknown/uncertain frames count as
failures. The supported event names are `rotation`, `stacking`, `incoming`,
`clear`; collectively the clips must cover all four. Human/assistant review
provenance and sampling method should be recorded explicitly.

At least 54 frames must have correct geometry and all visible outermost lane
colors. `evaluate` checks this gate itself and refuses to fit when it fails.
Changing extractor code or source dataset invalidates the review. Review values
are manually judged evidence, not an independent annotation benchmark.

## Features and history

`hextris-board-experiment-v1` uses 24 geometry features (height, incoming gap,
incoming radius, confidence for each lane), 30 color features (four top-color
indicators per lane and six adjacent matches), and 96 temporal features (growth,
apparent clears, incoming movement, confidence per lane at 1/2/4/8 seconds).
Incoming/clear values are heuristic estimates; they do not recover game state.
Unknown top colors have no one-hot bit set. Missing incoming blocks use the
outer observation limit, 2.1 board apothems, as a documented sentinel.

The gray board is located using the largest gray connected component and six
support lines. This first prototype uses narrow lane sectors and radial color
connectivity; it is susceptible to rotating and thin/JPEG-aliased layers. The
center and timer band are masked. No global brightness, score, wall time, or run
age enters the visual feature arrays.

History resets on invalid geometry, more than 8% geometry change, capture gaps
over 600 ms active / 800 ms wall time, and recorded resumes. Eight seconds and
at least 28 continuous valid frames are required. Every original eligible
anchor remains in the output, including invalid rows with an exclusion reason.

## Comparisons

Six candidates: geometry, geometry+colors, all features; each paired with
regularized logistic regression and shallow histogram gradient boosting. Fixed
settings match the preceding audit. Sigmoid calibration uses only the fold's
calibration games. Four references: average duration, conditional empirical
duration, unweighted elapsed logistic, elapsed tree. References and candidates
use identical fitting/calibration and common evaluation windows. Reference
scores on all eligible evaluation windows are also reported.

Chronological folds: 1–9 / 10–15 / 16–20; 1–14 / 15–20 / 21–25;
1–19 / 20–25 / 26–30. Every evaluation game receives equal weight. Bootstrap
intervals resample paired games 5,000 times; overlapping training histories and
selection make these descriptive, not fresh confirmation.

Alerts use three consecutive one-second predictions above .5 for onset and
three below .5 for offset. Gaps/segment changes reset them. Exactly .5 does not
advance either counter. Timely onset means 15–60 seconds before loss; >60 seconds
is a false alarm. Predictions are sampled by active-second buckets, and reset
when the observed interval exceeds 1.6 seconds. Rates and precision are reported
per game then averaged, with counts/exposure retained for alternative summaries.

The eight screening gates are implemented in `metrics.screen`. A qualifying
candidate is refit on runs 1–24 and calibrated on 25–30, saved only here with
`eligible_for_live=False`, and hashed. No test set is automatically collected.

## Outputs

- `frame_cache.sqlite3`: feature cache keyed by source content and extractor hash.
- `windows.npz`: visual arrays, original labels, IDs/timestamps, validity/reasons,
  history segment IDs, feature names and schema.
- `manifest.json`: source hashes, per-game coverage, exclusions and frame hashes.
- `review/`, `checkpoint.json`, `robustness.json`: extraction evidence.
- `results.json`: only after a passed checkpoint and successful evaluation.
- `REPORT.md`, `decision.json`: the result, including early extraction failure.

Tests:

```sh
PYTHONPATH=service:ml .venv/bin/python -m unittest discover -s service/tests -p 'test_board_experiment.py'
```
