# Native replay forecasting: compact development candidate

**Status:** best development result so far; no candidate passes all screening
requirements. This is offline research, not a live or prospective model release.
The consolidated report is `artifacts/experiments/board-compact-v1/REPORT.md`.

## Representations tested

1. `board_v2`: native replay extraction, synchronization and diagnostic review.
2. `board_v3`: recovery of crowded-board geometry and tall outer colors.
3. `board_invariant`: rotation-invariant crowding and color summaries.
4. `board_dynamics`: adjacent-frame incoming-speed and recent activity estimates.
5. `board_compact`: speed only; speed plus crowding; speed plus crowding/activity.

All use the same fixed regularized logistic and shallow boosted-tree settings.
There were 20 distinct development feature/model pairs, with no hyperparameter
search. All scored candidates use identical anchors, labels, valid windows and
chronological folds. Runs 31–40 are excluded. The current best representation has
80 features and uses logistic regression with separate probability calibration.

`board_stable` is a rejected core-ratio normalization experiment. It reduced
continuous coverage and was stopped before model fitting. Keep its artifacts as
negative evidence; do not substitute it into the current best pipeline.

## Reproduce

From `playlens`, set `PYTHONPATH=service:ml`. Install the local decoder dependency
and run extraction/review as documented in `../board_v2/README.md` and
`../board_v3/README.md`. Complete the exact-revision visual checkpoint before any
real model evaluation. Source recordings and review decisions remain private
local artifacts, so cloning source alone does not reproduce personal results.

After native v3 windows are built:

```sh
.venv/bin/python -m playlens_ml.board_v3.evaluate
.venv/bin/python -m playlens_ml.board_invariant.data
.venv/bin/python -m playlens_ml.board_invariant.evaluate
.venv/bin/python -m playlens_ml.board_dynamics.data
.venv/bin/python -m playlens_ml.board_dynamics.evaluate
.venv/bin/python -m playlens_ml.board_compact.data
.venv/bin/python -m playlens_ml.board_compact.evaluate
.venv/bin/python -m playlens_ml.board_compact.report
.venv/bin/python -m unittest discover -s service/tests -p 'test_board*.py' -v
```

Each stage writes into a separate experiment directory. `results.json` includes
all folds, per-game metrics, seven horizons, calibration, alert controls,
descriptive bootstrap intervals, unsupported coverage and screening decisions.
The compact experiment also saves `development_predictions.npz` for diagnosis.

## Interpretation boundaries

- Review labels are conservative assistant inspection, not independent ground
  truth. Five of sixty stills remain ambiguous/incorrect.
- Four-FPS frames are delayed by 0.5 seconds using empirically checked replay
  alignment; no score, clock value, or internal game state enters visual features.
- Incoming identity, speed, landings and clears remain estimates. Tracking-pair
  coverage is included explicitly. Rotation frames are excluded from speed pairs.
- Alert rates use observed active minutes; coverage is screened separately.
  Timely recall alone can reward repeated onsets after gaps, so false alerts,
  warning time, controls and extraction coverage must be read together.
- The best development average is not proof of generalization. No candidate was
  frozen for prospective testing and no live forecast was enabled.
