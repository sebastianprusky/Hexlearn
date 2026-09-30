# Clear measurement: a focused decision before another model change

## Goal

The project goal is unchanged: useful warnings at least 15 seconds before loss,
beating all named time baselines. This experiment answers one prerequisite:
**Can pixels identify actual clears reliably enough to test their predictive value?**

The direction is measurement → fixed model comparison → independent future test.
A failure at one stage stops progress to the next. More parameter search cannot
substitute for reliable measurements or a genuinely independent forecast test.

## Hypothesis and decision rules

A broad white flash on a previously colored stack face is evidence of clearing.
Simple stack-height decreases also occur during rotation and animation, so they
are unreliable event labels. Central score pixels and the outer timer are outside
the flash sampling region; this detector does not read text or internal state.

The original 12 reviewed clips are development material. A development check
found short flashes missed at 4 FPS, so the protocol was amended **before freeze**
to test one fixed 12-FPS version on existing video. The initial protocol and the
reason for the change are retained. No new gameplay or live capture is involved.

Twelve new six-second clips are selected deterministically from runs 1–30. Detector,
preparation, metric code, and protocol hashes are frozen before those clips are
prepared. Their labels are written while detector output is hidden. They are new
segments from already used development games, not independent generalization data.

Pass requires all of:

- At least 80% recall and 90% precision.
- At most one false clear per reviewed minute.
- At least eight confirmable clears and twenty negative seconds.
- At least 95% supported frames and at most 10% ambiguous frames.

Event matches are one-to-one. A one-frame annotation interval handles the time
between the preceding frame and first visible clear. Up to 0.5s causal detection
delay is permitted. The first and last 0.5s are censored. Ambiguity is explicit;
never/always event controls cannot pass merely because of their definitions.

If the detector passes, test its signal in one fixed forecasting ablation using
the retained compact representation, original folds and original forecast gates.
If it fails, do not train another forecasting model or tune to these validation
clips. Record the failure and the measurement constraint that it establishes.

## Reproduction

Use `.venv/bin/python` with `PYTHONPATH=service:ml` from `playlens`. Personal videos
and labels remain local. Artifacts are in `artifacts/experiments/clear-measurement-v1`.

Before freeze:

```sh
python -m playlens_ml.clear_measurement.data development --fps 12
python -m playlens_ml.clear_measurement.data freeze
```

Then:

```sh
python -m playlens_ml.clear_measurement.data validation
# Review raw contact sheets and write validation/labels.json before predictions.
python -m playlens_ml.clear_measurement.evaluate
```

Freeze and validation results cannot be silently overwritten. Inspect the saved
results to repeat an analysis. A new detector requires a new experiment and new
measurement validation material. Assistant labels are not independent ground truth.

## Completed measurement and fixed next stage

The reserved check passed: 23/26 confirmable clears, zero false detections,
100% supported scored frames, and 44.67 negative seconds. The detector and labels
remain frozen. This is evidence to test clear features, not evidence of useful
loss forecasting.

The one permitted ablation adds clear counts and event-confidence sums over
1, 2, 4, and 8 seconds to the retained 80-feature representation. It compares
with/without clears in both existing fixed model families on identical windows,
against all four named time baselines. The original screening gates still apply.
The source commitment is saved in `ablation-protocol.json` before scoring.

```sh
python -m playlens_ml.clear_measurement.ablation extract --workers 4
python -m playlens_ml.clear_measurement.ablation build
python -m playlens_ml.clear_measurement.ablation evaluate
python -m playlens_ml.clear_measurement.report
```

Extraction caches each run once and checks its video hash. Window exclusions are
recorded, including unsupported geometry and resets inside the full eight-second
history. A half-second feature delay is preserved. Completed forecasting results
cannot be overwritten by the evaluation command. If no candidate qualifies, stop
this experiment without expanding model search or requesting more runs.

## Outcome

The fixed comparison is complete. Adding clear features improved the tree by
only 0.00032 Brier and 0.036 seconds of last-minute error; descriptive paired
intervals include no improvement. Its Brier/error are 0.06660/11.140s, versus
best time-baseline values 0.06171/10.610s. Both clear candidates fail six of
eight screening gates. Worst evaluation-game window coverage is 78.1%.

Decision: close this experiment and improve extraction first. The clear detector
remains an experimental measurement tool. It does not justify a new forecasting
representation, prospective gameplay test, or broader parameter search.
The report identifies gray-boundary failures during animation/occlusion as the
next concrete measurement problem. All 64 focused tests pass.
