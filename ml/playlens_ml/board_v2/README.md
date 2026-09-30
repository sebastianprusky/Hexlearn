# Native replay board experiment

This isolated offline experiment uses only the original development runs 1–30.
Replays contain the canvas at 2940×1602. Extraction decodes them at 960×524,
retaining substantially more board detail than the original 160×160 captures.
No score, elapsed-time feature, game state, production schema, or live model is used.

## Reproduce

From `playlens`, set `PYTHONPATH=service:ml` for these commands:

```sh
.venv/bin/python -m pip install --target artifacts/experiments/board-features-v2/deps imageio-ffmpeg==0.6.0
.venv/bin/python -m playlens_ml.board_v2.review
.venv/bin/python -m playlens_ml.board_v2.audit robustness
.venv/bin/python -m playlens_ml.board_v2.audit clips
.venv/bin/python -m playlens_ml.board_v2.alignment
.venv/bin/python -m playlens_ml.board_v2.data --run 1
# Repeat --run for 2 through 30. Independent runs may execute concurrently.
.venv/bin/python -m playlens_ml.board_v2.data --build
.venv/bin/python -m playlens_ml.board_v2.evaluate
.venv/bin/python -m playlens_ml.board_v2.report
```

The exact extractor requires explicit review decisions at
`artifacts/experiments/board-features-v2/review/decisions.json`. Review all 60
stills and twelve clips; ambiguous assignments count as failures. Evaluation
refuses to fit unless the review and transformation checks pass. The checked-in
code never generates passing decisions automatically.

## Feature and timing contract

- Detect the outer gray hexagon from visible fragments; estimate orientation
  from the central dark hexagon without reading its score.
- Sample six rotating faces, preserve four block colors, distinguish disconnected
  blocks from attached stacks, and exclude the decorative outer timer band.
- Base features: six heights, incoming distances/gaps/confidence, outer colors,
  and adjacent color matches. Incoming blocks use fixed screen lanes; distances
  during rotation are approximate and require further validation.
- Temporal features compare **sorted height and incoming-distance distributions**
  over 1/2/4/8 seconds. They do not claim to track physical blocks or count true
  clears. Confidence is conservatively reduced. Sorting prevents a pure lane
  permutation from becoming apparent growth or clearing.
- Decode at four frames per second and delay feature availability by 0.5 seconds.
  This exceeds observed alignment offsets plus a codec/sampling margin. Additional
  image matches cover 10%/90% of each game and points after recorded resumes.
  This is empirical synchronization, not a hardware timestamp guarantee.
- Reset eight-second histories after unsupported geometry, changes in scale or
  center, pauses, and capture gaps. Keep every original eligible prediction anchor
  with an explicit validity flag and exclusion reason.
- Per-run caches include source-video hashes and extractor revisions. Originals
  and v1 artifacts remain separate. Runs 31–40 are excluded from all selection.

## Evaluation

Reuse the three chronological fitting/calibration/evaluation folds and six fixed
candidates from v1. Fit calibrators only on calibration games. Compare against
four time baselines on common windows, score games equally, and retain baseline
results across all eligible windows. Coverage is screened separately on all
15 evaluation games. The persistent alert definition is unchanged from v1.

A development win requires a frozen candidate and a new prospective test. Repeated
extractor/model changes on these same runs cannot establish generalization.
