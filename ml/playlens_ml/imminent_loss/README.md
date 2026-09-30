# Imminent loss v1

**Closed at the timing gate. No models trained.**

The approved experiment targets pixel-only warnings 2–5 active seconds before
loss. Five-second prediction is primary; three-second prediction is secondary.
Runs 1–30 are the only development cohort. Historical runs 31–40 are excluded.
This package is offline and does not modify production capture or inference.

## Finding

Run 5 has a 0.342-second gap between the last clearly active image and first
visible loss transition, exceeding the required 0.25 seconds. Runs 14 and 17
end before a visible loss transition. Full-stream decoding reproduces all
ending pixels from the seek-based review, ruling out seeking and frame-rate
conversion as explanations. See [RESULTS.md](RESULTS.md).

Score disappearance helps identify the outcome transition during annotation;
it is never proposed as a model feature. The center score remains visible in
the final saved images of runs 14 and 17. The saved video is a canvas recording,
so a later HTML game-over overlay cannot supply the missing visual boundary.
Review judgments were made by the assistant and are not independent ground truth.

The gate stopped work before active-time alignment, the 90-frame extraction
review, dataset assembly, training, or warning evaluation. Those stages have
not been implemented or claimed complete. No difficult games were removed.
The experiment does not request new gameplay or another repair/search loop.

## Reproduce the completed checkpoint

From the repository root, set `PYTHONPATH=service:ml` and use `.venv/bin/python`.
The original local recordings and existing ffmpeg dependency are required.

```sh
python -m playlens_ml.imminent_loss.review
python -m playlens_ml.imminent_loss.propose_boundaries
python -m playlens_ml.imminent_loss.verify_endings
python -m playlens_ml.imminent_loss.report
python -m unittest discover -s service/tests -p test_imminent_loss.py
```

Artifacts are written only under `artifacts/experiments/imminent-loss-v1`.
Review uses original presentation timestamps and disables frame duplication;
nominal frame-rate metadata is not used to infer outcome times. Automated
boundary proposals are inspection aids, not automatic annotation approval.
`reviewed_endings.json` contains the frozen visual judgments and image hashes.
The report refuses changed sources or review evidence. The raw recordings and
review images remain local and ignored by Git.

## Approved downstream protocol (not executed)

- Require a visual outcome interval at most 0.25 seconds wide for every game,
  video/active-time alignment within 0.5 seconds near endings and pauses, and
  at least 90% correct overlays at approximately 10, 5 and 2 seconds before loss.
- Build decision timestamps every 0.25 active seconds from recording and
  observation metadata. Inputs must precede decisions by at least 0.5 seconds;
  no visible loss or post-loss pixels. Horizon-straddling labels are uncertain.
- Reuse matching native-resolution 4-FPS extraction caches. Compare current
  geometry, heights, gaps and colors against the same features plus existing
  1- and 2-second changes. No clear-flash features or eight-second exclusion.
- Reset history at pauses, capture gaps, unsupported geometry or geometry
  changes. Allow two-second warm-up after start/resume; later failures remain
  in coverage. Report each game's overall and final-ten-second availability.
- Fit/calibrate/evaluate folds: 1–9/10–15/16–20;
  1–14/15–20/21–25; 1–19/20–25/26–30.
- Four candidates: two representations with regularized logistic regression
  and shallow boosted trees. Keep the existing fixed settings: logistic
  C=0.1, max_iter=2000, liblinear after variance filtering/scaling; trees
  80 iterations, 7 leaves, 40 minimum samples per leaf, L2=5, learning rate
  0.05, no early stopping, seed 42. Platt calibration C=1, liblinear,
  max_iter=1000 on calibration games only. Separate 3/5-second outputs.
- Baselines: average duration, empirical duration, elapsed logistic/trees,
  maximum normalized stack-height logistic, always and never warning.
  Common windows for fit/calibration/probability comparisons; actual method
  availability on full timelines for warnings. Weight games equally in scores.
- Report Brier, precision–recall, calibration, per-game/fold results, coverage,
  descriptive paired game-bootstrap intervals and warnings.
- Start alerts at the second consecutive probability above threshold; end
  after four below; reset on pauses, gaps and unavailable predictions.
  No backdating. Entire onset lead interval must lie within 2–5 seconds.
  Earlier onsets are false; later onsets are late; uncertain onsets cannot
  count as timely. Precision is timely/all onsets. Measure early occupancy.
- Calibration-only thresholds: 0.01, 0.02, 0.05 and 0.10 to 0.95 by 0.05.
  Maximize timely recall subject to false onsets ≤0.2/min, precision ≥60%,
  and early occupancy ≤10%; tie to higher threshold. If none qualifies,
  declare no usable threshold.
- Screen: evaluation recall ≥70%, false onsets ≤0.2/min, precision ≥60%,
  early occupancy ≤10%, overall and final-ten-second coverage ≥95% in every
  evaluation game; five-second Brier at least 10% below every time/crowding
  baseline and improvement in at least two folds; recall at least 10 points
  above the strongest baseline satisfying warning limits; five-second
  calibration error ≤0.12. Rank by recall, false alarms, then Brier.
- Always/never controls must fail. A passing candidate would be frozen before
  a separate independent test; passing never enables live warnings.

The approved stopping rule applies now: timing failure closes this experiment.
Predictive information and small-sample uncertainty were not evaluated.
