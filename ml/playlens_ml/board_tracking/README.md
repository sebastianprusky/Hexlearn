# Pixel tracking repair experiment

Offline development only, using runs 1–30. Historical runs 31–40 are excluded.
The experiment keeps the original three chronological folds, separate calibration
runs, eight-second causal history, 0.5-second availability delay, alert definition,
and promotion criteria. It does not change production features or capture.

## Changes under test

- Refit abrupt center jumps from current core/gray pixels.
- When a colored incoming ring briefly hides the gray edge, retain preceding
  geometry for at most two frames, only with confirming current core pixels.
  Pause/capture gaps clear this history. Genuine unsupported frames remain invalid.
- Detect incoming color strips by radial thickness rather than erasing an entire
  annulus around the timer. Suppress thin decorative timer fragments.
- Require repeated inward movement with the same color before estimating speed.
- Confirm activity changes using two consistent observations. This is deliberately
  conservative and can miss genuine clears; it remains an estimate.

## Reproduce

From `playlens`, use `PYTHONPATH=service:ml` and `.venv/bin/python`:

```sh
python -m playlens_ml.board_tracking.data --all
python -m playlens_ml.board_tracking.review
python -m playlens_ml.board_tracking.annotations
python -m playlens_ml.board_tracking.audit
python -m playlens_ml.board_tracking.data --build
python -m playlens_ml.board_tracking.windows
python -m playlens_ml.board_tracking.evaluate
python -m playlens_ml.board_tracking.report
```

The annotation module records explicit assistant judgments from the saved contact
sheets, including uncertain examples. These judgments are not independent ground
truth. Review any new repaired frames before updating this list; the checkpoint
must refuse unknown IDs or stale extractor hashes. The 60 original still geometry
and color outputs are verified byte-for-value equivalent to their prior review.
Incoming tracks, clear timing, and repaired geometry are reviewed separately.

Artifacts live in `artifacts/experiments/board-tracking-v1`. `protocol.json` records
the hypotheses before forecasting evaluation. `tracking-audit-initial.json` keeps
the initial timer failure. `review/broad-refit-diagnostics` keeps an exploratory
geometry diagnostic that was narrowed to abrupt jumps before model evaluation.

`results.json` separates common-window comparisons from the corrected model on
all supported windows. The common comparison includes the previous compact model,
pixel repairs with the old motion calculation, and the conservative motion model.
Both fixed families are used; no parameter search is performed. All named time
baselines and always/never warning controls use the same scored windows. Unsupported
coverage and baseline errors across all eligible windows remain in the report.
