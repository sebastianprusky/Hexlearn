# Exploratory imminent loss study: 27 recordings

**Closed at extraction review. Timing alignment passed; no models were trained.**
See [RESULTS.md](RESULTS.md) for the documented failures and limitations.

This is the separately authorized follow-up to the closed `imminent-loss-v1`
experiment. Runs 5, 14 and 17 are explicitly excluded because their visible
outcome timing failed the original checkpoint. The user authorized this subset;
no further exclusions, model search or gameplay are authorized by this protocol.

Keep original run numbers and chronological fold boundaries:

| Fold | Fit | Calibration | Evaluation |
|---|---|---|---|
| 1 | 1–9 except 5 | 10–15 except 14 | 16–20 except 17 |
| 2 | 1–14 except 5,14 | 15–20 except 17 | 21–25 |
| 3 | 1–19 except 5,14,17 | 20–25 | 26–30 |

All model families, feature groups, baseline comparisons, alert rules and
success thresholds remain those in the original experiment's approved
[protocol](../imminent_loss/README.md). Five seconds remains the primary target.
Runs 31–40 remain excluded. Success would be exploratory, not an independent
validation or authorization for live warnings.

Before any model fit, verify saved-observation/video alignment near each ending
and on both sides of recorded pauses, and visually review cached extraction at
approximately 10, 5 and 2 seconds before each loss. All reviewed alignments must
fit the original conservative 0.5-second allowance; at least 90% of reviewed
frames must have correct geometry, heights, incoming separation and colors.
If this checkpoint fails, stop and report; do not silently remove more games.

Artifacts are isolated under `artifacts/experiments/imminent-loss-27-v1`.
Original experiments and production behavior are preserved.

From the repository root with `PYTHONPATH=service:ml`:

```sh
.venv/bin/python -m playlens_ml.imminent_loss_27.audit
.venv/bin/python -m playlens_ml.imminent_loss_27.incoming_review
.venv/bin/python -m playlens_ml.imminent_loss_27.checkpoint
```

This produces review evidence only. It does not fit a forecasting model.
