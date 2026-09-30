"""Fail closed on incomplete, stale, or imprecise visual outcome annotations."""
import math


def timing_gate(rows):
    if sorted(row['run'] for row in rows) != list(range(1, 31)):
        raise ValueError('Review must contain each development run 1–30 exactly once')
    failures = []
    for row in rows:
        reason = None
        interval = row.get('lossIntervalVideoSeconds')
        if not row.get('visuallyReviewed'):
            reason = 'not_visually_reviewed'
        elif interval is None:
            reason = 'visible_loss_transition_missing'
        elif len(interval) != 2 or not all(math.isfinite(x) for x in interval):
            reason = 'invalid_loss_interval'
        elif interval[1] <= interval[0] or interval[1] - interval[0] > .25 + 1e-9:
            reason = 'loss_uncertainty_exceeds_0.25_seconds'
        if reason:
            failures.append({'run': row['run'], 'reason': reason})
    if failures:
        return {'status': 'closed_timing_failure', 'failures': failures,
                'forecastingAllowed': False}
    # Outcome interval precision alone does not validate active-time alignment
    # or geometry. Unknown reviews must never implicitly pass.
    pending = [row['run'] for row in rows if
               row.get('alignmentVerified') is not True or
               not isinstance(row.get('alignmentBoundSeconds'), (int, float)) or
               not 0 <= row['alignmentBoundSeconds'] <= .5]
    return {'status': 'pending_alignment' if pending else 'timing_passed',
            'pendingRuns': pending, 'forecastingAllowed': False}


def require_forecasting_gate(gate):
    if gate.get('forecastingAllowed') is not True:
        raise ValueError('Forecasting blocked: timing and extraction review must pass first')
