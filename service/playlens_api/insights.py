from __future__ import annotations

from typing import Any


def _segments(predictions: list[dict[str, Any]], predicate, minimum_active_ms: int, kind: str) -> list[dict[str, Any]]:
    segments = []
    active = []
    for prediction in predictions:
        if predicate(prediction):
            active.append(prediction)
        elif active:
            if int(active[-1].get("activeElapsedMs") or 0) - int(active[0].get("activeElapsedMs") or 0) >= minimum_active_ms:
                segments.append(
                    {
                        "kind": kind,
                        "startMs": active[0]["timestampMs"],
                        "endMs": active[-1]["timestampMs"],
                        "startActiveMs": active[0].get("activeElapsedMs"),
                        "endActiveMs": active[-1].get("activeElapsedMs"),
                    }
                )
            active = []
    if active and int(active[-1].get("activeElapsedMs") or 0) - int(active[0].get("activeElapsedMs") or 0) >= minimum_active_ms:
        segments.append(
            {
                "kind": kind,
                "startMs": active[0]["timestampMs"],
                "endMs": active[-1]["timestampMs"],
                "startActiveMs": active[0].get("activeElapsedMs"),
                "endActiveMs": active[-1].get("activeElapsedMs"),
            }
        )
    return segments


def derive_insights(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = [prediction for prediction in predictions if prediction.get("status") == "ready"]
    insights = _segments(
        ready,
        lambda prediction: prediction.get("lossWindow") not in {None, "none"},
        3_000,
        "loss_signal",
    )
    insights.extend(
        _segments(
            ready,
            lambda prediction: prediction.get("lossWindow") == "under_10s",
            1_000,
            "imminent",
        )
    )
    for index, current in enumerate(ready):
        earlier = next(
            (
                candidate
                for candidate in reversed(ready[:index])
                if int(current.get("activeElapsedMs") or 0) - int(candidate.get("activeElapsedMs") or 0) <= 3_000
            ),
            None,
        )
        if earlier is None:
            continue
        current_remaining = current.get("estimatedRemainingSeconds")
        earlier_remaining = earlier.get("estimatedRemainingSeconds")
        if current_remaining is not None and earlier_remaining is not None and earlier_remaining - current_remaining >= 15:
            insights.append(
                {
                    "kind": "forecast_shift",
                    "startMs": earlier["timestampMs"],
                    "endMs": current["timestampMs"],
                    "startActiveMs": earlier.get("activeElapsedMs"),
                    "endActiveMs": current.get("activeElapsedMs"),
                }
            )
    return sorted(insights, key=lambda insight: (insight["startMs"], insight["kind"]))
