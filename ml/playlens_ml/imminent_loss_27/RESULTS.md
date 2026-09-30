# Exploratory 27-run imminent-loss study

**Stopped at the extraction checkpoint. No forecasting models were trained.**

The user authorized excluding runs 5, 14 and 17 after the original timing audit failed. No further runs were excluded. Historical runs 31–40 were not used.

## Completed checks

- All 64 video/observation alignment checks passed the existing 0.5-second allowance, including checks near endings and on both sides of recorded pauses.
- The original precisely reviewed loss intervals were retained for the remaining 27 recordings.
- All 81 cached extraction overlays were reviewed at approximately 10, 5 and 2 seconds before loss. Additional panels show incoming colors, radii and gaps.
- The source-video hashes, observation hashes, cache hashes and extractor fingerprint were verified. No extractor repair was made.

## Extraction result

Four frames contain clear incoming-block errors and six have stack/color assignments that could not be certified during clears or transitions. Even if every remaining frame were correct, the rate would be only **71/81 = 87.7%**, below the required **90%**. This is an optimistic upper bound, not a measured accuracy estimate.
The existing cache marks every reviewed frame valid; that flag does not establish that its measurements are correct. Judgments are assistant visual review, not independent ground truth.

| Frame | Review | Finding |
|---|---|---|
| run-01-lead-10 | uncertain | Bottom lane is visibly green while the outermost-color label is blue; fading edge pixels prevent confident verification. |
| run-08-lead-10 | uncertain | Bottom stack ends in visible green while the reported outermost color is red; cannot certify the color/extent during the transition. |
| run-09-lead-2 | uncertain | All six stack heights collapse to the core and top colors become empty despite large visible block groups during a clear; stack versus detached-group assignment is not trustworthy. |
| run-13-lead-10 | uncertain | The large bottom block group is reported as an empty stack with an inner blue incoming boundary during a clear; visible outer extent cannot be certified. |
| run-15-lead-10 | uncertain | Lower-left blue/yellow/red group is reported as an empty stack and is not represented as the nearest incoming group; clear-state assignment is ambiguous. |
| run-19-lead-10 | incorrect | A separated green block above the core is visible, but lane 4 reports no incoming block and an empty stack. |
| run-27-lead-10 | incorrect | A separated green group above the core is visible, but lane 4 reports no incoming block; the reported red stack does not account for this group. |
| run-28-lead-10 | incorrect | A separated blue block on the upper-right world lane is visible during rotation, but incoming detection reports none in every lane. |
| run-28-lead-2 | uncertain | The large bottom block group is reported as an empty stack and an inner yellow incoming boundary; its visible outer extent is not represented reliably during the clear. |
| run-29-lead-5 | incorrect | A separated red block in the lower-right lane lies nearer than the outer blue block, but the reported incoming boundary is the outer blue block. |

## What this establishes

The narrower cohort passes the checked timing alignment requirement, but current board extraction does not meet the retained visual review standard. Visible detached blocks are sometimes missed or replaced by farther blocks, and clear animations can make reported stack heights/colors unreliable.
The pixels are visibly informative in the documented examples. This failure concerns the current extractor; it does not demonstrate insufficient predictive information or impossibility of a 2–5-second warning.

## Stopping point

Dataset assembly, model comparison, warning metrics and full-timeline coverage were not run after the extraction gate failed. There are no Brier, recall or baseline-superiority claims. Small-sample predictive uncertainty remains unassessed.
The original fold boundaries, four candidates, baselines and success thresholds were preserved in the protocol. No further game exclusions, model search, capture changes, new gameplay or live warning changes were made. The original 30-run experiment and historical reports remain intact.

Reproduce with the audit, incoming_review and checkpoint modules described in README.md. Local review images remain outside Git.
