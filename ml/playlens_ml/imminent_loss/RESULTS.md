# Imminent loss v1: stopped at the timing checkpoint

The recordings do not meet the predeclared outcome-timing requirements.
No models were trained; no predictive results or baseline wins are claimed.

| Run | Last clearly active video frame | First visible loss frame | Interval width |
|---|---:|---:|---:|
| 1 | 128.198 | 128.239 | 0.041 s |
| 2 | 191.199 | 191.262 | 0.063 s |
| 3 | 137.055 | 137.139 | 0.084 s |
| 4 | 170.408 | 170.475 | 0.067 s |
| 5 | 161.426 | 161.768 | 0.342 s |
| 6 | 107.864 | 107.948 | 0.084 s |
| 7 | 140.801 | 140.883 | 0.082 s |
| 8 | 110.266 | 110.333 | 0.067 s |
| 9 | 180.332 | 180.399 | 0.067 s |
| 10 | 193.628 | 193.694 | 0.066 s |
| 11 | 148.668 | 148.732 | 0.064 s |
| 12 | 165.265 | 165.332 | 0.067 s |
| 13 | 168.069 | 168.149 | 0.080 s |
| 14 | 164.864 | Not recorded | Unknown |
| 15 | 187.602 | 187.666 | 0.064 s |
| 16 | 166.336 | 166.399 | 0.063 s |
| 17 | 180.401 | Not recorded | Unknown |
| 18 | 116.081 | 116.148 | 0.067 s |
| 19 | 149.335 | 149.399 | 0.064 s |
| 20 | 160.865 | 160.932 | 0.067 s |
| 21 | 167.364 | 167.430 | 0.066 s |
| 22 | 156.526 | 156.610 | 0.084 s |
| 23 | 179.533 | 179.616 | 0.083 s |
| 24 | 173.202 | 173.283 | 0.081 s |
| 25 | 129.598 | 129.665 | 0.067 s |
| 26 | 148.736 | 148.815 | 0.079 s |
| 27 | 177.066 | 177.131 | 0.065 s |
| 28 | 102.999 | 103.066 | 0.067 s |
| 29 | 163.332 | 163.399 | 0.067 s |
| 30 | 165.137 | 165.200 | 0.063 s |

## Interpretation

- Run 5 has a 0.342-second gap across the transition; the limit is 0.250 seconds.
- Runs 14 and 17 end with the score still visible and no recorded loss transition.
- All 30 ending boundary pairs were visually reviewed. Native tail images and final-ten-second contact sheets were saved for every game.
- Full-stream decoding of runs 5, 14 and 17 reproduced every tail image, confirming these are not seek or frame-rate-conversion artifacts.
- Score disappearance is used only for outcome review. It is never a model input. Local renderer code corroborates this visible transition; no engine state was used as a feature.
- Times above are video presentation timestamps, not validated active-time labels. End/pause alignment and the extraction review were not completed after the timing gate failed.
- No games were silently removed. Runs 31–40 were not used. Production and prior experiment outputs remain unchanged.

## Answers

1. Timing is insufficient under the agreed rule. Near-loss extraction reliability remains unassessed.
2. Whether pixels beat time and crowding at this horizon is untested.
3. There is no validated basis for an independent test or live warnings.

This is a timing limitation, not evidence that imminent loss is unpredictable. Predictive information and small-sample uncertainty cannot be assessed without passing the input gate. The experiment is closed; no expanded search or new gameplay is requested.
