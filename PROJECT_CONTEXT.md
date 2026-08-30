# PlayLens project context

Read this file before making product, ML, data, or interface changes. Update the
Current status, Decisions, Change log, and Next milestone sections after every
material change.

## Product purpose

PlayLens is a local-first playtesting tool that tests whether Hextris canvas
pixels can forecast a specific player's loss window before simple elapsed-time
and average-run shortcuts can. It is a developer-facing diagnostic prototype,
not a competitive assistant, emotion detector, or boredom detector.

The portfolio claim must remain evidence-based:

> PlayLens evaluates whether a personal temporal vision model can forecast a
> Hextris loss at least 15 seconds ahead while outperforming elapsed-time,
> average-duration, and board-occupancy baselines on locked future runs.

## North-star experience

Initial data collection shows no experimental forecast:

- `CAPTURING RUN 12 / 50`
- `PAUSED`
- An explicit capture or validity error

After promotion, the live overlay shows exactly one forecast plus certainty:

- `NO FAILURE SIGNAL`
- `LOSS LIKELY IN 45–60s`, `30–45s`, `20–30s`, or `10–20s`
- `LOSS IMMINENT · <10s`
- `HIGH`, `MEDIUM`, or `LOW CERTAINTY`

No probabilities, risk bars, score estimates, sound, flashing, or automatic
movement appear during play.

## Lifecycle and data contract

- Production model input is only the Hextris canvas sampled at 4 FPS.
- Lifecycle detection may inspect visible page UI but never internal variables.
- The extension state machine is `idle → active ↔ paused → game_over`.
- P, Space, the visible pause control, and a hidden tab suspend capture,
  inference, recording, and the active clock without ending the session.
- Resuming clears temporal context and requires a fresh eight active seconds.
- Stillness never means game over. The visible Hextris game-over screen is the
  required completion signal.
- Each frame stores wall time and active-play time. Pause intervals are stored
  separately.
- A usable personal run requires confirmed game over, eight active seconds,
  enough valid frames, and at least 75% expected capture coverage.
- Legacy runs are reviewed conservatively for gaps and possible pauses before
  being counted.

## ML problem

Input:

- Eight seconds of pixel-derived visual and temporal features.
- Board occupancy, near-center pressure, sector imbalance, motion, contrast,
  saturation, edges, and recent change.
- No score, active elapsed time, average duration, or game state.

Targets:

- Failure within 5, 10, 15, 20, 30, 45, and 60 active seconds.
- Cumulative probabilities must be monotonic across horizons.
- The first calibrated 50% crossing creates the live loss window.
- A survival-curve integral supplies the report's estimated seconds remaining.

Models:

- Five-member run-bootstrap scikit-learn visual ensemble.
- Five-member PyTorch temporal GRU ensemble when PyTorch is installed.
- Constant-rate, elapsed-time-only, player-average-duration, and occupancy-only
  references are evaluated but can never enter the visual model.

## Data split and promotion

- The first 50 usable personal runs are chronological.
- Runs 1–32 fit models; runs 33–40 calibrate probabilities and certainty.
- Runs 41–50 are locked future tests and never influence fitting or selection.
- Experimental jobs run at 20, 25, 30, 35, and 40 runs; the final gate runs at
  50. The overlay remains collection-only throughout.
- Bot and instrumented runs are retained only for pipeline smoke tests.

All promotion gates must pass:

1. At least 10% Brier improvement over elapsed time and average duration.
2. At least 10% time-to-failure MAE improvement over both shortcuts.
3. A useful warning at least 15 seconds ahead on most locked runs.
4. Worst horizon calibration error no greater than 0.12.
5. No material final-ten-second regression versus elapsed time.
6. Local prediction latency below 500 ms at p95.
7. Certainty tiers show progressively lower empirical error; unsupported tiers
   are disabled.

Failure to pass keeps the overlay in collection mode and must be explained in
Model Lab. More data cannot be described as success by itself.

## Architecture

- Shared browser controller plus Chrome Manifest V3 extension 0.6.4: consent,
  pause-safe capture, lifecycle, recording, and loss-window overlay. The local
  server reserves and injects the controller as the authoritative clean-game
  path; a shared DOM ownership lock keeps extension and local execution worlds
  from duplicating capture.
- FastAPI on `127.0.0.1:8000`: Hextris, frames, active-time sessions, inference,
  recordings, training jobs, and quality status.
- scikit-learn and PyTorch personal survival pipelines under `ml/`.
- Next.js 16 dashboard on `127.0.0.1:3001`: session library, active-time loss
  graph, data-quality reasons, and Model Lab.
- SQLite, frames, recordings, jobs, and artifacts remain local.

## Current status — 2026-08-30

- The v3 pause-aware extension, active-time API/database migration, personal
  dataset builder, scikit-learn/PyTorch survival training, automatic milestone
  jobs, loss-window report, and human-first Model Lab are implemented.
- Unit tests cover pause/resume continuity, stillness versus game over,
  active-time quality labels, bot exclusion, seven-horizon monotonicity, and the
  32/8/10 split.
- Dashboard lint and the Next.js production build pass.
- PyTorch 2.8 is installed in the project virtual environment, so automatic
  milestone jobs can train both planned model families.
- Legacy user sessions remain preserved but are excluded from the official
  `collection-v3` cohort. The official counter therefore begins at zero and all
  fifty runs share pause-aware active-time telemetry.
- The completion overlay now reports `RUN ACCEPTED` with the official count and
  capture coverage, or `RUN EXCLUDED` with the specific quality reason.
- Model Lab exposes recent v3 frame coverage, frame counts, pauses, recording
  status, and acceptance so collection problems are visible immediately.
- Full local cohort checkpoints are created at runs 20, 40, and 50; interrupted
  milestone training is retried after the service restarts.
- Both real training families pass an isolated synthetic end-to-end smoke test.
  That test caught and fixed a scikit-learn latency-evaluation shape error before
  any personal milestone job could encounter it.
- Chrome repeatedly failed to inject the unpacked extension despite reloads, so
  the combined local server now supplies the same controller directly. Live
  verification found one launcher, two local scripts, and no browser errors.
- A later live run exposed that Chrome's extension and page scripts have separate
  global objects, allowing two controllers, two sessions, and doubled capture
  load. Version 0.6.1 adds a cross-world DOM lock and delayed fallback, reuses
  capture canvases, and records replay at 15 FPS/900 Kbps. Both simultaneous
  affected runs are preserved but quarantined; one clean v3 run currently counts.
- Live start testing then exposed that the motion threshold could miss gameplay
  after capture downscaling. Version 0.6.2 uses the visible pause control as the
  authoritative in-progress signal and keeps direct canvas motion as fallback.
- Local controller ownership made the Chrome toolbar toggle target an inactive
  extension-world controller. Version 0.6.3 auto-arms on the dedicated local
  PlayLens game and treats the visible Play-button interaction as a direct start
  signal, while preserving pause-control and motion fallbacks.
- Leaving and returning through Chrome's back/forward cache then exposed stale
  detector state after a navigation-finished session. Version 0.6.4 resets the
  detector synchronously when any session closes and reinitializes restored pages.
- Old v2 bot artifacts remain historical and cannot load as a v3 live model.
- No personal v3 model has been trained or promoted yet. The live experience is
  intentionally collection-only.
- Three usable personal runs are currently accepted, producing 417 eight-second
  training windows under the `hextris-stack-v2` feature schema. Forty-seven
  usable runs remain before the locked evaluation and promotion decision.
- The source repository is published as Hexlearn; local gameplay frames,
  recordings, SQLite data, checkpoints, and model artifacts are ignored and
  remain on the user's machine.

## Decisions

- 2026-08-29: Made personal clean-copy runs the only model-development and
  evaluation source; bots became smoke-test-only.
- 2026-08-29: Replaced exact seconds and dense risk signals with calibrated loss
  windows plus empirically gated certainty.
- 2026-08-29: Chose a seven-horizon survival curve so the report can derive time
  remaining without claiming false live precision.
- 2026-08-29: Froze the final ten of fifty runs as a future test and prohibited
  experimental forecasts during collection.
- 2026-08-29: Made active-play time authoritative and prohibited stillness-only
  game-over detection.
- 2026-08-29: Made the official fifty an all-v3 cohort and archived legacy runs
  outside model fitting, calibration, and locked evaluation.
- 2026-08-30: Made local capture independent of Chrome extension injection while
  preserving explicit opt-in and the extension-compatible product architecture.
- 2026-08-30: Made controller ownership cross-world and quarantined simultaneous
  duplicate captures so neither can contaminate personal fitting.

## Change log

- 2026-08-29: Added explicit paused/active lifecycle events, recorder suspension,
  active timestamps, resume warmup, and visible-UI game-over confirmation.
- 2026-08-29: Added session quality fields, conservative legacy review, exclusion
  reasons, and personal collection indices.
- 2026-08-29: Added 5/10/15/20/30/45/60-second labels, five-member ensembles,
  average-duration baseline, certainty contracts, locked-test gates, and local
  milestone training jobs.
- 2026-08-29: Replaced the overlay with collection, pause, no-signal, loss-window,
  certainty, and run-complete states.
- 2026-08-29: Rebuilt session reports around active-time truth, model estimate,
  uncertainty, pause markers, baseline, warning lead, window accuracy, and MAE.
- 2026-08-29: Rebuilt Model Lab around the 40-development/10-test personal
  collection, exclusions, background jobs, baselines, and outstanding gates.
- 2026-08-29: Added truthful completion receipts, per-run collection health,
  milestone archives, interrupted-job recovery, and isolated two-model smoke
  training.
- 2026-08-30: Added a server-injected local capture fallback and verified the
  launcher and armed overlay in the user's Chrome tab.
- 2026-08-30: Added delayed fallback election, a shared DOM ownership lock,
  reusable capture canvases, lower-cost replay recording, and reversible
  duplicate-run quarantine.
- 2026-08-30: Replaced threshold-only start detection with visible-UI start
  detection and restored direct 16x16 canvas motion sampling as fallback.
- 2026-08-30: Made the local clean-game bootstrap reserve controller ownership
  before cached extension scripts, so fixes take effect on refresh.
- 2026-08-30: Versioned local controller resource URLs and disabled their cache
  so browser refreshes cannot retain stale capture logic during development.
- 2026-08-30: Auto-armed the dedicated local controller and added direct mouse,
  touch, and Enter-key Play signals so collection no longer depends on the
  Chrome toolbar action or a motion threshold.
- 2026-08-30: Added navigation/BFCache lifecycle reset so leaving and returning
  cannot strand an armed controller without an open session.
- 2026-08-30: Replaced the overly broad inward-weighted radial mask with an
  observed central-stack mask, outward pressure, outer danger-band occupancy,
  and six-sector stack reach. Raw 4 FPS JPEGs remain the retraining source, so
  all accepted runs can be re-featurized without replaying them.
- 2026-08-30: Added the `hextris-stack-v2` feature-schema contract to datasets
  and model artifacts; live inference rejects stale feature dimensions instead
  of failing or silently mixing extractors.

## Next milestone

1. Refresh the local game; its 0.6.4 controller arms automatically.
2. Complete a three-run pilot: normal, in-game pause, and hidden-tab pause.
   Confirm acceptance, coverage, pause count, and recording status in Model Lab.
3. Continue to 20 official v3 runs for the first experimental candidate, then
   continue to 40 without exposing predictions.
4. Freeze the 40-run candidate and collect ten untouched future test runs.
5. Evaluate both model families at run 50 and promote only if every gate passes.
6. If held back, inspect failure bands, class balance, certainty, and visual
   features before deciding whether to collect more runs or revise the model.
