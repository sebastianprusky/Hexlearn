# Hexlearn project context

Read this file before making product, ML, data, or interface changes. Update the
Current status, Decisions, Change log, and Next milestone sections after every
material change.

## Product purpose

Hexlearn is a local-first playtesting tool that tests whether Hextris canvas
pixels can forecast a specific player's loss window before simple elapsed-time
and average-run shortcuts can. It is a developer-facing diagnostic prototype,
not a competitive assistant, emotion detector, or boredom detector.

The portfolio claim must remain evidence-based:

> Hexlearn evaluates whether a personal temporal vision model can forecast a
> Hextris loss at least 15 seconds ahead while outperforming elapsed-time,
> average-duration, and board-occupancy baselines on locked future runs.

## North-star experience

Initial data collection shows no experimental forecast:

- `CAPTURING RUN 12 / 40`
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

- The initial 40 usable personal runs are chronological.
- Runs 1–24 fit models; runs 25–30 calibrate probabilities and certainty.
- Runs 31–40 are locked future tests and never influence fitting or selection.
- Experimental jobs run at 20, 25, and 30 runs; the final gate runs at 40.
  At run 30, calibration metrics are provisional diagnostics, not test evidence.
- No automatic jobs run during the locked-test collection (31–39).
- Later gameplay remains recorded for future training rounds. The initial
  pipeline stays capped at 40; future models need a separately defined split.
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

## Current status — 2026-09-30

### Clear measurement experiment — 2026-09-30

- Direction: validate one pixel measurement, then run one fixed feature comparison, then consider an independent future test only if all original forecast gates pass.
- Added separate offline `clear_measurement` package. A frozen 12-FPS clear-flash detector passed a reserved clip check: 23/26 confirmable clears, zero false detections, and 100% supported scored frames across 60 scored seconds. Raw labels were written before detector predictions were displayed. These are assistant labels on new segments of previously used games.
- The 4-FPS development check missed brief flashes; sampling was amended to 12 FPS before protocol/source freeze. No threshold changes after reserved review.
- Completed one fixed ablation on 56,364 frames from runs 1–30: eight recent-clear features added to the retained 80-feature pixel representation, compared with unchanged features in the two original fixed families. Historical runs 31–40 remain excluded.
- On identical supported windows, adding clears improved the tree by only 0.00032 Brier and 0.036 seconds of last-minute error; descriptive paired intervals include no improvement. The clear tree scores Brier 0.06660 and MAE 11.140s, versus best time-baseline values 0.06171 and 10.610s. Its false-alert rate is 0.331/minute; final-ten-second error is 14.611s. Both clear candidates fail six of eight original gates.
- Coverage is 3,797/4,127 development windows (92.0%), with 2,016 supported evaluation windows and a worst evaluation-game coverage of 78.1%. Some evaluation games retain only 1–2 final-ten-second windows. Do not compare these scores directly with older reports scored on more windows.
- Decision: close this experiment; improve extraction first. Keep the clear detector as an experimental measurement tool, without replacing the forecasting representation or promoting a model. Four inspected failures reproduce uncertain gray-boundary fits during animation/occlusion; current recovery returns immediately on invalid native geometry.
- 64 focused tests pass; frozen measurement hashes and eleven original artifact hashes remain unchanged. Results are kept under `artifacts/experiments/clear-measurement-v1`.

### Tracking repair follow-up — 2026-09-30

- Added offline `board_tracking` experiment; report at `artifacts/experiments/board-tracking-v1/REPORT.md`.
- Reviewed 12 clips with explicit incoming/clear labels and all 33 repaired geometry frames. Original 60 still geometry/color outputs remain identical to v3; incoming target-color agreement is 56/60, and conservative repair geometry/color review is 30/33. These are assistant judgments, not independent labels.
- Current-pixel center refits and bounded prior-geometry recovery increased supported windows from 3,888 to 3,970 of 4,127 (96.2%). Worst evaluation-game coverage remains 87.9%, below the 95% per-game requirement.
- The pixel-repaired tree is effectively tied with the prior compact logistic on identical windows: Brier 0.07143 vs 0.07206, last-minute MAE 11.716s vs 11.719s; paired intervals include no improvement. On all supported windows it scores Brier 0.07143, MAE 11.665s, final-10s MAE 13.622s and 0.313 false alerts/minute. No candidate passes all gates.
- Stricter motion/clear confirmation removed five false clear detections but matched only 5 of 13 confirmable reviewed clears. It worsened forecasting and is rejected as a replacement; keep the prior motion representation as the reference.
- Four new feature/family combinations were examined with unchanged fixed settings (24 distinct combinations across the development program). No runs 31–40, live promotion, new collection or model search expansion.
- All 45 board tests pass. Eleven original dataset/model/report artifacts still match archived hashes.


- The user authorized continued pixel-only improvement using clearer existing
  replays, including a possible future need for higher-resolution live capture.
  Native recordings are 2940x1602; the new offline extractor decodes at 960x524.
- Native extraction and crowded-board recovery are implemented in `board_v2`
  and `board_v3`. Conservative review confirms 55/60 stills, with twelve clips
  and twelve additional crowded-board examples inspected. Resize/letterbox
  checks pass the review threshold. Sixty-six extra timing checks support a
  conservative 0.5-second feature delay; original active-time labels are reused.
- Native v3 supports 3,888/4,127 eligible development windows (94.2%); the lowest
  evaluation-game coverage is 86.3%, below the required 95%. A core-ratio scale
  variant reduced coverage and was rejected before model fitting.
- Tested rotation-invariant features, adjacent-frame speed/activity estimates,
  and compact speed/crowding/activity models using the same fixed logistic/tree
  settings and three chronological folds. Twenty distinct development pairs
  were evaluated; no hyperparameter search and no use of runs 31–40 occurred.
- The best development result is `board-compact-v1/speed_board_activity_logistic`:
  Brier 0.07192 and last-minute MAE 11.69 s, versus average duration 0.08396 and
  12.78 s. Stronger elapsed-time baselines remain better (best Brier 0.06694;
  best time error 11.16 s). Descriptive paired intervals include no improvement
  over average duration. No candidate passes all screening requirements.
- Best-candidate false alerts are 0.329 per observed active minute (cap 0.2),
  final-ten-second error is 16.16 s (best time baseline 9.53 s), and per-game
  coverage fails. Timely recall of 15/15 is not a success claim because repeated
  onsets, false alarms and unsupported intervals remain material problems.
- The complete report is `artifacts/experiments/board-compact-v1/REPORT.md`;
  reproduction instructions are `ml/playlens_ml/board_compact/README.md`.
  Thirty-six board-experiment tests pass. Original datasets, models, job state
  and evaluation reports match archived bytes/hashes. Live inference, API,
  extension and dashboard behavior were not changed by this offline work.

### Earlier extraction checkpoint — 2026-09-30

- The requested bounded pixel-only board experiment is implemented under
  `ml/playlens_ml/board_experiment/` and stopped at its extraction checkpoint.
  Conservative visual review confirmed 25/60 sampled frames (41.7%), below the
  required 90%. Only 329/4,127 eligible development windows (8.0%) retained eight
  seconds of valid continuous extraction; one game had zero valid windows.
- The gray hexagon generally localizes, but radial connectivity confuses inner
  and outer colors and loses assignments during rotation. Letterbox checks kept
  lane colors identical on 57/60 frames; resizing was less stable. These results
  diagnose this extractor, not a fundamental limit on visual prediction.
- Real model fitting was blocked as specified. The six-candidate evaluation,
  four time baselines, persistent-alert metrics, coverage gates, and conditional
  offline candidate freeze are implemented and tested with synthetic data.
  Fifteen new tests pass, including a complete synthetic comparison. Original
  datasets, model bundles, and evaluation reports match their prior hashes or
  archived bytes. No production inference, capture, API, or UI changed.
- Reproduction instructions are in the experimental package README; local
  review images, clips, decisions, cache, dataset, and final report are under
  `artifacts/experiments/board-features-v1/`. The report recommends improving
  extraction before any new model experiment or further gameplay collection.

- Initial collection and both training jobs completed at 40 accepted runs.
  Neither original candidate passed all promotion gates; live forecasts remain
  disabled.
- An offline audit compared ten configurations in three chronological folds
  using only runs 1–30. Visual-only candidates did not demonstrate a consistent
  advantage over stronger time baselines. Adding board features to a matched
  elapsed-time tree changed mean time error from 11.13 s to 11.11 s; its possible
  final-ten-second benefit remains uncertain.
- The original linear candidate saturates near 100% failure probability on
  runs 39–40, alongside large shifts in image-wide appearance features. Prioritize
  pixel-based board normalization and color/clear/incoming-block features before
  more bulk collection. Matrix-multiplication warnings reproduced with finite
  outputs agreeing with elementwise computation within 1e-15; warning removal
  alone is not an accuracy fix.
- Original reports and model files are archived locally under
  `artifacts/experiments/development-audit-2026-09-30/original/`. The audit report
  is `artifacts/experiments/development-audit-2026-09-30/REPORT.md`; reproducible
  commands are `playlens_ml.development_audit` and
  `playlens_ml.audit_original_candidate`. These do not promote models.
- Original test results have now informed diagnosis. Revisions motivated by
  these findings need a newly defined prospective test. The hybrid experiment
  does not change the visual-only production input contract.

### Historical status — 2026-09-29

- The user reduced initial collection to 40 runs; 25 accepted runs and a
  completed 25-run training job were observed before this change.
- Current policy is 24 fit / 6 calibration / 10 future test, with checkpoints
  at 20, 30, and 40. Promotion quality requirements are unchanged.
- Captures after 40 are preserved, but automatic continual retraining is not
  enabled. Training on a test run retires it as independent test evidence for
  the newly trained version.

### Historical status — 2026-09-04

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
  Hexlearn game and treats the visible Play-button interaction as a direct start
  signal, while preserving pause-control and motion fallbacks.
- Leaving and returning through Chrome's back/forward cache then exposed stale
  detector state after a navigation-finished session. Version 0.6.4 resets the
  detector synchronously when any session closes and reinitializes restored pages.
- Old v2 bot artifacts remain historical and cannot load as a v3 live model.
- No personal v3 model has been trained or promoted yet. The live experience is
  intentionally collection-only.
- Eleven usable personal runs are currently accepted, producing 1,247
  eight-second training windows under the `hextris-stack-v2` feature schema.
  Thirty-nine usable runs remain before the locked evaluation and promotion
  decision.
- The local controller now detects game over from Hextris's visible score
  container instead of its always-rendered wrapper, so the start screen no
  longer blocks Play-button run registration.
- Future runs persist every valid 4 FPS frame and calculate coverage from the
  saved JPEGs. This closes a timing-jitter gap that made three accepted runs
  contribute fewer dense eight-second windows than their receipts implied.
- The source repository is published as Hexlearn; local gameplay frames,
  recordings, SQLite data, checkpoints, and model artifacts are ignored and
  remain on the user's machine.

## Decisions

- 2026-09-30: The frozen clear measurement passed, but its completed fixed forecasting comparison showed no reliable incremental benefit and no qualifying candidate. Stop this experiment. Any further work should first address geometry continuity on crowded late-game frames using a separate measurement protocol; no broader model search, new gameplay or live promotion.

- 2026-09-30 tracking follow-up: retain causal pixel geometry/incoming fixes for research; reject strict activity confirmation as a forecasting replacement. Treat the tiny common-window gain as inconclusive and keep all original promotion gates.

- 2026-09-30: User explicitly prioritized the strongest pixel-only approach using
  native replay pixels. Continued targeted feature experiments with fixed model
  settings; retained the 15-second warning objective and all screening gates.
  Keep every failed/rejected variant as development evidence. No new prospective
  cohort or live promotion is justified by the current results.

- 2026-09-30: At the user's request, keep the next experiment offline and
  pixel-only, retain the 15-second objective, and stop before real training when
  extraction review is below 90%. The first board extractor failed this gate;
  no parameter search or additional collection followed.

- 2026-09-29: Reduced initial target to 40 at user request, retaining ten
  future test runs and all promotion gates. Later captures stay available.

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
- 2026-09-04: Made persisted frames authoritative for capture coverage and
  removed the redundant server-side cadence gate; client-side sampling remains
  fixed at 4 FPS.

## Change log

- 2026-09-30: Completed causal clear-flash detection, annotation with predictions hidden, frozen measurement evaluation and the fixed forecasting ablation. Measurement passed (23/26 clears, zero false detections); forecasting improvement was negligible with intervals spanning zero. No candidate qualifies. Report: `artifacts/experiments/clear-measurement-v1/REPORT.md`; 64 tests pass and original artifacts remain intact.

- 2026-09-30: added reviewed tracking repair, explicit clip annotations, common-window ablations, all-supported evaluation, prediction traces and nine regression tests under `board_tracking`; production remains unchanged.

- 2026-09-30: Added native replay decoding/alignment, crowded-board recovery,
  conservative review, rotation-invariant summaries, causal speed/activity
  estimates, compact model ablations, out-of-fold prediction export, and a
  consolidated report. Improved development results over average duration but
  did not clear stronger time baselines or reliability gates.

- 2026-09-30: Implemented the isolated board-feature experiment, deterministic
  extraction review, resize/letterbox checks, per-frame cache, continuous-history
  dataset, calibrated candidate/reference comparison, persistent-alert metrics,
  screening gates, report command, and fifteen tests. Executed extraction on all
  30 development games and issued an extraction-first stop decision.

- 2026-09-30: Added an offline development comparison and original-candidate
  postmortem, with tests for future-run exclusion, chronological folds, and
  warning timing. Preserved the first experiment's reports and models; made no
  production inference or promotion changes.

- 2026-09-29: Updated API, overlay, dataset, split, milestone jobs, checkpoints,
  and Model Lab to the initial 40-run protocol; added split boundary tests.

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
- 2026-09-03: Fixed local Play-button run registration by distinguishing the
  always-rendered game-over wrapper from its game-over-only score content.
- 2026-09-04: Fixed jitter-related frame persistence loss, capped collection
  receipts at 100% coverage, and added a 235 ms cadence regression test.
- 2026-08-30: Replaced the overly broad inward-weighted radial mask with an
  observed central-stack mask, outward pressure, outer danger-band occupancy,
  and six-sector stack reach. Raw 4 FPS JPEGs remain the retraining source, so
  all accepted runs can be re-featurized without replaying them.
- 2026-08-30: Added the `hextris-stack-v2` feature-schema contract to datasets
  and model artifacts; live inference rejects stale feature dimensions instead
  of failing or silently mixing extractors.

## Current direction and stopping point

The user approved a separate pixel-only imminent-danger experiment, targeting
warnings 2–5 active seconds before loss. Its five-second target is primary;
three seconds is secondary. The original long-horizon and clear-measurement
results remain historical evidence, with no qualifying candidate.

The imminent-loss experiment is now closed at its mandatory timing gate:

- Run 5: a 0.342-second interval across visible loss, above the 0.25-second limit.
- Runs 14 and 17: no visible loss transition before the saved video ends.
- Full-stream decoding confirms the same ending pixels as the review decoder.

All 30 ending boundary pairs were visually reviewed. End/pause alignment,
near-loss extraction validation and forecasting were not completed after the
timing failure. No games were removed to make the experiment pass, no models
were trained, and no new gameplay is requested. This is a recording/timing
limitation, not proof that short-horizon prediction cannot work.

See `ml/playlens_ml/imminent_loss/README.md` for the approved protocol and
`RESULTS.md` in that directory for the report. Do not resume model tuning or
claim a validated warning system from these results. Any later study needs a
separate explicit scope; the current experiment does not enable live warnings.

## Authorized 27-run follow-up

The user explicitly authorized excluding runs 5, 14 and 17 for a separate
exploratory study. `imminent-loss-27-v1` retains the original numerical run/fold
boundaries, four candidate definitions, baselines, warning rules and thresholds.
No further exclusions were made and historical runs 31–40 remain unused.

All 64 checked video/observation alignments passed. All 81 near-loss extraction
panels were inspected with additional incoming-block overlays. Four definite
incoming-block errors and six uncertifiable stack/color assignments imply an
optimistic correct-frame upper bound of 71/81 (87.7%), below the required 90%.
The extractor marked all 81 frames valid, so validity flags alone were inadequate.

This follow-up is closed at extraction review. Dataset/model/warning evaluation
was not run, and no new runs, repair loop or live warnings were enabled. See
`ml/playlens_ml/imminent_loss_27/RESULTS.md`. Timing is no longer the identified
blocker in this subset; reliable block separation through clears and rotations
remains unresolved. Predictive usefulness is still untested at this horizon.
