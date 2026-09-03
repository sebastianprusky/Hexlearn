import Link from "next/link";

import { getModelStatus } from "@/lib/api";
import type { EvaluationMetrics, ModelStatus } from "@/lib/types";

const percent = (value: number | undefined | null) => typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
const decimal = (value: number | undefined | null) => typeof value === "number" ? value.toFixed(3) : "—";

function MetricRow({ metrics }: { metrics: EvaluationMetrics }) {
  return (
    <div className="metric-strip">
      <span><small>Mean Brier</small><strong>{decimal(metrics.meanBrier)}</strong></span>
      <span><small>Time MAE</small><strong>{typeof metrics.timeToFailureMae === "number" ? `${metrics.timeToFailureMae.toFixed(1)}s` : "—"}</strong></span>
      <span><small>15s warning rate</small><strong>{percent(metrics.usefulWarningRateAt15s)}</strong></span>
      <span><small>Worst calibration</small><strong>{decimal(metrics.worstCalibrationError)}</strong></span>
    </div>
  );
}

function EmptyModelLab() {
  return (
    <main className="message-page">
      <p className="eyebrow">MODEL LAB</p><h1>Waiting for the local API.</h1>
      <p>Start the combined Hexlearn server on port 8000, then refresh this page.</p>
      <Link className="primary-link" href="/">Return to sessions</Link>
    </main>
  );
}

export default async function ModelPage() {
  let status: ModelStatus;
  try {
    status = await getModelStatus();
  } catch {
    return <EmptyModelLab />;
  }
  const collection = status.personalCollection;
  const progress = Math.min(100, (collection.usableRuns / collection.targetRuns) * 100);
  const exclusionSummary = Object.entries(collection.exclusionReasons);
  const officialRecentRuns = collection.recentRuns.filter((run) => run.officialV3);
  const latestCheckpoint = status.checkpoints.at(-1);

  return (
    <main className="model-page">
      <Link className="back-link" href="/">← Session reports</Link>
      <section className="model-hero">
        <div>
          <p className="eyebrow">MODEL LAB / PERSONAL SURVIVAL</p>
          <h1>Fifty clean runs.<br />One honest test.</h1>
          <p>Hexlearn learns only from your quality-gated canvas captures. The final ten runs stay locked until a visual model is ready to prove it beats elapsed time and average duration.</p>
        </div>
        <div className={`live-model-state ${status.liveModel.artifactPresent ? "validated" : "experimental"}`}>
          <span>Live overlay</span>
          <strong>{status.liveModel.artifactPresent ? "Personal loss windows" : "Collection only"}</strong>
          <small>{status.liveModel.artifactPresent ? "Promoted on the locked test" : "No unvalidated forecast is shown"}</small>
        </div>
      </section>

      <section className="collection-meter" aria-label={`${collection.usableRuns} of ${collection.targetRuns} usable personal runs`}>
        <div><span>{collection.usableRuns}</span><small>/ {collection.targetRuns} usable runs</small></div>
        <div className="lab-progress"><i style={{ width: `${progress}%` }} /></div>
        <p>{collection.remainingRuns > 0 ? `${collection.remainingRuns} clean runs remain` : "Initial collection complete"} · {collection.excludedRuns} excluded</p>
      </section>

      <section className="pipeline-grid" aria-label="Personal training pipeline">
        <article className="pipeline-card personal-card">
          <div className="pipeline-step"><span>01</span><b>Development set</b></div>
          <strong>{collection.developmentRuns} / {collection.developmentTarget}</strong>
          <p>Chronological personal runs used for model fitting and run-grouped calibration.</p>
          <div className="lab-progress personal"><i style={{ width: `${Math.min(100, collection.developmentRuns / collection.developmentTarget * 100)}%` }} /></div>
          <small>Experimental candidates refresh at 20, 25, 30, 35, and 40 runs.</small>
        </article>
        <article className="pipeline-card locked-card">
          <div className="pipeline-step"><span>02</span><b>Locked future test</b></div>
          <strong>{collection.lockedTestRuns} / {collection.lockedTestTarget}</strong>
          <p>Runs 41–50 are never used to fit features, thresholds, calibration, or certainty tiers.</p>
          <div className="lab-progress locked"><i style={{ width: `${Math.min(100, collection.lockedTestRuns / collection.lockedTestTarget * 100)}%` }} /></div>
          <small>{collection.phase === "locked_test" ? "Candidate frozen; keep playing with collection-only overlay." : collection.phase === "evaluation" ? "Locked test available for final gates." : "Unlocks after 40 usable runs."}</small>
        </article>
        <article className="pipeline-card">
          <div className="pipeline-step"><span>03</span><b>Local training job</b></div>
          <strong>{status.trainingJob.status === "running" ? "Training now" : status.trainingJob.status === "failed" ? "Needs attention" : status.trainingJob.status === "complete" ? `Built at ${status.trainingJob.usableRuns} runs` : "Waiting for milestone"}</strong>
          <p>{status.trainingJob.step ? `Current step: ${status.trainingJob.step}` : status.trainingJob.error || "scikit-learn runs first; PyTorch follows when installed."}</p>
          <small>{latestCheckpoint ? `Checkpoint secured at ${latestCheckpoint.usableRuns} runs.` : "Full local checkpoints are created at runs 20, 40, and 50."}</small>
        </article>
      </section>

      <section className="run-health-panel panel">
        <div className="panel-heading"><div><p className="eyebrow">COLLECTION RECEIPTS</p><h2>Recent v3 run health</h2></div><span>{officialRecentRuns.length} shown · {collection.archivedLegacyRuns} legacy archived</span></div>
        {officialRecentRuns.length === 0 ? (
          <div className="run-health-empty"><strong>No v3 run has completed yet.</strong><p>Use the first three runs to verify normal play, an in-game pause, and a hidden-tab pause before bulk collection.</p></div>
        ) : (
          <div className="run-health-list">
            {officialRecentRuns.map((run) => (
              <Link href={`/sessions/${run.id}`} key={run.id}>
                <span className={run.usable ? "receipt accepted" : "receipt excluded"}>{run.usable ? "Accepted" : "Excluded"}</span>
                <span><strong>{new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(run.startedAt))}</strong><small>{run.reason}</small></span>
                <span><strong>{run.expectedFrameCount ? `${Math.round(run.captureCoverage * 100)}%` : "—"}</strong><small>{run.validFrameCount} / {run.expectedFrameCount} frames</small></span>
                <span><strong>{run.pauseCount}</strong><small>pauses</small></span>
                <span><strong>{run.hasRecording ? "Ready" : "Pending"}</strong><small>recording</small></span>
              </Link>
            ))}
          </div>
        )}
      </section>

      {exclusionSummary.length > 0 ? <section className="exclusion-panel panel">
        <div className="panel-heading"><div><p className="eyebrow">DATA QUALITY</p><h2>Excluded runs</h2></div><span>{collection.excludedRuns} total</span></div>
        <div className="exclusion-list">{exclusionSummary.map(([reason, count]) => <p key={reason}><span>{reason}</span><strong>{count}</strong></p>)}</div>
      </section> : null}

      <section className="evaluation-section">
        <div className="section-heading"><div><p className="eyebrow">HELD-OUT EVALUATION</p><h2>Candidate models</h2></div><span>{status.evaluations.length} evaluated</span></div>
        {status.evaluations.length === 0 ? <div className="lab-empty"><strong>No personal candidate yet.</strong><p>The first automatic training job starts after 20 usable runs. Forecasts remain hidden.</p></div> : <div className="evaluation-list">
          {status.evaluations.map((evaluation) => {
            const failedGates = Object.entries(evaluation.gates).filter(([, passed]) => !passed).map(([name]) => name);
            return <article key={evaluation.id}>
              <div className="evaluation-heading"><div><small>{evaluation.split}</small><h3>{evaluation.model}</h3></div><span className={evaluation.eligibleForLive ? "gate-pass" : "gate-hold"}>{evaluation.eligibleForLive ? "Promoted" : "Held back"}</span></div>
              <MetricRow metrics={evaluation.metrics} />
              <p className="improvement-line">Brier vs elapsed {percent(evaluation.metrics.brierImprovements?.elapsed_time_only)} · Time MAE vs average duration {percent(evaluation.metrics.timeToFailureImprovements?.average_duration)} · p95 {typeof evaluation.metrics.latencyMsP95 === "number" ? `${evaluation.metrics.latencyMsP95.toFixed(0)} ms` : "—"}</p>
              {!evaluation.eligibleForLive && <p className="gate-reasons">Outstanding gates: {failedGates.join(", ") || "evaluation incomplete"}</p>}
            </article>;
          })}
        </div>}
      </section>
    </main>
  );
}
