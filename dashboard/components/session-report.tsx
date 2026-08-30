"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import type { Insight, LossWindow, SessionDetail } from "@/lib/types";

const insightLabel: Record<Insight["kind"], string> = {
  loss_signal: "Sustained loss signal",
  imminent: "Loss imminent",
  forecast_shift: "Forecast moved sharply",
};

const actualWindow = (remaining: number): LossWindow => {
  if (remaining <= 10) return "under_10s";
  if (remaining <= 20) return "10_to_20s";
  if (remaining <= 30) return "20_to_30s";
  if (remaining <= 45) return "30_to_45s";
  if (remaining <= 60) return "45_to_60s";
  return "none";
};

const formatSeconds = (seconds: number) => `${Math.max(0, Math.round(seconds))}s`;

function chartPath(points: Array<[number, number]>, durationMs: number) {
  return points.map(([activeMs, remaining], index) => {
    const x = (activeMs / Math.max(1, durationMs)) * 1000;
    const y = 180 - (Math.max(0, Math.min(60, remaining)) / 60) * 180;
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export default function SessionReport({ session, apiBase }: { session: SessionDetail; apiBase: string }) {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [deleting, setDeleting] = useState(false);
  const predictions = useMemo(
    () => session.predictions.filter((prediction) => prediction.activeElapsedMs !== null),
    [session.predictions],
  );
  const durationMs = Math.max(1, session.activeDurationMs || predictions.at(-1)?.activeElapsedMs || 1);

  const stats = useMemo(() => {
    const errors = predictions.flatMap((prediction) => prediction.estimatedRemainingSeconds === null ? [] : [
      Math.abs(prediction.estimatedRemainingSeconds - Math.min(60, (durationMs - (prediction.activeElapsedMs || 0)) / 1_000)),
    ]);
    const firstWarning = predictions.find((prediction) => prediction.lossWindow && prediction.lossWindow !== "none");
    const warningLead = firstWarning ? Math.max(0, (durationMs - (firstWarning.activeElapsedMs || 0)) / 1_000) : 0;
    const windowMatches = predictions.filter((prediction) => {
      const remaining = (durationMs - (prediction.activeElapsedMs || 0)) / 1_000;
      return prediction.lossWindow === actualWindow(remaining);
    }).length;
    const falseEarly = predictions.filter((prediction) => prediction.lossWindow !== "none" && (durationMs - (prediction.activeElapsedMs || 0)) / 1_000 > 60).length;
    const latencies = predictions.map((prediction) => prediction.latencyMs).sort((left, right) => left - right);
    return {
      mae: errors.length ? errors.reduce((total, value) => total + value, 0) / errors.length : null,
      warningLead,
      windowAccuracy: predictions.length ? windowMatches / predictions.length : null,
      falseEarly,
      latency: latencies.length ? latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * 0.95))] : null,
    };
  }, [durationMs, predictions]);

  const predictionPoints = predictions.flatMap((prediction) => prediction.estimatedRemainingSeconds === null ? [] : [
    [prediction.activeElapsedMs || 0, prediction.estimatedRemainingSeconds] as [number, number],
  ]);
  const truthStart = Math.min(60, durationMs / 1_000);
  const truthPoints: Array<[number, number]> = durationMs > 60_000
    ? [[0, 60], [durationMs - 60_000, 60], [durationMs, 0]]
    : [[0, truthStart], [durationMs, 0]];
  const baselineDuration = session.baselineDurationSeconds;
  const baselinePoints: Array<[number, number]> = baselineDuration === null ? [] : [
    [0, Math.min(60, baselineDuration)],
    [durationMs, Math.max(0, Math.min(60, baselineDuration - durationMs / 1_000))],
  ];
  const uncertaintyPoints = predictions.filter((prediction) => prediction.uncertaintyLowSeconds !== null && prediction.uncertaintyHighSeconds !== null);
  const uncertaintyPolygon = [
    ...uncertaintyPoints.map((prediction) => {
      const x = ((prediction.activeElapsedMs || 0) / durationMs) * 1000;
      const y = 180 - (Math.min(60, prediction.uncertaintyHighSeconds || 0) / 60) * 180;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }),
    ...[...uncertaintyPoints].reverse().map((prediction) => {
      const x = ((prediction.activeElapsedMs || 0) / durationMs) * 1000;
      const y = 180 - (Math.min(60, prediction.uncertaintyLowSeconds || 0) / 60) * 180;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }),
  ].join(" ");

  const seek = (activeElapsedMs: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, activeElapsedMs / 1_000);
    void videoRef.current.play();
  };

  const remove = async () => {
    if (!window.confirm("Delete this local recording, frames, and report? This cannot be undone.")) return;
    setDeleting(true);
    const response = await fetch(`${apiBase}/api/v1/sessions/${session.id}`, { method: "DELETE" });
    if (response.ok) router.push("/");
    else setDeleting(false);
  };

  return (
    <>
      <section className="report-title">
        <div><p className="eyebrow">HEXTRIS / RUN REPORT</p><h1>{new Intl.DateTimeFormat("en", { month: "long", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(session.startedAt))}</h1></div>
        <div className="report-actions"><span className={predictions.length ? "validated-badge" : "experimental-badge"}>{predictions.length ? "Validated forecast" : "Collection run"}</span><button className="delete-button" disabled={deleting} onClick={remove}>{deleting ? "Deleting…" : "Delete run"}</button></div>
      </section>

      <div className={session.usableForTraining ? "quality-banner usable" : "quality-banner excluded"}>
        <strong>{session.usableForTraining ? "Included in personal training." : "Not included in personal training."}</strong> {session.qualityReason || "Quality evaluation pending."}
      </div>
      {!predictions.length && <div className="truth-banner"><strong>No experimental forecast was shown.</strong> Initial collection runs capture clean behavior; forecasts remain hidden until a model passes the locked test.</div>}

      <section className="stat-grid">
        <article><span>Active duration</span><strong>{formatSeconds(durationMs / 1_000)}</strong></article>
        <article><span>{predictions.length ? "First-warning lead" : "Capture coverage"}</span><strong>{predictions.length ? (stats.warningLead ? formatSeconds(stats.warningLead) : "None") : `${Math.round(session.captureCoverage * 100)}%`}</strong></article>
        <article><span>Window accuracy</span><strong>{stats.windowAccuracy === null ? "—" : `${Math.round(stats.windowAccuracy * 100)}%`}</strong></article>
        <article><span>Time estimate MAE</span><strong>{stats.mae === null ? "—" : formatSeconds(stats.mae)}</strong></article>
      </section>

      <section className="analysis-grid">
        <article className="video-panel panel">
          <div className="panel-heading"><div><p className="eyebrow">GAMEPLAY</p><h2>Run replay</h2></div><span>{session.hasRecording ? "Active-time synchronized" : "Not recorded"}</span></div>
          {session.hasRecording ? <video ref={videoRef} controls preload="metadata" src={`${apiBase}/api/v1/sessions/${session.id}/recording`} /> : <div className="video-empty">The canvas recording was unavailable. Training-quality metadata is still preserved.</div>}
        </article>

        <article className="moments-panel panel">
          <div className="panel-heading"><div><p className="eyebrow">FORECAST MOMENTS</p><h2>{session.insights.length} to review</h2></div></div>
          <div className="moments-list">
            {session.insights.length === 0 ? <p className="quiet">No validated forecast moments were recorded in this collection run.</p> : session.insights.map((insight, index) => <button key={`${insight.kind}-${insight.startMs}-${index}`} onClick={() => seek(insight.startActiveMs || 0)}><span className={`moment-mark ${insight.kind}`} /><span><strong>{insightLabel[insight.kind]}</strong><small>{formatSeconds((insight.startActiveMs || 0) / 1_000)} active</small></span><i>▶</i></button>)}
          </div>
        </article>
      </section>

      <section className="timeline-panel panel">
        <div className="panel-heading"><div><p className="eyebrow">TIME TO FAILURE</p><h2>Forecast versus the true countdown</h2></div><div className="legend"><span><i className="estimate-key" /> Model</span><span><i className="truth-key" /> Truth</span><span><i className="baseline-key" /> Average-run baseline</span></div></div>
        <div className="chart-wrap">
          <svg className="timeline" role="img" aria-label="Estimated and true active seconds remaining over the run" viewBox="0 0 1000 220" preserveAspectRatio="none" onClick={(event) => { const box = event.currentTarget.getBoundingClientRect(); seek(((event.clientX - box.left) / box.width) * durationMs); }}>
            {[0, 15, 30, 45, 60].map((seconds) => <g key={seconds}><line x1="0" x2="1000" y1={180 - seconds * 3} y2={180 - seconds * 3} className="threshold"/><text x="2" y={174 - seconds * 3} className="axis-label">{seconds}s</text></g>)}
            {session.lifecycleEvents.filter((event) => event.state === "paused").map((event, index) => <line key={`${event.timestampMs}-${index}`} x1={(event.activeElapsedMs / durationMs) * 1000} x2={(event.activeElapsedMs / durationMs) * 1000} y1="0" y2="180" className="pause-marker" />)}
            {uncertaintyPolygon ? <polygon points={uncertaintyPolygon} className="uncertainty-band" /> : null}
            {baselinePoints.length > 0 && <path d={chartPath(baselinePoints, durationMs)} className="baseline-line" />}
            <path d={chartPath(truthPoints, durationMs)} className="truth-line" />
            {predictionPoints.length > 0 && <path d={chartPath(predictionPoints, durationMs)} className="estimate-line" />}
          </svg>
          <div className="chart-labels"><span>0:00 active</span><span>{formatSeconds(durationMs / 1_000)} active</span></div>
          {!predictions.length && <p className="chart-note">The diagonal is the true time remaining. A model line will appear only after promotion.</p>}
        </div>
      </section>

      <footer className="report-footer"><p>{session.validFrameCount} / {session.expectedFrameCount} frames · {session.lifecycleEvents.filter((event) => event.state === "paused").length} pause{session.lifecycleEvents.filter((event) => event.state === "paused").length === 1 ? "" : "s"} excluded from active time · {stats.falseEarly} false early warning{stats.falseEarly === 1 ? "" : "s"} · {stats.latency === null ? "No inference" : `${Math.round(stats.latency)} ms p95`}.</p><p>Raw gameplay and model outputs remain in the local PlayLens data directory.</p></footer>
    </>
  );
}
