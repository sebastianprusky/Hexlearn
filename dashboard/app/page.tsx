import Link from "next/link";

import { getSessions } from "@/lib/api";
import type { SessionSummary } from "@/lib/types";

function duration(session: SessionSummary) {
  if (session.activeDurationMs > 0) {
    const activeSeconds = Math.round(session.activeDurationMs / 1_000);
    return `${Math.floor(activeSeconds / 60)}m ${String(activeSeconds % 60).padStart(2, "0")}s active`;
  }
  if (!session.endedAt) return "Live now";
  const seconds = Math.max(0, Math.round((Date.parse(session.endedAt) - Date.parse(session.startedAt)) / 1_000));
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

export default async function Home() {
  let sessions: SessionSummary[] = [];
  let connected = true;
  try {
    sessions = await getSessions();
  } catch {
    connected = false;
  }

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">HEXTRIS PLAYTESTING</p>
          <h1>See a loss coming<br />before the run breaks.</h1>
          <p className="hero-copy">
            Personal gameplay pixels become a calibrated loss window—then a timeline you can inspect after every run.
          </p>
        </div>
        <aside className="model-card">
          <span className="model-label">Inference status</span>
          <strong>{connected ? "Service connected" : "Service offline"}</strong>
          <p>{connected ? "Collection, model training, and recordings stay on this computer." : "Start the combined Hexlearn server on port 8000."}</p>
          <span className={connected ? "status-line ready" : "status-line offline"}><i /> {connected ? "Ready to collect a run" : "Waiting for API"}</span>
          <Link className="model-link" href="/model">Open Model Lab <span>↗</span></Link>
        </aside>
      </section>

      <section className="sessions-section">
        <div className="section-heading">
          <div><p className="eyebrow">SESSION LIBRARY</p><h2>Recent runs</h2></div>
          <span>{sessions.length} captured</span>
        </div>

        {sessions.length === 0 ? (
          <div className="empty-state">
            <div className="empty-hex">⌁</div>
            <h3>No playtests yet</h3>
            <p>Open Hextris, click the Hexlearn extension, and begin a run. The first report appears here automatically.</p>
            <code>http://127.0.0.1:8000</code>
          </div>
        ) : (
          <div className="session-grid">
            {sessions.map((session, index) => (
              <Link className="session-card" href={`/sessions/${session.id}`} key={session.id}>
                <div className="run-index">{String(sessions.length - index).padStart(2, "0")}</div>
                <div className="session-main">
                  <div className="session-topline">
                    <span className={`run-state ${session.status}`}>{session.status}</span>
                    <time>{new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(session.startedAt))}</time>
                  </div>
                  <h3>Hextris run</h3>
                  <div className="session-meta"><span>{duration(session)}</span><span>{session.usableForTraining ? "Training eligible" : session.qualityReason || "Quality pending"}</span><span>{session.hasRecording ? "Recording ready" : "Signals only"}</span></div>
                </div>
                <span className="arrow">↗</span>
              </Link>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
