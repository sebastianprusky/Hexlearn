"use client";

export default function ErrorPage({ reset }: { reset: () => void }) {
  return <main className="message-page"><p className="eyebrow">LOCAL SERVICE</p><h1>The report could not load.</h1><p>Check that Hexlearn is running on port 8000.</p><button className="primary-link" onClick={reset}>Try again</button></main>;
}
