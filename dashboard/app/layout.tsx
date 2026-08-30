import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "PlayLens — Playtest diagnostics",
  description: "Local, pixel-only personal loss-window forecasting for Hextris.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <Link className="wordmark" href="/" aria-label="PlayLens home">
            <span className="wordmark-mark">P</span>
            <span>PLAYLENS</span>
          </Link>
          <nav className="site-nav" aria-label="PlayLens sections">
            <Link href="/">Sessions</Link>
            <Link href="/model">Model Lab</Link>
            <div className="local-chip"><i /> Local only</div>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
