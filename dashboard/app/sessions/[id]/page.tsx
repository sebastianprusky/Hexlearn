import Link from "next/link";
import { notFound } from "next/navigation";

import SessionReport from "@/components/session-report";
import { apiBase, getSession } from "@/lib/api";

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let session;
  try {
    session = await getSession(id);
  } catch (error) {
    if (error instanceof Error && error.message.includes("404")) notFound();
    throw error;
  }
  return (
    <main className="report-page">
      <Link className="back-link" href="/">← All sessions</Link>
      <SessionReport apiBase={apiBase} session={session} />
    </main>
  );
}
