import type { ModelStatus, SessionDetail, SessionSummary } from "@/lib/types";

export const apiBase = process.env.PLAYLENS_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`PlayLens API returned ${response.status}`);
  return response.json() as Promise<T>;
}

export const getSessions = () => request<SessionSummary[]>("/api/v1/sessions");
export const getSession = (id: string) => request<SessionDetail>(`/api/v1/sessions/${encodeURIComponent(id)}`);
export const getModelStatus = () => request<ModelStatus>("/api/v1/ml/status");
