import type { DemoCase, MetricsFile, Prediction } from "./types";

export function apiBase(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!raw) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is missing. Start the FastAPI process and set the origin in apps/web/.env.local.",
    );
  }
  return raw.replace(/\/$/, "");
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", ...init });
  } catch {
    throw new Error(`The PRISM API is not reachable at ${url}. Start FastAPI and retry.`);
  }
  if (!response.ok) {
    throw new Error(`The PRISM API returned ${response.status} from ${url}.`);
  }
  return (await response.json()) as T;
}

export async function loadCases(): Promise<DemoCase[]> {
  return fetchJson<DemoCase[]>(`${apiBase()}/v1/cases`);
}

export async function loadMetrics(): Promise<MetricsFile> {
  const card = await fetchJson<{ metricsFull?: MetricsFile }>(`${apiBase()}/v1/model-card`);
  if (!card.metricsFull) {
    throw new Error("The PRISM API did not return metrics on /v1/model-card.");
  }
  return card.metricsFull;
}

export async function loadLiveCase(id: string): Promise<DemoCase | null> {
  const item = (await loadCases()).find((entry) => entry.id === id);
  if (!item) return null;
  const prediction = await fetchJson<Prediction>(`${apiBase()}/v1/risk/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      eventId: item.id,
      cutoffHours: 48,
      messages: item.messages,
    }),
  });
  return { ...item, prediction };
}
