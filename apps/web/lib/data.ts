import { readFile } from "node:fs/promises";
import path from "node:path";
import type { DemoCase, MetricsFile, Prediction } from "./types";

export function apiBase(): string | undefined {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!raw) return undefined;
  return raw.replace(/\/$/, "");
}

function allowOfflineFallback(): boolean {
  return process.env.NODE_ENV !== "production";
}

async function readPublicJson<T>(name: string): Promise<T> {
  const file = path.join(process.cwd(), "public", name);
  const raw = await readFile(file, "utf8");
  return JSON.parse(raw) as T;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...init });
  if (!response.ok) {
    throw new Error(`${response.status} from ${url}`);
  }
  return (await response.json()) as T;
}

export async function loadCases(): Promise<DemoCase[]> {
  const api = apiBase();
  if (api) {
    try {
      return await fetchJson<DemoCase[]>(`${api}/v1/cases`);
    } catch (error) {
      if (!allowOfflineFallback()) throw error;
    }
  } else if (!allowOfflineFallback()) {
    throw new Error("NEXT_PUBLIC_API_URL is required in production so the exhibit can load live cases.");
  }
  return readPublicJson<DemoCase[]>("demo_cases.json");
}

export async function loadMetrics(): Promise<MetricsFile | null> {
  const api = apiBase();
  if (api) {
    try {
      const card = await fetchJson<{ metricsFull?: MetricsFile }>(`${api}/v1/model-card`);
      return card.metricsFull ?? null;
    } catch (error) {
      if (!allowOfflineFallback()) throw error;
    }
  } else if (!allowOfflineFallback()) {
    throw new Error("NEXT_PUBLIC_API_URL is required in production so the exhibit can load live metrics.");
  }
  try {
    return await readPublicJson<MetricsFile>("metrics.json");
  } catch {
    return null;
  }
}

export async function loadLiveCase(id: string): Promise<DemoCase | null> {
  const item = (await loadCases()).find((entry) => entry.id === id);
  if (!item) return null;
  const api = apiBase();
  if (!api) {
    if (!allowOfflineFallback()) {
      throw new Error("NEXT_PUBLIC_API_URL is required in production so cases can run live inference.");
    }
    return item;
  }
  const prediction = await fetchJson<Prediction>(`${api}/v1/risk/predict`, {
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
