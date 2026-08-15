import { readFile } from "node:fs/promises";
import path from "node:path";
import type { DemoCase, MetricsFile } from "./types";

async function readPublicJson<T>(name: string): Promise<T> {
  const file = path.join(process.cwd(), "public", name);
  const raw = await readFile(file, "utf8");
  return JSON.parse(raw) as T;
}

export async function loadCases(): Promise<DemoCase[]> {
  const api = process.env.NEXT_PUBLIC_API_URL;
  if (api) {
    try {
      const response = await fetch(`${api}/v1/cases`, { cache: "no-store" });
      if (response.ok) {
        return (await response.json()) as DemoCase[];
      }
    } catch {
      // fall through to offline cache
    }
  }
  return readPublicJson<DemoCase[]>("demo_cases.json");
}

export async function loadMetrics(): Promise<MetricsFile | null> {
  const api = process.env.NEXT_PUBLIC_API_URL;
  if (api) {
    try {
      const response = await fetch(`${api}/v1/model-card`, { cache: "no-store" });
      if (response.ok) {
        const card = await response.json();
        return card.metricsFull as MetricsFile;
      }
    } catch {
      // offline
    }
  }
  try {
    return await readPublicJson<MetricsFile>("metrics.json");
  } catch {
    return null;
  }
}

export function formatLogRisk(value: number): string {
  return value.toFixed(2);
}

export function formatPc(value: number): string {
  if (value <= 0) return "0";
  const exp = Math.log10(value);
  return `1 in ${Math.round(10 ** -exp).toLocaleString()}`;
}
