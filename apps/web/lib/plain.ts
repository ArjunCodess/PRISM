import { formatPc } from "./format";
import type { DemoCase } from "./types";

export function chanceWords(logRisk: number): string {
  const pc = formatPc(10 ** logRisk);
  return pc.startsWith("1 in") ? `about ${pc}` : pc;
}

export function hoursUntilClosest(days: number): string {
  return `${(days * 24).toFixed(0)} h`;
}

export function factorDirection(direction: string): string {
  return direction === "higher" ? "more worrying" : "calmer";
}

export function fallbackBriefing(
  item: Pick<DemoCase, "baselineRiskLog10" | "prediction">,
): string {
  const today = chanceWords(item.baselineRiskLog10);
  const guess = chanceWords(item.prediction.predictedFinalRiskLog10);
  if (item.prediction.abstained) {
    return `Today ${today}. Guesses cross the warning line.`;
  }
  return `Today ${today}. Forecast ${guess}.`;
}
