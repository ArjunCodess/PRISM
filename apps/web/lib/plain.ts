import { formatPc } from "./format";

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

