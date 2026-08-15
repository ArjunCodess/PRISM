export function formatLogRisk(value: number): string {
  return value.toFixed(2);
}

export function formatPc(value: number): string {
  if (value <= 0) return "0";
  const exp = Math.log10(value);
  if (!Number.isFinite(exp) || exp <= -9) return "less than 1 in a billion";
  return `1 in ${Math.round(10 ** -exp).toLocaleString("en-US")}`;
}
