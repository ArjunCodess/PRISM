export function Band({ value, abstained, compact = false }: { value: string; abstained: boolean; compact?: boolean }) {
  const label = abstained || value === "review" ? "Review required" : value === "high" ? "High warning" : "Low concern";
  const symbol = abstained || value === "review" ? "—" : value === "high" ? "↑" : "·";
  const tone = abstained || value === "review" ? "border-amber/25 bg-amber/[0.07] text-amber" : value === "high" ? "border-alert/25 bg-alert/[0.07] text-alert" : "border-safe/25 bg-safe/[0.07] text-safe";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border font-medium ${compact ? "px-2.5 py-1 text-[0.65rem]" : "px-3 py-1.5 text-xs"} ${tone}`} role="status">
      <span aria-hidden="true">{symbol}</span><span>{label}</span>
    </span>
  );
}
