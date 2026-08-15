export function Band({ value, abstained }: { value: string; abstained: boolean }) {
  const label = abstained ? "REVIEW REQUIRED" : value.toUpperCase();
  const tone =
    abstained || value === "review"
      ? "text-amber border-amber"
      : value === "high"
        ? "text-alert border-alert"
        : "text-cyan border-cyan";
  return (
    <span className={`rounded border px-2 py-1 font-mono text-xs ${tone}`} role="status">
      {label}
    </span>
  );
}
