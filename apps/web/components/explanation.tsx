import { factorDirection } from "@/lib/plain";
import type { Factor } from "@/lib/types";

export function Explanation({ text, factors }: { text?: string; factors: Factor[] }) {
  const max = Math.max(...factors.map((factor) => Math.abs(factor.contribution)), 0.01);
  return (
    <section className="panel p-5" aria-labelledby="explanation-title">
      <p className="eyebrow">Verified model explanation</p>
      <h2 id="explanation-title" className="display mt-2 text-3xl">Why the forecast moved</h2>
      <p className="mt-3 text-sm leading-6 text-stone-600">{text ?? "No explanation was exported for this case."}</p>
      <div className="mt-5 space-y-4">
        {factors.slice(0, 6).map((factor) => {
          const higher = factor.direction === "higher";
          return (
            <div key={factor.feature}>
              <div className="mb-1.5 flex items-start justify-between gap-4 text-xs">
                <span className="text-stone-700">{factor.label ?? factor.feature}</span>
                <span className={`telemetry shrink-0 ${higher ? "text-amber" : "text-safe"}`}>{higher ? "+" : "−"}{Math.abs(factor.contribution).toFixed(2)}</span>
              </div>
              <div className="h-1 rounded-full bg-stone-100">
                <div className={`h-full ${higher ? "bg-amber" : "bg-safe"}`} style={{ width: `${Math.max(4, Math.abs(factor.contribution) / max * 100)}%` }} />
              </div>
              <span className="sr-only">{factorDirection(factor.direction)}</span>
            </div>
          );
        })}
      </div>
      <p className="mt-5 border-t hairline pt-4 text-[0.68rem] leading-5 text-stone-500">SHAP describes this trained model’s output. It does not establish physical causation.</p>
    </section>
  );
}
