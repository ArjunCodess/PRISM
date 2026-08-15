import { factorDirection } from "@/lib/plain";
import type { Factor } from "@/lib/types";

export function Explanation({
  text,
  factors,
}: {
  text?: string;
  factors: Factor[];
}) {
  return (
    <section className="rounded-lg border border-white/10 bg-panel p-4">
      <h2 className="text-sm uppercase tracking-widest text-cyan">Why</h2>
      <p className="mt-2 text-sm leading-6 text-slate-200">{text}</p>
      <ul className="mt-4 space-y-2 text-sm">
        {factors.map((factor) => (
          <li key={factor.feature} className="flex items-start justify-between gap-3">
            <span className="text-slate-200">{factor.label ?? factor.feature}</span>
            <span className="shrink-0 text-xs uppercase text-cyan">{factorDirection(factor.direction)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
