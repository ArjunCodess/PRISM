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
      <h2 className="text-sm uppercase tracking-widest text-cyan">Why this forecast</h2>
      <p className="mt-2 text-sm leading-6 text-slate-200">{text}</p>
      <ul className="mt-4 space-y-2">
        {factors.map((factor) => (
          <li key={factor.feature} className="flex items-center justify-between gap-3 text-sm">
            <span>
              {factor.label ?? factor.feature}
              <span className="ml-2 text-xs uppercase text-slate-400">{factor.direction}</span>
            </span>
            <span className="telemetry font-mono text-cyan">{factor.contribution.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
