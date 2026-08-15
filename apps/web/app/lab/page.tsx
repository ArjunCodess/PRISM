import { Shell } from "@/components/shell";
import { loadMetrics } from "@/lib/data";

export default async function LabPage() {
  const metrics = await loadMetrics();
  const rows = [
    ["Ensemble MAE", metrics?.ensemble.mae],
    ["Persistence MAE", metrics?.persistence.mae],
    ["MAE improvement", metrics?.improvement.mae_improvement],
    ["ESA-style loss", metrics?.ensemble.esa_loss],
    ["F2", metrics?.ensemble.f2],
    ["PR-AUC", metrics?.warning.pr_auc],
    ["Brier", metrics?.warning.brier],
  ];
  return (
    <Shell title="Model laboratory">
      <section className="grid gap-3 md:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={String(label)} className="rounded-lg border border-white/10 bg-panel p-4">
            <p className="text-xs uppercase tracking-widest text-slate-400">{label}</p>
            <p className="telemetry mt-2 font-mono text-2xl">
              {typeof value === "number" ? value.toFixed(3) : "n/a"}
            </p>
          </div>
        ))}
      </section>
      <section className="rounded-lg border border-white/10 bg-panel p-4 text-sm leading-7 text-slate-300">
        <h2 className="text-cyan">Limitations</h2>
        <p>
          Local metrics use synthetic CDM histories that follow the ESA challenge schema, grouped
          event splits, and a T-48 cutoff. They are not a ranking on the original 2019 hidden test
          set. Persistence is a strong baseline. SHAP explains the trained model, not physical cause.
        </p>
      </section>
    </Shell>
  );
}
