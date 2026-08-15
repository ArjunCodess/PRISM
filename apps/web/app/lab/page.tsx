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
    ["Snapshot-only MAE", metrics?.ablation?.snapshot_mae],
  ];
  return (
    <Shell title="Model laboratory">
      <section className="grid gap-3 md:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={String(label)} className="rounded-lg border border-white/10 bg-panel p-4">
            <p className="text-xs uppercase tracking-widest text-slate-400">{label}</p>
            <p className="telemetry mt-2 font-mono text-2xl">
              {typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "n/a"}
            </p>
          </div>
        ))}
      </section>

      <section className="rounded-lg border border-white/10 bg-panel p-4">
        <h2 className="text-sm uppercase tracking-widest text-cyan">Calibration</h2>
        <p className="mt-2 text-sm text-slate-400">
          Predicted warning probability versus observed high-risk frequency on the hold-out set.
        </p>
        <div className="mt-4 space-y-2">
          {(metrics?.calibration ?? []).map((bin) => (
            <div key={bin.mid} className="grid grid-cols-[7rem_1fr_7rem] items-center gap-3 text-sm">
              <span className="telemetry font-mono">{(bin.predicted * 100).toFixed(0)}% pred</span>
              <div className="h-2 rounded bg-white/10">
                <div className="h-2 rounded bg-cyan" style={{ width: `${Math.min(100, bin.observed * 100)}%` }} />
              </div>
              <span className="telemetry font-mono text-right">{(bin.observed * 100).toFixed(0)}% obs · n={bin.n}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-panel p-4">
        <h2 className="text-sm uppercase tracking-widest text-cyan">Grouped feature importance</h2>
        <ul className="mt-3 space-y-2 text-sm">
          {(metrics?.featureGroups ?? []).map((item) => (
            <li key={item.group} className="flex justify-between gap-4">
              <span>{item.group}</span>
              <span className="telemetry font-mono text-cyan">{item.gain.toFixed(1)}</span>
            </li>
          ))}
        </ul>
      </section>

      <FailureTable title="Worst under-predictions" rows={metrics?.failures?.worstUnderpredictions} />
      <FailureTable title="Missed high-risk events" rows={metrics?.failures?.missedHighRisk} />
      <FailureTable title="False escalations" rows={metrics?.failures?.falseEscalations} />

      <section className="rounded-lg border border-white/10 bg-panel p-4 text-sm leading-7 text-slate-300">
        <h2 className="text-cyan">Provenance and limitations</h2>
        <p>
          Local metrics use synthetic CDM histories that follow the ESA challenge schema, grouped
          event splits, and a T-48 cutoff. They are not a ranking on the original 2019 hidden test
          set. Persistence is a strong baseline. SHAP explains the trained model, not physical cause.
          Educational threshold is log10 Pc = -6, not an ISRO rule.
        </p>
      </section>
    </Shell>
  );
}

function FailureTable({
  title,
  rows,
}: {
  title: string;
  rows?: Array<Record<string, number>>;
}) {
  if (!rows?.length) {
    return (
      <section className="rounded-lg border border-white/10 bg-panel p-4 text-sm text-slate-400">
        {title}: none in the hold-out set.
      </section>
    );
  }
  return (
    <section className="overflow-hidden rounded-lg border border-white/10">
      <h2 className="bg-white/5 px-4 py-3 text-sm uppercase tracking-widest text-cyan">{title}</h2>
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase tracking-widest text-slate-400">
          <tr>
            <th className="px-4 py-2">Event</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Persistence</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.eventId} className="border-t border-white/10">
              <td className="px-4 py-2 font-mono">{row.eventId}</td>
              <td className="font-mono">{row.actual.toFixed(2)}</td>
              <td className="font-mono">{row.predicted.toFixed(2)}</td>
              <td className="font-mono">{row.persistence.toFixed(2)}</td>
              <td className="font-mono">{row.error.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
