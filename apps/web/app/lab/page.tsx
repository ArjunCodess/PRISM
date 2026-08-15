import { Shell } from "@/components/shell";
import { loadMetrics } from "@/lib/data";

export default async function LabPage() {
  const metrics = await loadMetrics();
  const rows = [
    ["Copilot MAE", metrics?.ensemble.mae, "typical forecast error; lower is better"],
    ["Persistence MAE", metrics?.persistence.mae, "error if we copy today's number"],
    ["MAE improvement", metrics?.improvement.mae_improvement, "positive means the copilot beat persistence"],
    ["ESA-style loss", metrics?.ensemble.esa_loss, "high-risk error divided by F₂"],
    ["F₂", metrics?.ensemble.f2, "how well true high-risk events are caught"],
    ["PR-AUC", metrics?.warning.pr_auc, "warning ranking quality"],
    ["Brier", metrics?.warning.brier, "warning calibration; lower is better"],
    ["Snapshot-only MAE", metrics?.ablation?.snapshot_mae, "error using only the latest early message"],
  ] as const;
  return (
    <Shell title="Model laboratory">
      <p className="text-sm text-slate-400">
        Held-out synthetic events in the ESA schema, not the 2019 hidden test ranking.
      </p>
      <section className="grid gap-3 md:grid-cols-2">
        {rows.map(([label, value, hint]) => (
          <div key={label} className="rounded-lg border border-white/10 bg-panel p-4">
            <p className="text-xs uppercase tracking-widest text-slate-400">{label}</p>
            <p className="telemetry mt-2 font-mono text-2xl">
              {typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "n/a"}
            </p>
            <p className="mt-1 text-xs text-slate-500">{hint}</p>
          </div>
        ))}
      </section>

      <section className="rounded-lg border border-white/10 bg-panel p-4">
        <h2 className="text-sm uppercase tracking-widest text-cyan">Calibration</h2>
        <p className="mt-2 text-xs text-slate-400">Stated warning chance vs how often it came true.</p>
        <div className="mt-4 space-y-2">
          {(metrics?.calibration ?? []).map((bin) => (
            <div key={bin.mid} className="grid grid-cols-[7rem_1fr_8rem] items-center gap-3 text-sm">
              <span className="font-mono">{(bin.predicted * 100).toFixed(0)}% said</span>
              <div className="h-2 rounded bg-white/10">
                <div className="h-2 rounded bg-cyan" style={{ width: `${Math.min(100, bin.observed * 100)}%` }} />
              </div>
              <span className="font-mono text-right text-slate-400">
                {(bin.observed * 100).toFixed(0)}% happened · n={bin.n}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-panel p-4">
        <h2 className="text-sm uppercase tracking-widest text-cyan">What the model uses</h2>
        <ul className="mt-3 space-y-2 text-sm">
          {(metrics?.featureGroups ?? []).map((item) => (
            <li key={item.group} className="flex justify-between gap-4">
              <span>{item.group}</span>
              <span className="font-mono text-cyan">{item.gain.toFixed(1)}</span>
            </li>
          ))}
        </ul>
      </section>

      <FailureTable title="Worst under-predictions" rows={metrics?.failures?.worstUnderpredictions} />
      <FailureTable title="Missed high-risk" rows={metrics?.failures?.missedHighRisk} />
      <FailureTable title="False alarms" rows={metrics?.failures?.falseEscalations} />

      <section className="rounded-lg border border-white/10 bg-panel p-4 text-sm leading-6 text-slate-300">
        <h2 className="text-cyan">Limits</h2>
        <p>
          Synthetic CDMs, grouped event splits, T-48 cutoff. Persistence is a strong baseline. SHAP
          explains the model, not physics. Warning line −6 (about 1 in a million) is educational, not
          an ISRO rule.
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
            <th>Copilot</th>
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
