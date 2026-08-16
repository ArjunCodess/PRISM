import type { Metadata } from "next";
import { PrintButton } from "@/components/print-button";
import { Shell } from "@/components/shell";
import { loadMetrics } from "@/lib/data";

export const metadata: Metadata = { title: "Model laboratory" };

export default async function LabPage() {
  const metrics = await loadMetrics();
  if (!metrics) return <Shell title="Model evidence is unavailable." kicker="Model laboratory"><p className="panel p-6 text-stone-600">The frozen metrics file could not be loaded.</p></Shell>;

  const improvement = metrics.improvement.mae_improvement / metrics.persistence.mae * 100;
  const featureGroups = metrics.featureGroups ?? [];
  const maxGain = Math.max(...featureGroups.map((item) => item.gain), 1);
  const slices = metrics.robustness?.byMessageCount ?? {};

  return (
    <Shell title="What the model gets right—and where it fails." kicker="Model laboratory">
      <div className="space-y-14">
        <section className="grid gap-8 border-b hairline pb-10 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div>
            <div className="flex items-start justify-between gap-4"><p className="max-w-[62ch] text-lg leading-8 text-stone-600">The selected ensemble cuts average error by {improvement.toFixed(1)}%, but it ties persistence on the ESA high-risk loss. The project does not claim full baseline superiority.</p><PrintButton /></div>
            <dl className="mt-8 grid gap-6 sm:grid-cols-3">
              <HeroMetric label="Ensemble MAE" value={metrics.ensemble.mae} />
              <HeroMetric label="Persistence MAE" value={metrics.persistence.mae} />
              <HeroMetric label="ESA loss" value={metrics.ensemble.esa_loss} note={`persistence ${metrics.persistence.esa_loss.toFixed(3)}`} />
            </dl>
          </div>
          <aside className="rounded-lg bg-[#eeeae0] p-6 text-sm leading-6 text-stone-600">
            <p className="font-semibold text-ink">Current scientific status</p>
            <p className="mt-3">Real ESA data, event-disjoint splits, a separate calibration partition, mission holdout, failure galleries, and leakage tests are present.</p>
            <p className="mt-3">Operational generalization and calibrated predictive intervals are not established.</p>
          </aside>
        </section>

        <Section title="Baseline comparison" copy="Lower is better. One log unit is a tenfold probability error.">
          <div className="overflow-x-auto rounded-lg border hairline bg-panel">
            <table className="w-full min-w-[650px] text-left text-sm">
              <thead className="border-b hairline text-xs text-stone-500"><tr><th className="px-5 py-4">System</th><th>MAE</th><th>RMSE</th><th>High-risk MAE</th><th>Within 1 unit</th><th>ESA loss</th></tr></thead>
              <tbody><ModelRow name="Persistence" data={metrics.persistence} /><ModelRow name="Ridge" data={metrics.ridge} /><ModelRow name="XGBoost" data={metrics.xgboost} /><ModelRow name="Guarded ensemble" data={metrics.ensemble} selected /></tbody>
            </table>
          </div>
        </Section>

        <section className="grid gap-8 lg:grid-cols-2">
          <Section title="Model spread is under-calibrated" copy="Measured coverage is far below the nominal labels, so the case view calls these ranges model spread.">
            <div className="panel grid grid-cols-2 gap-6 p-6">
              <Coverage label="50% band" value={metrics.uncertainty?.interval50Coverage} width={metrics.uncertainty?.meanInterval50Width} />
              <Coverage label="90% band" value={metrics.uncertainty?.interval90Coverage} width={metrics.uncertainty?.meanInterval90Width} />
            </div>
          </Section>
          <Section title="Mission identity does not help" copy="The production model excludes mission_id because it slightly worsens ordinary hold-out MAE.">
            <div className="panel p-6"><Line label="Without mission ID" value={metrics.missionIdComparison?.withoutMissionId.mae?.toFixed(3) ?? "—"} /><Line label="With mission ID" value={metrics.missionIdComparison?.withMissionId.mae?.toFixed(3) ?? "—"} /><Line label="Held-out missions" value={metrics.missionHoldout?.heldOutMissions.join(", ") ?? "—"} /><Line label="Mission holdout events" value={metrics.missionHoldout?.testEvents.toLocaleString("en-US") ?? "—"} /></div>
          </Section>
        </section>

        <section className="grid gap-8 lg:grid-cols-2">
          <Section title="What the model uses" copy="Grouped XGBoost gain describes model behavior, not physical causation.">
            <div className="panel space-y-5 p-6">{featureGroups.map((item) => <div key={item.group}><div className="mb-2 flex justify-between text-xs"><span>{item.group}</span><span className="telemetry text-stone-500">{item.gain.toFixed(1)}</span></div><div className="h-1.5 rounded-full bg-stone-100"><div className="h-full rounded-full bg-cyan" style={{ width: `${Math.max(2, item.gain / maxGain * 100)}%` }} /></div></div>)}</div>
          </Section>
          <Section title="History-length robustness" copy="These slices expose how performance changes with the amount of cutoff-safe evidence.">
            <div className="panel p-6">{Object.entries(slices).map(([name, data]) => <Line key={name} label={`${plainSlice(name)} · n=${data.n ?? 0}`} value={typeof data.mae === "number" ? `MAE ${data.mae.toFixed(3)}` : "no events"} />)}</div>
          </Section>
        </section>

        <Section title="Failure gallery" copy="The worst errors remain visible because sparse histories and abrupt late changes can defeat an early forecast.">
          <div className="grid gap-5 xl:grid-cols-2"><FailureTable title="Worst under-predictions" rows={metrics.failures?.worstUnderpredictions} /><FailureTable title="Worst over-predictions" rows={metrics.failures?.worstOverpredictions} /><FailureTable title="Missed high-risk events" rows={metrics.failures?.missedHighRisk} /><FailureTable title="False escalations" rows={metrics.failures?.falseEscalations} /></div>
        </Section>

        <section className="grid gap-8 border-t hairline pt-10 lg:grid-cols-2">
          <div><p className="eyebrow">Provenance</p><h2 className="display mt-2 text-3xl">Frozen real-data run</h2><dl className="mt-6 max-w-xl"><Line label="Source" value={metrics.dataSource ?? "Unknown"} /><Line label="Source rows" value={(metrics.sourceRows ?? 0).toLocaleString("en-US")} /><Line label="Eligible events" value={(metrics.nEvents ?? 0).toLocaleString("en-US")} /><Line label="Train / validation" value={`${metrics.splits.train} / ${metrics.splits.validation}`} /><Line label="Calibration / test" value={`${metrics.splits.calibration} / ${metrics.splits.test}`} /></dl></div>
          <div><p className="eyebrow !text-amber">Limits</p><h2 className="display mt-2 text-3xl">Do not over-read the result</h2><p className="mt-5 text-sm leading-7 text-stone-600">Only 66 eligible events meet the configured high-risk class. The −6 line is an ESA challenge scoring threshold, not an ISRO rule. PRISM forecasts a later reported risk; it does not calculate collision probability from first principles or recommend a manoeuvre.</p></div>
        </section>
      </div>
    </Shell>
  );
}

function Section({ title, copy, children }: { title: string; copy: string; children: React.ReactNode }) { return <section><header className="mb-5"><h2 className="display text-3xl">{title}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-stone-500">{copy}</p></header>{children}</section>; }
function HeroMetric({ label, value, note }: { label: string; value: number; note?: string }) { return <div><dt className="text-xs text-stone-500">{label}</dt><dd className="telemetry mt-2 text-3xl text-ink">{value.toFixed(3)}</dd>{note ? <p className="mt-1 text-xs text-stone-500">{note}</p> : null}</div>; }
function Coverage({ label, value, width }: { label: string; value?: number; width?: number }) { return <div><p className="text-xs text-stone-500">{label}</p><p className="display mt-2 text-4xl">{typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"}</p><p className="mt-1 text-xs text-stone-500">mean width {width?.toFixed(2) ?? "—"}</p></div>; }
function ModelRow({ name, data, selected = false }: { name: string; data?: Record<string, number>; selected?: boolean }) { if (!data) return null; return <tr className={`border-b hairline last:border-0 ${selected ? "bg-[#edf4f1]" : ""}`}><th className="px-5 py-4 font-medium">{name}</th><Cell value={data.mae} /><Cell value={data.rmse} /><Cell value={data.mae_high_risk} /><Cell value={data.within_1_0} percent /><Cell value={data.esa_loss} /></tr>; }
function Cell({ value, percent = false }: { value?: number; percent?: boolean }) { return <td className="telemetry py-4 text-xs text-stone-600">{typeof value === "number" ? percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3) : "—"}</td>; }
function Line({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-5 border-b hairline py-3 first:pt-0 last:border-0 last:pb-0"><dt className="text-sm text-stone-500">{label}</dt><dd className="telemetry text-right text-xs text-ink">{value}</dd></div>; }
function plainSlice(value: string) { return value === "one" ? "1 message" : value === "twoToFive" ? "2–5 messages" : "6+ messages"; }
function FailureTable({ title, rows }: { title: string; rows?: Array<Record<string, number>> }) { return <section className="panel overflow-hidden"><h3 className="border-b hairline px-5 py-4 text-sm font-medium">{title}</h3>{rows?.length ? <div className="overflow-x-auto"><table className="w-full min-w-[430px] text-left text-xs"><thead className="text-stone-500"><tr><th className="px-5 py-3">Event</th><th>Actual</th><th>Forecast</th><th>Persistence</th><th>Error</th></tr></thead><tbody>{rows.slice(0, 5).map((row) => <tr key={row.eventId} className="border-t hairline"><td className="telemetry px-5 py-3">{row.eventId}</td><td className="telemetry">{row.actual.toFixed(2)}</td><td className="telemetry">{row.predicted.toFixed(2)}</td><td className="telemetry text-stone-500">{row.persistence.toFixed(2)}</td><td className="telemetry text-amber">{row.error > 0 ? "+" : ""}{row.error.toFixed(2)}</td></tr>)}</tbody></table></div> : <p className="p-5 text-sm text-stone-500">No cases in this category.</p>}</section>; }
