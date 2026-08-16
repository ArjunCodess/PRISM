import type { Metadata } from "next";
import { PrintButton } from "@/components/print-button";
import { Shell } from "@/components/shell";
import { loadMetrics } from "@/lib/data";

export const metadata: Metadata = { title: "Model laboratory" };
export const dynamic = "force-dynamic";

const FAMILY_LABEL: Record<string, string> = {
  snapshot: "Latest snapshot only",
  snapshot_history: "Snapshot + history",
  snapshot_history_covariance: "Snapshot + history + covariance",
  full: "Full PRISM features",
};

const MODE_LABEL: Record<string, string> = {
  accurate: "Accurate (|error| ≤ 0.5)",
  final_collapses_to_negligible: "Final risk collapses to the dataset floor",
  late_high_risk_jump: "Late high-risk jump",
  missed_high_risk: "Missed high-risk event",
  false_high_risk: "False high-risk call",
  sparse_history_error: "Sparse-history error",
  close_approach_error: "Close-approach error",
  underprediction: "Underprediction",
  overprediction: "Overprediction",
  moderate_error: "Moderate error",
};

export default async function LabPage() {
  const metrics = await loadMetrics();
  if (!metrics) return <Shell title="Model evidence is unavailable." kicker="Model laboratory"><p className="panel p-6 text-stone-600">The frozen metrics file could not be loaded.</p></Shell>;

  const improvement = metrics.improvement.mae_improvement / metrics.persistence.mae * 100;
  const featureGroups = metrics.featureGroups ?? [];
  const maxGain = Math.max(...featureGroups.map((item) => item.gain), 1);
  const slices = metrics.robustness?.byMessageCount ?? {};
  const families = metrics.ablation?.families ?? {};
  const horizons = (metrics.horizons ?? []).filter((row) => row.model);
  const operating = metrics.abstention?.operatingPoint ?? {};
  const coverage = typeof operating.coverage === "number" ? operating.coverage : null;
  const maeAccepted = typeof operating.maeAccepted === "number" ? operating.maeAccepted : null;
  const warning = metrics.warning;
  const nHighRiskTest = typeof warning.nHighRiskTest === "number" ? warning.nHighRiskTest : null;
  const nHighRiskEligible = typeof warning.nHighRiskEligible === "number" ? warning.nHighRiskEligible : metrics.nHighRiskEligible;
  const shapCorrect = metrics.shapContrast?.correct;
  const shapIncorrect = metrics.shapContrast?.incorrect;
  const shapNames = Array.from(new Set([
    ...(shapIncorrect?.groups.slice(0, 5).map((item) => item.group) ?? []),
    ...(shapCorrect?.groups.slice(0, 5).map((item) => item.group) ?? []),
  ])).slice(0, 5);
  const shapMax = Math.max(
    ...(shapCorrect?.groups.map((item) => item.meanAbsShap) ?? [0]),
    ...(shapIncorrect?.groups.map((item) => item.meanAbsShap) ?? [0]),
    0.01,
  );
  const failureModes = Object.entries(metrics.failureClusters?.modes ?? {}).sort((left, right) => (right[1].n ?? 0) - (left[1].n ?? 0));

  return (
    <Shell title="What the model gets right—and where it fails." kicker="Model laboratory">
      <div className="space-y-14">
        <section className="grid gap-8 border-b hairline pb-10 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div>
            <div className="flex items-start justify-between gap-4">
              <p className="max-w-[62ch] text-lg leading-8 text-stone-600">
                {metrics.researchQuestion ?? "Do pre-T−48 conjunction histories improve forecasts of later reported log10(Pc) over persistence?"}
              </p>
              <PrintButton />
            </div>
            <dl className="mt-8 grid gap-6 sm:grid-cols-3">
              <HeroMetric label="Ensemble MAE" value={metrics.ensemble.mae} note={`persistence ${metrics.persistence.mae.toFixed(3)}`} />
              <HeroMetric label="ESA-style loss" value={metrics.ensemble.esa_loss} note={`persistence ${metrics.persistence.esa_loss.toFixed(3)}`} />
              <HeroMetric label="90% band coverage" value={metrics.uncertainty?.interval90Coverage ?? Number.NaN} percent note="nominal 90%" />
            </dl>
            <p className="mt-6 max-w-[62ch] text-sm leading-7 text-stone-600">
              Average error falls {improvement.toFixed(1)}%, but ESA-style loss and F2 tie persistence exactly because the persistence guard copies any current report already at or above −6. The MAE gain is continuous-risk accuracy, not a better high-risk decision score. On missions never seen in training, high-risk MAE is {(metrics.missionHoldout?.model.mae_high_risk ?? Number.NaN).toFixed(1)}.
            </p>
          </div>
          <aside className="rounded-lg bg-[#eeeae0] p-6 text-sm leading-6 text-stone-600">
            <p className="font-semibold text-ink">What this page is for</p>
            <p className="mt-3">The horizon table is the main result: learned forecasting helps most when information is sparse. History versus snapshot and abstention follow.</p>
            <p className="mt-3">Not flight software. Not an operational decision system.</p>
          </aside>
        </section>

        {horizons.length > 0 ? (
          <Section title="The value of learned forecasting is highest when information is sparse" copy="Waiting helps persistence more than it helps PRISM. As the cutoff moves toward closest approach, the latest report becomes a stronger baseline and the model’s edge shrinks.">
            <div className="overflow-x-auto rounded-lg border hairline bg-panel">
              <table className="w-full min-w-[620px] text-left text-sm">
                <thead className="border-b hairline text-xs text-stone-500"><tr><th className="px-5 py-4">Horizon</th><th>Test events</th><th>XGBoost MAE</th><th>Persistence MAE</th><th>Improvement</th></tr></thead>
                <tbody>
                  {horizons.map((row) => (
                    <tr key={row.cutoffHours} className={`border-b hairline last:border-0 ${row.cutoffHours === 48 ? "bg-[#edf4f1]" : ""}`}>
                      <th className="px-5 py-4 font-medium">T−{row.cutoffHours} h</th>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.testEvents.toLocaleString("en-US")}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.model?.mae.toFixed(3) ?? "—"}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.persistence?.mae.toFixed(3) ?? "—"}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{typeof row.maeImprovement === "number" ? row.maeImprovement.toFixed(3) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        ) : null}

        {Object.keys(families).length > 0 ? (
          <Section title="Does history beat the latest snapshot?" copy="The history block consists of temporal transforms of variables already available in the latest snapshot, plus message count and recency. Covariance trends add little after those summaries are present.">
            <div className="overflow-x-auto rounded-lg border hairline bg-panel">
              <table className="w-full min-w-[620px] text-left text-sm">
                <thead className="border-b hairline text-xs text-stone-500"><tr><th className="px-5 py-4">Feature set</th><th>Features</th><th>MAE</th><th>ESA loss</th><th>Δ MAE from previous</th></tr></thead>
                <tbody>
                  {Object.entries(FAMILY_LABEL).map(([key, label]) => {
                    const row = families[key];
                    if (!row) return null;
                    const delta = row.deltaFromPreviousMae;
                    return (
                      <tr key={key} className="border-b hairline last:border-0">
                        <th className="px-5 py-4 font-medium">{label}</th>
                        <td className="telemetry py-4 text-xs text-stone-600">{typeof row.nFeatures === "number" ? row.nFeatures : "—"}</td>
                        <td className="telemetry py-4 text-xs text-stone-600">{typeof row.mae === "number" ? row.mae.toFixed(3) : "—"}</td>
                        <td className="telemetry py-4 text-xs text-stone-600">{typeof row.esa_loss === "number" ? row.esa_loss.toFixed(3) : "—"}</td>
                        <td className="telemetry py-4 text-xs text-stone-600">{typeof delta === "number" ? `${delta > 0 ? "−" : "+"}${Math.abs(delta).toFixed(3)}` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-stone-500">
              {metrics.ablation?.historyHelps
                ? `Historical evolution reduces MAE by ${(metrics.ablation.historyDeltaMae ?? 0).toFixed(3)} versus snapshot only.`
                : "On this split, adding historical summaries does not clearly beat the latest snapshot."}
            </p>
          </Section>
        ) : null}

        <Section title="When should the model refuse?" copy="The −6 class follows ESA. The persistence guard and 1.25 disagreement threshold were fixed before test evaluation. False reassurance is an accepted forecast below −6 while the final reported value is ≥ −6.">
          <div className="grid gap-6 lg:grid-cols-3">
            <HeroBox label="Coverage" value={coverage === null ? "—" : `${(coverage * 100).toFixed(1)}%`} note={coverage === null ? "events that receive a firm forecast" : `${((1 - coverage) * 100).toFixed(1)}% sent to review`} />
            <HeroBox label="Accepted MAE" value={maeAccepted === null ? "—" : maeAccepted.toFixed(3)} note={`all-event MAE ${metrics.ensemble.mae.toFixed(3)}`} />
            <HeroBox label="High-risk events" value={nHighRiskTest === null ? "—" : nHighRiskTest.toString()} note="all nine are either flagged or sent to review" />
          </div>
        </Section>

        <section className="grid gap-8 lg:grid-cols-2">
          <Section title="Ensemble disagreement is not equivalent to calibrated uncertainty" copy="A 90% label that covers 48.6% of outcomes is not a little off. The interface therefore calls these ranges model spread.">
            <div className="panel grid grid-cols-2 gap-6 p-6">
              <Coverage label="50% band" value={metrics.uncertainty?.interval50Coverage} width={metrics.uncertainty?.meanInterval50Width} />
              <Coverage label="90% band" value={metrics.uncertainty?.interval90Coverage} width={metrics.uncertainty?.meanInterval90Width} />
            </div>
          </Section>
          <Section title="Mission identity adds little" copy={metrics.missionIdComparison?.why ?? "Adding mission_id provides negligible improvement and does not materially change performance, so it is excluded from the deployed exhibit."}>
            <div className="panel p-6">
              <Line label="Without mission ID" value={metrics.missionIdComparison?.withoutMissionId.mae?.toFixed(3) ?? "—"} />
              <Line label="With mission ID" value={metrics.missionIdComparison?.withMissionId.mae?.toFixed(3) ?? "—"} />
              <Line label="Held-out missions" value={metrics.missionHoldout?.heldOutMissions.join(", ") ?? "—"} />
              <Line label="Mission holdout high-risk events" value={(metrics.missionHoldout?.nHighRiskTest ?? 0).toString()} />
              <Line label="Holdout model high-risk MAE" value={metrics.missionHoldout?.model.mae_high_risk?.toFixed(3) ?? "—"} />
            </div>
          </Section>
        </section>

        {failureModes.length > 0 ? (
          <Section title="How failures cluster" copy="These are mutually exclusive tags on the guarded ensemble. Accurate cases are included so the shares sum to the test set.">
            <div className="overflow-x-auto rounded-lg border hairline bg-panel">
              <table className="w-full min-w-[640px] text-left text-sm">
                <thead className="border-b hairline text-xs text-stone-500"><tr><th className="px-5 py-4">Mode</th><th>n</th><th>Share</th><th>MAE</th><th>Mean messages</th><th>Mean miss (m)</th></tr></thead>
                <tbody>
                  {failureModes.map(([name, row]) => (
                    <tr key={name} className="border-b hairline last:border-0">
                      <th className="px-5 py-4 font-medium">{MODE_LABEL[name] ?? name}</th>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.n}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{`${((row.share ?? 0) * 100).toFixed(1)}%`}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.mae?.toFixed(3) ?? "—"}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.meanMessages?.toFixed(1) ?? "—"}</td>
                      <td className="telemetry py-4 text-xs text-stone-600">{row.meanMissDistanceM?.toFixed(0) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        ) : null}

        {shapNames.length > 0 ? (
          <Section title="Explanations on correct vs incorrect cases" copy="Tracking-completeness features have higher mean |SHAP| among errors of two or more log units than among accurate cases. That is an association in model attribution, not a physical cause.">
            <div className="panel space-y-5 p-6">
              {shapNames.map((name) => {
                const correct = shapCorrect?.groups.find((item) => item.group === name)?.meanAbsShap ?? 0;
                const incorrect = shapIncorrect?.groups.find((item) => item.group === name)?.meanAbsShap ?? 0;
                return (
                  <div key={name}>
                    <div className="mb-2 flex justify-between text-xs"><span>{name}</span><span className="text-stone-500">right {correct.toFixed(2)} · wrong {incorrect.toFixed(2)}</span></div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="h-1.5 rounded-full bg-stone-100"><div className="h-full rounded-full bg-cyan" style={{ width: `${Math.max(2, correct / shapMax * 100)}%` }} /></div>
                      <div className="h-1.5 rounded-full bg-stone-100"><div className="h-full rounded-full bg-amber" style={{ width: `${Math.max(2, incorrect / shapMax * 100)}%` }} /></div>
                    </div>
                  </div>
                );
              })}
              <p className="text-xs text-stone-500">Left bars: |error| ≤ 0.5 ({shapCorrect?.n ?? 0} events). Right bars: |error| ≥ 2.0 ({shapIncorrect?.n ?? 0} events).</p>
            </div>
          </Section>
        ) : null}

        <Section title="Baseline comparison" copy="Lower is better. One log unit is a tenfold probability error. The guarded ensemble is selected because raw XGBoost misses the ESA high-risk class.">
          <div className="overflow-x-auto rounded-lg border hairline bg-panel">
            <table className="w-full min-w-[650px] text-left text-sm">
              <thead className="border-b hairline text-xs text-stone-500"><tr><th className="px-5 py-4">System</th><th>MAE</th><th>RMSE</th><th>High-risk MAE</th><th>Within 1 unit</th><th>ESA loss</th></tr></thead>
              <tbody><ModelRow name="Persistence" data={metrics.persistence} /><ModelRow name="Ridge" data={metrics.ridge} /><ModelRow name="XGBoost" data={metrics.xgboost} /><ModelRow name="Guarded ensemble" data={metrics.ensemble} selected /></tbody>
            </table>
          </div>
        </Section>

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
          <div>
            <p className="eyebrow">Provenance</p>
            <h2 className="display mt-2 text-3xl">Frozen real-data run</h2>
            <dl className="mt-6 max-w-xl">
              <Line label="Source" value={metrics.dataSource ?? "Unknown"} />
              <Line label="Source rows" value={(metrics.sourceRows ?? 0).toLocaleString("en-US")} />
              <Line label="Eligible events" value={(metrics.nEvents ?? 0).toLocaleString("en-US")} />
              <Line label="Train / validation" value={`${metrics.splits.train} / ${metrics.splits.validation}`} />
              <Line label="Calibration / test" value={`${metrics.splits.calibration} / ${metrics.splits.test}`} />
            </dl>
          </div>
          <div>
            <p className="eyebrow !text-amber">Limits</p>
            <h2 className="display mt-2 text-3xl">The high-risk estimate is scarce</h2>
            <p className="mt-5 text-sm leading-7 text-stone-600">
              Only {nHighRiskEligible ?? 66} eligible events meet the ESA challenge class log10(Pc) ≥ −6, including {nHighRiskTest ?? 9} in the frozen test split.
              That line is a competition scoring class, not an ISRO operational threshold. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.
              PRISM forecasts the later reported `log10(Pc)`; it does not calculate collision probability from first principles or recommend a manoeuvre.
            </p>
          </div>
        </section>
      </div>
    </Shell>
  );
}

function Section({ title, copy, children }: { title: string; copy: string; children: React.ReactNode }) { return <section><header className="mb-5"><h2 className="display text-3xl">{title}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-stone-500">{copy}</p></header>{children}</section>; }
function HeroMetric({ label, value, note, percent = false }: { label: string; value: number; note?: string; percent?: boolean }) { return <div><dt className="text-xs text-stone-500">{label}</dt><dd className="telemetry mt-2 text-3xl text-ink">{Number.isFinite(value) ? percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3) : "—"}</dd>{note ? <p className="mt-1 text-xs text-stone-500">{note}</p> : null}</div>; }
function HeroBox({ label, value, note }: { label: string; value: string; note: string }) { return <div className="panel p-6"><p className="text-xs text-stone-500">{label}</p><p className="display mt-2 text-4xl">{value}</p><p className="mt-2 text-xs text-stone-500">{note}</p></div>; }
function Coverage({ label, value, width }: { label: string; value?: number; width?: number }) { return <div><p className="text-xs text-stone-500">{label}</p><p className="display mt-2 text-4xl">{typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"}</p><p className="mt-1 text-xs text-stone-500">mean width {width?.toFixed(2) ?? "—"}</p></div>; }
function ModelRow({ name, data, selected = false }: { name: string; data?: Record<string, number>; selected?: boolean }) { if (!data) return null; return <tr className={`border-b hairline last:border-0 ${selected ? "bg-[#edf4f1]" : ""}`}><th className="px-5 py-4 font-medium">{name}</th><Cell value={data.mae} /><Cell value={data.rmse} /><Cell value={data.mae_high_risk} /><Cell value={data.within_1_0} percent /><Cell value={data.esa_loss} /></tr>; }
function Cell({ value, percent = false }: { value?: number; percent?: boolean }) { return <td className="telemetry py-4 text-xs text-stone-600">{typeof value === "number" ? percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3) : "—"}</td>; }
function Line({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-5 border-b hairline py-3 first:pt-0 last:border-0 last:pb-0"><dt className="text-sm text-stone-500">{label}</dt><dd className="telemetry text-right text-xs text-ink">{value}</dd></div>; }
function plainSlice(value: string) { return value === "one" ? "1 message" : value === "twoToFive" ? "2–5 messages" : "6+ messages"; }
function FailureTable({ title, rows }: { title: string; rows?: Array<Record<string, number>> }) { return <section className="panel overflow-hidden"><h3 className="border-b hairline px-5 py-4 text-sm font-medium">{title}</h3>{rows?.length ? <div className="overflow-x-auto"><table className="w-full min-w-[430px] text-left text-xs"><thead className="text-stone-500"><tr><th className="px-5 py-3">Event</th><th>Actual</th><th>Forecast</th><th>Persistence</th><th>Error</th></tr></thead><tbody>{rows.slice(0, 5).map((row) => <tr key={row.eventId} className="border-t hairline"><td className="telemetry px-5 py-3">{row.eventId}</td><td className="telemetry">{row.actual.toFixed(2)}</td><td className="telemetry">{row.predicted.toFixed(2)}</td><td className="telemetry text-stone-500">{row.persistence.toFixed(2)}</td><td className="telemetry text-amber">{row.error > 0 ? "+" : ""}{row.error.toFixed(2)}</td></tr>)}</tbody></table></div> : <p className="p-5 text-sm text-stone-500">No cases in this category.</p>}</section>; }
