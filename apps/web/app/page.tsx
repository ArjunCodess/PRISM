import Link from "next/link";
import { Band } from "@/components/band";
import { Shell } from "@/components/shell";
import { loadCases, loadMetrics } from "@/lib/data";
import { chanceWords } from "@/lib/plain";

const storyLabel: Record<string, string> = {
  low: "Stable low risk",
  escalate: "Escalating",
  deescalate: "De-escalating",
  uncertain: "Uncertain",
  failure: "Known failure",
};

export default async function QueuePage() {
  const [cases, metrics] = await Promise.all([loadCases(), loadMetrics()]);
  const reviewCount = cases.filter((item) => item.prediction.abstained).length;

  return (
    <Shell title="Five encounters, frozen before the future is known." kicker="Conjunction event queue">
      <section className="mb-12 grid gap-8 border-b hairline pb-10 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <p className="max-w-[64ch] text-lg leading-8 text-stone-600">
          Every forecast uses only messages available at least 48 hours before closest approach. Choose a case, read the model’s reasons, then reveal the final update.
        </p>
        <dl className="grid grid-cols-3 gap-4 text-sm">
          <Summary label="Cases" value={cases.length.toString()} />
          <Summary label="Need review" value={reviewCount.toString()} />
          <Summary label="Cutoff" value="48 h" />
        </dl>
      </section>

      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <h2 className="display text-3xl">Curated cases</h2>
        <p className="text-xs text-stone-500">High-risk class: log₁₀(Pc) ≥ −6</p>
      </div>

      <section className="overflow-hidden rounded-lg border hairline bg-panel" aria-label="Curated conjunction cases">
        {cases.map((item, index) => {
          const delta = item.prediction.predictedFinalRiskLog10 - item.baselineRiskLog10;
          return (
            <article key={item.id} className="grid gap-4 border-b hairline p-5 last:border-b-0 md:grid-cols-[2rem_minmax(14rem,1fr)_9rem_9rem_auto] md:items-center">
              <span className="telemetry text-xs text-stone-400">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <p className="mb-1 text-xs text-cyan">{storyLabel[item.story] ?? item.story}</p>
                <h3 className="display text-2xl">{item.title}</h3>
                <p className="mt-1 max-w-[55ch] text-sm leading-6 text-stone-500">{item.blurb}</p>
              </div>
              <Value label="Current" value={item.baselineRiskLog10.toFixed(2)} note={chanceWords(item.baselineRiskLog10)} />
              <Value label="Forecast" value={item.prediction.predictedFinalRiskLog10.toFixed(2)} note={`${delta >= 0 ? "+" : ""}${delta.toFixed(2)} change`} />
              <div className="flex items-center justify-between gap-4 md:justify-end">
                <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} compact />
                <Link href={`/cases/${item.id}`} className="interactive rounded-md bg-ink px-4 py-2.5 text-sm text-white hover:bg-cyan">Open</Link>
              </div>
            </article>
          );
        })}
      </section>

      <p className="mt-5 max-w-3xl text-xs leading-5 text-stone-500">
        Source: {metrics?.dataSource ?? "frozen training artifacts"}. These historical anonymized cases explain early forecasting; they are not operational predictions.
      </p>
    </Shell>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-stone-500">{label}</dt><dd className="display mt-1 text-3xl text-ink">{value}</dd></div>;
}

function Value({ label, value, note }: { label: string; value: string; note: string }) {
  return <div><p className="text-[0.65rem] uppercase tracking-wider text-stone-400">{label}</p><p className="telemetry mt-1 text-sm text-ink">{value}</p><p className="mt-1 text-[0.65rem] text-stone-500">{note}</p></div>;
}
