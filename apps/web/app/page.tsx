import Link from "next/link";
import { Band } from "@/components/band";
import { Shell } from "@/components/shell";
import { loadCases, loadMetrics } from "@/lib/data";
import { chanceWords } from "@/lib/plain";

export const dynamic = "force-dynamic";

const storyLabel: Record<string, string> = {
  low: "Low",
  uncertain: "Human approval",
  high: "High",
};

export default async function QueuePage() {
  const [cases, metrics] = await Promise.all([loadCases(), loadMetrics()]);
  const lowCount = cases.filter((item) => item.story === "low").length;
  const reviewCount = cases.filter((item) => item.prediction.abstained).length;
  const highCount = cases.filter((item) => item.story === "high").length;

  return (
    <Shell title="Six encounters, frozen before T−48." kicker="Conjunction event queue">
      <section className="mb-10 grid gap-8 border-b hairline pb-10 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <p className="max-w-[64ch] text-lg leading-8 text-stone-600">
          At T−48, what can we infer about the later reported `log10(Pc)`? Open a case, read the reasons, then reveal the later update.
        </p>
        <dl className="grid grid-cols-3 gap-4 text-sm">
          <Summary label="Low" value={lowCount.toString()} />
          <Summary label="Human approval" value={reviewCount.toString()} />
          <Summary label="High" value={highCount.toString()} />
        </dl>
      </section>

      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <h2 className="display text-3xl">Event queue</h2>
        <p className="text-xs text-stone-500">ESA class log₁₀(Pc) ≥ −6 · 9 test positives, not an operational threshold</p>
      </div>

      <section className="overflow-x-auto rounded-lg border hairline bg-panel" aria-label="Curated conjunction cases">
        <table className="w-full min-w-[46rem] border-collapse text-left">
          <thead>
            <tr className="border-b hairline text-[0.65rem] uppercase tracking-[0.14em] text-stone-500">
              <th scope="col" className="w-12 px-4 py-3 font-medium">#</th>
              <th scope="col" className="px-3 py-3 font-medium">Encounter</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Today</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Forecast</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Change</th>
              <th scope="col" className="px-3 py-3 font-medium">Status</th>
              <th scope="col" className="w-24 px-4 py-3 text-right font-medium"><span className="sr-only">Open</span></th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item, index) => {
              const delta = item.prediction.predictedFinalRiskLog10 - item.baselineRiskLog10;
              const rail =
                item.prediction.abstained || item.prediction.riskBand === "review"
                  ? "bg-amber"
                  : item.prediction.riskBand === "high"
                    ? "bg-alert"
                    : "bg-safe";
              return (
                <tr key={item.id} className="group border-b hairline last:border-b-0 hover:bg-[#f7f4ec]">
                  <td className="relative px-4 py-4 align-middle">
                    <span className={`absolute inset-y-3 left-0 w-0.5 ${rail}`} aria-hidden="true" />
                    <span className="telemetry text-xs text-stone-400">{String(index + 1).padStart(2, "0")}</span>
                  </td>
                  <td className="max-w-[22rem] px-3 py-4 align-middle">
                    <p className="text-[0.65rem] uppercase tracking-[0.12em] text-cyan">{storyLabel[item.story] ?? item.story}</p>
                    <Link href={`/cases/${item.id}`} className="display mt-1 block text-xl leading-tight text-ink hover:text-cyan">
                      {item.title}
                    </Link>
                    <p className="mt-1 text-sm leading-5 text-stone-500">{item.blurb}</p>
                  </td>
                  <td className="px-3 py-4 text-right align-middle">
                    <p className="telemetry text-sm text-ink">{item.baselineRiskLog10.toFixed(2)}</p>
                    <p className="mt-1 text-[0.65rem] text-stone-500">{chanceWords(item.baselineRiskLog10)}</p>
                  </td>
                  <td className="px-3 py-4 text-right align-middle">
                    <p className="telemetry text-sm text-ink">{item.prediction.predictedFinalRiskLog10.toFixed(2)}</p>
                    <p className="mt-1 text-[0.65rem] text-stone-500">{chanceWords(item.prediction.predictedFinalRiskLog10)}</p>
                  </td>
                  <td className="px-3 py-4 text-right align-middle">
                    <p className="telemetry text-sm text-ink">{`${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`}</p>
                    <p className="mt-1 text-[0.65rem] text-stone-500">vs today</p>
                  </td>
                  <td className="px-3 py-4 align-middle">
                    <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} compact />
                  </td>
                  <td className="px-4 py-4 text-right align-middle">
                    <Link href={`/cases/${item.id}`} className="interactive text-sm text-stone-500 hover:text-cyan">
                      Open →
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <p className="mt-5 max-w-3xl text-xs leading-5 text-stone-500">
        Source: {metrics?.dataSource ?? "frozen training artifacts"}. These historical anonymized cases explain an early forecasting experiment; they are not operational predictions. High-risk class: ESA challenge log₁₀(Pc) ≥ −6, not an operational threshold.
      </p>
    </Shell>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-stone-500">{label}</dt>
      <dd className="display mt-1 text-3xl text-ink">{value}</dd>
    </div>
  );
}
