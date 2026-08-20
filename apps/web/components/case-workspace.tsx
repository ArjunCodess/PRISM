"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { Band } from "@/components/band";
import { Explanation } from "@/components/explanation";
import { chanceWords } from "@/lib/plain";
import type { CdmMessage, DemoCase } from "@/lib/types";

type CutoffSafeCase = Omit<DemoCase, "futureMessages" | "actualFinalRiskLog10">;
type Outcome = { actualFinalRiskLog10: number; futureMessages: CdmMessage[] };

const RiskChart = dynamic(() => import("@/components/risk-chart").then((module) => module.RiskChart), {
  ssr: false,
  loading: () => <div className="panel min-h-[25rem] animate-pulse" aria-label="Loading risk history" />,
});

export function CaseWorkspace({ item }: { item: CutoffSafeCase }) {
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [revealError, setRevealError] = useState("");
  const [isRevealing, setIsRevealing] = useState(false);
  const latest = item.messages.at(-1)!;
  const forecast = item.prediction.predictedFinalRiskLog10;
  const persist = item.baselineRiskLog10;
  const [low90, high90] = item.prediction.interval90Log10;
  const [low50, high50] = item.prediction.interval50Log10 ?? item.prediction.interval90Log10;
  const delta = forecast - persist;

  async function revealOutcome() {
    setIsRevealing(true);
    setRevealError("");
    try {
      const response = await fetch(`/api/cases/${item.id}/outcome`);
      if (!response.ok) throw new Error("Outcome could not be loaded");
      setOutcome((await response.json()) as Outcome);
    } catch {
      setRevealError("The outcome could not be loaded. The cutoff-safe forecast is still available.");
    } finally {
      setIsRevealing(false);
    }
  }

  return (
    <div className="space-y-8">
      <section className="flex flex-col gap-4 border-b hairline pb-8 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="max-w-[70ch] text-lg leading-8 text-stone-600">{item.briefing}</p>
          <p className="mt-2 text-xs text-stone-500">Locked at T−{(latest.timeToTcaDays * 24).toFixed(1)} hours · {item.messages.length} messages used · forecast of the final reported log₁₀(Pc)</p>
        </div>
        <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} />
      </section>
      {item.prediction.abstained ? (
        <p className="max-w-[70ch] text-sm leading-6 text-amber">
          Review required: {item.prediction.abstentionReasons?.join("; ") || "the 90% conformal band crosses the ESA class, or a critical field is missing"}.
        </p>
      ) : null}

      <dl className="grid overflow-hidden rounded-lg border hairline bg-panel sm:grid-cols-2 xl:grid-cols-5">
        <Metric label="Time to TCA" value={`${(latest.timeToTcaDays * 24).toFixed(1)} h`} note="selected message" />
        <Metric label="Today's report" value={persist.toFixed(2)} note={`${chanceWords(persist)} · persistence`} />
        <Metric label="Forecast" value={forecast.toFixed(2)} note={`${delta >= 0 ? "+" : ""}${delta.toFixed(2)} vs today`} accent />
        <Metric label="50% / 90% interval" value={`${low50.toFixed(1)}…${high50.toFixed(1)}`} note={`outer ${low90.toFixed(1)}…${high90.toFixed(1)} · conformal`} />
        <Metric label="High-risk estimate" value={`${(item.prediction.configuredHighRiskProbability * 100).toFixed(0)}%`} note="ESA class ≥ −6 · 9 test positives" />
      </dl>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(19rem,.75fr)]">
        <RiskChart
          messages={item.messages}
          future={outcome?.futureMessages ?? []}
          revealed={Boolean(outcome)}
          forecast={forecast}
          persistence={persist}
        />
        <Explanation text={item.prediction.explanation} factors={item.prediction.topFactors} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <section className="panel p-6">
          <p className="eyebrow">Encounter state</p>
          <h2 className="display mt-2 text-3xl">Geometry and observations</h2>
          <dl className="mt-6 grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
            <Telemetry label="Miss distance" value={`${latest.missDistanceM.toFixed(0)} m`} />
            <Telemetry label="Relative speed" value={`${(latest.relativeSpeedMps / 1000).toFixed(2)} km/s`} />
            <Telemetry label="Radial uncertainty" value={`${(latest.tSigmaR ?? 0).toFixed(0)} / ${(latest.cSigmaR ?? 0).toFixed(0)} m`} />
            <Telemetry label="Observations used" value={`${((latest.tObsUsed ?? 0) + (latest.cObsUsed ?? 0)).toFixed(0)}`} />
          </dl>
          <p className="mt-6 max-w-3xl border-t hairline pt-4 text-xs leading-5 text-stone-500">A small miss distance alone does not imply high collision probability. Geometry, covariance, and reported probability must be read together.</p>
        </section>

        <aside className="panel p-6">
          <p className="eyebrow">Later update</p>
          <h2 className="display mt-2 text-3xl">Reveal when ready</h2>
          <p className="mt-4 text-sm leading-6 text-stone-600">
            Today&apos;s report and the forecast are already in the strip above. Persistence is today&apos;s value carried forward. The later messages stay hidden until you ask.
          </p>
          <button type="button" onClick={revealOutcome} disabled={Boolean(outcome) || isRevealing} className="interactive mt-6 w-full rounded-md bg-ink px-4 py-3 text-sm text-white hover:bg-cyan disabled:opacity-50">
            {outcome ? "Outcome revealed" : isRevealing ? "Loading outcome…" : "Reveal final outcome"}
          </button>
          {revealError ? <p role="alert" className="mt-3 text-xs text-alert">{revealError}</p> : null}
        </aside>
      </div>

      {outcome ? (
        <section className="rounded-lg border border-amber/30 bg-amber/[0.05] p-6" aria-live="polite">
          <p className="eyebrow !text-amber">Final update</p>
          <h2 className="display mt-2 text-3xl">Reported risk finished at {outcome.actualFinalRiskLog10.toFixed(2)}.</h2>
          <p className="mt-2 text-sm text-stone-600">{chanceWords(outcome.actualFinalRiskLog10)} · forecast error {Math.abs(forecast - outcome.actualFinalRiskLog10).toFixed(2)} log units · persistence error {Math.abs(persist - outcome.actualFinalRiskLog10).toFixed(2)}</p>
        </section>
      ) : null}

      <section className="rounded-lg border hairline bg-[#fbf8ef] p-5 text-sm leading-6 text-stone-600">
        <strong className="font-semibold text-ink">Educational recommendation:</strong> {item.prediction.abstained ? "Seek another observation and prioritize human review." : "Use this forecast to prioritize review; a human analyst must approve any response."} {item.prediction.disclaimer}
      </section>
    </div>
  );
}

function Metric({ label, value, note, accent = false }: { label: string; value: string; note: string; accent?: boolean }) {
  return <div className="border-b hairline p-5 last:border-b-0 sm:border-r xl:border-b-0"><dt className="text-[0.65rem] uppercase tracking-wider text-stone-500">{label}</dt><dd className={`telemetry mt-3 text-xl ${accent ? "text-cyan" : "text-ink"}`}>{value}</dd><p className="mt-1 text-[0.68rem] text-stone-500">{note}</p></div>;
}

function Telemetry({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-xs text-stone-500">{label}</dt><dd className="telemetry mt-2 text-sm text-ink">{value}</dd></div>;
}
