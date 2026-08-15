"use client";

import { useState } from "react";
import { Band } from "@/components/band";
import { Explanation } from "@/components/explanation";
import { RiskChart } from "@/components/risk-chart";
import { formatLogRisk } from "@/lib/format";
import { chanceWords, fallbackBriefing } from "@/lib/plain";
import type { DemoCase } from "@/lib/types";

export function CaseWorkspace({ item }: { item: DemoCase }) {
  const [revealed, setRevealed] = useState(false);
  const [showBaseline, setShowBaseline] = useState(false);
  const latest = item.messages[item.messages.length - 1];
  const displayRisk = showBaseline ? item.baselineRiskLog10 : item.prediction.predictedFinalRiskLog10;
  const hoursLeft = latest.timeToTcaDays * 24;

  return (
    <div className="space-y-4">
      <p className="text-sm leading-6 text-slate-200">{item.briefing ?? fallbackBriefing(item)}</p>

      <section className="grid gap-3 rounded-lg border border-white/10 bg-panel p-4 md:grid-cols-3 lg:grid-cols-5">
        <Metric label="Today" value={formatLogRisk(item.baselineRiskLog10)} hint={chanceWords(item.baselineRiskLog10)} />
        <Metric label="Hours left" value={hoursLeft.toFixed(0)} />
        <Metric
          label={showBaseline ? "Persistence" : "Forecast"}
          value={formatLogRisk(displayRisk)}
          hint={chanceWords(displayRisk)}
        />
        <Metric
          label="90% range"
          value={`${item.prediction.interval90Log10[0].toFixed(1)} to ${item.prediction.interval90Log10[1].toFixed(1)}`}
        />
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">Warning</p>
          <div className="mt-1 flex items-center gap-2">
            <span className="font-mono text-lg">
              {(item.prediction.configuredHighRiskProbability * 100).toFixed(0)}%
            </span>
            <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} />
          </div>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <RiskChart messages={item.messages} future={item.futureMessages} revealed={revealed} />
        <Explanation text={item.prediction.explanation} factors={item.prediction.topFactors} />
      </div>

      <section className="grid gap-3 md:grid-cols-3">
        <Metric label="Miss distance" value={`${latest.missDistanceM.toFixed(0)} m`} />
        <Metric label="Relative speed" value={`${latest.relativeSpeedMps.toFixed(0)} m/s`} />
        <Metric
          label="Position uncertainty"
          value={`t ${Number(latest.tSigmaR ?? 0).toFixed(0)} / c ${Number(latest.cSigmaR ?? 0).toFixed(0)} m`}
        />
      </section>

      <div className="flex flex-wrap gap-3">
        <button
          className="rounded border border-cyan px-4 py-2 text-sm text-cyan"
          onClick={() => setShowBaseline((value) => !value)}
          type="button"
        >
          {showBaseline ? "Show copilot" : "Show persistence"}
        </button>
        <button
          className="rounded border border-amber px-4 py-2 text-sm text-amber"
          onClick={() => setRevealed(true)}
          type="button"
        >
          Reveal outcome
        </button>
      </div>

      {revealed ? (
        <p className="rounded border border-white/10 bg-white/5 p-3 font-mono text-sm">
          Later update: {formatLogRisk(item.actualFinalRiskLog10)} ({chanceWords(item.actualFinalRiskLog10)})
        </p>
      ) : (
        <p className="text-sm text-slate-400">Post T-48 updates stay hidden until reveal.</p>
      )}
      <p className="text-xs text-slate-500">{item.prediction.disclaimer}</p>
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-slate-400">{label}</p>
      <p className="telemetry mt-1 font-mono text-lg">{value}</p>
      {hint ? <p className="mt-1 text-xs text-slate-400">{hint}</p> : null}
    </div>
  );
}
