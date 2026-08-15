"use client";

import { useMemo, useState } from "react";
import { Band } from "@/components/band";
import { Explanation } from "@/components/explanation";
import { RiskChart } from "@/components/risk-chart";
import { formatLogRisk, formatPc } from "@/lib/format";
import type { DemoCase } from "@/lib/types";

export function CaseWorkspace({ item }: { item: DemoCase }) {
  const [revealed, setRevealed] = useState(false);
  const [showBaseline, setShowBaseline] = useState(false);
  const latest = item.messages[item.messages.length - 1];
  const displayRisk = showBaseline ? item.baselineRiskLog10 : item.prediction.predictedFinalRiskLog10;
  const pc = useMemo(() => 10 ** displayRisk, [displayRisk]);

  return (
    <div className="space-y-4">
      <section className="grid gap-3 rounded-lg border border-white/10 bg-panel p-4 md:grid-cols-5">
        <Metric label="Current log-risk" value={formatLogRisk(item.baselineRiskLog10)} />
        <Metric label="Hours to TCA" value={`${(latest.timeToTcaDays * 24).toFixed(1)} h`} />
        <Metric label={showBaseline ? "Baseline forecast" : "Copilot forecast"} value={formatLogRisk(displayRisk)} />
        <Metric label="Approx Pc" value={formatPc(pc)} />
        <Metric
          label="90% spread"
          value={`${item.prediction.interval90Log10[0].toFixed(2)} to ${item.prediction.interval90Log10[1].toFixed(2)}`}
        />
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">Warning</p>
          <div className="mt-1 flex items-center gap-2">
            <span className="telemetry font-mono">
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

      <section className="grid gap-3 md:grid-cols-4">
        <Metric label="Miss distance" value={`${latest.missDistanceM.toFixed(0)} m`} />
        <Metric label="Relative speed" value={`${latest.relativeSpeedMps.toFixed(0)} m/s`} />
        <Metric
          label="Position uncertainty"
          value={`t ${Number(latest.tSigmaR ?? 0).toFixed(0)} m / c ${Number(latest.cSigmaR ?? 0).toFixed(0)} m`}
        />
        <Metric label="Eligible CDMs" value={`${item.messages.length}`} />
      </section>

      <div className="flex flex-wrap gap-3">
        <button
          className="rounded border border-cyan px-4 py-2 text-sm text-cyan"
          onClick={() => setShowBaseline((value) => !value)}
          type="button"
        >
          {showBaseline ? "Show model" : "Show persistence baseline"}
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
          Actual final log-risk: {formatLogRisk(item.actualFinalRiskLog10)} (Pc {formatPc(10 ** item.actualFinalRiskLog10)})
        </p>
      ) : (
        <p className="text-sm text-slate-400">Post T-48 observations stay hidden until you reveal the outcome.</p>
      )}
      <p className="text-xs text-slate-500">{item.prediction.disclaimer}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-slate-400">{label}</p>
      <p className="telemetry mt-1 font-mono text-lg">{value}</p>
    </div>
  );
}
