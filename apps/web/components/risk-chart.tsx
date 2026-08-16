"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CdmMessage } from "@/lib/types";

export function RiskChart({ messages, future, revealed, forecast }: { messages: CdmMessage[]; future: CdmMessage[]; revealed: boolean; forecast: number }) {
  const observed = messages.map((item) => ({ t: item.timeToTcaDays, observed: item.riskLog10, final: null as number | null }));
  const later = future.map((item) => ({ t: item.timeToTcaDays, observed: null as number | null, final: revealed ? item.riskLog10 : null }));
  const data = [...observed, ...later].sort((a, b) => b.t - a.t);
  const allRisks = [...messages.map((m) => m.riskLog10), forecast, ...(revealed ? future.map((m) => m.riskLog10) : [])];
  const minRisk = Math.floor(Math.min(-6.5, ...allRisks) - 0.5);
  const maxRisk = Math.ceil(Math.max(-3.5, ...allRisks) + 0.5);
  const maxDays = Math.ceil(Math.max(...messages.map((m) => m.timeToTcaDays)));

  return (
    <section className="panel min-h-[25rem] p-4 sm:p-5" aria-labelledby="risk-history-title">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="eyebrow">Time series</p>
          <h2 id="risk-history-title" className="display mt-2 text-3xl">Reported risk history</h2>
        </div>
        <div className="flex flex-wrap gap-3 text-[0.65rem] text-stone-500">
          <Key color="bg-cyan" label="Known by T−48" />
          <Key color="bg-amber" label="Copilot forecast" />
          {revealed ? <Key color="bg-ink" label="Later updates" /> : <Key color="bg-stone-300" label="Future hidden" />}
        </div>
      </div>
      <div className="h-72 w-full" role="img" aria-label="Log collision risk over days to closest approach, with later observations hidden until reveal">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 18, right: 24, bottom: 12, left: 0 }}>
            <CartesianGrid stroke="#e8e6df" vertical={false} />
            <XAxis dataKey="t" type="number" domain={[0, maxDays]} reversed tick={{ fill: "#77776f", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#d9d7cf" }} label={{ value: "days to closest approach", position: "insideBottom", offset: -8, fill: "#77776f", fontSize: 10 }} />
            <YAxis domain={[minRisk, maxRisk]} tick={{ fill: "#77776f", fontSize: 11 }} tickLine={false} axisLine={false} width={38} label={{ value: "log₁₀(Pc)", angle: -90, position: "insideLeft", fill: "#77776f", fontSize: 10 }} />
            <Tooltip content={<RiskTooltip revealed={revealed} />} />
            <ReferenceArea x1={0} x2={2} fill="#f7eedf" stroke="none" label={{ value: revealed ? "REVEALED" : "HIDDEN FUTURE", fill: "#8a7458", fontSize: 9, position: "insideTop" }} />
            <ReferenceLine x={2} stroke="#9a682a" strokeDasharray="4 4" label={{ value: "T−48", fill: "#9a682a", fontSize: 10, position: "top" }} />
            <ReferenceLine y={-6} stroke="#b27373" strokeDasharray="3 5" label={{ value: "ESA class −6", fill: "#986363", fontSize: 9, position: "insideTopRight" }} />
            <ReferenceLine y={-5} stroke="#dfd2bf" strokeDasharray="2 7" />
            <ReferenceLine y={-4} stroke="#e4cece" strokeDasharray="2 7" />
            <Line type="monotone" dataKey="observed" stroke="#356b67" strokeWidth={2.4} dot={{ r: 3, fill: "#fffefd", stroke: "#356b67", strokeWidth: 2 }} activeDot={{ r: 5 }} connectNulls={false} isAnimationActive={false} />
            {revealed ? <Line type="monotone" dataKey="final" stroke="#242521" strokeWidth={1.8} strokeDasharray="5 5" dot={{ r: 2.5, fill: "#242521" }} connectNulls isAnimationActive={false} /> : null}
            <ReferenceDot x={0.12} y={forecast} r={5} fill="#9a682a" stroke="#fffefd" strokeWidth={2} label={{ value: "forecast", fill: "#9a682a", fontSize: 10, position: "right" }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-[0.68rem] leading-5 text-stone-500">The −6 line is the ESA challenge scoring class. The −5 and −4 guides are educational display bands, not ISRO rules.</p>
    </section>
  );
}

function Key({ color, label }: { color: string; label: string }) {
  return <span className="flex items-center gap-1.5"><span className={`size-1.5 rounded-full ${color}`} />{label}</span>;
}

function RiskTooltip({ active, payload, label, revealed }: { active?: boolean; payload?: Array<{ dataKey: string; value: number | null }>; label?: number; revealed: boolean }) {
  if (!active) return null;
  const value = payload?.find((entry) => entry.value !== null)?.value;
  return (
    <div className="rounded border hairline bg-white px-3 py-2 text-xs">
      <p className="telemetry text-stone-500">T−{Number(label).toFixed(2)} days</p>
      <p className="telemetry mt-1 text-ink">{typeof value === "number" ? `log risk ${value.toFixed(2)}` : revealed ? "No update" : "Hidden until reveal"}</p>
    </div>
  );
}
