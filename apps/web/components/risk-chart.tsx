"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CdmMessage } from "@/lib/types";

export function RiskChart({
  messages,
  future,
  revealed,
}: {
  messages: CdmMessage[];
  future: CdmMessage[];
  revealed: boolean;
}) {
  const history = messages.map((item) => ({
    t: item.timeToTcaDays,
    risk: item.riskLog10,
    kind: "observed",
  }));
  const hidden = revealed
    ? future.map((item) => ({ t: item.timeToTcaDays, risk: item.riskLog10, kind: "future" }))
    : future.map((item) => ({ t: item.timeToTcaDays, risk: null, kind: "future" }));
  const data = [...history, ...hidden].sort((a, b) => b.t - a.t);

  return (
    <div className="h-72 rounded-lg border border-white/10 bg-panel p-3">
      <p className="mb-2 text-xs uppercase tracking-widest text-cyan">Risk history</p>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="t" reversed tick={{ fill: "#9db0c9", fontSize: 12 }} />
          <YAxis tick={{ fill: "#9db0c9", fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: "#10182a", border: "1px solid #3de2ff" }}
            formatter={(value) => [value ?? "hidden until reveal", "log10 Pc"]}
          />
          <ReferenceLine x={2} stroke="#f5c16c" label={{ value: "T-48", fill: "#f5c16c" }} />
          <Line type="monotone" dataKey="risk" stroke="#3de2ff" dot={false} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
