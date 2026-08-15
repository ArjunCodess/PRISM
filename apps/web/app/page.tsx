import { loadCases } from "@/lib/data";
import { formatLogRisk } from "@/lib/data";
import { Band } from "@/components/band";
import { Shell } from "@/components/shell";
import Link from "next/link";

export default async function QueuePage() {
  const cases = await loadCases();
  return (
    <Shell title="Event queue">
      <div className="overflow-hidden rounded-lg border border-white/10">
        <table className="w-full text-left text-sm">
          <thead className="bg-white/5 text-xs uppercase tracking-widest text-slate-400">
            <tr>
              <th className="px-4 py-3">Case</th>
              <th>Mission</th>
              <th>Current log-risk</th>
              <th>Forecast band</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item) => (
              <tr key={item.id} className="border-t border-white/10">
                <td className="px-4 py-3">
                  <div className="font-medium">{item.title}</div>
                  <div className="font-mono text-xs text-slate-400">{item.id}</div>
                </td>
                <td>{item.missionAlias}</td>
                <td className="telemetry font-mono">{formatLogRisk(item.baselineRiskLog10)}</td>
                <td>
                  <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} />
                </td>
                <td className="px-4 py-3 text-right">
                  <Link className="text-cyan" href={`/cases/${item.id}`}>
                    Open case
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
