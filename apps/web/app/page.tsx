import { loadCases } from "@/lib/data";
import { formatLogRisk } from "@/lib/format";
import { chanceWords, hoursUntilClosest } from "@/lib/plain";
import { Band } from "@/components/band";
import { Shell } from "@/components/shell";
import Link from "next/link";

export default async function QueuePage() {
  const cases = await loadCases();
  return (
    <Shell title="Event queue">
      <p className="text-sm text-slate-400">
        Five close-approach cases. The copilot may use only updates from 48 hours or more before
        closest approach. −6 is about 1 in a million.
      </p>
      <div className="overflow-hidden rounded-lg border border-white/10">
        <table className="w-full text-left text-sm">
          <thead className="bg-white/5 text-xs uppercase tracking-widest text-slate-400">
            <tr>
              <th className="px-4 py-3">Case</th>
              <th>Mission</th>
              <th>Hours left</th>
              <th>Today&apos;s chance</th>
              <th>Reading</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item) => {
              const latest = item.messages[item.messages.length - 1];
              return (
                <tr key={item.id} className="border-t border-white/10">
                  <td className="px-4 py-3">
                    <div className="font-medium">{item.title}</div>
                    <div className="text-xs text-slate-400">{item.blurb}</div>
                  </td>
                  <td>{item.missionAlias}</td>
                  <td className="font-mono">{hoursUntilClosest(latest.timeToTcaDays)}</td>
                  <td>
                    <div className="font-mono">{formatLogRisk(item.baselineRiskLog10)}</div>
                    <div className="text-xs text-slate-400">{chanceWords(item.baselineRiskLog10)}</div>
                  </td>
                  <td>
                    <Band value={item.prediction.riskBand} abstained={item.prediction.abstained} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Link className="text-cyan" href={`/cases/${item.id}`}>
                      Open
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
