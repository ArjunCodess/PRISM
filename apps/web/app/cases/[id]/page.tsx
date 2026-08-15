import { CaseWorkspace } from "@/components/case-workspace";
import { Shell } from "@/components/shell";
import { loadCases } from "@/lib/data";
import { notFound } from "next/navigation";

export default async function CasePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const cases = await loadCases();
  const item = cases.find((entry) => entry.id === id);
  if (!item) notFound();
  return (
    <Shell title={item.title}>
      <p className="font-mono text-sm text-slate-400">
        {item.id} · {item.missionAlias} · cutoff 48h
      </p>
      <CaseWorkspace item={item} />
    </Shell>
  );
}
