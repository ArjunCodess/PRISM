import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { CaseWorkspace } from "@/components/case-workspace";
import { Shell } from "@/components/shell";
import { loadCases, loadLiveCase } from "@/lib/data";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const item = (await loadCases()).find((entry) => entry.id === id);
  return { title: item?.title ?? "Case not found" };
}

export default async function CasePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const item = await loadLiveCase(id);
  if (!item) notFound();
  const { futureMessages: _futureMessages, actualFinalRiskLog10: _actualFinalRiskLog10, ...cutoffSafeItem } = item;
  return (
    <Shell title={item.title} kicker={`${item.missionAlias} · ${item.id} · live model`}>
      <Link href="/" className="no-print interactive mb-6 inline-flex items-center gap-2 text-sm text-stone-500 hover:text-cyan">
        <span aria-hidden="true">←</span> Back to event queue
      </Link>
      <CaseWorkspace item={cutoffSafeItem} />
    </Shell>
  );
}
