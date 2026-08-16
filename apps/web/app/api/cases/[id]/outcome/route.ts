import { NextResponse } from "next/server";
import { loadCases } from "@/lib/data";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const item = (await loadCases()).find((entry) => entry.id === id);
  if (!item) return NextResponse.json({ message: "Case not found" }, { status: 404 });
  return NextResponse.json({
    actualFinalRiskLog10: item.actualFinalRiskLog10,
    futureMessages: item.futureMessages,
  });
}
