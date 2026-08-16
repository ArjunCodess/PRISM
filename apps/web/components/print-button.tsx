"use client";

export function PrintButton() {
  return <button type="button" onClick={() => window.print()} className="no-print interactive shrink-0 rounded-md border hairline px-4 py-2.5 text-xs text-stone-600 hover:border-cyan/40 hover:text-cyan">Print</button>;
}
