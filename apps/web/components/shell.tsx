import Link from "next/link";

export function Shell({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-cyan/20 pb-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-cyan">PRISM / ISTRAC-style copilot</p>
          <h1 className="mt-1 text-3xl font-semibold">{title}</h1>
        </div>
        <nav className="flex gap-3 text-sm text-cyan">
          <Link href="/">Queue</Link>
          <Link href="/lab">Laboratory</Link>
        </nav>
      </header>
      <p className="rounded border border-amber/40 bg-amber/10 px-3 py-2 text-sm text-amber">
        Educational research prototype. Not for operational decisions. Human approval is required.
      </p>
      {children}
    </div>
  );
}
