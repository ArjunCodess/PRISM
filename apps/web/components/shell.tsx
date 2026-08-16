import Link from "next/link";
import { PrimaryNav } from "@/components/primary-nav";

export function Shell({ title, kicker, children }: { title: string; kicker?: string; children: React.ReactNode }) {
  return (
    <div className="mx-auto flex min-h-[100dvh] max-w-[1240px] flex-col px-5 sm:px-8">
      <header className="no-print flex min-h-20 items-center justify-between gap-5 border-b hairline">
        <Link href="/" className="flex items-baseline gap-3" aria-label="PRISM event queue">
          <span className="display text-2xl">PRISM</span>
          <span className="hidden text-[0.65rem] uppercase tracking-[0.14em] text-stone-500 sm:block">T−48 risk forecast</span>
        </Link>
        <PrimaryNav />
      </header>

      <main id="main-content" className="flex-1 py-10 sm:py-14">
        <header className="mb-10 max-w-4xl">
          {kicker ? <p className="eyebrow mb-3">{kicker}</p> : null}
          <h1 className="display text-4xl leading-[1.05] sm:text-6xl">{title}</h1>
        </header>
        {children}
      </main>

      <footer className="no-print grid gap-2 border-t hairline py-6 text-xs leading-5 text-stone-500 sm:grid-cols-2">
        <p>Educational research prototype · offline-ready</p>
        <p className="sm:text-right">Forecasts support human review and never authorize a manoeuvre.</p>
      </footer>
    </div>
  );
}
