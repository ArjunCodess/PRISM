import Link from "next/link";
import { PrimaryNav } from "@/components/primary-nav";

export function Shell({
  title,
  kicker,
  children,
  compact = false,
}: {
  title: string;
  kicker?: string;
  children: React.ReactNode;
  compact?: boolean;
}) {
  return (
    <div className="mx-auto flex min-h-[100dvh] max-w-[1240px] flex-col px-5 sm:px-8">
      <header className="no-print flex min-h-20 items-center justify-between gap-5 border-b hairline">
        <Link href="/" className="flex items-baseline gap-3" aria-label="PRISM event queue">
          <span className="display text-2xl">PRISM</span>
          <span className="hidden text-[0.65rem] uppercase tracking-[0.14em] text-stone-500 sm:block">T−48 risk forecast</span>
        </Link>
        <PrimaryNav />
      </header>

      <main id="main-content" className={`flex-1 ${compact ? "py-6" : "py-10 sm:py-14"}`}>
        <header className={`max-w-4xl ${compact ? "mb-5" : "mb-10"}`}>
          {kicker ? <p className={`eyebrow ${compact ? "mb-2" : "mb-3"}`}>{kicker}</p> : null}
          <h1 className={`display leading-[1.05] ${compact ? "text-3xl sm:text-4xl" : "text-4xl sm:text-6xl"}`}>{title}</h1>
        </header>
        {children}
      </main>

      <footer className="no-print grid gap-2 border-t hairline py-6 text-xs leading-5 text-stone-500 sm:grid-cols-2">
        <p>Research prototype · not flight software</p>
        <p className="sm:text-right">Forecasts support human review and never authorize a manoeuvre.</p>
      </footer>
    </div>
  );
}
