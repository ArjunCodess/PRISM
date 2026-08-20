import type { Metadata } from "next";
import Link from "next/link";
import { Shell } from "@/components/shell";

export const metadata: Metadata = { title: "About" };

export default function InfoPage() {
  return (
    <Shell title="A research copilot that forecasts debris risk 48 hours early." kicker="Project brief" compact>
      <div className="max-w-3xl space-y-5 text-[0.95rem] leading-7 text-stone-600">
        <p>
          Representing the <I>CMS Kanpur Road Campus</I> for <I>National Space Day</I>, I built <B>PRISM</B>, a <I>smart, offline research copilot</I> designed to forecast space-debris collision risk. This project is directly inspired by the vital space-tracking work done by <I>ISRO&apos;s NETRA control centre</I>.
        </p>
        <p>
          Currently, satellite operators get warning messages as debris gets close, but <I>waiting for the final, most accurate data</I> leaves them with almost no time to plan a safe escape. Therefore, PRISM <B>freezes the clock exactly 48 hours before closest approach</B>.
        </p>
        <p>
          Instead of just <I>carrying the latest warning snapshot forward</I>, it reads the history of those warnings and forecasts the later reported collision chance. The live model is a two-part floor hurdle: it can call a later collapse to the dataset floor, otherwise it adjusts today&apos;s report. SHAP explains the residual forecast in named quantities, not a black box.
        </p>
        <p className="pt-2 text-sm text-stone-500">
          <Link href="/" className="interactive text-ink hover:text-cyan">
            Event queue
          </Link>
          <span className="mx-3">·</span>
          <Link href="/lab" className="interactive text-ink hover:text-cyan">
            Model lab
          </Link>
        </p>
      </div>
    </Shell>
  );
}

function B({ children }: { children: React.ReactNode }) {
  return <strong className="font-semibold text-ink">{children}</strong>;
}

function I({ children }: { children: React.ReactNode }) {
  return <em>{children}</em>;
}
