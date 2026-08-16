import Link from "next/link";
import { Shell } from "@/components/shell";

export default function NotFound() {
  return (
    <Shell title="This case is not in the frozen exhibit." kicker="404">
      <p className="max-w-xl text-lg leading-8 text-stone-600">Return to the event queue and choose one of the five versioned cases.</p>
      <Link href="/" className="interactive mt-7 inline-flex rounded-md bg-ink px-5 py-3 text-sm text-white hover:bg-cyan">Return to event queue</Link>
    </Shell>
  );
}
