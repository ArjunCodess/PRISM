"use client";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="mx-auto flex min-h-[70vh] max-w-3xl flex-col justify-center px-6 py-16">
      <p className="text-[0.65rem] uppercase tracking-[0.14em] text-stone-500">PRISM</p>
      <h1 className="display mt-3 text-4xl text-ink">API unavailable</h1>
      <p className="mt-4 max-w-[62ch] text-lg leading-8 text-stone-600">
        This exhibit loads cases, metrics, and live forecasts from FastAPI only. There is no frozen JSON fallback.
      </p>
      <p className="mt-4 max-w-[62ch] text-sm leading-6 text-stone-500">{error.message}</p>
      <button
        type="button"
        onClick={reset}
        className="interactive mt-8 w-fit rounded-md bg-ink px-4 py-2 text-sm text-white"
      >
        Retry
      </button>
    </main>
  );
}
