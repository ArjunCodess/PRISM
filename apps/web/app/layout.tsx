import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://prism.local"),
  title: { default: "PRISM — conjunction risk copilot", template: "%s · PRISM" },
  description: "An offline, explainable T-48-hour conjunction-risk forecasting research prototype.",
  applicationName: "PRISM",
  robots: { index: false, follow: false },
  openGraph: {
    title: "PRISM — conjunction risk copilot",
    description: "Forecasting final conjunction risk 48 hours early, with uncertainty and explanations.",
    type: "website",
  },
};

export const viewport: Viewport = { colorScheme: "light", themeColor: "#f5f3ee" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <a href="#main-content" className="fixed left-4 top-4 z-50 -translate-y-24 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white transition-transform focus:translate-y-0">
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}
