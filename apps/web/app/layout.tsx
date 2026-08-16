import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://prism.local"),
  title: { default: "PRISM — T−48 conjunction-risk forecast", template: "%s · PRISM" },
  description: "Predicting final reported conjunction risk from information available 48 hours before closest approach.",
  applicationName: "PRISM",
  robots: { index: false, follow: false },
  openGraph: {
    title: "PRISM — T−48 conjunction-risk forecast",
    description: "Predicting final reported conjunction risk from information available 48 hours before closest approach.",
    type: "website",
  },
};

export const viewport: Viewport = { colorScheme: "light", themeColor: "#f5f3ee" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased`}>
        <a href="#main-content" className="fixed left-4 top-4 z-50 -translate-y-24 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white transition-transform focus:translate-y-0">
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}
