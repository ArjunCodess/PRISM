import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PRISM",
  description: "Predictive Risk Intelligence for Space Monitoring",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <div className="mx-auto max-w-7xl px-6 py-6">{children}</div>
      </body>
    </html>
  );
}
