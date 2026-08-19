"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Event queue" },
  { href: "/lab", label: "Model lab" },
  { href: "/info", label: "About" },
];

export function PrimaryNav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Primary navigation" className="flex items-center gap-5">
      {items.map((item) => {
        const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        return (
          <Link key={item.href} href={item.href} aria-current={active ? "page" : undefined} className={`interactive border-b py-2 text-sm ${active ? "border-ink text-ink" : "border-transparent text-stone-500 hover:text-ink"}`}>
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
