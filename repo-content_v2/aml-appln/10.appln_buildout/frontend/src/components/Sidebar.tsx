"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ListChecks,
  Telescope,
  Users,
  GitCompare,
  Beaker,
  ShieldAlert,
  BookText,
  Trophy,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/alerts", label: "Alert Queue", icon: ListChecks },
  { href: "/cockpit", label: "Investigation Cockpit", icon: Telescope },
  { href: "/entities", label: "Entity 360", icon: Users },
  { href: "/tools", label: "Compliance Tools", icon: BookText },
  { href: "/skills", label: "Skill Playgrounds", icon: Beaker },
  { href: "/compare", label: "Model Comparison", icon: GitCompare },
  { href: "/leaderboard", label: "Leaderboard", icon: Trophy },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r divider surface">
      <div className="flex items-center gap-2 px-4 h-14 border-b divider">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-nv-green/15 text-nv-green">
          <ShieldAlert size={16} />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">AML Investigator</div>
          <div className="text-[10px] uppercase tracking-wider text-muted">
            NeMo Agent Toolkit
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto py-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname?.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 mx-2 px-3 py-2 rounded-md text-sm transition-colors",
                active
                  ? "bg-nv-green/10 text-nv-green border border-nv-green/30"
                  : "text-muted hover:text-[rgb(var(--fg))] hover:bg-[rgb(var(--line))] border border-transparent",
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t divider p-3 text-[11px] text-muted">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-nv-green animate-pulse-slow" />
          Backend online
        </div>
        <div className="mt-1 truncate font-mono">localhost:8010 • aml-custom-task-nim</div>
      </div>
    </aside>
  );
}
