"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Moon, Sun, Activity } from "lucide-react";
import useSWR from "swr";
import { health } from "@/lib/api";

export function TopBar() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const { data, error } = useSWR("health", health, { refreshInterval: 30_000 });
  const ok = !!data && !error;
  return (
    <header className="h-14 shrink-0 border-b divider surface flex items-center px-4 gap-4">
      <div className="text-xs text-muted">
        AML Investigation Platform · NeMo Agent Toolkit
      </div>
      <div className="ml-auto flex items-center gap-3">
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-md surface-muted text-xs">
          <Activity size={14} className={ok ? "text-nv-green" : "text-rose-500"} />
          <span className="font-mono">
            {ok
              ? `${data?.n_entities.toLocaleString()} entities · ${data?.n_transactions.toLocaleString()} tx`
              : "backend offline"}
          </span>
        </div>
        <button
          aria-label="Toggle theme"
          className="btn btn-outline"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {mounted && theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
          <span className="hidden md:inline">
            {mounted ? (theme === "dark" ? "Light" : "Dark") : "Theme"}
          </span>
        </button>
      </div>
    </header>
  );
}
