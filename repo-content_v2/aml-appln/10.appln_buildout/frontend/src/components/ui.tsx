"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  className,
  children,
  actions,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  className?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section
      className={clsx(
        "surface rounded-xl shadow-panel overflow-hidden animate-fade-in",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between px-4 py-3 border-b divider">
          <div>
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {subtitle && (
              <p className="text-xs text-muted mt-0.5">{subtitle}</p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div>{children}</div>
    </section>
  );
}

export function Kpi({
  label,
  value,
  delta,
  icon,
  hint,
  highlight,
}: {
  label: string;
  value: ReactNode;
  delta?: string;
  icon?: ReactNode;
  hint?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={clsx(
        "surface rounded-xl p-4 shadow-panel flex flex-col gap-1.5",
        highlight && "border-nv-green/40 shadow-nv-glow",
      )}
    >
      <div className="flex items-center justify-between text-xs uppercase tracking-wide text-muted">
        <span>{label}</span>
        {icon}
      </div>
      <div className="text-2xl font-semibold tracking-tight">{value}</div>
      <div className="flex items-baseline gap-2">
        {delta && <span className="text-xs chip chip-brand">{delta}</span>}
        {hint && <span className="text-[11px] text-muted">{hint}</span>}
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="px-6 pt-6 pb-4 flex items-start justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  cta,
}: {
  title: string;
  hint?: string;
  cta?: ReactNode;
}) {
  return (
    <div className="p-10 text-center border divider rounded-xl surface-muted">
      <div className="text-sm font-medium">{title}</div>
      {hint && <p className="text-xs text-muted mt-1">{hint}</p>}
      {cta && <div className="mt-4 flex justify-center">{cta}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 p-8 text-xs text-muted">
      <span className="h-2 w-2 rounded-full bg-nv-green animate-ping" />
      {label}
    </div>
  );
}
