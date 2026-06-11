export const fmtNum = (n: number, opts: Intl.NumberFormatOptions = {}) =>
  new Intl.NumberFormat("en-US", opts).format(n);

export const fmtUsd = (n: number, frac = 0) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: frac,
  }).format(n);

export const fmtPct = (n: number, frac = 1) =>
  `${(n * 100).toFixed(frac)}%`;

export const fmtMs = (ms: number) => {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
};

export const titleCase = (s: string) =>
  s
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

export const riskColor = (rating?: string) => {
  switch ((rating || "").toLowerCase()) {
    case "low":
      return "text-emerald-500";
    case "medium":
      return "text-amber-500";
    case "high":
      return "text-orange-500";
    case "enhanced":
      return "text-red-500";
    case "prohibited":
      return "text-red-700";
    default:
      return "text-muted";
  }
};

export const statusChip = (s: string) => {
  switch (s) {
    case "open":
      return "chip chip-info";
    case "in_progress":
      return "chip chip-warn";
    case "closed":
      return "chip chip-success";
    default:
      return "chip chip-neutral";
  }
};

export const typologyHue: Record<string, string> = {
  structuring: "#76b900",
  smurfing: "#22c55e",
  layering: "#06b6d4",
  trade_based_ml: "#a855f7",
  shell_company: "#f97316",
  human_trafficking: "#ef4444",
  terrorist_financing: "#b91c1c",
  elder_exploitation: "#eab308",
  none: "#64748b",
  unknown: "#475569",
};

export const palette = [
  "#76b900",
  "#06b6d4",
  "#a855f7",
  "#f97316",
  "#22c55e",
  "#ef4444",
  "#eab308",
  "#64748b",
  "#0ea5e9",
  "#ec4899",
];
