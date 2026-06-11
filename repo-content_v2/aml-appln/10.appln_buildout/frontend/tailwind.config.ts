import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // NVIDIA brand
        nv: {
          green: "#76b900",       // primary
          "green-bright": "#8fdc00",
          "green-dim": "#5a8e00",
          black: "#0a0a0a",
          slate: "#111827",
          panel: "#161b22",
          line: "#1f2937",
        },
        // Semantic risk palette
        risk: {
          low: "#22c55e",
          medium: "#eab308",
          high: "#f97316",
          enhanced: "#ef4444",
          prohibited: "#b91c1c",
        },
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Fira Code"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        "nv-glow": "0 0 0 1px rgba(118,185,0,0.4), 0 0 24px -4px rgba(118,185,0,0.25)",
        panel: "0 1px 0 0 rgba(255,255,255,0.04), 0 8px 24px -12px rgba(0,0,0,0.4)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "fade-in": "fadeIn 0.18s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(2px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
