import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // Base surfaces — a near-black city-at-night palette, not pure
        // #000, which reads as harsh and flattens depth in the 3D scene.
        background: "#0a0e14",
        surface: {
          DEFAULT: "#11161f",
          raised: "#161c28",
          overlay: "#1c2330",
        },
        border: {
          DEFAULT: "#242b3a",
          subtle: "#1a202c",
        },
        foreground: {
          DEFAULT: "#e6e9ef",
          muted: "#8b93a7",
          subtle: "#5a6377",
        },
        // Risk scale drives both the UI (risk list, badges) and the 3D
        // building glow shader — one scale, used everywhere, so a file
        // reads as "the same risk" whether you're looking at the city or
        // a sidebar list.
        risk: {
          low: "#3fb950",
          medium: "#d29922",
          high: "#e8584a",
          critical: "#f85149",
        },
        // Single accent color, used sparingly (primary actions, active
        // states, the import-edge particle flow) — not a rainbow of
        // "brand colors," which is what makes a UI look templated.
        accent: {
          DEFAULT: "#5b8ef4",
          muted: "#3a5a9e",
          foreground: "#eaf0ff",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        // A deliberate type scale (1.25 ratio) rather than ad hoc
        // text-sm/text-lg choices scattered per component.
        xs: ["0.75rem", { lineHeight: "1.1rem" }],
        sm: ["0.875rem", { lineHeight: "1.3rem" }],
        base: ["1rem", { lineHeight: "1.5rem" }],
        lg: ["1.25rem", { lineHeight: "1.7rem" }],
        xl: ["1.563rem", { lineHeight: "2rem" }],
        "2xl": ["1.953rem", { lineHeight: "2.3rem" }],
        "3xl": ["2.441rem", { lineHeight: "2.7rem" }],
      },
      borderRadius: {
        sm: "0.25rem",
        DEFAULT: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
      },
      boxShadow: {
        panel: "0 8px 30px rgba(0, 0, 0, 0.4)",
        glow: "0 0 24px rgba(91, 142, 244, 0.35)",
      },
      animation: {
        "pulse-risk": "pulse-risk 2.2s ease-in-out infinite",
        "fade-in": "fade-in 0.3s ease-out",
      },
      keyframes: {
        "pulse-risk": {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;