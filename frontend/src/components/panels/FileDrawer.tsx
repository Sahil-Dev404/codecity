/**
 * Slide-in detail panel for the selected building. The visible payoff of
 * selection — without this, clicking a building only moves the camera
 * with no confirmation of what got selected or why it matters.
 */

"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X, GitCommit, Users, Clock, Code2, AlertTriangle } from "lucide-react";
import { useCityStore } from "@/store/cityStore";
import type { RiskLevel } from "@/types/city";

const RISK_LABEL: Record<RiskLevel, string> = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
  critical: "Critical risk",
};

// Tailwind can't interpolate dynamic class names like `text-risk-${level}`
// at build time (it needs literal strings to scan for), so each risk
// level's classes are spelled out explicitly here rather than templated.
const RISK_STYLES: Record<RiskLevel, { text: string; bg: string; border: string }> = {
  low: { text: "text-risk-low", bg: "bg-risk-low/10", border: "border-risk-low/30" },
  medium: { text: "text-risk-medium", bg: "bg-risk-medium/10", border: "border-risk-medium/30" },
  high: { text: "text-risk-high", bg: "bg-risk-high/10", border: "border-risk-high/30" },
  critical: {
    text: "text-risk-critical",
    bg: "bg-risk-critical/10",
    border: "border-risk-critical/30",
  },
};

export function FileDrawer() {
  const getSelectedNode = useCityStore((s) => s.getSelectedNode);
  const selectedNodeId = useCityStore((s) => s.selectedNodeId);
  const selectNode = useCityStore((s) => s.selectNode);

  const node = getSelectedNode();

  return (
    <AnimatePresence>
      {node && (
        <motion.aside
          key={selectedNodeId} // re-trigger enter animation when switching files directly
          initial={{ x: 32, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 32, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="glass-panel absolute top-4 right-4 bottom-4 w-80 rounded-lg p-5 flex flex-col overflow-y-auto"
        >
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <p className="font-mono text-xs text-foreground-subtle truncate">
                {node.path.split("/").slice(0, -1).join("/")}
              </p>
              <h2 className="font-mono text-sm text-foreground font-medium mt-0.5 truncate">
                {node.path.split("/").pop()}
              </h2>
            </div>
            <button
              onClick={() => selectNode(null)}
              className="shrink-0 p-1 rounded-md text-foreground-subtle hover:text-foreground hover:bg-surface-overlay transition-colors"
              aria-label="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div
            className={`mt-4 flex items-center gap-2 rounded-md border px-3 py-2 ${RISK_STYLES[node.riskLevel].bg} ${RISK_STYLES[node.riskLevel].border}`}
          >
            <AlertTriangle className={`w-4 h-4 shrink-0 ${RISK_STYLES[node.riskLevel].text}`} />
            <div>
              <p className={`text-sm font-medium ${RISK_STYLES[node.riskLevel].text}`}>
                {RISK_LABEL[node.riskLevel]}
              </p>
              <p className="text-xs text-foreground-subtle">
                {Math.round(node.riskScore * 100)}% predicted probability
              </p>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3">
            <Stat icon={<Code2 className="w-3.5 h-3.5" />} label="Lines of code" value={node.linesOfCode.toLocaleString()} />
            <Stat icon={<AlertTriangle className="w-3.5 h-3.5" />} label="Complexity" value={node.complexity.toString()} />
            <Stat icon={<Users className="w-3.5 h-3.5" />} label="Authors" value={node.authorCount.toString()} />
            <Stat
              icon={<Clock className="w-3.5 h-3.5" />}
              label="Last changed"
              value={formatDaysAgo(node.lastChangedDaysAgo)}
            />
          </div>

          <div className="mt-6 pt-5 border-t border-border">
            <button
              className="w-full flex items-center justify-center gap-2 text-sm font-medium text-foreground-muted hover:text-foreground bg-surface-overlay hover:bg-surface-raised rounded-md py-2.5 transition-colors"
              disabled
              title="Wired up once explain/gnn_explainer.py is served via the API"
            >
              <GitCommit className="w-3.5 h-3.5" />
              Why is this flagged?
            </button>
            <p className="mt-2 text-xs text-center text-foreground-subtle">
              Explanation available once the backend is connected
            </p>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-surface-overlay rounded-md px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-foreground-subtle">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <p className="mt-1 text-sm font-mono text-foreground">{value}</p>
    </div>
  );
}

function formatDaysAgo(days: number): string {
  if (days === 0) return "today";
  if (days === 1) return "1 day ago";
  if (days < 30) return `${days} days ago`;
  if (days < 365) return `${Math.round(days / 30)} mo ago`;
  return `${Math.round(days / 365)}y ago`;
}