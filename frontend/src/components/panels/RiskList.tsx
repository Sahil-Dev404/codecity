/**
 * Ranked list of every file by risk score — the triage view. Clicking a
 * row calls the same selectNode action Buildings.tsx uses on click, so it
 * gets the camera fly-to and FileDrawer open for free: one selection
 * system, driven from two completely different UI surfaces.
 */

"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, ArrowUpDown } from "lucide-react";
import { useCityStore } from "@/store/cityStore";
import type { CityNode, RiskLevel } from "@/types/city";

const RISK_DOT: Record<RiskLevel, string> = {
  low: "bg-risk-low",
  medium: "bg-risk-medium",
  high: "bg-risk-high",
  critical: "bg-risk-critical",
};

type SortMode = "risk" | "complexity" | "recent";

const SORT_LABEL: Record<SortMode, string> = {
  risk: "Risk",
  complexity: "Complexity",
  recent: "Recently changed",
};

function sortNodes(nodes: CityNode[], mode: SortMode): CityNode[] {
  const copy = [...nodes];
  switch (mode) {
    case "risk":
      return copy.sort((a, b) => b.riskScore - a.riskScore);
    case "complexity":
      return copy.sort((a, b) => b.complexity - a.complexity);
    case "recent":
      return copy.sort((a, b) => a.lastChangedDaysAgo - b.lastChangedDaysAgo);
  }
}

export function RiskList() {
  const city = useCityStore((s) => s.city);
  const selectedNodeId = useCityStore((s) => s.selectedNodeId);
  const hoveredNodeId = useCityStore((s) => s.hoveredNodeId);
  const selectNode = useCityStore((s) => s.selectNode);
  const hoverNode = useCityStore((s) => s.hoverNode);

  const [sortMode, setSortMode] = useState<SortMode>("risk");

  const sorted = useMemo(() => {
    if (!city) return [];
    return sortNodes(city.nodes, sortMode);
  }, [city, sortMode]);

  if (!city) return null;

  const flaggedCount = city.nodes.filter(
    (n) => n.riskLevel === "high" || n.riskLevel === "critical"
  ).length;

  function cycleSortMode() {
    const modes: SortMode[] = ["risk", "complexity", "recent"];
    const next = modes[(modes.indexOf(sortMode) + 1) % modes.length];
    setSortMode(next);
  }

  return (
    <div className="glass-panel absolute top-4 left-4 bottom-4 w-72 rounded-lg flex flex-col overflow-hidden">
      <div className="p-4 pb-3 border-b border-border">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-foreground">Files</h2>
          <button
            onClick={cycleSortMode}
            className="flex items-center gap-1.5 text-xs text-foreground-subtle hover:text-foreground transition-colors"
          >
            <ArrowUpDown className="w-3 h-3" />
            {SORT_LABEL[sortMode]}
          </button>
        </div>
        {flaggedCount > 0 && (
          <p className="mt-1.5 flex items-center gap-1.5 text-xs text-risk-high">
            <AlertTriangle className="w-3 h-3" />
            {flaggedCount} file{flaggedCount === 1 ? "" : "s"} flagged
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {sorted.map((node, i) => (
          <RiskRow
            key={node.id}
            node={node}
            index={i}
            isSelected={node.id === selectedNodeId}
            isHovered={node.id === hoveredNodeId}
            onSelect={() => selectNode(node.id === selectedNodeId ? null : node.id)}
            onHover={() => hoverNode(node.id)}
            onUnhover={() => hoverNode(null)}
          />
        ))}
      </div>
    </div>
  );
}

function RiskRow({
  node,
  index,
  isSelected,
  isHovered,
  onSelect,
  onHover,
  onUnhover,
}: {
  node: CityNode;
  index: number;
  isSelected: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: () => void;
  onUnhover: () => void;
}) {
  return (
    <motion.button
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2, delay: Math.min(index * 0.015, 0.3) }}
      onClick={onSelect}
      onMouseEnter={onHover}
      onMouseLeave={onUnhover}
      className={`w-full flex items-center gap-2.5 px-4 py-2 text-left transition-colors ${
        isSelected
          ? "bg-accent/15"
          : isHovered
            ? "bg-surface-overlay"
            : "hover:bg-surface-overlay/60"
      }`}
    >
      <span className={`w-2 h-2 rounded-full shrink-0 ${RISK_DOT[node.riskLevel]}`} />
      <span className="flex-1 min-w-0 font-mono text-xs text-foreground truncate">
        {node.path.split("/").pop()}
      </span>
      <span className="shrink-0 text-xs font-mono text-foreground-subtle">
        {Math.round(node.riskScore * 100)}%
      </span>
    </motion.button>
  );
}