/**
 * Shared types describing a city's data — the frontend's typed view of the
 * backend's `GET /repos/{id}/city` response (schemas/city.py, Phase 5).
 *
 * Deliberately mirrors ml/codecity_ml/graph/builder.py's RepoGraph shape
 * (nodes, features, edges) but flattened and simplified for rendering:
 * the frontend doesn't need every training feature, just what drives the
 * visuals and the inspector panel.
 */

export type RiskLevel = "low" | "medium" | "high" | "critical";

/** One file, rendered as one building. */
export interface CityNode {
  id: string; // repo-relative file path, matches the backend's node key
  path: string; // same as id, kept separate for display vs. lookup use
  language: string;
  linesOfCode: number;
  complexity: number; // max_cyclomatic_complexity, drives building height
  riskScore: number; // 0..1, model's predicted probability
  riskLevel: RiskLevel; // bucketed from riskScore, for color lookup
  authorCount: number;
  lastChangedDaysAgo: number;
  // Treemap layout position, computed server-side or client-side —
  // optional because mock/demo data may omit it until layout.worker.ts
  // (later file) computes it.
  layout?: {
    x: number;
    z: number;
    width: number;
    depth: number;
  };
}

/** One edge between two files — the type distinguishes which "layer" of
 * the graph it came from, since imports and co-change render differently
 * (DependencyArcs.tsx will style them distinctly). */
export interface CityEdge {
  source: string; // CityNode.id
  target: string; // CityNode.id
  kind: "import" | "cochange";
  weight: number; // import count, or co-change commit count
}

/** A directory becomes a "district" in the city — a group of buildings
 * with a shared ground plane. */
export interface CityDistrict {
  id: string; // directory path
  name: string; // last path segment, for display
  nodeIds: string[];
  layout?: {
    x: number;
    z: number;
    width: number;
    depth: number;
  };
}

/** Why a file was flagged — the explainability payload from
 * explain/gnn_explainer.py, shown when a building is clicked. */
export interface RiskExplanation {
  nodeId: string;
  topFactors: Array<{
    label: string; // e.g. "3 co-change neighbors are also high-risk"
    contribution: number; // relative weight, for a simple bar display
  }>;
  influencingNeighbors: string[]; // CityNode ids
}

/** One complete city, ready to render — what the Zustand store holds. */
export interface CityData {
  repoSlug: string;
  repoUrl: string;
  snapshotAt: string; // ISO date string
  nodes: CityNode[];
  edges: CityEdge[];
  districts: CityDistrict[];
  modelMetrics?: {
    prAuc: number;
    recallAt10Pct: number;
  };
}

export function riskLevelFromScore(score: number): RiskLevel {
  if (score >= 0.75) return "critical";
  if (score >= 0.5) return "high";
  if (score >= 0.25) return "medium";
  return "low";
}