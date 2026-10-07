/**
 * Generates a realistic mock CityData for frontend development, before the
 * real backend (Phase 5) exists. Deliberately NOT a hand-written static
 * JSON file — generating it programmatically means it's easy to regenerate
 * with different sizes/seeds while building and testing the 3D components,
 * and it forces this file to produce genuinely valid CityData, exercising
 * the same types every real API response will need to satisfy.
 *
 * Swap-out point: once GET /repos/{id}/city exists (Phase 5), lib/api.ts
 * calls that instead. Components should never import this file directly —
 * they go through useCityData.ts (next file), which decides mock vs. real.
 */

import type { CityData, CityDistrict, CityEdge, CityNode } from "@/types/city";
import { riskLevelFromScore } from "@/types/city";

// A believable directory layout for a mid-sized Python web framework —
// modeled loosely on Flask's actual structure, not copied from it.
const MOCK_STRUCTURE: Record<string, string[]> = {
  "src/app/core": ["config.py", "routing.py", "context.py", "errors.py"],
  "src/app/auth": ["login.py", "tokens.py", "permissions.py", "oauth.py"],
  "src/app/db": ["session.py", "models.py", "migrations.py", "pool.py"],
  "src/app/api": ["users.py", "posts.py", "comments.py", "search.py", "admin.py"],
  "src/app/utils": ["validators.py", "formatters.py", "cache.py"],
  "src/app/templates": ["render.py", "filters.py"],
  "src/app/cli": ["commands.py", "scaffold.py"],
};

// Simple seeded PRNG so the "random" mock is reproducible across reloads —
// using Math.random() directly would make the city reshuffle every refresh,
// which makes visual debugging ("did my change do anything?") harder.
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateMockCity(seed = 42): CityData {
  const rand = mulberry32(seed);
  const nodes: CityNode[] = [];
  const districts: CityDistrict[] = [];

  for (const [dirPath, files] of Object.entries(MOCK_STRUCTURE)) {
    const nodeIds: string[] = [];

    for (const fileName of files) {
      const id = `${dirPath}/${fileName}`;
      const loc = Math.floor(40 + rand() * 460);
      const complexity = Math.floor(2 + rand() * rand() * 38); // skewed low, a few spikes
      // Auth and db files skew riskier in this mock — mirrors the real-world
      // pattern that security/data-layer code tends to accumulate bugs.
      const riskBias = dirPath.includes("auth") || dirPath.includes("db") ? 0.25 : 0;
      const riskScore = Math.min(0.97, rand() * 0.65 + riskBias);

      nodes.push({
        id,
        path: id,
        language: "python",
        linesOfCode: loc,
        complexity,
        riskScore: Number(riskScore.toFixed(3)),
        riskLevel: riskLevelFromScore(riskScore),
        authorCount: Math.floor(1 + rand() * 5),
        lastChangedDaysAgo: Math.floor(rand() * 400),
      });
      nodeIds.push(id);
    }

    districts.push({
      id: dirPath,
      name: dirPath.split("/").pop() ?? dirPath,
      nodeIds,
    });
  }

  const edges: CityEdge[] = [];

  // Import edges: mostly within a district, occasionally crossing into
  // src/app/core (a realistic "everything depends on config/routing" shape).
  for (const node of nodes) {
    const sameDistrict = districts.find((d) => d.nodeIds.includes(node.id));
    if (!sameDistrict) continue;

    const importsCore = rand() > 0.5;
    if (importsCore && !node.id.startsWith("src/app/core")) {
      const coreTargets = districts.find((d) => d.id === "src/app/core")?.nodeIds ?? [];
      const target = coreTargets[Math.floor(rand() * coreTargets.length)];
      if (target && target !== node.id) {
        edges.push({ source: node.id, target, kind: "import", weight: 1 });
      }
    }

    const siblings = sameDistrict.nodeIds.filter((id) => id !== node.id);
    if (siblings.length > 0 && rand() > 0.6) {
      const target = siblings[Math.floor(rand() * siblings.length)];
      edges.push({ source: node.id, target, kind: "import", weight: 1 });
    }
  }

  // Co-change edges: sparser, occasionally crossing district boundaries
  // entirely — mirrors "hidden coupling" being the point of this edge type.
  for (let i = 0; i < Math.floor(nodes.length * 0.4); i++) {
    const a = nodes[Math.floor(rand() * nodes.length)];
    const b = nodes[Math.floor(rand() * nodes.length)];
    if (a && b && a.id !== b.id) {
      edges.push({
        source: a.id,
        target: b.id,
        kind: "cochange",
        weight: Math.floor(2 + rand() * 10),
      });
    }
  }

  return {
    repoSlug: "github.com__demo__mock-flask",
    repoUrl: "https://github.com/demo/mock-flask",
    snapshotAt: new Date().toISOString(),
    nodes,
    edges,
    districts,
    modelMetrics: {
      prAuc: 0.412,
      recallAt10Pct: 0.58,
    },
  };
}