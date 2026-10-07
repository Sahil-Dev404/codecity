/**
 * Renders CityEdges as visible connections. Import edges (structural) draw
 * as raised arcs between buildings; co-change edges (behavioral, "hidden
 * coupling") draw as thin, low ground lines — a deliberate visual echo of
 * the same distinction graph/edges.py drew on the Python side: imports are
 * what the code says, co-change is what the commit history says.
 *
 * Only edges touching the SELECTED node are rendered, not all edges at
 * once — a repo can easily have hundreds of edges, and drawing them all
 * simultaneously would be unreadable visual noise rather than insight.
 */

"use client";

import { useMemo } from "react";
import * as THREE from "three";
import { useCityStore } from "@/store/cityStore";
import { useCityLayout } from "@/hooks/useCityLayout";

const IMPORT_COLOR = "#5b8ef4"; // accent token — structural edges feel "active"
const COCHANGE_COLOR = "#8b93a7"; // foreground-muted token — behavioral edges feel "quiet"
const ARC_HEIGHT = 6;
const ARC_SEGMENTS = 32;

interface ResolvedEdge {
  id: string;
  kind: "import" | "cochange";
  weight: number;
  from: THREE.Vector3;
  to: THREE.Vector3;
}

export function DependencyArcs() {
  const city = useCityStore((s) => s.city);
  const selectedNodeId = useCityStore((s) => s.selectedNodeId);
  const { nodePositions } = useCityLayout();

  const edges = useMemo<ResolvedEdge[]>(() => {
    if (!city || !selectedNodeId) return [];

    const relevant = city.edges.filter(
      (e) => e.source === selectedNodeId || e.target === selectedNodeId
    );

    const resolved: ResolvedEdge[] = [];
    for (const e of relevant) {
      const fromPos = nodePositions.get(e.source);
      const toPos = nodePositions.get(e.target);
      if (!fromPos || !toPos) continue; // skip edges to not-yet-laid-out nodes

      resolved.push({
        id: `${e.kind}-${e.source}-${e.target}`,
        kind: e.kind,
        weight: e.weight,
        from: new THREE.Vector3(fromPos.x, 0.3, fromPos.z),
        to: new THREE.Vector3(toPos.x, 0.3, toPos.z),
      });
    }
    return resolved;
  }, [city, selectedNodeId, nodePositions]);

  if (edges.length === 0) return null;

  return (
    <group>
      {edges.map((e) =>
        e.kind === "import" ? (
          <ImportArc key={e.id} from={e.from} to={e.to} weight={e.weight} />
        ) : (
          <CoChangeLine key={e.id} from={e.from} to={e.to} weight={e.weight} />
        )
      )}
    </group>
  );
}

/** A raised, curved arc — reads as "energy flowing between two points,"
 * appropriate for a structural import relationship. */
function ImportArc({ from, to, weight }: { from: THREE.Vector3; to: THREE.Vector3; weight: number }) {
  const points = useMemo(() => {
    const mid = from.clone().lerp(to, 0.5);
    mid.y += ARC_HEIGHT * Math.min(1, from.distanceTo(to) / 40 + 0.3);

    const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
    return curve.getPoints(ARC_SEGMENTS);
  }, [from, to]);

  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  // Thicker lines for heavily-weighted edges (more import references
  // between the two files), capped so one outlier doesn't dominate.
  const lineWidth = Math.min(3, 1 + weight * 0.3);

  return (
    <line geometry={geometry}>
      <lineBasicMaterial color={IMPORT_COLOR} transparent opacity={0.55} linewidth={lineWidth} />
    </line>
  );
}

/** A flat, direct ground-level line — deliberately less visually
 * prominent than an import arc, since co-change is a softer, statistical
 * signal rather than a hard structural dependency. */
function CoChangeLine({ from, to, weight }: { from: THREE.Vector3; to: THREE.Vector3; weight: number }) {
  const geometry = useMemo(
    () => new THREE.BufferGeometry().setFromPoints([from, to]),
    [from, to]
  );

  const opacity = Math.min(0.5, 0.15 + weight * 0.02);

  return (
    <line geometry={geometry}>
      <lineDashedMaterial
        color={COCHANGE_COLOR}
        transparent
        opacity={opacity}
        dashSize={0.6}
        gapSize={0.4}
      />
    </line>
  );
}