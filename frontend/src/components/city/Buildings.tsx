/**
 * Renders every CityNode as a building, positioned by the real squarified
 * treemap from useCityLayout (layout.worker.ts) instead of the placeholder
 * grid this file started with. Still one InstancedMesh for the whole city —
 * only WHERE each instance sits has changed, not HOW it's drawn.
 */

"use client";

import { useMemo, useRef, useState } from "react";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useCityStore } from "@/store/cityStore";
import { useCityLayout } from "@/hooks/useCityLayout";
import type { CityNode, RiskLevel } from "@/types/city";

const RISK_COLORS: Record<RiskLevel, string> = {
  low: "#3fb950",
  medium: "#d29922",
  high: "#e8584a",
  critical: "#f85149",
};

const MAX_HEIGHT = 14;
const MIN_HEIGHT = 0.8;

interface LaidOutNode extends CityNode {
  x: number;
  z: number;
  width: number;
  depth: number;
}

function heightFromComplexity(complexity: number, maxComplexity: number): number {
  const t = maxComplexity > 0 ? complexity / maxComplexity : 0;
  return MIN_HEIGHT + t * (MAX_HEIGHT - MIN_HEIGHT);
}

export function Buildings() {
  const city = useCityStore((s) => s.city);
  const selectedNodeId = useCityStore((s) => s.selectedNodeId);
  const hoveredNodeId = useCityStore((s) => s.hoveredNodeId);
  const selectNode = useCityStore((s) => s.selectNode);
  const hoverNode = useCityStore((s) => s.hoverNode);

  const { nodePositions, isComputing } = useCityLayout();

  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [pulsePhase, setPulsePhase] = useState(0);

  // Combine CityNode data with the worker's computed positions. Nodes the
  // worker hasn't placed yet (still computing, or a race on first load)
  // are filtered out rather than rendered at a fallback (0,0) — a pile of
  // boxes at the origin looks like a bug, an empty scene for a moment
  // doesn't.
  const laidOut = useMemo<LaidOutNode[]>(() => {
    if (!city) return [];
    const result: LaidOutNode[] = [];
    for (const node of city.nodes) {
      const pos = nodePositions.get(node.id);
      if (!pos) continue;
      result.push({ ...node, x: pos.x, z: pos.z, width: pos.width, depth: pos.depth });
    }
    return result;
  }, [city, nodePositions]);

  const maxComplexity = useMemo(
    () => Math.max(1, ...laidOut.map((n) => n.complexity)),
    [laidOut]
  );

  const colorArray = useMemo(() => {
    const colors = new Float32Array(laidOut.length * 3);
    laidOut.forEach((node, i) => {
      const baseColor = new THREE.Color(RISK_COLORS[node.riskLevel]);
      const isSelected = node.id === selectedNodeId;
      const isHovered = node.id === hoveredNodeId;
      const emphasis = isSelected ? 1.0 : isHovered ? 1.3 : 1.0;
      baseColor.multiplyScalar(emphasis);
      baseColor.toArray(colors, i * 3);
    });
    return colors;
  }, [laidOut, selectedNodeId, hoveredNodeId]);

  // Transforms now use each node's real treemap width/depth, not a fixed
  // footprint — a file with more lines of code visibly occupies more
  // ground, matching the original "Footprint: Lines of code" plan.
  useMemo(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();

    laidOut.forEach((node, i) => {
      const height = heightFromComplexity(node.complexity, maxComplexity);
      dummy.position.set(node.x, height / 2, node.z);
      dummy.scale.set(node.width, height, node.depth);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [laidOut, maxComplexity]);

  useFrame((_, delta) => {
    setPulsePhase((p) => p + delta);
    const mesh = meshRef.current;
    if (!mesh) return;

    const pulse = 0.75 + 0.25 * Math.sin(pulsePhase * 2.2);
    laidOut.forEach((node, i) => {
      if (node.riskLevel !== "high" && node.riskLevel !== "critical") return;
      const baseColor = new THREE.Color(RISK_COLORS[node.riskLevel]);
      baseColor.multiplyScalar(pulse);
      mesh.setColorAt(i, baseColor);
    });
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  function handleClick(e: ThreeEvent<MouseEvent>) {
    e.stopPropagation();
    const id = e.instanceId;
    if (id === undefined) return;
    const node = laidOut[id];
    if (node) selectNode(node.id === selectedNodeId ? null : node.id);
  }

  function handlePointerOver(e: ThreeEvent<PointerEvent>) {
    e.stopPropagation();
    const id = e.instanceId;
    if (id === undefined) return;
    const node = laidOut[id];
    if (node) {
      hoverNode(node.id);
      document.body.style.cursor = "pointer";
    }
  }

  function handlePointerOut() {
    hoverNode(null);
    document.body.style.cursor = "auto";
  }

  if (!city || laidOut.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, laidOut.length]}
      castShadow
      receiveShadow
      onClick={handleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      {/* Base geometry is a unit cube (1x1x1) — per-instance scale now
          does ALL the sizing work: width/depth from the treemap footprint,
          height from complexity. No separate boxGeometry args needed. */}
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial vertexColors roughness={0.55} metalness={0.15} />
      <instancedBufferAttribute attach="instanceColor" args={[colorArray, 3]} />
    </instancedMesh>
  );
}