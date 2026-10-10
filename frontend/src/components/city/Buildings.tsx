/**
 * Renders every CityNode as a building, positioned by the real squarified
 * treemap from useCityLayout. Height additionally scales by
 * timelapseProgress (from TimeLapse.tsx's slider) — buildings grow from
 * zero to full height as the slider advances, approximating repo history
 * growth (see TimeLapse.tsx's docstring for the mock-mode limitation this
 * approximates around).
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
  const timelapseProgress = useCityStore((s) => s.timelapseProgress);

  const { nodePositions } = useCityLayout();

  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [pulsePhase, setPulsePhase] = useState(0);

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

  // Now re-runs when timelapseProgress changes, not just on layout/
  // complexity changes — the growth effect needs transforms recomputed
  // every time the slider moves, same reasoning as layout changes needing
  // a recompute, just a different trigger.
  useMemo(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();

    laidOut.forEach((node, i) => {
      const fullHeight = heightFromComplexity(node.complexity, maxComplexity);
      const height = Math.max(0.02, fullHeight * timelapseProgress);
      dummy.position.set(node.x, height / 2, node.z);
      dummy.scale.set(node.width, height, node.depth);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [laidOut, maxComplexity, timelapseProgress]);

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
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial vertexColors roughness={0.55} metalness={0.15} />
      <instancedBufferAttribute attach="instanceColor" args={[colorArray, 3]} />
    </instancedMesh>
  );
}