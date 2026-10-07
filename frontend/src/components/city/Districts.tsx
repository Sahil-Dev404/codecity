/**
 * Renders each directory as a ground-plane "district" beneath its
 * buildings, using districtRects from useCityLayout — the half of the
 * worker's output Buildings.tsx doesn't need but this component does.
 * Uses regular (non-instanced) meshes: unlike buildings, district count
 * stays small (one per directory, typically single digits to a few dozen)
 * so instancing's draw-call savings wouldn't meaningfully matter here,
 * and plain meshes let each district have its own clickable label.
 */

"use client";

import { useMemo, useState } from "react";
import { Text } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { useCityStore } from "@/store/cityStore";
import { useCityLayout } from "@/hooks/useCityLayout";

const PLATFORM_HEIGHT = 0.12;

export function Districts() {
  const city = useCityStore((s) => s.city);
  const { districtRects } = useCityLayout();

  const districts = useMemo(() => {
    if (!city) return [];
    return city.districts
      .map((d) => {
        const rect = districtRects.get(d.id);
        if (!rect) return null;
        return { ...d, ...rect };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);
  }, [city, districtRects]);

  if (districts.length === 0) return null;

  return (
    <group>
      {districts.map((d) => (
        <DistrictPlatform key={d.id} id={d.id} name={d.name} x={d.x} z={d.z} width={d.width} depth={d.depth} />
      ))}
    </group>
  );
}

function DistrictPlatform({
  id,
  name,
  x,
  z,
  width,
  depth,
}: {
  id: string;
  name: string;
  x: number;
  z: number;
  width: number;
  depth: number;
}) {
  const [hovered, setHovered] = useState(false);

  function handlePointerOver(e: ThreeEvent<PointerEvent>) {
    e.stopPropagation();
    setHovered(true);
  }

  function handlePointerOut() {
    setHovered(false);
  }

  // districtRects gives a corner (x, z) + size, but buildings are
  // positioned by their CENTER (set in layout.worker.ts's squarify
  // output). Converting here keeps that center-vs-corner distinction
  // local to this one file rather than leaking into useCityLayout's
  // shared return shape, which both Buildings and Districts consume.
  const centerX = x + width / 2;
  const centerZ = z + depth / 2;

  return (
    <group>
      <mesh
        position={[centerX, -PLATFORM_HEIGHT / 2, centerZ]}
        receiveShadow
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <boxGeometry args={[width, PLATFORM_HEIGHT, depth]} />
        <meshStandardMaterial
          color={hovered ? "#1c2330" : "#161c28"}
          roughness={0.85}
          metalness={0.05}
        />
      </mesh>

      {/* Thin edge outline so adjacent districts read as distinct blocks
          even when their platform colors are nearly identical. */}
      <lineSegments position={[centerX, 0, centerZ]}>
        <edgesGeometry args={[new BoxEdgesGeometryArgs(width, PLATFORM_HEIGHT, depth)]} />
        <lineBasicMaterial color="#242b3a" transparent opacity={0.6} />
      </lineSegments>

      {/* Label floats just above the platform, always facing the camera */}
      <Text
        position={[centerX, 0.4, z - 0.6]}
        fontSize={Math.min(1.1, width * 0.09)}
        color={hovered ? "#e6e9ef" : "#5a6377"}
        anchorX="center"
        anchorY="bottom"
        font="/fonts/JetBrainsMono-Regular.woff"
        maxWidth={width}
      >
        {name}
      </Text>
    </group>
  );
}

// Tiny helper so edgesGeometry gets a real BoxGeometry to derive edges
// from, without re-instantiating one inline on every render.
function BoxEdgesGeometryArgs(width: number, height: number, depth: number) {
  const THREE = require("three");
  return new THREE.BoxGeometry(width, height, depth);
}