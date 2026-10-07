/**
 * Flies the camera to focus on the selected building, using REAL treemap
 * positions from useCityLayout — replacing the hashToRange placeholder
 * flagged in the original version of this file. Still doesn't know or
 * care WHO selected a node, only that something did.
 */

"use client";

import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { useCityStore } from "@/store/cityStore";
import { useCityLayout } from "@/hooks/useCityLayout";

const FLY_DURATION = 0.9;
const FOCUS_DISTANCE = 9;
const FOCUS_HEIGHT_OFFSET = 5;
const DEFAULT_CAMERA_POS = new THREE.Vector3(40, 35, 40);
const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);

export function CameraRig() {
  const { camera, controls } = useThree((state) => ({
    camera: state.camera,
    controls: state.controls as OrbitControlsImpl | null,
  }));

  const city = useCityStore((s) => s.city);
  const selectedNodeId = useCityStore((s) => s.selectedNodeId);
  const { nodePositions } = useCityLayout();

  const animRef = useRef<{
    active: boolean;
    elapsed: number;
    fromPos: THREE.Vector3;
    toPos: THREE.Vector3;
    fromTarget: THREE.Vector3;
    toTarget: THREE.Vector3;
  }>({
    active: false,
    elapsed: 0,
    fromPos: new THREE.Vector3(),
    toPos: new THREE.Vector3(),
    fromTarget: new THREE.Vector3(),
    toTarget: new THREE.Vector3(),
  });

  useEffect(() => {
    if (!city) return;

    const anim = animRef.current;
    anim.fromPos.copy(camera.position);
    anim.fromTarget.copy(controls?.target ?? DEFAULT_TARGET);

    if (selectedNodeId === null) {
      anim.toPos.copy(DEFAULT_CAMERA_POS);
      anim.toTarget.copy(DEFAULT_TARGET);
    } else {
      const node = city.nodes.find((n) => n.id === selectedNodeId);
      const pos = nodePositions.get(selectedNodeId);

      // The worker may not have computed this node's position yet (e.g.
      // selection happened via a fast double-click right after a new city
      // loaded, before layout finished). Rather than fly to a wrong/fake
      // position, skip the animation this cycle — the selection itself
      // (FileDrawer, RiskList highlight) still works correctly even if
      // the camera doesn't move.
      if (!node || !pos) return;

      const target = new THREE.Vector3(pos.x, 1.5, pos.z);

      anim.toTarget.copy(target);
      anim.toPos.copy(
        target
          .clone()
          .add(new THREE.Vector3(FOCUS_DISTANCE, FOCUS_HEIGHT_OFFSET, FOCUS_DISTANCE))
      );
    }

    anim.elapsed = 0;
    anim.active = true;
  }, [selectedNodeId, city, camera, controls, nodePositions]);

  useFrame((_, delta) => {
    const anim = animRef.current;
    if (!anim.active) return;

    anim.elapsed += delta;
    const t = Math.min(anim.elapsed / FLY_DURATION, 1);
    const eased = easeInOutCubic(t);

    camera.position.lerpVectors(anim.fromPos, anim.toPos, eased);

    if (controls) {
      controls.target.lerpVectors(anim.fromTarget, anim.toTarget, eased);
      controls.update();
    }

    if (t >= 1) {
      anim.active = false;
    }
  });

  return null;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}