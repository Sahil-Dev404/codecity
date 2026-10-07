/**
 * The 3D scene root. Sets up the canvas, camera, lighting and ground plane
 * that every city component (Buildings, Districts, DependencyArcs,
 * CameraRig) renders inside as a child. This file knows nothing about
 * CityData — it's pure "stage," not "actors."
 */

"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, AdaptiveDpr, AdaptiveEvents } from "@react-three/drei";
import * as THREE from "three";

interface CityCanvasProps {
  children: React.ReactNode;
}

export function CityCanvas({ children }: CityCanvasProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        shadows
        camera={{ position: [40, 35, 40], fov: 45, near: 0.1, far: 500 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        className="bg-background"
      >
        {/* Fog hides the hard edge where the city meets empty space, and
            gives distant buildings a subtle depth cue — cheap and effective. */}
        <fog attach="fog" args={["#0a0e14", 60, 220]} />

        <Lighting />
        <GroundPlane />

        {/* Suspense because drei's Environment and any later-added GLTF
            assets load asynchronously; this boundary keeps the canvas from
            throwing before those are ready. */}
        <Suspense fallback={null}>{children}</Suspense>

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          minDistance={8}
          maxDistance={160}
          maxPolarAngle={Math.PI / 2.1} // stop just short of going underground
        />

        {/* Adaptive performance: automatically lowers pixel ratio / pauses
            event handling under load, so a large city (hundreds of
            buildings) degrades gracefully instead of freezing the tab. */}
        <AdaptiveDpr pixelated={false} />
        <AdaptiveEvents />
      </Canvas>
    </div>
  );
}

function Lighting() {
  return (
    <>
      {/* Soft ambient fill so shadowed faces aren't pure black */}
      <ambientLight intensity={0.35} color="#8ba3d6" />

      {/* Key light, like a low evening sun — casts the shadows that give
          the city its sense of depth and scale */}
      <directionalLight
        position={[30, 45, 20]}
        intensity={1.4}
        color="#fff4e0"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-60}
        shadow-camera-right={60}
        shadow-camera-top={60}
        shadow-camera-bottom={-60}
        shadow-camera-far={150}
        shadow-bias={-0.0005}
      />

      {/* Cool rim light from the opposite side, reinforcing the
          "night city" mood established by the dark theme tokens */}
      <directionalLight position={[-25, 20, -30]} intensity={0.3} color="#5b8ef4" />
    </>
  );
}

function GroundPlane() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} receiveShadow>
      <planeGeometry args={[400, 400]} />
      <meshStandardMaterial color="#0d1219" roughness={0.9} metalness={0.1} />
    </mesh>
  );
}