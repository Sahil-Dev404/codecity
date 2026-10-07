/**
 * Runs layout.worker.ts against the current city and returns computed
 * positions, keeping the worker's lifecycle (create, post, terminate)
 * contained in one hook so components never touch the Worker API directly.
 */

"use client";

import { useEffect, useRef, useState } from "react";
import { useCityStore } from "@/store/cityStore";

interface LayoutResult {
  nodePositions: Map<string, { x: number; z: number; width: number; depth: number }>;
  districtRects: Map<string, { x: number; z: number; width: number; depth: number }>;
  isComputing: boolean;
}

const CITY_SIZE = 120; // world units — must match CityCanvas's ground plane scale

export function useCityLayout(): LayoutResult {
  const city = useCityStore((s) => s.city);
  const [nodePositions, setNodePositions] = useState<LayoutResult["nodePositions"]>(new Map());
  const [districtRects, setDistrictRects] = useState<LayoutResult["districtRects"]>(new Map());
  const [isComputing, setIsComputing] = useState(false);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    workerRef.current = new Worker(new URL("../workers/layout.worker.ts", import.meta.url));

    workerRef.current.onmessage = (e: MessageEvent) => {
      const positions = new Map(
        e.data.nodes.map((n: any) => [n.id, { x: n.x, z: n.z, width: n.width, depth: n.depth }])
      );
      const districts = new Map(
        e.data.districts.map((d: any) => [d.id, { x: d.x, z: d.z, width: d.width, depth: d.depth }])
      );
      setNodePositions(positions);
      setDistrictRects(districts);
      setIsComputing(false);
    };

    return () => workerRef.current?.terminate();
  }, []);

  useEffect(() => {
    if (!city || !workerRef.current) return;

    setIsComputing(true);
    workerRef.current.postMessage({
      nodes: city.nodes.map((n) => {
        const district = n.id.split("/").slice(0, -1).join("/");
        return { id: n.id, districtId: district, weight: Math.max(1, n.linesOfCode) };
      }),
      citySize: CITY_SIZE,
    });
  }, [city]);

  return { nodePositions, districtRects, isComputing };
}