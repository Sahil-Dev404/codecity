/**
 * The real city page, at /repo/[id]. Everything built across Phase 3 —
 * CityCanvas, Buildings, Districts, DependencyArcs, CameraRig, RiskList,
 * FileDrawer, PipelineAnimation, TimeLapseSlider — comes together here as
 * a permanent route.
 */

"use client";

import { use } from "react";
import { CityCanvas } from "@/components/city/CityCanvas";
import { Districts } from "@/components/city/Districts";
import { Buildings } from "@/components/city/Buildings";
import { DependencyArcs } from "@/components/city/DependencyArcs";
import { CameraRig } from "@/components/city/CameraRig";
import { TimeLapseSlider } from "@/components/city/TimeLapse";
import { RiskList } from "@/components/panels/RiskList";
import { FileDrawer } from "@/components/panels/FileDrawer";
import { PipelineAnimation } from "@/components/pipeline/PipelineAnimation";
import { useCityData } from "@/hooks/useCityData";
import { useCityStore } from "@/store/cityStore";

export default function RepoPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const { isLoading, error } = useCityData(id);
  const city = useCityStore((s) => s.city);

  if (error) {
    return (
      <main className="min-h-screen flex items-center justify-center px-6">
        <div className="text-center">
          <p className="text-risk-high text-sm font-medium">Couldn't load this city</p>
          <p className="mt-2 text-foreground-subtle text-sm">{error}</p>
        </div>
      </main>
    );
  }

  if (isLoading || !city) {
    return (
      <main className="min-h-screen flex items-center justify-center px-6">
        <PipelineAnimation />
      </main>
    );
  }

  return (
    <main className="h-screen w-screen relative overflow-hidden">
      <CityCanvas>
        <Districts />
        <Buildings />
        <DependencyArcs />
        <CameraRig />
      </CityCanvas>

      <RiskList />
      <FileDrawer />
      <TimeLapseSlider />

      <header className="absolute top-4 left-1/2 -translate-x-1/2 glass-panel rounded-lg px-4 py-2">
        <p className="font-mono text-xs text-foreground-muted">{city.repoSlug}</p>
      </header>
    </main>
  );
}