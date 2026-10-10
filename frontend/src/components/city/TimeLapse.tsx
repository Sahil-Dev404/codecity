/**
 * History slider UI. Writes timelapseProgress into the store; Buildings.tsx
 * reads it and scales building heights, so the city "grows" as the slider
 * moves.
 *
 * Mock-mode limitation: a real time-lapse would use per-snapshot RepoGraphs
 * (graph/builder.py), where each file appears when it was created and has
 * its complexity as of that date. The API doesn't expose snapshots yet, so
 * this approximates the effect by scaling ALL current building heights
 * uniformly by progress.
 */

"use client";

import { useEffect, useState } from "react";
import { Play, Pause } from "lucide-react";
import { useCityStore } from "@/store/cityStore";

const PLAYBACK_SPEED = 0.15; // progress units per second while playing

export function TimeLapseSlider() {
  const city = useCityStore((s) => s.city);
  const timelapseProgress = useCityStore((s) => s.timelapseProgress);
  const setTimelapseProgress = useCityStore((s) => s.setTimelapseProgress);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!isPlaying) return;

    let rafId = 0;
    let lastTime = performance.now();

    function tick(now: number) {
      const delta = (now - lastTime) / 1000;
      lastTime = now;

      // Read the CURRENT progress from the store each frame. Capturing
      // `timelapseProgress` from the render closure would freeze it at the
      // value it had when playback started, and the animation would never
      // accumulate.
      const current = useCityStore.getState().timelapseProgress;
      const next = Math.min(1, current + delta * PLAYBACK_SPEED);
      setTimelapseProgress(next);

      if (next >= 1) {
        setIsPlaying(false);
        return;
      }
      rafId = requestAnimationFrame(tick);
    }

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, setTimelapseProgress]);

  if (!city) return null;

  const approxLabel = formatApproxDate(timelapseProgress, new Date(city.snapshotAt));

  function togglePlay() {
    if (!isPlaying && timelapseProgress >= 1) {
      setTimelapseProgress(0); // replay from the start after finishing
    }
    setIsPlaying((p) => !p);
  }

  return (
    <div className="glass-panel absolute bottom-4 left-1/2 -translate-x-1/2 rounded-lg px-4 py-3 flex items-center gap-4 w-[420px]">
      <button
        onClick={togglePlay}
        className="shrink-0 w-8 h-8 rounded-full bg-accent hover:bg-accent-muted text-accent-foreground flex items-center justify-center transition-colors"
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
      </button>

      <div className="flex-1 flex flex-col gap-1.5">
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={timelapseProgress}
          onChange={(e) => {
            setIsPlaying(false);
            setTimelapseProgress(Number(e.target.value));
          }}
          className="w-full h-1 rounded-full appearance-none bg-surface-overlay accent-accent cursor-pointer"
        />
        <div className="flex items-center justify-between text-xs text-foreground-subtle font-mono">
          <span>{approxLabel}</span>
          <span>{Math.round(timelapseProgress * 100)}%</span>
        </div>
      </div>
    </div>
  );
}

function formatApproxDate(progress: number, snapshotDate: Date): string {
  // Maps the slider onto "3 years ago" -> "today", since mock mode has no
  // real per-file history dates to interpolate between.
  const yearsBack = 3 * (1 - progress);
  if (yearsBack < 0.05) return "Today";
  const approx = new Date(snapshotDate);
  approx.setFullYear(approx.getFullYear() - Math.floor(yearsBack));
  return approx.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}