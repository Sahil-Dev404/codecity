/**
 * The loading sequence shown while a repo is being analyzed: clone, parse,
 * graph, predict. Replaces the plain "Building city…" placeholder in
 * repo/[id]/page.tsx. Stages advance automatically on a timer in mock
 * mode; once the real backend exists (Phase 5), this will instead receive
 * actual stage/progress updates over SSE (see jobs.py's planned stream)
 * and stop advancing on its own.
 */

"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GitBranch, FileSearch, Share2, Brain, Check } from "lucide-react";

export interface PipelineStage {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const STAGES: PipelineStage[] = [
  { id: "clone", label: "Cloning repository", icon: <GitBranch className="w-4 h-4" /> },
  { id: "parse", label: "Parsing source files", icon: <FileSearch className="w-4 h-4" /> },
  { id: "graph", label: "Building dependency graph", icon: <Share2 className="w-4 h-4" /> },
  { id: "predict", label: "Scoring risk with the GNN", icon: <Brain className="w-4 h-4" /> },
];

// Mock-mode timing only — real durations will vary hugely by repo size
// once wired to the actual backend, which is exactly why this constant
// lives here and nowhere else: one place to delete when mock mode goes.
const MOCK_STAGE_DURATION_MS = 900;

interface PipelineAnimationProps {
  /** When provided, the component becomes controlled — it displays
   * whatever stage index is passed in rather than advancing itself.
   * This is the seam Phase 5's real SSE progress will plug into. */
  currentStageIndex?: number;
}

export function PipelineAnimation({ currentStageIndex }: PipelineAnimationProps) {
  const [internalIndex, setInternalIndex] = useState(0);
  const isControlled = currentStageIndex !== undefined;
  const activeIndex = isControlled ? currentStageIndex : internalIndex;

  useEffect(() => {
    if (isControlled) return; // real progress is driving this; don't self-advance

    if (internalIndex >= STAGES.length - 1) return;
    const timer = setTimeout(() => {
      setInternalIndex((i) => Math.min(i + 1, STAGES.length - 1));
    }, MOCK_STAGE_DURATION_MS);

    return () => clearTimeout(timer);
  }, [internalIndex, isControlled]);

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="relative w-12 h-12">
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-accent/20"
        />
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-t-accent border-r-transparent border-b-transparent border-l-transparent"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
        />
      </div>

      <div className="flex flex-col gap-3 w-64">
        {STAGES.map((stage, i) => {
          const isDone = i < activeIndex;
          const isActive = i === activeIndex;
          const isPending = i > activeIndex;

          return (
            <div key={stage.id} className="flex items-center gap-3">
              <div
                className={`shrink-0 w-6 h-6 rounded-full flex items-center justify-center transition-colors ${
                  isDone
                    ? "bg-accent text-accent-foreground"
                    : isActive
                      ? "bg-accent/20 text-accent"
                      : "bg-surface-overlay text-foreground-subtle"
                }`}
              >
                <AnimatePresence mode="wait">
                  {isDone ? (
                    <motion.div
                      key="check"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Check className="w-3.5 h-3.5" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="icon"
                      initial={{ scale: 0.8, opacity: 0.6 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.2 }}
                    >
                      {stage.icon}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <span
                className={`text-sm transition-colors ${
                  isActive
                    ? "text-foreground font-medium"
                    : isDone
                      ? "text-foreground-muted"
                      : "text-foreground-subtle"
                }`}
              >
                {stage.label}
                {isActive && (
                  <motion.span
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  >
                    …
                  </motion.span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}