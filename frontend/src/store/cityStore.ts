/**
 * Global city state, read by 3D components (Buildings, CameraRig) and UI
 * panels (FileDrawer, RiskList) alike. Centralizing this in one Zustand
 * store — instead of prop-drilling CityData through CityCanvas down into
 * every mesh, or duplicating "which node is selected" in multiple
 * components' local state — is what lets a RiskList item and a 3D building
 * both react to the same click, in sync, with no coordination code.
 */

import { create } from "zustand";
import type { CityData, CityNode, RiskExplanation } from "@/types/city";

interface CityState {
  city: CityData | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  explanation: RiskExplanation | null;
  isLoadingExplanation: boolean;
  // Playback state for the time-lapse feature (TimeLapse.tsx, later file) —
  // lives here rather than component-local state, since both the slider
  // UI and the 3D growth animation need to read/drive the same value.
  timelapseProgress: number; // 0..1

  setCity: (city: CityData) => void;
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  setExplanation: (explanation: RiskExplanation | null) => void;
  setLoadingExplanation: (loading: boolean) => void;
  setTimelapseProgress: (progress: number) => void;

  // Derived lookup, not stored state — avoids the selected node ever
  // going stale relative to the main `city.nodes` array.
  getSelectedNode: () => CityNode | null;
}

export const useCityStore = create<CityState>((set, get) => ({
  city: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  explanation: null,
  isLoadingExplanation: false,
  timelapseProgress: 1, // defaults to "fully grown" (present day)

  setCity: (city) =>
    set({ city, selectedNodeId: null, explanation: null, timelapseProgress: 1 }),

  selectNode: (nodeId) => set({ selectedNodeId: nodeId, explanation: null }),

  hoverNode: (nodeId) => set({ hoveredNodeId: nodeId }),

  setExplanation: (explanation) => set({ explanation, isLoadingExplanation: false }),

  setLoadingExplanation: (loading) => set({ isLoadingExplanation: loading }),

  setTimelapseProgress: (progress) => set({ timelapseProgress: progress }),

  getSelectedNode: () => {
    const { city, selectedNodeId } = get();
    if (!city || !selectedNodeId) return null;
    return city.nodes.find((n) => n.id === selectedNodeId) ?? null;
  },
}));