/**
 * The one hook every component uses to get city data. Decides mock vs.
 * real backend in exactly one place, so swapping to the live API later
 * (Phase 5) means changing this file only — every component downstream
 * (CityCanvas, FileDrawer, RiskList) stays untouched.
 *
 * Also owns pushing the result into the Zustand store, so components can
 * either call this hook directly (to trigger a load) or read
 * useCityStore() directly (to just consume already-loaded data) without
 * needing to know which one "owns" the fetch.
 */

"use client";

import { useEffect, useState } from "react";
import { useCityStore } from "@/store/cityStore";
import { generateMockCity } from "@/lib/mockCity";

// Flips to false once backend/app/api/v1/city.py exists and
// NEXT_PUBLIC_API_URL points at a real server. Centralized here, not
// scattered as ad hoc checks, so turning mock mode off is a one-line change.
const USE_MOCK_DATA = true;

interface UseCityDataResult {
  isLoading: boolean;
  error: string | null;
}

/**
 * Loads a city by repo slug (or the demo mock, if `repoSlug` is omitted)
 * and pushes it into the global store. Call this once per page that needs
 * city data (e.g. the repo detail page) — components further down the
 * tree should read `useCityStore()` directly rather than calling this
 * hook again, to avoid redundant fetches.
 */
export function useCityData(repoSlug?: string): UseCityDataResult {
  const setCity = useCityStore((s) => s.setCity);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);

      try {
        if (USE_MOCK_DATA || !repoSlug) {
          // Simulate realistic network latency so loading states
          // (skeletons, the pipeline animation) are actually visible and
          // testable during development, instead of resolving instantly.
          await new Promise((resolve) => setTimeout(resolve, 400));
          const city = generateMockCity();
          if (!cancelled) setCity(city);
        } else {
          const res = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/api/v1/repos/${repoSlug}/city`
          );
          if (!res.ok) {
            throw new Error(`Failed to load city (${res.status})`);
          }
          const city = await res.json();
          if (!cancelled) setCity(city);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load city");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [repoSlug, setCity]);

  return { isLoading, error };
}