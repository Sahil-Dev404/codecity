/**
 * Computes a squarified treemap layout off the main thread. Takes raw
 * node sizes (grouped by district), returns x/z positions and footprint
 * dimensions for each — this is the real implementation Buildings.tsx's
 * computeGridLayout placeholder was always meant to be replaced by.
 *
 * Runs in a Web Worker so laying out a large repo (hundreds of files)
 * never blocks the main thread / freezes the 3D scene or UI while it
 * computes — directly addressing the performance plan from the start of
 * this project.
 */

interface LayoutInputNode {
  id: string;
  districtId: string;
  weight: number; // proportional to footprint area — we use linesOfCode
}

interface LayoutRect {
  id: string;
  x: number;
  z: number;
  width: number;
  depth: number;
}

interface DistrictLayoutRect {
  id: string;
  x: number;
  z: number;
  width: number;
  depth: number;
}

interface LayoutRequest {
  nodes: LayoutInputNode[];
  citySize: number; // total city footprint, in world units (e.g. 120)
}

interface LayoutResponse {
  nodes: LayoutRect[];
  districts: DistrictLayoutRect[];
}

/** Classic squarified treemap (Bruls, Huizing, van Wijk). Lays out
 * `items` (each with a .value) inside the given rectangle, minimizing
 * aspect ratio so cells stay roughly square rather than thin slivers. */
function squarify<T extends { value: number }>(
  items: T[],
  x: number,
  y: number,
  width: number,
  height: number
): Array<T & { rx: number; ry: number; rw: number; rh: number }> {
  const results: Array<T & { rx: number; ry: number; rw: number; rh: number }> = [];
  const sorted = [...items].sort((a, b) => b.value - a.value);
  const total = sorted.reduce((sum, i) => sum + i.value, 0);
  if (total <= 0 || sorted.length === 0) return results;

  const scale = (width * height) / total;
  let remaining = [...sorted];
  let rx = x, ry = y, rw = width, rh = height;

  while (remaining.length > 0) {
    const shortSide = Math.min(rw, rh);
    let row: typeof remaining = [];
    let rowSum = 0;
    let bestWorst = Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = [...row, remaining[i]];
      const candidateSum = rowSum + remaining[i].value * scale;
      const worst = worstAspectRatio(candidate.map((c) => c.value * scale), shortSide, candidateSum);
      if (worst <= bestWorst) {
        row = candidate;
        rowSum = candidateSum;
        bestWorst = worst;
      } else {
        break;
      }
    }

    const rowLength = rowSum / shortSide;
    let offset = 0;
    const horizontal = rw <= rh;

    for (const item of row) {
      const itemArea = item.value * scale;
      const itemLength = itemArea / rowLength;
      if (horizontal) {
        results.push({ ...item, rx: rx + offset, ry, rw: itemLength, rh: rowLength });
      } else {
        results.push({ ...item, rx, ry: ry + offset, rw: rowLength, rh: itemLength });
      }
      offset += itemLength;
    }

    if (horizontal) {
      rx += rowLength;
      rw -= rowLength;
    } else {
      ry += rowLength;
      rh -= rowLength;
    }
    remaining = remaining.slice(row.length);
  }

  return results;
}

function worstAspectRatio(areas: number[], shortSide: number, sum: number): number {
  const max = Math.max(...areas);
  const min = Math.min(...areas);
  return Math.max(
    (shortSide * shortSide * max) / (sum * sum),
    (sum * sum) / (shortSide * shortSide * min)
  );
}

function computeLayout(request: LayoutRequest): LayoutResponse {
  const { nodes, citySize } = request;

  const byDistrict = new Map<string, LayoutInputNode[]>();
  for (const node of nodes) {
    const group = byDistrict.get(node.districtId) ?? [];
    group.push(node);
    byDistrict.set(node.districtId, group);
  }

  const districtWeights = Array.from(byDistrict.entries()).map(([id, group]) => ({
    id,
    value: group.reduce((sum, n) => sum + n.weight, 0),
  }));

  const districtRects = squarify(
    districtWeights,
    -citySize / 2,
    -citySize / 2,
    citySize,
    citySize
  );

  const nodeRects: LayoutRect[] = [];
  const districts: DistrictLayoutRect[] = [];

  for (const dRect of districtRects) {
    // Small inset so districts have a visible gap between them, rather
    // than buildings from adjacent districts touching directly.
    const inset = Math.min(dRect.rw, dRect.rh) * 0.06;
    const innerX = dRect.rx + inset;
    const innerY = dRect.ry + inset;
    const innerW = Math.max(0.1, dRect.rw - inset * 2);
    const innerH = Math.max(0.1, dRect.rh - inset * 2);

    districts.push({ id: dRect.id, x: dRect.rx, z: dRect.ry, width: dRect.rw, depth: dRect.rh });

    const groupNodes = (byDistrict.get(dRect.id) ?? []).map((n) => ({ ...n, value: n.weight }));
    const nodeLayout = squarify(groupNodes, innerX, innerY, innerW, innerH);

    for (const n of nodeLayout) {
      const padding = Math.min(n.rw, n.rh) * 0.1;
      nodeRects.push({
        id: n.id,
        x: n.rx + n.rw / 2,
        z: n.ry + n.rh / 2,
        width: Math.max(0.3, n.rw - padding * 2),
        depth: Math.max(0.3, n.rh - padding * 2),
      });
    }
  }

  return { nodes: nodeRects, districts };
}

self.onmessage = (e: MessageEvent<LayoutRequest>) => {
  const result = computeLayout(e.data);
  self.postMessage(result);
};