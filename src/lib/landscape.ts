/**
 * Toy error surface for the step-size demo: two "valleys" in the plane of two weights (a, b).
 * The deep one (global minimum) and a shallow one (local minimum), plus a gentle bowl so
 * gradient descent cannot escape to infinity. Gradient descent: (a, b) ← (a, b) − η · ∇L(a, b).
 */
export interface Basin { a: number; b: number; depth: number; width: number }

export const GLOBAL: Basin = { a: 1.2, b: 1.0, depth: 1.0, width: 1.6 };
export const LOCAL: Basin = { a: -1.5, b: -1.1, depth: 0.55, width: 0.3 };
export const BOWL = 0.05;
export const DOMAIN: [number, number] = [-3, 3];

const g = (x: Basin, a: number, b: number) => x.depth * Math.exp(-((a - x.a) ** 2 + (b - x.b) ** 2) / x.width);

export function surface(a: number, b: number): number {
  return 1 - g(GLOBAL, a, b) - g(LOCAL, a, b) + BOWL * (a * a + b * b);
}

/** Exact gradient (∂L/∂a, ∂L/∂b). */
export function gradient(a: number, b: number): [number, number] {
  let da = 2 * BOWL * a;
  let db = 2 * BOWL * b;
  for (const x of [GLOBAL, LOCAL]) {
    const k = (2 * g(x, a, b)) / x.width;
    da += k * (a - x.a);
    db += k * (b - x.b);
  }
  return [da, db];
}

export interface DescentPoint { a: number; b: number; loss: number }

export function descend(a0: number, b0: number, eta: number, steps: number): DescentPoint[] {
  const path: DescentPoint[] = [{ a: a0, b: b0, loss: surface(a0, b0) }];
  let a = a0;
  let b = b0;
  for (let i = 0; i < steps; i++) {
    const [da, db] = gradient(a, b);
    a -= eta * da;
    b -= eta * db;
    if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) > 1e6) break;
    path.push({ a, b, loss: surface(a, b) });
  }
  return path;
}

/** Which valley a point ended in (by distance), or null if far from both. */
export function basinOf(p: DescentPoint, tol = 0.35): 'global' | 'local' | null {
  const d = (x: Basin) => Math.hypot(p.a - x.a, p.b - x.b);
  if (d(GLOBAL) < tol) return 'global';
  if (d(LOCAL) < tol) return 'local';
  return null;
}
