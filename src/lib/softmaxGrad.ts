/**
 * Output-layer error with softmax, with and without the cross-entropy "trick".
 *
 *   softmax      p_k = e^{z_k} / Σ_j e^{z_j}
 *   Jacobian     ∂p_k/∂z_i = p_k (1[k=i] − p_i)        J = diag(p) − p pᵀ (symmetric)
 *   cross-entropy L = −Σ_k d_k ln p_k                  ⇒ δ_i = ∂L/∂z_i = p_i − d_i
 *   squared error L = Σ_k (p_k − d_k)²                 ⇒ δ = Jᵀ · 2(p − d)   (needs the whole Jacobian)
 */
import { softmax } from './mlp.ts';

export function softmaxJacobian(p: number[]): number[][] {
  return p.map((pk, k) => p.map((pi, i) => pk * ((k === i ? 1 : 0) - pi)));
}

export const ceLoss = (p: number[], d: number[]) => -d.reduce((a, dk, k) => a + (dk ? dk * Math.log(p[k]) : 0), 0);
export const mseLoss = (p: number[], d: number[]) => p.reduce((a, pk, k) => a + (pk - d[k]) ** 2, 0);

/** δ for cross-entropy + softmax: the simple form. */
export const ceDelta = (p: number[], d: number[]) => p.map((pk, k) => pk - d[k]);

/** δ for cross-entropy written the long way: Σ_k ∂L/∂p_k · ∂p_k/∂z_i, with ∂L/∂p_k = −d_k/p_k. */
export function ceDeltaLong(p: number[], d: number[]): number[] {
  const J = softmaxJacobian(p);
  const g = p.map((pk, k) => -d[k] / pk);
  return p.map((_, i) => g.reduce((a, gk, k) => a + gk * J[k][i], 0));
}

/** δ for squared error + softmax: ∂L/∂p_k = 2(p_k − d_k), then the full Jacobian. */
export function mseDelta(p: number[], d: number[]): { g: number[]; J: number[][]; delta: number[] } {
  const J = softmaxJacobian(p);
  const g = p.map((pk, k) => 2 * (pk - d[k]));
  return { g, J, delta: p.map((_, i) => g.reduce((a, gk, k) => a + gk * J[k][i], 0)) };
}

/** Same product without building the matrix: Jᵀg = p ⊙ (g − (p·g)), O(C) instead of O(C²). */
export function jacobianTimes(p: number[], g: number[]): number[] {
  const pg = p.reduce((a, pk, k) => a + pk * g[k], 0);
  return p.map((pk, k) => pk * (g[k] - pg));
}

/** Logits that give a chosen probability vector (any constant shift gives the same p). */
export const logitsFor = (p: number[]) => p.map(Math.log);

/** Numerical ∂L/∂z for a loss of the softmax output. */
export function numericDelta(z: number[], d: number[], loss: (p: number[], d: number[]) => number, h = 1e-6): number[] {
  return z.map((_, i) => {
    const up = [...z]; up[i] += h;
    const dn = [...z]; dn[i] -= h;
    return (loss(softmax(up), d) - loss(softmax(dn), d)) / (2 * h);
  });
}
