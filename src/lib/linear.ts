/**
 * Lesson 3: deep linear networks and the chain rule, with plain matrices.
 *
 *   f(x) = W_Lᵀ ⋯ W_2ᵀ W_1ᵀ x,   W_ℓ ∈ ℝ^{d_{ℓ−1} × d_ℓ}
 *   J(W_1, …, W_L) = Σ_i ‖f(x_i) − y_i‖²
 *
 * Backpropagation (eqs. 4–8 of the page):
 *   forward  h_0 = x,  h_ℓ = W_ℓᵀ h_{ℓ−1}            (stored: the "book-keeping")
 *   output   δ_L = 2 (h_L − y)
 *   backward δ_ℓ = W_{ℓ+1} δ_{ℓ+1}
 *   gradient ∇_{W_ℓ} J = Σ_i h_{ℓ−1,i} δ_{ℓ,i}ᵀ
 */
export type Mat = number[][];
export type Vec = number[];

export const matT = (A: Mat): Mat => A[0].map((_, j) => A.map((row) => row[j]));
export const matVec = (A: Mat, v: Vec): Vec => A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
export const outer = (u: Vec, v: Vec): Mat => u.map((a) => v.map((b) => a * b));
export const addM = (A: Mat, B: Mat): Mat => A.map((row, i) => row.map((a, j) => a + B[i][j]));
export const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array(c).fill(0));

export interface Sample { x: Vec; y: Vec }

export interface LinearPass {
  /** h[ℓ] for ℓ = 0..L (h[0] = x) */
  h: Vec[];
  r: Vec;
  loss: number;
  /** δ[ℓ − 1] = δ_ℓ for ℓ = 1..L */
  delta: Vec[];
}

export function linearPass(W: Mat[], s: Sample): LinearPass {
  const h: Vec[] = [s.x];
  for (const Wl of W) h.push(matVec(matT(Wl), h[h.length - 1]));
  const out = h[h.length - 1];
  const r = out.map((v, k) => v - s.y[k]);
  const L = W.length;
  const delta: Vec[] = new Array(L);
  delta[L - 1] = r.map((v) => 2 * v);
  for (let l = L - 2; l >= 0; l--) delta[l] = matVec(W[l + 1], delta[l + 1]);
  return { h, r, loss: r.reduce((a, v) => a + v * v, 0), delta };
}

export function linearLoss(W: Mat[], data: Sample[]): number {
  return data.reduce((a, s) => a + linearPass(W, s).loss, 0);
}

/** ∇_{W_ℓ} J for every layer, by backpropagation. */
export function linearGradient(W: Mat[], data: Sample[]): Mat[] {
  let G = W.map((Wl) => zeros(Wl.length, Wl[0].length));
  for (const s of data) {
    const p = linearPass(W, s);
    G = G.map((Gl, l) => addM(Gl, outer(p.h[l], p.delta[l])));
  }
  return G;
}

/** Central finite differences: 2 evaluations of J per weight. */
export function numericGradient(W: Mat[], data: Sample[], eps = 1e-6): { grad: Mat[]; evaluations: number } {
  let evaluations = 0;
  const grad = W.map((Wl, l) => Wl.map((row, a) => row.map((_, b) => {
    const up = W.map((M) => M.map((r) => [...r]));
    const dn = W.map((M) => M.map((r) => [...r]));
    up[l][a][b] += eps;
    dn[l][a][b] -= eps;
    evaluations += 2;
    return (linearLoss(up, data) - linearLoss(dn, data)) / (2 * eps);
  })));
  return { grad, evaluations };
}

export const countWeights = (W: Mat[]) => W.reduce((a, M) => a + M.length * M[0].length, 0);

/** Example used on the page and in the Python file: L = 3, widths 2 → 2 → 2 → 1, four samples. */
export const EXAMPLE_W: Mat[] = [
  [[0.5, -0.3], [0.2, 0.8]],
  [[0.7, 0.1], [-0.4, 0.6]],
  [[0.9], [-0.5]],
];
export const EXAMPLE_DATA: Sample[] = [
  { x: [1, 0], y: [1] },
  { x: [0, 1], y: [-1] },
  { x: [1, 1], y: [0] },
  { x: [0.5, -1], y: [1.5] },
];

/** J(a, b, c) = (c·b·a − 1)²: three scalar layers. At 0 the gradient and the Hessian are zero, yet it is a saddle. */
export const threeJ = (a: number, b: number, c: number) => (a * b * c - 1) ** 2;
