// Checks every formula and number of lesson 3 (backpropagation and the optimization landscape).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  countWeights, EXAMPLE_DATA, EXAMPLE_W, linearGradient, linearLoss, linearPass, matT, matVec, numericGradient, threeJ,
  type Mat,
} from '../src/lib/linear.ts';
import { descend, eig2, linearGrad, linearHessian, linearJ } from '../src/lib/landscape.ts';

const maxAbs = (A: Mat[], B: Mat[]) => Math.max(...A.flatMap((M, l) => M.flatMap((r, i) => r.map((v, j) => Math.abs(v - B[l][i][j])))));

test('eq. (7): backpropagation gradient equals finite differences for the 3-layer linear example', () => {
  const bp = linearGradient(EXAMPLE_W, EXAMPLE_DATA);
  const fd = numericGradient(EXAMPLE_W, EXAMPLE_DATA);
  assert.ok(maxAbs(bp, fd.grad) < 1e-7);
  assert.equal(countWeights(EXAMPLE_W), 10);
  assert.equal(fd.evaluations, 20);
});

test('eq. (8): closed form δ_ℓ = W_{ℓ+1} ⋯ W_L · 2r for a linear network', () => {
  const s = EXAMPLE_DATA[3];
  const p = linearPass(EXAMPLE_W, s);
  const twoR = p.r.map((v) => 2 * v);
  // δ_1 = W_2 W_3 (2r)
  const d1 = matVec(EXAMPLE_W[1], matVec(EXAMPLE_W[2], twoR));
  d1.forEach((v, i) => assert.ok(Math.abs(v - p.delta[0][i]) < 1e-12));
  // h_2 = W_2ᵀ W_1ᵀ x
  const h2 = matVec(matT(EXAMPLE_W[1]), matVec(matT(EXAMPLE_W[0]), s.x));
  h2.forEach((v, i) => assert.ok(Math.abs(v - p.h[2][i]) < 1e-12));
});

test('one layer: ∇_W J = 2 Σ x_i r_iᵀ (eq. 2)', () => {
  const W = [EXAMPLE_W[2]];
  const data = EXAMPLE_DATA.map((s) => ({ x: [s.x[0] / 2, s.x[1]], y: s.y }));
  const g = linearGradient(W, data)[0];
  const manual = W[0].map((row, a) => row.map((_, b) => data.reduce((acc, s) => {
    const r = matVec(matT(W[0]), s.x)[b] - s.y[b];
    return acc + 2 * s.x[a] * r;
  }, 0)));
  g.forEach((row, a) => row.forEach((v, b) => assert.ok(Math.abs(v - manual[a][b]) < 1e-12)));
});

test('worked example: w1 = w2 = 0.5, x = y = 1 → J = 0.5625, both partial derivatives −0.75, then 0.575', () => {
  assert.equal(linearJ(0.5, 0.5), 0.5625);
  assert.deepEqual(linearGrad(0.5, 0.5), [-0.75, -0.75]);
  const p = descend(0.5, 0.5, 0.1, 1, linearJ, linearGrad);
  assert.ok(Math.abs(p[1].a - 0.575) < 1e-12 && Math.abs(p[1].b - 0.575) < 1e-12);
  assert.ok(Math.abs(p[1].loss - (0.575 * 0.575 - 1) ** 2) < 1e-15);
});

test('not convex: J(1,1) = J(−1,−1) = 0 but J(0,0) = 1 at the midpoint', () => {
  assert.equal(linearJ(1, 1), 0);
  assert.equal(linearJ(-1, -1), 0);
  assert.equal(linearJ(0, 0), 1);
});

test('critical points: saddle at 0 (eigenvalues −2, 2); minima on w1·w2 = 1 with a flat direction', () => {
  assert.ok(linearGrad(0, 0).every((g) => g === 0));
  assert.deepEqual(eig2(linearHessian(0, 0)), [-2, 2]);
  const [lo, hi] = eig2(linearHessian(2, 0.5));
  assert.ok(Math.abs(lo) < 1e-12 && Math.abs(hi - 2 * (4 + 0.25)) < 1e-12);
  for (const [a, b] of [[2, 0.5], [-1, -1], [0.8, 1.25]]) {
    const g = linearGrad(a, b);
    assert.ok(Math.abs(g[0]) < 1e-12 && Math.abs(g[1]) < 1e-12);
  }
});

test('gradient descent (η = 0.1): stuck at the saddle from 0, escapes from 0.001, every other run reaches J = 0', () => {
  const at = (s: [number, number]) => descend(s[0], s[1], 0.1, 150, linearJ, linearGrad);
  assert.equal(at([0, 0]).at(-1)!.loss, 1);
  const slow = at([0.001, 0.001]);
  const fast = at([0.05, 0.02]);
  const first = (p: { loss: number }[]) => p.findIndex((q) => q.loss < 1e-3);
  assert.ok(first(slow) > first(fast));
  for (const s of [[0.001, 0.001], [-1.4, 1.6], [-1.5, -0.2], [1.8, -0.3]] as [number, number][]) {
    const e = at(s).at(-1)!;
    assert.ok(e.loss < 1e-10 && Math.abs(e.a * e.b - 1) < 1e-5);
  }
});

test('three scalar layers: at 0 gradient and Hessian vanish ("bad" saddle), yet J goes below and above 1', () => {
  const h = 1e-4;
  const H: number[][] = [0, 1, 2].map((i) => [0, 1, 2].map((j) => {
    const e = (k: number, d: number) => [0, 1, 2].map((q) => (q === k ? d : 0));
    const f = (p: number[]) => threeJ(p[0], p[1], p[2]);
    const pp = e(i, h).map((v, q) => v + e(j, h)[q]);
    const pm = e(i, h).map((v, q) => v - e(j, h)[q]);
    const mp = e(i, -h).map((v, q) => v + e(j, h)[q]);
    const mm = e(i, -h).map((v, q) => v - e(j, h)[q]);
    return (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h);
  }));
  for (const row of H) for (const v of row) assert.ok(Math.abs(v) < 1e-6);
  assert.ok(threeJ(0.1, 0.1, 0.1) < 1 && threeJ(-0.1, -0.1, -0.1) > 1);
});

test('cost: finite differences need 2 evaluations per weight; for 15 and 21 weights that is 30 and 42', () => {
  assert.equal(numericGradient(EXAMPLE_W, EXAMPLE_DATA).evaluations, 2 * countWeights(EXAMPLE_W));
  assert.ok(linearLoss(EXAMPLE_W, EXAMPLE_DATA) > 0);
});
