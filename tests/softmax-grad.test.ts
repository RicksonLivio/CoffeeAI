// δ_o = p − d: the cross-entropy + softmax derivation and the "without the trick" comparison.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initNet, softForward, softmax } from '../src/lib/mlp.ts';
import {
  ceDelta, ceDeltaLong, ceLoss, jacobianTimes, logitsFor, mseDelta, mseLoss, numericDelta, softmaxJacobian,
} from '../src/lib/softmaxGrad.ts';

const close = (a: number[], b: number[], tol = 1e-7) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < tol, `${v} vs ${b[i]}`));
const d = [0, 0, 1];
const z = softForward(initNet(2), 80, 83).zo; // Venusaur, initial weights
const p = softmax(z);

test('softmax Jacobian: p_k(1[k=i] − p_i), symmetric, rows add to 0', () => {
  const J = softmaxJacobian(p);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(J[k].reduce((a, b) => a + b, 0)) < 1e-15);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(J[k][i] - J[i][k]) < 1e-15);
    assert.ok(Math.abs(J[k][k] - p[k] * (1 - p[k])) < 1e-15);
  }
});

test('cross-entropy: the long chain rule, the short form p − d and the numerical derivative agree', () => {
  close(ceDeltaLong(p, d), ceDelta(p, d), 1e-12);
  close(numericDelta(z, d, ceLoss), ceDelta(p, d));
});

test('squared error + softmax: δ = Jᵀ·2(p − d), checked numerically, and different from p − d', () => {
  const m = mseDelta(p, d);
  close(numericDelta(z, d, mseLoss), m.delta);
  close(jacobianTimes(p, m.g), m.delta, 1e-12);
  assert.ok(Math.abs(m.delta[2] - ceDelta(p, d)[2]) > 0.1);
});

test('confidently wrong: cross-entropy keeps a large error, squared error almost vanishes', () => {
  const q = [0.98, 0.01, 0.01]; // says Fire with 98%, the answer is Grass
  const ce = ceDelta(q, d);
  const mse = mseDelta(q, d).delta;
  close(numericDelta(logitsFor(q), d, ceLoss), ce);
  assert.ok(Math.abs(ce[2]) > 0.98);
  assert.ok(Math.abs(mse[2]) < 0.05 && Math.abs(ce[2]) / Math.abs(mse[2]) > 20);
});
