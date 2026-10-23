// Checks the claims of the newer parts of lesson 2: epoch player, update order, step size, two hidden layers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { backprop, initNet, STARTERS, trainMlp } from '../src/lib/mlp.ts';
import {
  deepAccuracy, deepBackprop, deepCurve, deepEpoch, deepForward, deepHistory, deepLoss, flatDeep, initDeep, mapDeep, toDeep,
  type DeepNet,
} from '../src/lib/deep.ts';
import { basinOf, descend, gradient, surface, GLOBAL, LOCAL } from '../src/lib/landscape.ts';
import { buildCachedMlp, buildCachedMlp2, MLP2_CONFIG, MLP_CONFIG } from '../src/lib/model.ts';

const V = STARTERS.find((s) => s.id === 'venusaur')!;
const maxDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

test('generic network gives the same gradient as the one-hidden-layer code', () => {
  const net = initNet(2);
  for (const s of STARTERS) {
    const g1 = backprop(net, s).grad;
    const g2 = deepBackprop(toDeep(net), s).grad;
    assert.ok(maxDiff(flatDeep(toDeep(g1)), flatDeep(g2)) < 1e-12, s.id);
  }
});

test('generic training reproduces the cached one-hidden-layer run', () => {
  const m = buildCachedMlp();
  const h = deepHistory(toDeep(m.init), STARTERS, m.config.eta, m.config.epochs);
  assert.ok(maxDiff(flatDeep(h[m.config.epochs]), flatDeep(toDeep(m.final))) < 1e-9);
  const old = trainMlp(STARTERS, initNet(MLP_CONFIG.seed), MLP_CONFIG.eta, 50, [], 1);
  const dc = deepCurve(toDeep(initNet(MLP_CONFIG.seed)), STARTERS, MLP_CONFIG.eta, 50, 1);
  dc.curve.forEach((c, i) => assert.ok(Math.abs(c.loss - old.curve[i].loss) < 1e-12));
});

function numericGrad(m: DeepNet, h = 1e-6): number[] {
  const base = flatDeep(m);
  const unflat = (v: number[]): DeepNet => {
    let i = 0;
    return { layers: m.layers.map((layer) => layer.map((n) => ({ w: n.w.map(() => v[i++]), theta: v[i++] }))) };
  };
  return base.map((_, k) => {
    const up = [...base]; up[k] += h;
    const dn = [...base]; dn[k] -= h;
    const l = (v: number[]) => deepLoss(deepForward(unflat(v), V.x1, V.x2), 'grass');
    return (l(up) - l(dn)) / (2 * h);
  });
}

test('two hidden layers: backpropagation equals the numerical gradient for all 21 weights', () => {
  const m = initDeep(MLP2_CONFIG.seed, MLP2_CONFIG.sizes);
  const g = flatDeep(deepBackprop(m, V).grad);
  assert.equal(g.length, 21);
  assert.ok(maxDiff(g, numericGrad(m)) < 1e-6);
});

test('an epoch computes every δ with the old weights, then updates all weights at once', () => {
  const m = toDeep(initNet(2));
  const st = deepEpoch(m, STARTERS, 2, 1);
  for (const b of st.samples) assert.deepEqual(b.forward, deepForward(m, b.forward.a[0][0] * 100, b.forward.a[0][1] * 100));
  const expected = mapDeep(m, st.grad, (w, g) => w - 2 * g);
  assert.ok(maxDiff(flatDeep(st.after), flatDeep(expected)) < 1e-15);
});

test('updating the output layer first would change the hidden δ (so it must use the old weights)', () => {
  const m = toDeep(initNet(2));
  const b = deepBackprop(m, V);
  const outFirst: DeepNet = { layers: [m.layers[0], m.layers[1].map((n, k) => ({ w: n.w.map((w, j) => w - 2 * b.grad.layers[1][k].w[j]), theta: n.theta }))] };
  const wrong = deepBackprop(outFirst, V).delta[0];
  assert.ok(maxDiff(wrong, b.delta[0]) > 1e-3);
});

test('two-hidden-layer network learns all nine; its initial version calls Venusaur Water', () => {
  const m = buildCachedMlp2();
  assert.equal(deepAccuracy(m.final, STARTERS), 1);
  const p = deepForward(m.init, 80, 83).p;
  assert.equal(p.indexOf(Math.max(...p)), 1);
});

test('two hidden layers: the error signal shrinks from the output back to the first layer', () => {
  const b = deepBackprop(initDeep(MLP2_CONFIG.seed, MLP2_CONFIG.sizes), V);
  const mag = b.delta.map((d) => Math.max(...d.map(Math.abs)));
  assert.ok(mag[0] < mag[1] && mag[1] < mag[2]);
});

test('MLP with η = 10 gets stuck: hidden sigmoids saturate, the error stops reaching them, one class for all', () => {
  const r = deepCurve(toDeep(initNet(2)), STARTERS, 10, 1500, 1500);
  assert.ok(Math.abs(deepAccuracy(r.final, STARTERS) - 1 / 3) < 1e-9);
  const st = deepEpoch(r.final, STARTERS, 10, 0);
  for (const b of st.samples) for (const h of b.forward.a[1]) assert.ok(h < 0.01);
  const hiddenGrad = Math.sqrt(st.grad.layers[0].flatMap((n) => [...n.w, n.theta]).reduce((acc, g) => acc + g * g, 0));
  assert.ok(hiddenGrad < 1e-3);
});

test('MLP with η = 0.2 is slower than η = 2', () => {
  const slow = deepCurve(toDeep(initNet(2)), STARTERS, 0.2, 1500, 1);
  const fast = deepCurve(toDeep(initNet(2)), STARTERS, 2, 1500, 1);
  const first = (c: { accuracy: number; epoch: number }[]) => c.find((q) => q.accuracy === 1)!.epoch;
  assert.ok(first(slow.curve) > first(fast.curve));
  assert.ok(slow.curve[slow.curve.length - 1].loss > fast.curve[fast.curve.length - 1].loss);
});

test('error surface: exact gradient matches the numerical one; global is deeper than local', () => {
  const h = 1e-6;
  for (const [a, b] of [[0.3, -0.7], [-2.3, -2], [1, 1]]) {
    const [da, db] = gradient(a, b);
    assert.ok(Math.abs(da - (surface(a + h, b) - surface(a - h, b)) / (2 * h)) < 1e-7);
    assert.ok(Math.abs(db - (surface(a, b + h) - surface(a, b - h)) / (2 * h)) < 1e-7);
  }
  assert.ok(surface(GLOBAL.a, GLOBAL.b) < surface(LOCAL.a, LOCAL.b));
});

test('from (−2.3, −2.0): η = 0.1 stops in the local valley, η = 1.5 reaches the global one, η = 0.7 never settles', () => {
  assert.equal(basinOf(descend(-2.3, -2, 0.1, 300).at(-1)!), 'local');
  assert.equal(basinOf(descend(-2.3, -2, 1.5, 300).at(-1)!), 'global');
  const p = descend(-2.3, -2, 0.7, 300);
  const [x, y] = p.slice(-2);
  assert.ok(Math.hypot(x.a - y.a, x.b - y.b) > 0.1);
});
