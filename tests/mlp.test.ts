// Checks every claim of lesson 2 (Learn with me → MLP).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  HAND_NET, STARTERS, TYPES, accuracy, backprop, fireWaterHull, insideConvex, initNet, loss, meanGrad,
  sgdStep, softForward, softmax, stepForward, type Net, type Poke,
} from '../src/lib/mlp.ts';
import { classify, train, type Sample } from '../src/lib/perceptron.ts';
import { buildCachedMlp } from '../src/lib/model.ts';

const byId = (id: string) => STARTERS.find((s) => s.id === id)!;

test('every Fire starter has Speed − Defense = 22, every Water one −22, Grass between −4 and −3', () => {
  for (const s of STARTERS) {
    const d = s.x1 - s.x2;
    if (s.type === 'fire') assert.equal(d, 22);
    if (s.type === 'water') assert.equal(d, -22);
    if (s.type === 'grass') assert.ok(d >= -4 && d <= -3);
  }
});

test('lesson 1 weights put Venusaur on the Water side', () => {
  assert.equal(classify({ w1: 22, w2: -22, theta: 0.5 }, 80, 83).y, 0);
});

test('Ivysaur and Venusaur are inside the region enclosed by Fire and Water; Bulbasaur is not', () => {
  const hull = fireWaterHull();
  assert.equal(hull.length, 4);
  assert.ok(insideConvex(hull, [60, 63]));
  assert.ok(insideConvex(hull, [80, 83]));
  assert.ok(!insideConvex(hull, [45, 49]));
});

test('one perceptron for "Grass or not" never reaches zero errors (not linearly separable)', () => {
  const samples: Sample[] = STARTERS.map((s) => ({ id: s.id, x1: s.x1, x2: s.x2, d: s.type === 'grass' ? 1 : 0 }));
  const r = train(samples, { w1: 0, w2: 0, theta: 0.5 }, 1, 300);
  assert.equal(r.converged, false);
  assert.ok(r.epochs.every((e) => e.errors > 0));
});

test('hand-made MLP: Venusaur gives h = (0, 0) and Grass, with the numbers of the lesson', () => {
  const f = stepForward(HAND_NET, 80, 83);
  assert.deepEqual(f.hiddenNet, [-14, -8]);
  assert.deepEqual(f.h, [0, 0]);
  assert.deepEqual(f.outputNet, [-0.5, -0.5, 0.5]);
  assert.equal(f.type, 'grass');
});

test('hand-made MLP: Charizard → h = (1, 0) → Fire; Blastoise → h = (0, 1) → Water', () => {
  const c = stepForward(HAND_NET, 100, 78);
  assert.deepEqual(c.hiddenNet, [11, -33]);
  assert.deepEqual(c.outputNet, [0.5, -0.5, -0.5]);
  assert.equal(c.type, 'fire');
  const b = stepForward(HAND_NET, 78, 100);
  assert.deepEqual(b.hiddenNet, [-33, 11]);
  assert.equal(b.type, 'water');
});

test('hand-made MLP classifies all nine starters', () => {
  for (const s of STARTERS) assert.equal(stepForward(HAND_NET, s.x1, s.x2).type, s.type, s.id);
});

test('softmax sums to 1', () => {
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

/** Numerical gradient (central differences) of the loss of one sample. */
function numericGrad(net: Net, s: Poke, h = 1e-6): number[] {
  const flat = (n: Net) => [...n.hidden, ...n.output].flatMap((u) => [...u.w, u.theta]);
  const unflat = (v: number[]): Net => {
    let i = 0;
    const take = (u: { w: number[] }) => ({ w: u.w.map(() => v[i++]), theta: v[i++] });
    return { hidden: net.hidden.map(take), output: net.output.map(take) };
  };
  const base = flat(net);
  return base.map((_, k) => {
    const up = [...base]; up[k] += h;
    const dn = [...base]; dn[k] -= h;
    const l = (v: number[]) => loss(softForward(unflat(v), s.x1, s.x2), s.type);
    return (l(up) - l(dn)) / (2 * h);
  });
}

test('backpropagation gradient equals the numerical gradient for every weight', () => {
  const net = initNet(2);
  for (const s of STARTERS) {
    const g = backprop(net, s).grad;
    const analytic = [...g.hidden, ...g.output].flatMap((u) => [...u.w, u.theta]);
    numericGrad(net, s).forEach((v, k) => assert.ok(Math.abs(v - analytic[k]) < 1e-6, `${s.id} weight ${k}`));
  }
});

test('one gradient step on Venusaur lowers its loss', () => {
  const net = initNet(2);
  const v = byId('venusaur');
  const b = backprop(net, v);
  const after = sgdStep(net, b.grad, 2);
  assert.ok(loss(softForward(after, v.x1, v.x2), 'grass') < b.loss);
});

test('initial network of the lesson calls Venusaur Water', () => {
  const p = softForward(initNet(2), 80, 83).p;
  assert.equal(TYPES[p.indexOf(Math.max(...p))], 'water');
});

test('trained MLP classifies all nine starters and its loss went down', () => {
  const m = buildCachedMlp();
  assert.equal(accuracy(m.final, STARTERS), 1);
  assert.ok(m.curve[m.curve.length - 1].loss < 0.05);
  assert.ok(m.curve[m.curve.length - 1].loss < m.curve[0].loss);
  assert.equal(meanGrad(m.final, STARTERS).loss, m.curve[m.curve.length - 1].loss);
});

test('trained hidden neurons react to Speed − Defense with opposite signs', () => {
  const { hidden } = buildCachedMlp().final;
  const s = hidden.map((n) => Math.sign(n.w[0] - n.w[1]));
  assert.deepEqual([...s].sort(), [-1, 1]);
  for (const n of hidden) assert.ok(Math.sign(n.w[0]) === -Math.sign(n.w[1]));
});

test('the threshold 11 of the hand-made net can be anything in (4, 22]', () => {
  const withC = (c: number): Net => ({ hidden: [{ w: [1, -1], theta: c }, { w: [-1, 1], theta: c }], output: HAND_NET.output });
  for (const c of [4.01, 11, 22]) for (const s of STARTERS) assert.equal(stepForward(withC(c), s.x1, s.x2).type, s.type, `c=${c} ${s.id}`);
  for (const c of [4, 22.01]) assert.ok(STARTERS.some((s) => stepForward(withC(c), s.x1, s.x2).type !== s.type), `c=${c}`);
});

test('without activations the hand-made Grass output is the same (22.5) for every Pokémon', () => {
  for (const s of STARTERS) {
    const h = HAND_NET.hidden.map((n) => n.w[0] * s.x1 + n.w[1] * s.x2 - n.theta);
    const g = HAND_NET.output[2];
    assert.equal(g.w[0] * h[0] + g.w[1] * h[1] - g.theta, 22.5);
  }
});

test('a single hidden sigmoid neuron (2 → 1 → 3) also learns the nine starters', async () => {
  const { deepCurve, initDeep } = await import('../src/lib/deep.ts');
  const r = deepCurve(initDeep(2, [1, 3]), STARTERS, 2, 600, 1);
  assert.equal(r.curve[r.curve.length - 1].accuracy, 1);
});

test('with the step, one hidden neuron gives only two values, so it cannot output three types', () => {
  // h ∈ {0, 1}: whatever the output weights, at most two different output patterns exist
  const patterns = new Set(STARTERS.map((s) => stepForward(HAND_NET, s.x1, s.x2).h[0]));
  assert.ok(patterns.size <= 2);
});
