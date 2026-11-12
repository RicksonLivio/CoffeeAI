// Checks the claims of lesson 3 (CNN): the block examples, output sizes, backprop and the cached card model.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  BACK_DELTA, BACK_K, BACK_X, CARDNET_SHAPES, CLASSES, CNN_CARDS_PATH, CNN_CONFIG, CNN_MODEL_PATH, EDGE_PATCH, KERNELS, POOL_PATCH,
  backward, colorBaseline, conv2dGrid, countParams, decodePx, evaluate, flatNet, forward, initCardNet, inputGrad, kernelGrad, loss, maxPoolBack,
  outSize, pool2dGrid, roundNet, tensor, toTensor, trainCardNet, type CachedCnn, type CardSample,
} from '../src/lib/cnn.ts';
import { rng } from '../src/lib/mlp.ts';

const read = (p: string) => JSON.parse(readFileSync(new URL(`../public/${p}`, import.meta.url), 'utf8'));
const cards: CardSample[] = read(CNN_CARDS_PATH);
const model: CachedCnn = read(CNN_MODEL_PATH);
const prep = (c: CardSample) => ({ x: toTensor(decodePx(c.px)), y: CLASSES.indexOf(c.type) });

test('vertical-edge kernel on the 6×6 patch: first windows 1 and 23, output 4×4', () => {
  const y = conv2dGrid(EDGE_PATCH, KERNELS.vertical);
  assert.equal(y.length, 4); assert.equal(y[0].length, 4);
  assert.equal(y[0][0], 1);   // (9+9+9) − (9+9+8)
  assert.equal(y[0][1], 23);  // (9+9+9) − (1+1+2)
  assert.deepEqual(y, [[1, 23, 23, 1], [2, 21, 22, 3], [3, 19, 20, 5], [3, 18, 20, 5]]);
  assert.ok(y.every((r) => r[1] > 15 && r[2] > 15 && r[0] < 6 && r[3] < 6), 'the two middle columns (the edge) light up');
});

test('output size ⌊(n + 2p − k)/s⌋ + 1, and the card network shapes', () => {
  assert.equal(outSize(6, 3, 0, 1), 4);
  assert.equal(outSize(6, 3, 1, 1), 6);   // "same" padding
  assert.equal(outSize(6, 3, 0, 2), 2);
  assert.equal(outSize(6, 3, 1, 2), 3);
  assert.equal(conv2dGrid(EDGE_PATCH, KERNELS.vertical, 1, 2).length, 3);
  assert.equal(outSize(32, 5, 0, 1), 28); assert.equal(28 / 2, 14);
  assert.equal(outSize(14, 3, 0, 1), 12); assert.equal(12 / 2, 6);
  assert.equal(6 * 6 * 8, 288);
  assert.deepEqual(CARDNET_SHAPES.map((s) => s.s), [32, 28, 14, 12, 6, 288, 3]);
  const f = forward(initCardNet(1), tensor(3, 32, 32, 0.1));
  assert.deepEqual([f.z1.c, f.z1.h, f.p1.h, f.z2.c, f.z2.h, f.p2.h, f.flat.length, f.p.length], [6, 28, 14, 8, 12, 6, 288, 3]);
});

test('pooling on the 4×4 patch: max, average and min', () => {
  assert.deepEqual(pool2dGrid(POOL_PATCH, 2, 2, 'max'), [[6, 9], [7, 8]]);
  assert.deepEqual(pool2dGrid(POOL_PATCH, 2, 2, 'avg'), [[3.75, 3], [2.5, 4.75]]);
  assert.deepEqual(pool2dGrid(POOL_PATCH, 2, 2, 'min'), [[1, 0], [0, 3]]);
  assert.deepEqual(maxPoolBack(POOL_PATCH, [[1, 2], [3, 4]]), [[0, 0, 0, 2], [0, 1, 0, 0], [3, 0, 0, 0], [0, 0, 4, 0]]);
});

test('parameter count: 456 + 440 + 867 = 1763, against 9219 for a dense layer on the raw pixels', () => {
  assert.equal(6 * (3 * 5 * 5) + 6, 456);
  assert.equal(8 * (6 * 3 * 3) + 8, 440);
  assert.equal(3 * 288 + 3, 867);
  assert.equal(countParams(initCardNet(1)), 1763);
  assert.equal(3 * (32 * 32 * 3) + 3, 9219);
});

test('hand backprop through a 2×2 kernel matches a numerical gradient', () => {
  const y = conv2dGrid(BACK_X, BACK_K);
  assert.deepEqual(y, [[1, 8], [1, 0]]);
  const g = kernelGrad(BACK_X, BACK_DELTA, 2);
  assert.deepEqual(g, [[-1.25, 1.75], [-0.75, -2.25]]);
  // L = Σ δ·y is the loss whose gradient with respect to y is δ
  const L = (K: number[][]) => conv2dGrid(BACK_X, K).reduce((s, r, i) => s + r.reduce((t, v, j) => t + v * BACK_DELTA[i][j], 0), 0);
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
    const Kp = BACK_K.map((r) => [...r]), Km = BACK_K.map((r) => [...r]);
    Kp[a][b] += 1e-6; Km[a][b] -= 1e-6;
    assert.ok(Math.abs((L(Kp) - L(Km)) / 2e-6 - g[a][b]) < 1e-6);
  }
  assert.deepEqual(inputGrad(BACK_K, BACK_DELTA, 3, 3), [[0.5, -1.5, 1], [0, 1.25, -2.25], [0, 0, 0.5]]);
});

test('card network backprop equals finite differences on every layer', () => {
  const net = initCardNet(5), r = rng(9), x = tensor(3, 32, 32);
  x.d = x.d.map(() => r() - 0.5);
  const g = flatNet(backward(net, forward(net, x), 1).grad);
  const groups: number[][] = [net.conv1.w, net.conv1.b, net.conv2.w, net.conv2.b, net.fc.w, net.fc.b];
  let off = 0, worst = 0;
  for (const a of groups) {
    for (const i of [0, Math.floor(a.length / 2), a.length - 1]) {
      const o = a[i];
      a[i] = o + 1e-5; const lp = loss(forward(net, x).p, 1);
      a[i] = o - 1e-5; const lm = loss(forward(net, x).p, 1);
      a[i] = o;
      worst = Math.max(worst, Math.abs((lp - lm) / 2e-5 - g[off + i]));
    }
    off += a.length;
  }
  assert.ok(worst < 1e-7, String(worst));
});

test('dataset: 93 cards, 69 train / 24 test, balanced, lesson cards in training', () => {
  assert.equal(cards.length, 93);
  for (const t of CLASSES) {
    assert.equal(cards.filter((c) => c.type === t && c.split === 'train').length, 23);
    assert.equal(cards.filter((c) => c.type === t && c.split === 'test').length, 8);
  }
  assert.ok(['charizard', 'blastoise', 'venusaur'].every((id) => cards.find((c) => c.id === id)?.split === 'train'));
  assert.equal(new Set(cards.map((c) => c.id)).size, 93);
  assert.ok(cards.every((c) => decodePx(c.px).length === 32 * 32 * 3));
});

test('cached card model: same config, matches a fresh training run, 69/69 train and 23/24 test', () => {
  assert.deepEqual(model.config, CNN_CONFIG);
  assert.equal(model.params, 1763);
  const train = cards.filter((c) => c.split === 'train').map(prep), testSet = cards.filter((c) => c.split === 'test').map(prep);
  const fresh = roundNet(trainCardNet(train, testSet, CNN_CONFIG).net);
  const a = flatNet(fresh), b = flatNet(model.net);
  assert.ok(a.every((v, i) => Math.abs(v - b[i]) <= 2e-5), 'run npm run train-cnn if this fails');
  assert.equal(evaluate(model.net, train).accuracy, 1);
  const ev = evaluate(model.net, testSet);
  assert.equal(Math.round(ev.accuracy * 24), 23);
  assert.deepEqual(ev.confusion, model.test.confusion);
  const miss = model.samples.filter((s) => s.split === 'test' && CLASSES[s.p.indexOf(Math.max(...s.p))] !== s.type);
  assert.deepEqual(miss.map((s) => [s.name, s.type]), [['Orbeetle V', 'grass']]);
  for (const s of model.samples) {
    const p = forward(model.net, prep(cards.find((c) => c.id === s.id)!).x).p;
    p.forEach((v, j) => assert.ok(Math.abs(v - s.p[j]) < 1e-4, s.id));
  }
});

test('average-colour baseline (no convolution) gets 22/24 test cards: colour carries most of the signal', () => {
  const train = cards.filter((c) => c.split === 'train').map(prep), testSet = cards.filter((c) => c.split === 'test');
  const { predictColor } = colorBaseline(train);
  const miss = testSet.filter((c) => CLASSES[predictColor(prep(c).x)] !== c.type).map((c) => c.name);
  assert.deepEqual(miss.sort(), ['Orbeetle V', 'Torkoal V']);
});
