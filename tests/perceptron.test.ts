// Checks every number written in the lesson (Learn with me → Perceptron).
// Run with: npm test  (Node ≥ 22.6, type stripping, no extra dependencies)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BLASTOISE, CHARIZARD, LESSON_ETA, LESSON_INIT, LESSON_SAMPLES,
  boundarySegment, classify, clean, fireRegion, train, trainStep,
} from '../src/lib/perceptron.ts';

test('epoch 1, sample 1 (Charizard): net = −0.5, y = 0, e = 1, weights → (100, 78, −0.5)', () => {
  const s = trainStep(LESSON_INIT, CHARIZARD, LESSON_ETA, 1, 0);
  assert.deepEqual(s.products, [0, 0, -0.5]);
  assert.equal(s.net, -0.5);
  assert.equal(s.y, 0);
  assert.equal(s.e, 1);
  assert.deepEqual(s.delta, { w1: 100, w2: 78, theta: -1 });
  assert.deepEqual(s.after, { w1: 100, w2: 78, theta: -0.5 });
});

test('epoch 1, sample 2 (Blastoise): net = 15600.5, y = 1, e = −1, weights → (22, −22, 0.5)', () => {
  const s = trainStep({ w1: 100, w2: 78, theta: -0.5 }, BLASTOISE, LESSON_ETA, 1, 1);
  assert.deepEqual(s.products, [7800, 7800, 0.5]);
  assert.equal(s.net, 15600.5);
  assert.equal(s.y, 1);
  assert.equal(s.e, -1);
  assert.deepEqual(s.after, { w1: 22, w2: -22, theta: 0.5 });
});

test('epoch 2 has no errors: Charizard net = 483.5, Blastoise net = −484.5', () => {
  const r = train(LESSON_SAMPLES, LESSON_INIT, LESSON_ETA);
  assert.equal(r.converged, true);
  assert.equal(r.steps.length, 4);
  assert.deepEqual(r.epochs.map((e) => e.errors), [2, 0]);
  assert.equal(r.steps[2].net, 483.5);
  assert.equal(r.steps[2].y, 1);
  assert.equal(r.steps[2].e, 0);
  assert.equal(r.steps[3].net, -484.5);
  assert.equal(r.steps[3].y, 0);
  assert.equal(r.steps[3].e, 0);
  assert.deepEqual(r.final, { w1: 22, w2: -22, theta: 0.5 });
});

test('weights do not change when the answer is right', () => {
  const s = trainStep({ w1: 22, w2: -22, theta: 0.5 }, CHARIZARD, 1, 2, 0);
  assert.deepEqual(s.after, s.before);
});

test('Venusaur (Speed 80, Defense 83) falls on the Water side: net = −66.5', () => {
  assert.deepEqual(classify({ w1: 22, w2: -22, theta: 0.5 }, 80, 83), { net: -66.5, y: 0 });
});

test('clean removes floating-point noise', () => {
  assert.equal(clean(0.1 + 0.2), 0.3);
  assert.ok(Object.is(clean(-0), 0));
});

test('final boundary x2 = x1 − 0.5/22 crosses the square; fire region is below it', () => {
  const w = { w1: 22, w2: -22, theta: 0.5 };
  const seg = boundarySegment(w, { min: 0, max: 150 });
  assert.ok(seg);
  for (const [x1, x2] of seg!) assert.ok(Math.abs(22 * x1 - 22 * x2 - 0.5) < 1e-6);
  const poly = fireRegion(w, { min: 0, max: 150 });
  for (const [x1, x2] of poly) assert.ok(22 * x1 - 22 * x2 - 0.5 >= -1e-6);
  assert.ok(poly.some(([x1, x2]) => x1 === 150 && x2 === 0));
});

test('with all weights zero there is no boundary', () => {
  assert.equal(boundarySegment({ w1: 0, w2: 0, theta: 0.5 }, { min: 0, max: 150 }), null);
});

test('a smaller learning rate also converges on this pair', () => {
  const r = train(LESSON_SAMPLES, LESSON_INIT, 0.1);
  assert.equal(r.converged, true);
  for (const s of LESSON_SAMPLES) assert.equal(classify(r.final, s.x1, s.x2).y, s.d);
});
