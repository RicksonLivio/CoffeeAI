import { test } from 'node:test';
import assert from 'node:assert/strict';
import { auc, report } from '../src/lib/metrics.ts';
import { softForward, STARTERS } from '../src/lib/mlp.ts';
import { deepForward } from '../src/lib/deep.ts';
import { buildCachedMlp, buildCachedMlp2 } from '../src/lib/model.ts';

test('AUC: perfect ranking = 1, reversed = 0, ties = 0.5', () => {
  assert.equal(auc([0.9, 0.8, 0.1], [true, true, false]), 1);
  assert.equal(auc([0.1, 0.2, 0.9], [true, true, false]), 0);
  assert.equal(auc([0.5, 0.5], [true, false]), 0.5);
});

test('report on a hand-made confusion: precision, recall, specificity and F1', () => {
  // 3 Fire predicted Fire; Water: 2 right, 1 called Fire; Grass: all called Water
  const probs: Record<string, number[]> = {
    charmander: [1, 0, 0], charmeleon: [1, 0, 0], charizard: [1, 0, 0],
    squirtle: [0, 1, 0], wartortle: [0, 1, 0], blastoise: [1, 0, 0],
    bulbasaur: [0, 1, 0], ivysaur: [0, 1, 0], venusaur: [0, 1, 0],
  };
  const r = report(STARTERS, (s) => probs[s.id]);
  assert.deepEqual(r.confusion, [[3, 0, 0], [1, 2, 0], [0, 3, 0]]);
  const [fire, water, grass] = r.perClass;
  assert.equal(fire.precision, 3 / 4);
  assert.equal(fire.recall, 1);
  assert.equal(water.precision, 2 / 5);
  assert.equal(water.recall, 2 / 3);
  assert.equal(water.specificity, 3 / 6);
  assert.equal(grass.recall, 0);
  assert.equal(grass.f1, 0);
  assert.equal(r.accuracy, 5 / 9);
  assert.ok(Math.abs(fire.f1 - (2 * 0.75 * 1) / 1.75) < 1e-12);
});

test('trained networks: every metric is 1 on the nine training Pokémon', () => {
  const m1 = buildCachedMlp();
  const m2 = buildCachedMlp2();
  for (const r of [report(STARTERS, (s) => softForward(m1.final, s.x1, s.x2).p), report(STARTERS, (s) => deepForward(m2.final, s.x1, s.x2).p)]) {
    assert.equal(r.accuracy, 1);
    for (const c of r.perClass) {
      assert.equal(c.precision, 1);
      assert.equal(c.recall, 1);
      assert.equal(c.specificity, 1);
      assert.equal(c.f1, 1);
      assert.equal(c.auc, 1);
    }
  }
});

test('initial network: accuracy 3/9, it calls every Pokémon the same type', () => {
  const m = buildCachedMlp();
  const r = report(STARTERS, (s) => softForward(m.init, s.x1, s.x2).p);
  assert.equal(r.accuracy, 3 / 9);
  assert.equal(r.confusion.filter((row) => row.some((v) => v === 3)).length, 3);
});
