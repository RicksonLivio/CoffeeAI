// The "adding smooth steps" simulation: each preset does what its text says.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { curveY, INITIAL, isolated, PRESETS } from '../src/lib/curves.ts';

const byId = (id: string) => PRESETS.find((p) => p.id === id)!.p;

test('initial values: a bump that isolates Grass (y ≈ 0.96 for Grass, ≈ 0.01 for Fire and Water)', () => {
  assert.equal(isolated(INITIAL).type, 'grass');
  assert.ok(Math.abs(curveY(INITIAL, -3) - 0.957) < 0.01);
  assert.ok(curveY(INITIAL, 22) < 0.02 && curveY(INITIAL, -22) < 0.02);
});

test('a single step only rises: Grass and Fire end up together, nothing is isolated', () => {
  const p = byId('one');
  assert.equal(isolated(p).type, null);
  for (let x = -40; x < 40; x++) assert.ok(curveY(p, x + 1) >= curveY(p, x));
});

test('hard step still isolates Grass; soft step flattens the bump below the 0.3 gap', () => {
  assert.equal(isolated(byId('hard')).type, 'grass');
  const soft = isolated(byId('soft'));
  assert.equal(soft.type, null);
  assert.ok(soft.gap > 0 && soft.gap < 0.3);
});

test('moving the bump to x = +22 turns it into a Fire detector', () => {
  assert.equal(isolated(byId('fire')).type, 'fire');
});
