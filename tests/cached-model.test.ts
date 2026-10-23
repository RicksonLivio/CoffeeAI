// The cached models in public/ must be what the library trains today.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  buildCachedMlp, buildCachedMlp2, buildCachedModel, CACHED_MLP2_PATH, CACHED_MLP_PATH, CACHED_MODEL_PATH,
} from '../src/lib/model.ts';

const read = (p: string) => JSON.parse(readFileSync(new URL(`../public/${p}`, import.meta.url), 'utf8'));

/** Deep equality with a tolerance for floats (Math.exp may differ in the last bits across engines). */
function close(a: unknown, b: unknown, path = '$'): void {
  if (typeof a === 'number' && typeof b === 'number') {
    assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)), `${path}: ${a} vs ${b}`);
  } else if (Array.isArray(a) && Array.isArray(b)) {
    assert.equal(a.length, b.length, `${path}: length`);
    a.forEach((v, i) => close(v, b[i], `${path}[${i}]`));
  } else if (a && b && typeof a === 'object' && typeof b === 'object') {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), `${path}: keys`);
    for (const k of Object.keys(a)) close((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`);
  } else {
    assert.equal(a, b, path);
  }
}

test('perceptron cache matches src/lib/perceptron.ts (run npm run export-model if this fails)', () => {
  assert.deepEqual(read(CACHED_MODEL_PATH), JSON.parse(JSON.stringify(buildCachedModel())));
});

test('MLP cache matches src/lib/mlp.ts (run npm run export-model if this fails)', () => {
  close(read(CACHED_MLP_PATH), JSON.parse(JSON.stringify(buildCachedMlp())));
});

test('two-hidden-layer MLP cache matches src/lib/deep.ts (run npm run export-model if this fails)', () => {
  close(read(CACHED_MLP2_PATH), JSON.parse(JSON.stringify(buildCachedMlp2())));
});
