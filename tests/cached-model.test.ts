// The cached model in public/ must be exactly what the library trains today.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildCachedModel, CACHED_MODEL_PATH } from '../src/lib/model.ts';

test('public cached model matches src/lib/perceptron.ts (run npm run export-model if this fails)', () => {
  const file = JSON.parse(readFileSync(new URL(`../public/${CACHED_MODEL_PATH}`, import.meta.url), 'utf8'));
  assert.deepEqual(file, JSON.parse(JSON.stringify(buildCachedModel())));
});
