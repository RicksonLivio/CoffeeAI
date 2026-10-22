// Exports the "cached model" used by the Learn-with-me sheet, like web_app/backend/modelos/ in the NFH app:
// the site never trains on page load, it reads this file. The test suite checks that the file
// still matches what src/lib/perceptron.ts produces, so the two cannot drift apart.
// Run with: npm run export-model
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildCachedModel, CACHED_MODEL_PATH } from '../src/lib/model.ts';

const model = buildCachedModel();
const out = new URL(`../public/${CACHED_MODEL_PATH}`, import.meta.url);
mkdirSync(new URL('.', out), { recursive: true });
writeFileSync(out, JSON.stringify(model, null, 2) + '\n');
console.log(`wrote public/${CACHED_MODEL_PATH}: ${model.trace.length} steps, converged=${model.converged}, final=${JSON.stringify(model.final)}`);
