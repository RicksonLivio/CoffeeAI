// Exports the "cached models" used by the Learn-with-me sheet, like web_app/backend/modelos/ in the NFH app:
// the site never trains them on page load, it reads these files. The tests check that the files
// still match what src/lib/ produces, so code and cache cannot drift apart.
// Run with: npm run export-model
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildCachedMlp, buildCachedModel, CACHED_MLP_PATH, CACHED_MODEL_PATH } from '../src/lib/model.ts';

function write(path: string, obj: unknown) {
  const out = new URL(`../public/${path}`, import.meta.url);
  mkdirSync(new URL('.', out), { recursive: true });
  writeFileSync(out, JSON.stringify(obj, null, 2) + '\n');
}

const p = buildCachedModel();
write(CACHED_MODEL_PATH, p);
console.log(`wrote public/${CACHED_MODEL_PATH}: ${p.trace.length} steps, converged=${p.converged}, final=${JSON.stringify(p.final)}`);

const m = buildCachedMlp();
write(CACHED_MLP_PATH, m);
const last = m.curve[m.curve.length - 1];
console.log(`wrote public/${CACHED_MLP_PATH}: ${m.config.epochs} epochs, loss=${last.loss.toFixed(4)}, accuracy=${last.accuracy}`);
