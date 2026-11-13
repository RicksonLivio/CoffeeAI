// Checks that every number Misty says in the Lesson 3 audio comes from the lesson-3 code and the cached card model.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  BACK_DELTA, BACK_K, BACK_X, CLASSES, CNN_CARDS_PATH, CNN_CONFIG, CNN_MODEL_PATH, EDGE_PATCH, KERNELS, POOL_PATCH,
  colorBaseline, conv2dGrid, countParams, decodePx, initCardNet, kernelGrad, outSize, pool2dGrid, toTensor,
  type CachedCnn, type CardSample,
} from '../src/lib/cnn.ts';

const read = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const script = read('../src/content/misty-lesson3.json') as { sections: { id: string; title: [string, string]; text: [string[], string[]] }[] };
const cards: CardSample[] = read(`../public/${CNN_CARDS_PATH}`);
const model: CachedCnn = read(`../public/${CNN_MODEL_PATH}`);

const said = (id: string, lang: 0 | 1) => {
  const s = script.sections.find((q) => q.id === id);
  assert.ok(s, `section ${id}`);
  return s.text[lang].join(' ');
};
/** How the script writes a number: Portuguese decimal comma, "minus"/"menos" for negatives. */
const spoken = (v: number, lang: 0 | 1) => {
  const s = String(Math.abs(v)).replace('.', lang === 0 ? ',' : '.');
  return v < 0 ? `${lang === 0 ? 'menos' : 'minus'} ${s}` : s;
};
function mentions(id: string, values: number[]) {
  for (const lang of [0, 1] as const) {
    const text = said(id, lang);
    for (const v of values) {
      const w = spoken(v, lang);
      // whole token: "0,5" must not match inside "0,55", "23" not inside "2,23"
      const re = new RegExp(`(^|[^\\d,.])${w.replace(/[.]/g, '\\.')}(?![\\d]|[,.]\\d)`);
      assert.ok(re.test(text), `${id} [${lang ? 'en' : 'pt'}] should say ${w}`);
    }
  }
}
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);

test('both languages have the same sections and paragraph counts', () => {
  for (const s of script.sections) {
    assert.equal(s.text[0].length, s.text[1].length, s.id);
    assert.ok(s.title[0] && s.title[1]);
  }
  assert.equal(script.sections.length, 11);
});

test('the picture: 32 × 32 × 3 = 3072 numbers, the Charizard pixels, 9219 dense weights', () => {
  assert.equal(32 * 32 * 3, 3072);
  assert.equal(3 * 3072 + 3, 9219);
  const px = decodePx(cards.find((c) => c.id === 'charizard')!.px);
  const corner = [0, 1, 2].map((c) => px[c]);
  const mid = [0, 1, 2].map((c) => px[(14 * 32 + 16) * 3 + c]);
  assert.deepEqual(corner, [130, 130, 132]);
  assert.deepEqual(mid, [91, 47, 35]);
  assert.ok(mid[0] > mid[1] && mid[0] > mid[2], 'the middle pixel is mostly red');
  mentions('imagem', [32, 3, 3072, 255, 130, 132, 91, 47, 35, 0.5, -0.5, 100, 9219]);
});

test('vertical-edge filter: 27 − 26 = 1, 27 − 4 = 23, a 4 × 4 map with 18–23 in the middle and 1–5 outside', () => {
  const col = (x: number) => EDGE_PATCH.slice(0, 3).reduce((s, r) => s + r[x], 0);
  assert.deepEqual([col(0), col(2), col(1), col(3)], [27, 26, 27, 4]);
  const y = conv2dGrid(EDGE_PATCH, KERNELS.vertical);
  assert.equal(y[0][0], 1);
  assert.equal(y[0][1], 23);
  assert.equal(y.length, 4);
  const mids = y.flatMap((r) => [r[1], r[2]]), outs = y.flatMap((r) => [r[0], r[3]]);
  assert.deepEqual([Math.min(...mids), Math.max(...mids)], [18, 23]);
  assert.deepEqual([Math.min(...outs), Math.max(...outs)], [1, 5]);
  assert.deepEqual(KERNELS.vertical.map((r) => r[0]), [1, 1, 1]);
  assert.deepEqual(KERNELS.vertical.map((r) => r[2]), [-1, -1, -1]);
  mentions('convolucao', [3, 6, 9, 8, 27, 26, 1, 2, 4, 23, 18, 5]);
});

test('output size: 4, 6 with padding 1, 2 with stride 2, 3 with both; 32 − 5 + 1 = 28', () => {
  assert.equal(outSize(6, 3, 0, 1), 4);
  assert.equal(outSize(6, 3, 1, 1), 6);
  assert.equal(outSize(6, 3, 0, 2), 2);
  assert.equal(outSize(6, 3, 1, 2), 3);
  assert.equal(6 + 2 * 1, 8);
  assert.equal(outSize(32, 5, 0, 1), 28);
  mentions('tamanho', [4, 6, 3, 1, 2, 8, 5, 32, 28]);
});

test('pooling: first block 1, 3, 5, 6 → max 6, mean 3.75, min 1; blocks 6, 9, 7, 8; 28 → 14', () => {
  const block = [POOL_PATCH[0][0], POOL_PATCH[0][1], POOL_PATCH[1][0], POOL_PATCH[1][1]];
  assert.deepEqual([...block].sort((a, b) => a - b), [1, 3, 5, 6]);
  assert.deepEqual(pool2dGrid(POOL_PATCH, 2, 2, 'max').flat(), [6, 9, 7, 8]);
  assert.equal(pool2dGrid(POOL_PATCH, 2, 2, 'avg')[0][0], 3.75);
  assert.equal(pool2dGrid(POOL_PATCH, 2, 2, 'min')[0][0], 1);
  mentions('relu_pool', [0, 1, 3, 5, 6, 9, 7, 8, 3.75, 28, 14]);
});

test('architecture: 28 → 14 → 12 → 6, 288 numbers; 456 + 440 + 867 = 1763 weights', () => {
  assert.equal(outSize(14, 3, 0, 1), 12);
  assert.equal(6 * 6 * 8, 288);
  assert.equal(3 * 5 * 5, 75);
  assert.equal(6 * 3 * 3, 54);
  assert.equal(6 * 75 + 6, 456);
  assert.equal(8 * 54 + 8, 440);
  assert.equal(3 * 288 + 3, 867);
  assert.equal(countParams(initCardNet(1)), 1763);
  assert.equal(model.params, 1763);
  mentions('arquitetura', [32, 3, 6, 5, 28, 14, 8, 12, 288, 75, 456, 54, 440, 867, 1763, 9219]);
});

test('hand backprop: input, filter and error as said; ∂L/∂K = −1.25 (top left), −2.25 (bottom right); 2 → 2.225', () => {
  assert.deepEqual(BACK_X, [[1, 2, 0], [0, 1, 3], [2, 1, 1]]);
  assert.deepEqual(BACK_K, [[1, -1], [0, 2]]);
  assert.deepEqual(BACK_DELTA, [[0.5, -1], [0, 0.25]]);
  const g = kernelGrad(BACK_X, BACK_DELTA, 2);
  assert.equal(g[0][0], -1.25);
  assert.equal(g[1][1], -2.25);
  assert.equal(0.5 * 1 + -1 * 2 + 0 * 0 + 0.25 * 1, -1.25);
  assert.equal(0.5 * 1 + -1 * 3 + 0 * 1 + 0.25 * 1, -2.25);
  const updated = Number((BACK_K[1][1] - 0.1 * g[1][1]).toFixed(6));
  assert.equal(updated, 2.225);
  mentions('backprop', [3, 2, 1, 0, -1, 0.5, 0.25, -1.25, -2.25, 0.1, 0.225, 2.225]);
});

test('training: 93 cards (90 from 20 sets, 1999–2023), 31 per type, 69/24 split, the config and the curve', () => {
  const sets = cards.filter((c) => c.set !== 'lesson');
  assert.equal(cards.length, 93);
  assert.equal(sets.length, 90);
  assert.equal(new Set(sets.map((c) => c.set)).size, 20);
  for (const t of CLASSES) assert.equal(cards.filter((c) => c.type === t).length, 31);
  const train = cards.filter((c) => c.split === 'train'), testSet = cards.filter((c) => c.split === 'test');
  assert.deepEqual([train.length, testSet.length], [69, 24]);
  for (const t of CLASSES) assert.equal(testSet.filter((c) => c.type === t).length, 8);
  assert.deepEqual(CNN_CONFIG, { seed: 11, epochs: 40, lr: 0.002, momentum: 0.9 });
  assert.deepEqual(model.config, CNN_CONFIG);
  const e1 = model.curve.find((c) => c.epoch === 1)!, e5 = model.curve.find((c) => c.epoch === 5)!;
  assert.equal(Number(e1.loss.toFixed(2)), 0.93);
  assert.equal(Math.round(e5.testAcc * 24), 23);
  assert.equal(Math.round(model.train.accuracy * model.train.n), 69);
  assert.equal(Math.round(model.test.accuracy * model.test.n), 23);
  mentions('treino', [93, 90, 20, 1999, 2023, 31, 69, 24, 8, 2, 0.002, 0.9, 40, 11, 0.93, 23]);
});

test('the miss is Orbeetle (Grass, ~100 % Fire); the colour baseline gets 22/24, the CNN 23/24; Ekans and Koffing are Grass', () => {
  const miss = model.samples.filter((s) => s.split === 'test' && s.p.indexOf(Math.max(...s.p)) !== CLASSES.indexOf(s.type));
  assert.equal(miss.length, 1);
  assert.ok(miss[0].name.startsWith('Orbeetle'));
  assert.equal(miss[0].type, 'grass');
  assert.ok(miss[0].p[0] > 0.99);
  const prep = (c: CardSample) => ({ x: toTensor(decodePx(c.px)), y: CLASSES.indexOf(c.type) });
  const { predictColor } = colorBaseline(cards.filter((c) => c.split === 'train').map(prep));
  const hits = cards.filter((c) => c.split === 'test').filter((c) => predictColor(prep(c).x) === CLASSES.indexOf(c.type)).length;
  assert.equal(hits, 22);
  for (const n of ['Ekans', 'Koffing']) assert.equal(cards.find((c) => c.name === n)?.type, 'grass');
  mentions('honestidade', [100, 22, 24, 23]);
  for (const lang of [0, 1] as const) for (const n of ['Orbeetle', 'Ekans', 'Koffing']) assert.ok(said('honestidade', lang).includes(n));
});

test('the Pokédex section: 24 test cards, 32 × 32, 100 percent', () => {
  mentions('pokedex', [24, 32, 100]);
  assert.equal(sum(model.test.confusion.flat()), 24);
});

test('the audio metadata, once generated, matches the script', () => {
  let audio: Record<'pt' | 'en', { starts: number[]; checked: { paragraphs: number; number_errors: number; final_mp3_sections_ok: number } }>;
  try {
    audio = read('../src/content/misty-lesson3-audio.json');
  } catch {
    return; // not generated yet
  }
  const paragraphs = script.sections.reduce((s, q) => s + q.text[0].length, 0);
  for (const lang of ['pt', 'en'] as const) {
    if (!audio[lang]?.duration) continue; // that language is not generated yet
    assert.equal(audio[lang].starts.length, script.sections.length);
    assert.equal(audio[lang].checked.paragraphs, paragraphs);
    assert.equal(audio[lang].checked.number_errors, 0);
    assert.equal(audio[lang].checked.final_mp3_sections_ok, script.sections.length);
  }
});
