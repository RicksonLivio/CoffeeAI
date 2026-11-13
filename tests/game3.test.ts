// Checks that the Pokédex mini-game, phase 3, plays exactly the lesson-3 numbers and the cached card model.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  buildGame3, convWindow, COLOR_BASELINE_HITS, PADDED_CORNER, questionCount3, type LensEncounter, type ShortcutEncounter,
} from '../src/lib/game3.ts';
import {
  CLASSES, CNN_CARDS_PATH, CNN_MODEL_PATH, colorBaseline, decodePx, forward, toTensor, type CachedCnn, type CardSample,
} from '../src/lib/cnn.ts';

const read = (p: string) => JSON.parse(readFileSync(new URL(`../public/${p}`, import.meta.url), 'utf8'));
const model: CachedCnn = read(CNN_MODEL_PATH);
const cards: CardSample[] = read(CNN_CARDS_PATH);
const game = buildGame3(model, cards);
const lens = game.filter((g): g is LensEncounter => g.kind === 'lens');
const [cut] = game.filter((g): g is ShortcutEncounter => g.kind === 'shortcut');

test('three test cards, one per type, then Orbeetle V', () => {
  assert.deepEqual(game.map((g) => g.name), ['Fennekin', 'Totodile', 'Rowlet', 'Orbeetle V']);
  assert.deepEqual(lens.map((g) => g.type), ['fire', 'water', 'grass']);
  for (const g of game) assert.equal(cards.find((c) => c.id === g.id)?.split, 'test', `${g.name} is a test card`);
  assert.equal(questionCount3(game), 16);
});

test('the three lens cards are caught with the Ball of their real type', () => {
  for (const g of lens) {
    assert.equal(g.ball, g.type);
    assert.ok(g.caught);
    // the cached probabilities match a fresh forward pass on the 32 × 32 card
    const c = cards.find((k) => k.id === g.id)!;
    const p = forward(model.net, toTensor(decodePx(c.px))).p;
    assert.equal(CLASSES[p.indexOf(Math.max(...p))], g.type);
  }
});

test('lens questions: 3072, 23, 28, 6, 0 and 456, each with 3 distinct options that include the answer', () => {
  const byKey = Object.fromEntries(lens.flatMap((g) => g.questions).map((q) => [q.key, q]));
  assert.deepEqual(
    ['pixels', 'conv', 'size', 'pool', 'relu', 'params'].map((k) => byKey[k].answer),
    [3072, 23, 28, 6, 0, 456],
  );
  for (const q of Object.values(byKey)) {
    assert.equal(q.options.length, 3, q.key);
    assert.equal(new Set(q.options).size, 3, q.key);
    assert.ok(q.options.includes(q.answer), q.key);
  }
  assert.deepEqual(byKey.conv.options.slice().sort((a, b) => a - b), [-23, 23, 31]);
  assert.deepEqual(byKey.pool.options.slice().sort((a, b) => a - b), [1, 3.75, 6]);
  assert.deepEqual(byKey.relu.options.slice().sort((a, b) => a - b), [-18, 0, 18]);
  assert.deepEqual(byKey.params.options.slice().sort((a, b) => a - b), [75, 450, 456]);
  assert.ok(new Set(Object.values(byKey).map((q) => q.options.indexOf(q.answer))).size > 1, 'right answer moves around');
  const w = convWindow();
  assert.deepEqual([w.left, w.right, w.net], [27, 4, 23]);
  assert.equal(PADDED_CORNER, -18);
});

test('Orbeetle V: the CNN loads the Fire Ball and it breaks free; L = 23.15, δ = (1, 0, −1)', () => {
  assert.equal(cut.name, 'Orbeetle V');
  assert.equal(cut.type, 'grass');
  assert.equal(cut.ball, 'fire');
  assert.equal(cut.caught, false);
  assert.deepEqual(cut.p, [1, 0, 0]);
  const c = cards.find((k) => k.id === cut.id)!;
  const p = forward(model.net, toTensor(decodePx(c.px))).p;
  assert.equal(cut.loss, Number((-Math.log(p[2])).toFixed(2)));
  assert.equal(cut.loss, 23.15);
  assert.deepEqual(cut.lossOptions.slice().sort((a, b) => a - b), [-23.15, 0, 23.15]);
  assert.deepEqual(cut.delta, [1, 0, -1]);
  assert.equal(cut.deltaOptions.length, 3);
  assert.equal(new Set(cut.deltaOptions.map((d) => d.join('|'))).size, 3);
  assert.ok(cut.poolOptions.includes('winner'));
});

test('the hand gradient and update are the lesson-3 numbers: −2.25 and 2 → 2.225', () => {
  assert.equal(cut.grad, -2.25);
  assert.deepEqual(cut.gradTerms, [[0.5, 1], [-1, 3], [0, 1], [0.25, 1]]);
  assert.equal(cut.gradTerms.reduce((s, [d, x]) => s + d * x, 0), cut.grad);
  assert.deepEqual(cut.gradOptions.slice().sort((a, b) => a - b), [-2.25, -1.25, 2.25]);
  assert.deepEqual([cut.k, cut.kAfter], [2, 2.225]);
  assert.deepEqual(cut.kOptions.slice().sort((a, b) => a - b), [1.775, 2.225, 4.25]);
});

test('the colour clue: the baseline with no convolution gets 22 of 24, the CNN 23', () => {
  const prep = (c: CardSample) => ({ x: toTensor(decodePx(c.px)), y: CLASSES.indexOf(c.type) });
  const { predictColor } = colorBaseline(cards.filter((c) => c.split === 'train').map(prep));
  const hits = cards.filter((c) => c.split === 'test').filter((c) => predictColor(prep(c).x) === CLASSES.indexOf(c.type)).length;
  assert.equal(hits, COLOR_BASELINE_HITS);
  assert.deepEqual([cut.baselineHits, cut.cnnHits, cut.testCards], [22, 23, 24]);
  // the baseline also calls Orbeetle Fire: its average colour is closer to the Fire cards
  const orb = cards.find((c) => c.id === cut.id)!;
  assert.equal(CLASSES[predictColor(prep(orb).x)], 'fire');
  assert.ok(cut.clueOptions.includes('colour'));
});
