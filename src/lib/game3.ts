/**
 * Pokédex mini-game, phase 3: the Pokédex gets a camera and reads the type from the card picture with the CNN of lesson 3.
 *
 * Part A, "the lens": three test cards the network never saw (one per type). For each one the player works out two
 * pieces of the CNN (pixels, a convolution window, the output size, max pooling, ReLU, the weight count), then loads
 * the Ball of the type with the highest softmax probability. All three are caught.
 * Part B, "the shortcut": Orbeetle V, a Grass full-art card covered by a red Pokémon. The CNN says Fire with ~100 %,
 * the Fire Ball fails, and the player follows the error back: loss, δ = p − d, pooling, the gradient of a filter weight
 * and its update (the lesson's 3 × 3 example), and finally why it failed: the network mostly reads the card colour.
 * Wrong options are the usual mistakes, so each one teaches something.
 */
import {
  BACK_DELTA, BACK_K, BACK_X, CLASSES, EDGE_PATCH, KERNELS, POOL_PATCH,
  conv2dGrid, decodePx, forward, kernelGrad, outSize, pool2dGrid, toTensor, type CachedCnn, type CardClass, type CardSample,
} from './cnn.ts';

export type Ball3 = CardClass;
export type Vec3 = [number, number, number];

/** Rounds to 2 decimals (and turns −0 into 0), the precision the game shows. */
export const r2 = (v: number) => {
  const r = Math.round(v * 100) / 100;
  return r === 0 ? 0 : r;
};
const r3 = (v: number) => {
  const r = Math.round(v * 1000) / 1000;
  return r === 0 ? 0 : r;
};

function rotate<T>(items: T[], k: number): T[] {
  const s = ((k % items.length) + items.length) % items.length;
  return [...items.slice(s), ...items.slice(0, s)];
}

// --------------------------------------------------------------------------- //
// Part A: the pieces of the CNN, one card at a time
// --------------------------------------------------------------------------- //

export type LensKey = 'pixels' | 'conv' | 'size' | 'pool' | 'relu' | 'params';

export interface LensQuestion {
  key: LensKey;
  answer: number;
  /** the right answer plus two common mistakes, rotated */
  options: number[];
}

/** The six questions of part A, each computed from src/lib/cnn.ts. */
export function lensQuestions(): Record<LensKey, LensQuestion> {
  const q = (key: LensKey, answer: number, wrong: number[], k: number): LensQuestion =>
    ({ key, answer, options: rotate([answer, ...wrong], k) });

  // 1. the card squeezed to 32 × 32 with 3 colours; mistakes: forgot the colours, or 32 × 3
  const pixels = 32 * 32 * 3;
  // 2. vertical-edge filter at window (0, 1): left column − right column; mistakes: added everything, flipped the sign
  const vk = KERNELS.vertical;
  const win = EDGE_PATCH.slice(0, 3).map((r) => r.slice(1, 4));
  const conv = conv2dGrid(EDGE_PATCH, vk)[0][1];
  const absSum = win.reduce((s, r, y) => s + r.reduce((t, v, x) => t + v * Math.abs(vk[y][x]), 0), 0);
  // 3. first layer: 32 − 5 + 1; mistakes: forgot the + 1, or thought the size stays ("same" needs padding)
  const size = outSize(32, 5, 0, 1);
  // 4. max pooling of the first block; mistakes: average and min pooling
  const pool = pool2dGrid(POOL_PATCH, 2, 2, 'max')[0][0];
  // 5. ReLU of the corner of the padded convolution (net = −18); mistakes: no change, absolute value
  const net = conv2dGrid(EDGE_PATCH, vk, 1, 1)[0][0];
  // 6. first-layer weights: 6 filters × (3 · 5 · 5) + 6 biases; mistakes: forgot the biases, one filter only
  const params = 6 * (3 * 5 * 5) + 6;

  return {
    pixels: q('pixels', pixels, [32 * 32, 32 * 3], 0),
    conv: q('conv', conv, [absSum, -conv], 1),
    size: q('size', size, [32 - 5, 32], 2),
    pool: q('pool', pool, [pool2dGrid(POOL_PATCH, 2, 2, 'avg')[0][0], pool2dGrid(POOL_PATCH, 2, 2, 'min')[0][0]], 1),
    relu: q('relu', Math.max(0, net), [net, Math.abs(net)], 2),
    params: q('params', params, [6 * (3 * 5 * 5), 3 * 5 * 5], 0),
  };
}

/** The window and the products of question 2, for the hint and the worked line. */
export function convWindow() {
  const win = EDGE_PATCH.slice(0, 3).map((r) => r.slice(1, 4));
  const left = win.reduce((s, r) => s + r[0], 0), right = win.reduce((s, r) => s + r[2], 0);
  return { win, left, right, net: left - right };
}

/** net of the padded corner (question 5). */
export const PADDED_CORNER = conv2dGrid(EDGE_PATCH, KERNELS.vertical, 1, 1)[0][0];

export interface LensEncounter {
  kind: 'lens';
  n: number;
  id: string;
  name: string;
  type: CardClass;
  file: string;
  /** softmax probabilities of the cached network, rounded to 2 decimals */
  p: Vec3;
  ball: Ball3;
  caught: boolean;
  questions: LensQuestion[];
}

/** One test card per type, with the CNN pieces each one teaches. */
export const LENS_CARDS: { name: string; keys: [LensKey, LensKey] }[] = [
  { name: 'Fennekin', keys: ['pixels', 'conv'] },
  { name: 'Totodile', keys: ['size', 'pool'] },
  { name: 'Rowlet', keys: ['relu', 'params'] },
];

const ballOf = (p: number[]): Ball3 => CLASSES[p.indexOf(Math.max(...p))];

/** Exact softmax probabilities of the cached network for one card (a real forward pass on its 32 × 32 pixels). */
function scan(model: CachedCnn, card: CardSample): number[] {
  return forward(model.net, toTensor(decodePx(card.px))).p;
}

function testCard(cards: CardSample[], pick: (c: CardSample) => boolean, what: string): CardSample {
  const c = cards.find((k) => k.split === 'test' && pick(k));
  if (!c) throw new Error(`test card ${what} not found`);
  return c;
}

export function buildLens(model: CachedCnn, cards: CardSample[]): LensEncounter[] {
  const qs = lensQuestions();
  return LENS_CARDS.map(({ name, keys }, i) => {
    const s = testCard(cards, (c) => c.name === name, name);
    const p = scan(model, s), ball = ballOf(p);
    return {
      kind: 'lens', n: i + 1, id: s.id, name: s.name, type: s.type, file: s.file,
      p: p.map(r2) as Vec3, ball, caught: ball === s.type, questions: keys.map((k) => qs[k]),
    };
  });
}

// --------------------------------------------------------------------------- //
// Part B: the card it gets wrong, and the error going back
// --------------------------------------------------------------------------- //

export type PoolChoice = 'winner' | 'all' | 'none';
export type ClueChoice = 'colour' | 'shape' | 'text';

/** Test cards the colour-only baseline (no convolution) gets right; tests/game3.test.ts recomputes it from cards32.json. */
export const COLOR_BASELINE_HITS = 22;
export const ETA_BY_HAND = 0.1;

export interface ShortcutEncounter {
  kind: 'shortcut';
  n: number;
  id: string;
  name: string;
  type: CardClass;
  file: string;
  p: Vec3;
  ball: Ball3;
  caught: boolean;
  /** exact probability the network gave the real type (≈ 8.8e−11 for Orbeetle V) */
  pTrue: number;
  /** L = −ln p_Grass, with the exact probability */
  loss: number;
  lossOptions: number[];
  /** δ = p − d, d = (0, 0, 1) */
  delta: Vec3;
  deltaOptions: Vec3[];
  poolOptions: PoolChoice[];
  /** the lesson's 3 × 3 example: ∂L/∂K[1][1] = Σ δ[i][j]·x[i + 1][j + 1] */
  grad: number;
  gradTerms: [number, number][];
  gradOptions: number[];
  k: number;
  kAfter: number;
  kOptions: number[];
  clueOptions: ClueChoice[];
  baselineHits: number;
  cnnHits: number;
  testCards: number;
}

export function buildShortcut(model: CachedCnn, cards: CardSample[], n: number): ShortcutEncounter {
  const tests = cards.filter((c) => c.split === 'test');
  const scans = new Map(tests.map((c) => [c.id, scan(model, c)]));
  const miss = tests.filter((c) => ballOf(scans.get(c.id)!) !== c.type);
  if (miss.length !== 1) throw new Error(`expected exactly one test miss, got ${miss.length}`);
  const s = miss[0], exact = scans.get(s.id)!;
  const t = CLASSES.indexOf(s.type);
  const loss = r2(-Math.log(exact[t]));
  const p = exact.map(r2) as Vec3;
  const delta = p.map((v, j) => r2(v - (j === t ? 1 : 0))) as Vec3;
  const g = kernelGrad(BACK_X, BACK_DELTA, 2);
  const grad = g[1][1];
  const gradTerms: [number, number][] = [0, 1].flatMap((i) => [0, 1].map((j) => [BACK_DELTA[i][j], BACK_X[i + 1][j + 1]] as [number, number]));
  const k = BACK_K[1][1];
  const kAfter = r3(k - ETA_BY_HAND * grad);
  return {
    kind: 'shortcut', n, id: s.id, name: s.name, type: s.type, file: s.file, p, ball: ballOf(exact), caught: ballOf(exact) === s.type,
    pTrue: exact[t],
    loss,
    // right; "used the predicted class" (−ln p_Fire ≈ 0); forgot the minus sign
    lossOptions: rotate([loss, r2(-Math.log(Math.max(...exact))), r2(-loss)], 1),
    delta,
    // right; d − p (sign flipped); forgot to subtract d
    deltaOptions: rotate([delta, delta.map((v) => r2(-v)) as Vec3, p], 2),
    poolOptions: ['all', 'winner', 'none'],
    grad,
    gradTerms,
    // right; sign flipped; the top-left weight's gradient (the wrong window pixels)
    gradOptions: rotate([grad, -grad, g[0][0]], 2),
    k,
    kAfter,
    // right; added η·g instead of subtracting; forgot η
    kOptions: rotate([kAfter, r3(k + ETA_BY_HAND * grad), r3(k - grad)], 1),
    clueOptions: ['shape', 'text', 'colour'],
    baselineHits: COLOR_BASELINE_HITS,
    cnnHits: tests.length - miss.length,
    testCards: tests.length,
  };
}

export type Encounter3 = LensEncounter | ShortcutEncounter;

/** The whole game: three test cards read with the lens, then the one the CNN gets wrong. */
export function buildGame3(model: CachedCnn, cards: CardSample[]): Encounter3[] {
  const lens = buildLens(model, cards);
  return [...lens, buildShortcut(model, cards, lens.length + 1)];
}

/** Lens: two CNN questions and the Ball. Shortcut: Ball, loss, δ, pooling, gradient, update, the clue. */
export function questionCount3(enc: Encounter3[]): number {
  return enc.reduce((s, c) => s + (c.kind === 'lens' ? c.questions.length + 1 : 7), 0);
}

export const XP_PER_ANSWER = 10;
